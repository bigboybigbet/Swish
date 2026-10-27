"""
Rating engine for the player_elo variant: Elo for players, on the point scale.

Every player i carries
    r_i   impact rating (points of team margin per 48 on-court minutes),
    P_i   its variance: the Kalman "K".  New players start wide (fast moves),
          every game shrinks it (veterans move slowly), q per game and
          v_season per offseason widen it again,
    box_i box rating: gamma * decayed per-48 Game Score above the current
          league average, shrunk over box_prior minutes, plus a decaying
          replacement-level offset for players who debut,
    em_i  expected minutes if he plays (EWMA over games played),
    ea_i  expected minutes if he dresses (same EWMA, 0 for an unused game);
          the live app only knows the active roster, so it uses ea.

Pre-game, from the players available (R_i = r_i + box_i, Rbar = league
minute-weighted mean of R):

    w_i       = 5 * em_i^a / sum_team(em^a)              (shares, sum 5)
    T_team    = sum_i w_i (R_i - Rbar)       (points vs an average team)
    E[margin] = hfa + T_home - T_away
    sd        = sqrt(sigma^2 + sum_i w_i^2 P_i)

After the game, with the minutes actually played (h_i = 5 * min_i / team
minutes, negative for the away side):
  1. team margin, Kalman with a diagonal covariance:
       e = clip(M, ±cap) - E[margin | h],   S = sigma^2 + sum h^2 P
       r_i += P_i h_i e / S,   P_i -= (P_i h_i)^2 / S
     i.e. the error is split in proportion to minutes, with K_i = P_i / S;
  2. on-court plus-minus, within team: with f_i the share of the game on
     court, d_i = PM_i - f_i M_team has mean f_i (1 - f_i)(R_i - Rbar_-i)
     (Rbar_-i = teammates' minute-weighted mean), so
       r_i += P_i (d_i - E d_i) / (f_i (1 - f_i) P_i + sigma_pm^2)
     re-centred so the team's minute-weighted total is unchanged;
  3. box rating, expected minutes and counters are refreshed; P_i += q.

At a new season: r *= carry (partial regression toward the mean),
P += v_season (capped at p_max), box evidence *= box_carry.  Unseen players
start at a replacement-level prior (r0 < 0, box offset gamma * b0 < 0).
"""
import pickle

import numpy as np

# Box composite: Hollinger Game Score weights on the raw box line
BOX_STATS = ["PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB",
             "AST", "STL", "BLK", "TOV", "PF"]
GAME_SCORE = np.array([1.0, 0.4, -0.7, 0.4, -0.4, 0.7, 0.3,
                       0.7, 1.0, 0.7, -1.0, -0.4])

PARAMS = dict(
    sigma=12.0,       # sd of a game margin around its expectation (points)
    cap=30.0,         # margins are clipped to ±cap in the update
    hfa=2.4,          # home advantage (points), start value
    hfa_k=0.0,        # online learning rate of the home advantage (per game)
    r0=-3.0,          # replacement-level prior rating of an unseen player
    p0=2.0,           # prior variance of an unseen player's rating
    q=0.0015,         # variance added per game played (form drift)
    v_season=0.5,     # variance added at a new season
    p_max=6.0,        # cap on a player's rating variance
    carry=0.75,       # share of the impact rating kept at a new season
    young=0.0,        # rating bump at rollover in a player's first seasons
    sigma_pm=15.0,    # noise sd of the plus-minus observation; 0 = off
    gamma=0.4,        # points per Game Score unit (scale of the box prior)
    box_hl=80.0,      # box halflife in games played
    box_prior=600.0,  # prior minutes of the box rating (shrinkage)
    box_carry=1.0,    # box evidence kept at a new season
    b0=-6.0,          # box offset of an unseen player (Game Score units / 48)
    min_hl=20.0,      # expected-minutes halflife (games)
    min_pow=1.5,      # weights ∝ expected minutes ** min_pow
    new_min=14.0,     # expected minutes of a player with no history
    team_p0=0.0,      # prior variance of a team effect; 0 = off
    team_q=0.0,       # team-effect variance added per game
    team_carry=0.5,   # share of the team effect kept at a new season
)

_ARRAYS = ("r", "P", "n", "em", "ea", "bn", "bd", "off", "ns", "known", "team",
           "last_day")


class PlayerElo:
    """Player ratings.  Feed games in date order: new_season() at a season
    boundary, predict() / team_strength() before a game, update() after."""

    def __init__(self, **params):
        self.p = p = {**PARAMS, **params}
        self.idx, self.ids, self.names = {}, [], []
        cap = 1024
        self.r = np.zeros(cap)              # impact rating (pts / 48)
        self.P = np.zeros(cap)              # variance of r
        self.n = np.zeros(cap)              # games played
        self.em = np.zeros(cap)             # expected minutes if he plays
        self.ea = np.zeros(cap)             # expected minutes if he dresses
        self.bn = np.zeros(cap)             # decayed Game Score sum
        self.bd = np.zeros(cap)             # decayed minutes
        self.off = np.zeros(cap)            # replacement offset (pts / 48)
        self.ns = np.zeros(cap)             # completed seasons since debut
        self.known = np.zeros(cap, bool)    # debut seen in the data
        self.team = np.zeros(cap, np.int64)
        self.last_day = np.full(cap, -10 ** 6)
        self.tr, self.tP = {}, {}           # team effects and variances
        self.hfa = p["hfa"]
        self.xm = 0.0                       # league Game Score per 48
        self.rmean = 0.0                    # league minute-weighted mean of R
        self.season, self.n_seasons, self.n_games = None, 0, 0
        self._a_min = 1 - 0.5 ** (1 / p["min_hl"])
        self._a_box = 0.5 ** (1 / p["box_hl"])

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def rows(self, players, names=None) -> np.ndarray:
        """Internal rows for player ids; unseen players are created at the
        replacement-level prior."""
        out = np.empty(len(players), np.int64)
        for k, pid in enumerate(players):
            j = self.idx.get(pid)
            out[k] = self._new(pid, None if names is None else names[k]) \
                if j is None else j
        return out

    def _new(self, pid, name):
        j = len(self.ids)
        if j == len(self.r):
            for a in _ARRAYS:
                v = getattr(self, a)
                setattr(self, a, np.concatenate([v, np.zeros_like(v)]))
        p = self.p
        debut = self.n_seasons > 1          # first data season: unknown veterans
        self.idx[pid] = j
        self.ids.append(pid)
        self.names.append(name or str(pid))
        self.r[j] = p["r0"] if debut else 0.0
        self.P[j] = p["p0"] if debut else p["p_max"]
        self.n[j], self.bd[j], self.bn[j] = 0, 0.0, 0.0
        self.em[j] = self.ea[j] = p["new_min"] if debut else 2 * p["new_min"]
        self.off[j] = p["gamma"] * p["b0"] if debut else 0.0
        self.ns[j], self.known[j] = 0, debut
        self.team[j], self.last_day[j] = 0, -10 ** 6
        return j

    def box(self, j) -> np.ndarray:
        """Box part of the rating (points / 48): shrunk per-48 Game Score
        above the league mean, plus the decaying replacement offset."""
        k = self.p["box_prior"]
        return (self.p["gamma"] * (48 * self.bn[j] - self.bd[j] * self.xm)
                + k * self.off[j]) / (self.bd[j] + k)

    def rating(self, j) -> np.ndarray:
        """Full player rating R = r + box (points / 48)."""
        return self.r[j] + self.box(j)

    def weights(self, j, minutes=None, play_prob=None, dressed=False) -> np.ndarray:
        """Minute shares summing to 5 (five on court).  Without explicit
        minutes, each player's recent average (if he plays, or if he dresses
        when `dressed`) is used, raised to min_pow; play_prob (0..1 per
        player, from an injury report) scales it."""
        em = self.ea if dressed else self.em
        m = em[j] ** self.p["min_pow"] if minutes is None \
            else np.asarray(minutes, float)
        if play_prob is not None:
            m = m * np.asarray(play_prob, float)
        s = m.sum()
        return 5 * m / s if s > 0 else np.full(len(j), 5 / max(len(j), 1))

    # ── pre-game ─────────────────────────────────────────────────────────────
    def side(self, j, team=None, minutes=None, play_prob=None,
             dressed=False) -> dict:
        """Strength of one side from internal rows j (see team_strength)."""
        w = self.weights(j, minutes, play_prob, dressed)
        imp, box = float(w @ self.r[j]), float(w @ self.box(j))
        t = self.tr.get(team, 0.0)
        var = float(w ** 2 @ self.P[j] + self.tP.get(team, 0.0))
        # strength is centred on the league-average player (5 * rmean)
        return {"strength": imp + box + t - 5 * self.rmean, "impact": imp,
                "box": box, "team": t, "var": var}

    def team_strength(self, team, players, minutes=None, play_prob=None,
                      dressed=True) -> dict:
        """Strength (points vs an average team) of `team` with an arbitrary
        set of available players.  Expected minutes default to each player's
        recent average rescaled to 240 minutes; this is how injuries and rest
        enter: leave out players ruled out, give doubtful / questionable
        players a play_prob below 1.  `dressed=True` (live default) treats
        `players` as the active roster, False as the players known to play."""
        return self.side(self.rows(players), team, minutes, play_prob, dressed)

    def predict(self, home, away, home_players, away_players,
                home_prob=None, away_prob=None, dressed=True) -> dict:
        """Expected home margin and its predictive sd before tip-off."""
        h = self.team_strength(home, home_players, play_prob=home_prob, dressed=dressed)
        a = self.team_strength(away, away_players, play_prob=away_prob, dressed=dressed)
        sd = np.sqrt(self.p["sigma"] ** 2 + h["var"] + a["var"])
        return {"margin": self.hfa + h["strength"] - a["strength"],
                "sd": float(sd), "hfa": self.hfa, "home": h, "away": a}

    # ── post-game ────────────────────────────────────────────────────────────
    def update(self, home, away, margin, jh, ja, min_h, min_a, pm_h, pm_a,
               box_h, box_a, day=0, unused_h=(), unused_a=()):
        """Learn from a finished game.  jh / ja: rows of the players who
        played; min_*: their minutes; pm_*: plus-minus; box_*: Game Score
        (box_line @ GAME_SCORE); unused_*: rows of players who dressed but
        did not play.  Returns the expected margin given minutes."""
        p = self.p
        j = np.concatenate([jh, ja])
        mins = np.concatenate([min_h, min_a])
        sh, sa = 5 * min_h / min_h.sum(), 5 * min_a / min_a.sum()
        h = np.concatenate([sh, -sa])
        R = self.rating(j)
        th, ta = self.tr.get(home, 0.0), self.tr.get(away, 0.0)
        mu = self.hfa + h @ R + th - ta
        e = np.clip(margin, -p["cap"], p["cap"]) - mu

        # 1. team margin → players, in proportion to minutes and variance
        Pj = self.P[j]
        tph, tpa = self.tP.get(home, 0.0), self.tP.get(away, 0.0)
        S = p["sigma"] ** 2 + (h ** 2) @ Pj + tph + tpa
        dr = Pj * h * e / S
        Pj = Pj - (Pj * h) ** 2 / S
        if p["team_p0"] > 0:
            self.tr[home], self.tr[away] = th + tph * e / S, ta - tpa * e / S
            self.tP[home] = tph - tph ** 2 / S + p["team_q"]
            self.tP[away] = tpa - tpa ** 2 / S + p["team_q"]
        self.hfa += p["hfa_k"] * e

        # 2. on-court plus-minus, redistributed within each team
        if p["sigma_pm"] > 0:
            nh = len(jh)
            for sl, s, m_team, pm in ((slice(0, nh), sh, margin, pm_h),
                                      (slice(nh, None), sa, -margin, pm_a)):
                f = np.clip(s, 0.0, 0.98)          # share of the game on court
                Rt = R[sl]
                rbar = (s @ Rt - s * Rt) / np.maximum(5 - s, 1e-9)
                g = f * (1 - f)
                u = np.clip(pm - f * m_team - g * (Rt - rbar), -p["cap"], p["cap"])
                Pt = Pj[sl]
                k = Pt / (g * Pt + p["sigma_pm"] ** 2)
                d = k * u
                d -= (s @ d) / 5                   # team total unchanged
                dr[sl] += d
                Pj[sl] = Pt - k * g * Pt
        self.r[j] += dr
        self.P[j] = np.minimum(Pj + p["q"], p["p_max"])

        # 3. box rating, league mean, expected minutes, counters
        gs = np.concatenate([box_h, box_a])
        ok = mins > 0
        self.n_games += 1
        rate = 1 / min(self.n_games, 500)
        self.xm += rate * (48 * gs.sum() / max(mins.sum(), 1.0) - self.xm)
        self.rmean += rate * (np.abs(h) @ R / 10 - self.rmean)
        jj = j[ok]
        self.bn[jj] = self._a_box * self.bn[jj] + gs[ok]
        self.bd[jj] = self._a_box * self.bd[jj] + mins[ok]
        self.em[jj] += self._a_min * (mins[ok] - self.em[jj])
        self.ea[jj] += self._a_min * (mins[ok] - self.ea[jj])
        for t, u in ((home, unused_h), (away, unused_a)):
            if len(u):
                self.ea[u] -= self._a_min * self.ea[u]
                self.team[u] = t
        self.n[jj] += 1
        self.team[jh], self.team[ja] = home, away
        self.last_day[j] = day
        return mu

    def new_season(self, season):
        """Offseason: regress toward the mean, widen variances, age the box
        evidence."""
        p = self.p
        if self.season is not None:
            m = len(self.ids)
            self.r[:m] *= p["carry"]
            self.r[:m] += p["young"] * (self.known[:m] & (self.ns[:m] < 3))
            self.P[:m] = np.minimum(self.P[:m] + p["v_season"], p["p_max"])
            self.bn[:m] *= p["box_carry"]
            self.bd[:m] *= p["box_carry"]
            self.ns[:m] += self.n[:m] > 0
            self.tr = {t: v * p["team_carry"] for t, v in self.tr.items()}
            self.tP = {t: p["team_p0"] for t in self.tP}
        self.season = season
        self.n_seasons += 1

    def ensure_team(self, team):
        if self.p["team_p0"] > 0 and team not in self.tr:
            self.tr[team], self.tP[team] = 0.0, self.p["team_p0"]

    # ── inspection / persistence ─────────────────────────────────────────────
    def leaderboard(self, n=10, min_games=20, active_since=None) -> list:
        """Top players by full rating (points per 48 above the average
        player = impact + box)."""
        m = len(self.ids)
        ok = self.n[:m] >= min_games
        if active_since is not None:
            ok &= self.last_day[:m] >= active_since
        j = np.flatnonzero(ok)
        R, B = self.rating(j) - self.rmean, self.box(j) - self.rmean
        o = np.argsort(-R)[:n]
        return [{"player_id": self.ids[k], "name": self.names[k],
                 "rating": round(float(R[i]), 2),
                 "impact": round(float(self.r[k]), 2),
                 "box": round(float(B[i]), 2),
                 "sd": round(float(np.sqrt(self.P[k])), 2),
                 "exp_min": round(float(self.em[k]), 1),
                 "games": int(self.n[k]), "team": int(self.team[k])}
                for i, k in zip(o, j[o])]

    def roster(self, team, since_day=None) -> list:
        """Player ids whose most recent game was for `team`."""
        m = len(self.ids)
        ok = self.team[:m] == team
        if since_day is not None:
            ok &= self.last_day[:m] >= since_day
        return [self.ids[k] for k in np.flatnonzero(ok)]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        eng = cls.__new__(cls)
        with open(path, "rb") as f:
            eng.__dict__.update(pickle.load(f))
        return eng


# ── batch replay (research, and rebuilding the live state from history) ──────

def _pack(games, rows):
    """Box rows → (rows sorted by game, side; CSR pointer per game side)."""
    import pandas as pd
    gi = pd.Series(np.arange(len(games)), index=games["GAME_ID"].values)
    b = rows.assign(G=rows["GAME_ID"].map(gi)).dropna(subset=["G"])
    b["G"] = b["G"].astype(np.int64)
    home = games["TEAM_ID_H"].values[b["G"].values]
    away = games["TEAM_ID_A"].values[b["G"].values]
    b["SIDE"] = np.where(b["TEAM_ID"].values == home, 0,
                         np.where(b["TEAM_ID"].values == away, 1, -1))
    b = b[b["SIDE"] >= 0].sort_values(["G", "SIDE", "MIN"],
                                      ascending=[True, True, False])
    key = b["G"].values * 2 + b["SIDE"].values
    return b, np.searchsorted(key, np.arange(2 * len(games) + 1))


def prepare(games, box) -> dict:
    """Pack one-row-per-game `games` (GAME_ID, GAME_DATE, SEASON, TEAM_ID_H,
    TEAM_ID_A, MARGIN; date-sorted) and player box rows (GAME_ID, TEAM_ID,
    PLAYER_ID, PLAYER_NAME, MIN, PM, COMMENT + BOX_STATS) into flat arrays:
    players with minutes > 0, and players who dressed but did not play
    (DNP - Coach's Decision)."""
    import pandas as pd
    games = games.reset_index(drop=True)
    b, ptr = _pack(games, box[box["MIN"] > 0])
    dnp = box["COMMENT"].astype(str).str.startswith("DNP - Coach") & (box["MIN"] <= 0)
    u, uptr = _pack(games, box[dnp])
    day = (pd.to_datetime(games["GAME_DATE"]) - pd.Timestamp("2000-01-01")).dt.days
    return {
        "games": games, "ptr": ptr,
        "day": day.values, "season": games["SEASON"].values,
        "home": games["TEAM_ID_H"].values, "away": games["TEAM_ID_A"].values,
        "margin": games["MARGIN"].values.astype(float),
        "pid": b["PLAYER_ID"].values, "name": b["PLAYER_NAME"].values,
        "min": b["MIN"].values.astype(float), "pm": b["PM"].values.astype(float),
        "box": b[BOX_STATS].values.astype(float) @ GAME_SCORE,
        "uptr": uptr, "upid": u["PLAYER_ID"].values, "uname": u["PLAYER_NAME"].values,
    }


def team_prepare(games) -> dict:
    """Same packing with every team as one pseudo-player playing all 240
    minutes: replaying it (sigma_pm=0, gamma=0) is a plain Kalman team Elo
    on the same engine, the like-for-like comparison."""
    import pandas as pd
    games = games.reset_index(drop=True)
    n = len(games)
    day = (pd.to_datetime(games["GAME_DATE"]) - pd.Timestamp("2000-01-01")).dt.days
    pid = np.empty(2 * n, np.int64)
    pid[0::2], pid[1::2] = games["TEAM_ID_H"].values, games["TEAM_ID_A"].values
    margin = games["MARGIN"].values.astype(float)
    pm = np.empty(2 * n)
    pm[0::2], pm[1::2] = 5 * margin, -5 * margin
    return {
        "games": games, "ptr": np.arange(2 * n + 1), "day": day.values,
        "season": games["SEASON"].values,
        "home": games["TEAM_ID_H"].values, "away": games["TEAM_ID_A"].values,
        "margin": margin, "pid": pid, "name": pid.astype(str),
        "min": np.full(2 * n, 240.0), "pm": pm,
        "box": np.zeros(2 * n), "uptr": np.zeros(2 * n + 1, np.int64),
        "upid": np.zeros(0, np.int64), "uname": np.zeros(0, str),
    }


def replay(prep: dict, roster_games: int = 10, **params):
    """Chronological pass over every game.  Each output row holds the state
    BEFORE that game; all games of a day are predicted before any of them
    is learned from.  Recorded per side:
      PE_H / PE_A          strength of the players who play (minutes > 0),
      PE_LIVE_H / _A       strength of the active roster (players who play +
                           dressed-but-unused), weighted by minutes-if-dressed:
                           what the live app can compute from inactive lists,
      PE_FULL_H / _A       full strength: regulars of the last `roster_games`
                           games still on the roster (PE_DIFF - PE_FULL_DIFF
                           is the effect of absences).
    Returns (DataFrame aligned with prep['games'], final engine)."""
    import pandas as pd
    from collections import deque
    eng = PlayerElo(**params)
    ptr, uptr = prep["ptr"], prep["uptr"]
    day, season = prep["day"], prep["season"]
    n = len(day)
    cols = {k: np.zeros(n) for k in ("PE_H", "PE_A", "PE_HFA", "PE_VAR",
                                      "PE_IMP", "PE_BOX", "PE_TEAM",
                                      "PE_FULL_H", "PE_FULL_A",
                                      "PE_LIVE_H", "PE_LIVE_A")}
    recent = {}                                       # team -> rows of last games
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    for s0, s1 in zip(starts, np.r_[starts[1:], n]):
        if season[s0] != eng.season:
            eng.new_season(season[s0])
        seg = []
        for g in range(s0, s1):                       # pre-game, whole day
            a, m, z = ptr[2 * g], ptr[2 * g + 1], ptr[2 * g + 2]
            j = eng.rows(prep["pid"][a:z], prep["name"][a:z])
            ua, um, uz = uptr[2 * g], uptr[2 * g + 1], uptr[2 * g + 2]
            u = eng.rows(prep["upid"][ua:uz], prep["uname"][ua:uz])
            sides = []
            for jj, uu, t, sfx in ((j[:m - a], u[:um - ua], prep["home"][g], "H"),
                                   (j[m - a:], u[um - ua:], prep["away"][g], "A")):
                eng.ensure_team(t)
                s = eng.side(jj, t)
                cols["PE_LIVE_" + sfx][g] = eng.side(np.r_[jj, uu], t,
                                                     dressed=True)["strength"]
                past = recent.setdefault(t, deque(maxlen=roster_games))
                pool = np.unique(np.concatenate([jj, *past]))
                full = np.union1d(jj, pool[eng.team[pool] == t])
                cols["PE_" + sfx][g] = s["strength"]
                cols["PE_FULL_" + sfx][g] = eng.side(full, t)["strength"]
                sides.append(s)
            hs, aws = sides
            cols["PE_IMP"][g] = hs["impact"] - aws["impact"]
            cols["PE_BOX"][g] = hs["box"] - aws["box"]
            cols["PE_TEAM"][g] = hs["team"] - aws["team"]
            cols["PE_VAR"][g] = hs["var"] + aws["var"]
            cols["PE_HFA"][g] = eng.hfa
            seg.append((g, a, m, z, j[:m - a], j[m - a:], u, um - ua))
        for g, a, m, z, jh, ja, u, nu in seg:         # post-game learning
            eng.update(prep["home"][g], prep["away"][g], prep["margin"][g],
                       jh, ja, prep["min"][a:m], prep["min"][m:z],
                       prep["pm"][a:m], prep["pm"][m:z],
                       prep["box"][a:m], prep["box"][m:z], day[g],
                       u[:nu], u[nu:])
            recent[prep["home"][g]].append(jh)
            recent[prep["away"][g]].append(ja)
    out = pd.DataFrame(cols, index=prep["games"].index)
    out["PE_DIFF"] = out["PE_HFA"] + out["PE_H"] - out["PE_A"]
    out["PE_FULL_DIFF"] = out["PE_HFA"] + out["PE_FULL_H"] - out["PE_FULL_A"]
    out["PE_LIVE_DIFF"] = out["PE_HFA"] + out["PE_LIVE_H"] - out["PE_LIVE_A"]
    out["PE_SD"] = np.sqrt(eng.p["sigma"] ** 2 + out["PE_VAR"])
    return out, eng
