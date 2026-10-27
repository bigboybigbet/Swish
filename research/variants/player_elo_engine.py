"""
Rating engine for the player_elo variant: Elo for players, on the point scale.

Every player i carries
    r_i   impact rating, points of team margin per 48 on-court minutes above an
          average player (so a lineup of five +2 players is a +10 team),
    P_i   variance of r_i (the Kalman "K" — big for new players, small for
          veterans, widened at each new season),
    b_i   box-score rating: shrunk, decayed per-48 average of a linear box
          composite (Game Score style), centred on the league average,
    em_i  expected minutes, an EWMA of the minutes he played in past games.

A team's pre-game strength comes from the players who will play:

    w_i      = 5 * em_i / sum_team(em)                 (expected share, sum 5)
    R_i      = r_i + gamma * b_i                       (full player rating)
    T_team   = sum_i w_i * R_i + t_team                (t = small team effect)
    E[margin]= hfa + T_home - T_away
    sd       = sqrt(sigma^2 + sum_i w_i^2 P_i)

After the game (never before) three updates run, with the minutes actually
played (h_i = 5 * min_i / team minutes, negative for the away side):
  1. team margin (Kalman, diagonal):  e = clip(M, ±cap) - E[margin | h]
       r_i += P_i h_i e / S,  P_i -= (P_i h_i)^2 / S,  S = sigma^2 + sum h^2 P
     so the error is split in proportion to minutes and to each player's
     uncertainty (new players move fast, veterans slowly);
  2. on-court plus-minus (within team, zero-sum): with f_i the share of the
     game a player was on court, d_i = PM_i - f_i * M_team has expectation
     f_i (1 - f_i) (R_i - Rbar_team_without_i), so
       r_i += P_i (d_i - E[d_i]) / (f_i (1 - f_i) P_i + sigma_pm^2)
     re-centred so the team's minute-weighted total does not move;
  3. box score: b_i, em_i and games played are refreshed; P_i += q.

At a new season ratings regress toward the mean (r *= carry), variances
widen, old box evidence is down-weighted, young players get a development
bump.  Unseen players start at a replacement-level prior (r0, b0 < 0).
"""
import numpy as np

# Box composite: Hollinger Game Score weights on the raw box line.
BOX_STATS = ["PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB",
             "AST", "STL", "BLK", "TOV", "PF"]
BOX_W = np.array([1.0, 0.4, -0.7, 0.4, -0.4, 0.7, 0.3,
                  0.7, 1.0, 0.7, -1.0, -0.4])

PARAMS = dict(
    sigma=12.0,       # sd of a game margin around its expectation (points)
    cap=30.0,         # margins are clipped to ±cap in the update
    hfa=2.4,          # home advantage (points), start value
    hfa_k=0.0,        # online learning rate of the home advantage (per game)
    r0=-1.5,          # replacement-level prior rating for an unseen player
    p0=8.0,           # prior variance of an unseen player's rating
    q=0.05,           # variance added per game played (form drift)
    v_season=2.0,     # variance added at a new season
    p_max=12.0,       # cap on a player's rating variance
    carry=0.75,       # share of the impact rating kept at a new season
    young=0.0,        # rating bump at rollover for players in their first seasons
    sigma_pm=0.0,     # noise sd of the plus-minus observation; 0 = off
    gamma=0.0,        # weight of the box rating in R (points per box unit); 0 = off
    box_hl=40.0,      # box halflife in games played
    box_prior=300.0,  # prior minutes of the box rating (shrinkage)
    box_carry=0.6,    # box evidence kept at a new season
    b0=-2.0,          # box prior of an unseen player, relative to league average
    min_hl=6.0,       # expected-minutes halflife (games)
    new_min=14.0,     # expected minutes of a player with no history
    team_p0=0.0,      # prior variance of the team effect; 0 = off
    team_q=0.0,       # team-effect variance added per game
    team_carry=0.5,   # share of the team effect kept at a new season
)


class PlayerElo:
    """Player ratings.  Feed games in date order: predict() before a game,
    update() after it, new_season() at a season boundary."""

    def __init__(self, **params):
        self.p = {**PARAMS, **params}
        p = self.p
        self.idx, self.ids, self.names = {}, [], []
        cap = 1024
        self.r = np.zeros(cap)        # impact rating (pts / 48)
        self.P = np.zeros(cap)        # variance of r
        self.n = np.zeros(cap)        # games played
        self.em = np.zeros(cap)       # expected minutes
        self.bn = np.zeros(cap)       # decayed sum of minutes * box48
        self.bd = np.zeros(cap)       # decayed minutes
        self.bp = np.zeros(cap)       # box prior (absolute)
        self.ns = np.zeros(cap)       # seasons played (0 = debut season)
        self.known = np.zeros(cap, bool)  # debut seen in the data (age proxy)
        self.team = np.zeros(cap, np.int64)
        self.last_day = np.full(cap, -10 ** 6)
        self.tr, self.tP = {}, {}     # team effects and their variances
        self.hfa = p["hfa"]
        self.box_mean = 15.0          # league per-48 box average (running)
        self._n_games = 0
        self.season = None
        self.n_seasons = 0
        self._a_min = 1 - 0.5 ** (1 / p["min_hl"])
        self._a_box = 0.5 ** (1 / p["box_hl"])

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def rows(self, players, names=None, team=None) -> np.ndarray:
        """Internal rows for player ids, creating unseen players at the
        replacement-level prior."""
        out = np.empty(len(players), np.int64)
        for k, pid in enumerate(players):
            j = self.idx.get(pid)
            if j is None:
                j = self._new(pid, None if names is None else names[k])
            out[k] = j
        return out

    def _new(self, pid, name):
        j = len(self.ids)
        if j == len(self.r):
            for a in ("r", "P", "n", "em", "bn", "bd", "bp", "ns", "known",
                      "team", "last_day"):
                v = getattr(self, a)
                setattr(self, a, np.concatenate([v, np.zeros_like(v)]))
        p = self.p
        debut = self.n_seasons > 1        # first data season: unknown veterans
        self.idx[pid] = j
        self.ids.append(pid)
        self.names.append(name or str(pid))
        self.r[j] = p["r0"] if debut else 0.0
        self.P[j] = p["p0"] if debut else p["p_max"]
        self.n[j] = 0
        self.em[j] = p["new_min"] if debut else 2 * p["new_min"]
        self.bn[j], self.bd[j] = 0.0, 0.0
        self.bp[j] = self.box_mean + (p["b0"] if debut else 0.0)
        self.ns[j], self.known[j] = 0, debut
        self.last_day[j] = -10 ** 6
        return j

    def box(self, j) -> np.ndarray:
        """Box rating relative to the league average (box units / 48)."""
        k = self.p["box_prior"]
        return (self.bn[j] + k * self.bp[j]) / (self.bd[j] + k) - self.box_mean

    def rating(self, j) -> np.ndarray:
        """Full player rating R = r + gamma * box (points / 48)."""
        return self.r[j] + self.p["gamma"] * self.box(j)

    def weights(self, j, minutes=None) -> np.ndarray:
        """Minute shares that sum to 5 (five players on court)."""
        m = self.em[j] if minutes is None else np.asarray(minutes, float)
        s = m.sum()
        return 5 * m / s if s > 0 else np.full(len(j), 5 / max(len(j), 1))

    # ── pre-game ─────────────────────────────────────────────────────────────
    def team_strength(self, team, players, minutes=None) -> dict:
        """Strength (points vs an average team) of `team` with the given set
        of available players; expected minutes default to each player's
        recent average, rescaled to fill 240 minutes."""
        j = self.rows(players)
        w = self.weights(j, minutes)
        R = self.rating(j)
        t = self.tr.get(team, 0.0)
        return {"strength": float(w @ R + t), "var": float(w ** 2 @ self.P[j]
                + self.tP.get(team, 0.0)), "impact": float(w @ self.r[j]),
                "box": float(self.p["gamma"] * w @ self.box(j)), "team": t}

    def predict(self, home, away, home_players, away_players,
                home_minutes=None, away_minutes=None) -> dict:
        """Expected home margin and its predictive sd before tip-off."""
        h = self.team_strength(home, home_players, home_minutes)
        a = self.team_strength(away, away_players, away_minutes)
        mu = self.hfa + h["strength"] - a["strength"]
        sd = np.sqrt(self.p["sigma"] ** 2 + h["var"] + a["var"])
        return {"margin": mu, "sd": float(sd), "hfa": self.hfa,
                "home": h, "away": a}

    # ── post-game ────────────────────────────────────────────────────────────
    def update(self, home, away, margin, jh, ja, min_h, min_a, pm_h, pm_a,
               box_h, box_a, day=0):
        """Learn from a finished game.  jh / ja are rows (from rows()) of the
        players who played, min_* their minutes, pm_* their plus-minus and
        box_* their box composite (Game Score points, not per 48)."""
        p = self.p
        j = np.concatenate([jh, ja])
        mins = np.concatenate([min_h, min_a])
        sh = 5 * min_h / min_h.sum()
        sa = 5 * min_a / min_a.sum()
        h = np.concatenate([sh, -sa])
        R = self.rating(j)
        th, ta = self.tr.get(home, 0.0), self.tr.get(away, 0.0)
        mu = self.hfa + h @ R + th - ta
        e = np.clip(margin, -p["cap"], p["cap"]) - mu

        # 1. team margin → players in proportion to minutes and variance
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
            for sl, s, mt, pm in ((slice(0, nh), sh, margin, pm_h),
                                  (slice(nh, None), sa, -margin, pm_a)):
                f = np.clip(s, 0.0, 0.98)       # share of the game on court
                Rt, ww = R[sl], s
                rbar = (ww @ Rt - ww * Rt) / np.maximum(5 - ww, 1e-9)
                g = f * (1 - f)
                u = (pm - f * mt) - g * (Rt - rbar)
                u = np.clip(u, -p["cap"], p["cap"])
                Pt = Pj[sl]
                k = Pt / (g * Pt + p["sigma_pm"] ** 2)
                d = k * u
                d -= (ww @ d) / 5               # keep team total unchanged
                dr[sl] += d
                Pj[sl] = Pt - k * g * Pt

        self.r[j] += dr
        self.P[j] = np.minimum(Pj + p["q"], p["p_max"])

        # 3. box rating, expected minutes, counters
        ok = mins > 0
        box48 = np.concatenate([box_h, box_a]) * 48 / np.maximum(mins, 1e-9)
        g48 = np.concatenate([box_h, box_a]).sum() * 48 / max(mins.sum(), 1.0)
        self._n_games += 1
        self.box_mean += (g48 - self.box_mean) / min(self._n_games, 500)
        jj = j[ok]
        self.bn[jj] = self._a_box * self.bn[jj] + mins[ok] * box48[ok]
        self.bd[jj] = self._a_box * self.bd[jj] + mins[ok]
        self.em[jj] += self._a_min * (mins[ok] - self.em[jj])
        self.n[jj] += 1
        self.team[j[:len(jh)]], self.team[j[len(jh):]] = home, away
        self.last_day[j] = day
        return mu

    def new_season(self, season):
        """Offseason: regress toward the mean, widen variances, age box data."""
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

    # ── inspection ───────────────────────────────────────────────────────────
    def leaderboard(self, n=10, min_games=20, active_since=None) -> list:
        """Top players by full rating R (points per 48 above average)."""
        m = len(self.ids)
        j = np.arange(m)
        ok = self.n[:m] >= min_games
        if active_since is not None:
            ok &= self.last_day[:m] >= active_since
        j = j[ok]
        R = self.rating(j)
        o = np.argsort(-R)[:n]
        return [{"player_id": self.ids[k], "name": self.names[k],
                 "rating": round(float(R[i]), 2),
                 "impact": round(float(self.r[k]), 2),
                 "box": round(float(self.p["gamma"] * self.box(np.array([k]))[0]), 2),
                 "sd": round(float(np.sqrt(self.P[k])), 2),
                 "exp_min": round(float(self.em[k]), 1),
                 "games": int(self.n[k]), "team": int(self.team[k])}
                for i, k in zip(o, j[o])]

    def roster(self, team, since_day=None) -> list:
        """Players whose most recent game was for `team`."""
        m = len(self.ids)
        ok = self.team[:m] == team
        if since_day is not None:
            ok &= self.last_day[:m] >= since_day
        return [self.ids[k] for k in np.flatnonzero(ok)]


# ── batch replay (research + rebuilding the live state from history) ─────────

def prepare(games, box) -> dict:
    """Pack one-row-per-game `games` (GAME_ID, GAME_DATE, SEASON, TEAM_ID_H,
    TEAM_ID_A, MARGIN; date-sorted) and player box rows (GAME_ID, TEAM_ID,
    PLAYER_ID, PLAYER_NAME, MIN, PM + BOX_STATS) into flat arrays.  Only
    players with minutes > 0 are kept."""
    import pandas as pd
    games = games.reset_index(drop=True)
    gi = pd.Series(np.arange(len(games)), index=games["GAME_ID"].values)
    b = box[box["MIN"] > 0].copy()
    b["G"] = b["GAME_ID"].map(gi)
    b = b.dropna(subset=["G"])
    b["G"] = b["G"].astype(np.int64)
    home = games["TEAM_ID_H"].values[b["G"].values]
    away = games["TEAM_ID_A"].values[b["G"].values]
    b["SIDE"] = np.where(b["TEAM_ID"].values == home, 0,
                         np.where(b["TEAM_ID"].values == away, 1, -1))
    b = b[b["SIDE"] >= 0].sort_values(["G", "SIDE", "MIN"],
                                      ascending=[True, True, False])
    key = b["G"].values * 2 + b["SIDE"].values
    ptr = np.searchsorted(key, np.arange(2 * len(games) + 1))
    day = (pd.to_datetime(games["GAME_DATE"]) - pd.Timestamp("2000-01-01")).dt.days
    return {
        "games": games, "ptr": ptr, "day": day.values,
        "season": games["SEASON"].values,
        "home": games["TEAM_ID_H"].values, "away": games["TEAM_ID_A"].values,
        "margin": games["MARGIN"].values.astype(float),
        "pid": b["PLAYER_ID"].values, "name": b["PLAYER_NAME"].values,
        "min": b["MIN"].values.astype(float), "pm": b["PM"].values.astype(float),
        "box": b[BOX_STATS].values.astype(float) @ BOX_W,
    }


def replay(prep: dict, **params):
    """Chronological pass over every game.  Each output row holds the state
    BEFORE that game; all games of a day are predicted before any of them
    is learned from.  Returns (DataFrame aligned with prep['games'], engine)."""
    import pandas as pd
    eng = PlayerElo(**params)
    gamma = eng.p["gamma"]
    ptr, day, season = prep["ptr"], prep["day"], prep["season"]
    n = len(day)
    cols = {k: np.zeros(n) for k in ("PE_H", "PE_A", "PE_HFA", "PE_VAR",
                                      "PE_IMP", "PE_BOX", "PE_TEAM", "PE_NEW")}
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]
    seg = []
    for s0, s1 in zip(starts, ends):
        if season[s0] != eng.season:
            eng.new_season(season[s0])
        seg.clear()
        for g in range(s0, s1):                       # pre-game, whole day
            a, m, z = ptr[2 * g], ptr[2 * g + 1], ptr[2 * g + 2]
            j = eng.rows(prep["pid"][a:z], prep["name"][a:z])
            jh, ja = j[:m - a], j[m - a:]
            hm, aw = prep["home"][g], prep["away"][g]
            eng.ensure_team(hm), eng.ensure_team(aw)
            wh, wa = eng.weights(jh), eng.weights(ja)
            bh, ba = eng.box(jh), eng.box(ja)
            ih, ia = wh @ eng.r[jh], wa @ eng.r[ja]
            th, ta = eng.tr.get(hm, 0.0), eng.tr.get(aw, 0.0)
            cols["PE_IMP"][g] = ih - ia
            cols["PE_BOX"][g] = gamma * (wh @ bh - wa @ ba)
            cols["PE_TEAM"][g] = th - ta
            cols["PE_H"][g] = ih + gamma * wh @ bh + th
            cols["PE_A"][g] = ia + gamma * wa @ ba + ta
            cols["PE_HFA"][g] = eng.hfa
            cols["PE_VAR"][g] = (wh ** 2 @ eng.P[jh] + wa ** 2 @ eng.P[ja]
                                 + eng.tP.get(hm, 0.0) + eng.tP.get(aw, 0.0))
            cols["PE_NEW"][g] = (wh @ (eng.n[jh] < 10) - wa @ (eng.n[ja] < 10)) / 5
            seg.append((g, a, m, z, jh, ja))
        for g, a, m, z, jh, ja in seg:                # post-game learning
            eng.update(prep["home"][g], prep["away"][g], prep["margin"][g],
                       jh, ja, prep["min"][a:m], prep["min"][m:z],
                       prep["pm"][a:m], prep["pm"][m:z],
                       prep["box"][a:m], prep["box"][m:z], day[g])
    out = pd.DataFrame(cols, index=prep["games"].index)
    out["PE_DIFF"] = out["PE_HFA"] + out["PE_H"] - out["PE_A"]
    out["PE_SD"] = np.sqrt(eng.p["sigma"] ** 2 + out["PE_VAR"])
    return out, eng


def team_prepare(games) -> dict:
    """Same packing with every team as one pseudo-player playing all 240
    minutes: replaying it gives a plain (Kalman) team Elo on the same engine,
    used as the like-for-like comparison."""
    games = games.reset_index(drop=True)
    n = len(games)
    import pandas as pd
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
        "min": np.full(2 * n, 240.0), "pm": pm, "box": np.zeros(2 * n),
    }
