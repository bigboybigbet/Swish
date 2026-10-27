"""
Rating engine for the player_rapm variant: regularized adjusted plus-minus
(RAPM) at game level, with a box-score prior fitted jointly (the BPM / xRAPM
idea), exponential time decay and periodic refits.

Player i's impact, in points of team margin per 48 minutes on court (≈ the
per-100-possessions scale of RAPM at ~100 possessions a game), is

    beta_i(t) = x_i(t)' theta + u_i

  x_i(t)  box-score profile from games BEFORE t: decayed per-48 rates of
          PTS, 3PM, FGA, FTA, OREB, DREB, AST, STL, BLK, TOV, PF above the
          league average, shrunk toward 0 over box_k minutes; unknown-ness
          box_k / (minutes + box_k) (replacement level of new players);
          minutes per game (coaches' assessment); first / second NBA season
          indicators (development);
  theta   box-score weights (the BPM regression), fitted on the same decayed
          past games;
  u_i     plus-minus impact beyond the box profile, ridge-shrunk toward
          rho * (his u at the end of last season) with prior sd tau.

Observations from a finished game, with minute shares f_i = 5 min_i / team
minutes (sum 5 per side, f_i = share of the game on court):

  margin   M = hfa + sum_H f_i beta_i - sum_A f_j beta_j + c_H - c_A + e,
           var(e) = sigma^2                       (c = optional team effects)
  on/off   d_i = PM_i - f_i M_team
               = f_i (1 - f_i) (beta_i - sum_{k != i} f_k beta_k / (5 - f_i)) + e_i,
           var(e_i) = f_i (1 - f_i) sigma^2 / w_pm
           (on-court minus off-court margin rate, assuming teammates' minutes
           are spread in proportion to their shares and opponents do not
           depend on i; hfa, opponents and team effects cancel).

With phi = (hfa, theta, c, u) every refit solves the weighted ridge problem

    min  sum_rows 2^(-age_r / hl_days) (y_r - z_r' phi)^2 / var_r
         + sum_k (phi_k - mu_k)^2 / tau_k^2

over the games played so far (normal equations accumulated incrementally,
dense Cholesky on the players seen in the last `horizon` days).  The
posterior covariance (same matrix inverted) gives the predictive sd.

Pre-game, for the players who play (or the available roster):

    w_i        = 5 * em_i^a / sum_team em^a      (expected minutes em, EWMA)
    T_team     = sum_i w_i beta_i + c_team - Tbar   (Tbar: average pre-game
                                                  T of recent games)
    E[margin]  = hfa + T_home - T_away
    sd^2       = sigma^2 + v' Sigma v            (v = d E[margin] / d phi)
"""
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpotri

STATS = ["PTS", "FG3M", "FGA", "FTA", "OREB", "DREB", "AST", "STL", "BLK",
         "TOV", "PF"]
FEATURES = [f"{s}_48" for s in STATS] + ["UNK", "MPG", "EXP1", "EXP2"]
N_STATS, N_FEAT = len(STATS), len(FEATURES)
MAX_TEAMS = 40

PARAMS = dict(
    sigma=12.0,         # sd of a game margin around its expectation (points)
    cap=30.0,           # margins clipped to ±cap in the fit
    pm_cap=40.0,        # on/off residuals d_i clipped to ±pm_cap
    hl_days=550.0,      # half-life of a game's weight (calendar days)
    season_carry=1.0,   # extra weight factor applied to old games at a new season
    tau=2.5,            # prior sd of u (points / 48)
    rho=0.0,            # prior mean of u = rho * u at the end of last season
    w_pm=0.3,           # weight of the on/off rows relative to their variance
    theta_tau=0.5,      # prior sd of the box-score weights (around 0)
    hfa0=2.5,           # prior mean of the home advantage (points)
    hfa_tau=1.5,        # prior sd of the home advantage
    team_tau=0.0,       # prior sd of a team effect; 0 = off
    box_k=800.0,        # pseudo-minutes shrinking box rates toward average
    box_hl=60.0,        # half-life of box evidence (games played)
    min_hl=30.0,        # half-life of the expected-minutes EWMA (games)
    min_pow=1.5,        # pre-game weights ∝ expected minutes ** min_pow
    new_min=20.0,       # expected minutes of a player with no history
    refit_days=7,       # refit the ridge every refit_days days
    horizon=730,        # players seen in the last `horizon` days are solved for
)

# PARAMS were tuned on the dev window 2014-15 … 2016-17 (coordinate descent on
# walk-forward log loss); rho, season_carry, team_tau and refit_days had no
# effect there and stay at their neutral values.

_ARRAYS = ("S", "D", "em", "ea", "ngames", "ns", "known", "played", "team",
           "last_day", "u", "u_prev")


class PlayerRAPM:
    """Player ratings.  Feed games in date order: new_season() at a season
    boundary, refit() when due (maybe_refit), predict() / team_strength()
    before a game, update() after it."""

    def __init__(self, **params):
        self.p = p = {**PARAMS, **params}
        self.idx, self.ids, self.names = {}, [], []
        self.tidx = {}
        self.K = 1 + N_FEAT + MAX_TEAMS          # columns before the players
        cap = 512
        self.S = np.zeros((cap, N_STATS))        # decayed box sums
        self.D = np.zeros(cap)                   # decayed minutes
        self.em = np.zeros(cap)                  # expected minutes if he plays
        self.ea = np.zeros(cap)                  # expected minutes if he dresses
        self.ngames = np.zeros(cap)
        self.ns = np.zeros(cap)                  # completed seasons since debut
        self.known = np.zeros(cap, bool)         # debut seen in the data
        self.played = np.zeros(cap, bool)        # played this season
        self.team = np.zeros(cap, np.int64)
        self.last_day = np.full(cap, -10 ** 6)
        self.u = np.zeros(cap)                   # current solution
        self.u_prev = np.zeros(cap)              # u at the end of last season
        n = self.K + cap
        self.A = np.zeros((n, n))                # accumulated X'WX (inflated)
        self.b = np.zeros(n)                     # accumulated X'Wy (inflated)
        self.kappa = np.log(2) / p["hl_days"]
        self.ref_day = None
        self.L = np.zeros(N_STATS)               # league per-48 rates
        self.n_lg = 0
        self.hfa, self.theta = p["hfa0"], np.zeros(N_FEAT)
        self.ceff = np.zeros(MAX_TEAMS)
        self.bbar = 0.0                          # league minute-weighted beta
        self.sbar = 0.0                          # average pre-game side strength
        self.cols, self.pos, self.Sigma = None, None, None
        self.season, self.n_seasons, self.last_refit = None, 0, None
        self._a_min = 1 - 0.5 ** (1 / p["min_hl"])
        self._a_box = 0.5 ** (1 / p["box_hl"])

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def rows(self, players, names=None) -> np.ndarray:
        """Internal rows for player ids (unseen players are created)."""
        out = np.empty(len(players), np.int64)
        for k, pid in enumerate(players):
            j = self.idx.get(pid)
            out[k] = self._new(pid, None if names is None else names[k]) \
                if j is None else j
        return out

    def _new(self, pid, name):
        j = len(self.ids)
        if j == len(self.D):
            self._grow()
        debut = self.n_seasons > 1             # first data season: unknown vets
        self.idx[pid] = j
        self.ids.append(pid)
        self.names.append(name or str(pid))
        self.em[j] = self.ea[j] = self.p["new_min"]
        self.known[j] = debut
        return j

    def _grow(self):
        cap = len(self.D)
        for a in _ARRAYS:
            v = getattr(self, a)
            setattr(self, a, np.concatenate([v, np.zeros_like(v)]))
        self.last_day[cap:] = -10 ** 6
        n = self.K + 2 * cap
        A = np.zeros((n, n))
        A[:self.K + cap, :self.K + cap] = self.A
        self.A, self.b = A, np.r_[self.b, np.zeros(cap)]
        if self.pos is not None:
            self.pos = np.r_[self.pos, np.full(cap, -1)]

    def tcol(self, team) -> int:
        """Column of a team effect."""
        if team not in self.tidx:
            self.tidx[team] = min(len(self.tidx), MAX_TEAMS - 1)
        return 1 + N_FEAT + self.tidx[team]

    # ── ratings ──────────────────────────────────────────────────────────────
    def features(self, j) -> np.ndarray:
        """Box-score profile x_i (len(j) x N_FEAT) from games already played."""
        k = self.p["box_k"]
        D = self.D[j]
        dev = 48 * (self.S[j] - np.outer(D, self.L) / 48) / (D + k)[:, None]
        unk = k / (D + k)
        mpg = (self.em[j] - 24) / 12
        e1 = (self.known[j] & (self.ns[j] == 0)).astype(float)
        e2 = (self.known[j] & (self.ns[j] == 1)).astype(float)
        return np.column_stack([dev, unk, mpg, e1, e2])

    def beta(self, j, X=None) -> np.ndarray:
        """Player impact (points / 48 on court) = box part + plus-minus part."""
        X = self.features(j) if X is None else X
        return X @ self.theta + self.u[j]

    def weights(self, j, minutes=None, play_prob=None, dressed=False) -> np.ndarray:
        """Minute shares summing to 5.  Default: recent minutes when playing
        (em, or em-if-dressed `ea`) ** min_pow; play_prob scales them."""
        em = self.ea if dressed else self.em
        m = em[j] ** self.p["min_pow"] if minutes is None \
            else np.asarray(minutes, float)
        if play_prob is not None:
            m = m * np.asarray(play_prob, float)
        s = m.sum()
        return 5 * m / s if s > 0 else np.full(len(j), 5 / max(len(j), 1))

    # ── pre-game ─────────────────────────────────────────────────────────────
    def side(self, j, team=None, minutes=None, play_prob=None, dressed=False,
             with_var=False) -> dict:
        """Strength of one side from internal rows j (see team_strength)."""
        w = self.weights(j, minutes, play_prob, dressed)
        X = self.features(j)
        box, imp = float(w @ (X @ self.theta)), float(w @ self.u[j])
        tc = self.ceff[self.tidx[team]] if team in self.tidx else 0.0
        out = {"strength": box + imp + tc - self.sbar, "box": box,
               "impact": imp, "team": tc, "w": w, "j": j}
        if with_var:
            out["xw"] = w @ X
        return out

    def variance(self, h: dict, a: dict, home=None, away=None) -> float:
        """Posterior variance of E[margin] for sides h, a (from side(...,
        with_var=True)): v' Sigma v over the solved parameters plus tau^2 per
        unsolved player."""
        if self.Sigma is None:
            return 0.0
        K = self.K
        loc = [0, *range(1, 1 + N_FEAT)]
        val = [1.0, *(h["xw"] - a["xw"])]
        if home in self.tidx and away in self.tidx:
            loc += [self.tcol(home), self.tcol(away)]
            val += [1.0, -1.0]
        g = np.r_[np.asarray(loc), K + h["j"], K + a["j"]]
        v = np.r_[np.asarray(val), h["w"], -a["w"]]
        ps = self.pos[g]
        ok = ps >= 0
        var = float(v[~ok] ** 2 @ np.full((~ok).sum(), self.p["tau"] ** 2))
        vs, ps = v[ok], ps[ok]
        return var + float(vs @ self.Sigma[np.ix_(ps, ps)] @ vs)

    def team_strength(self, team, players, minutes=None, play_prob=None,
                      dressed=True) -> dict:
        """Strength (points vs an average team) of `team` with any set of
        available players.  Expected minutes default to each player's recent
        average rescaled to 240; leave out players ruled out, give doubtful /
        questionable ones a play_prob below 1.  dressed=True (live default)
        uses minutes-if-dressed, False minutes-if-he-plays."""
        return self.side(self.rows(players), team, minutes, play_prob, dressed)

    def predict(self, home, away, home_players, away_players, home_prob=None,
                away_prob=None, dressed=True) -> dict:
        """Expected home margin and predictive sd before tip-off."""
        h = self.side(self.rows(home_players), home, None, home_prob, dressed, True)
        a = self.side(self.rows(away_players), away, None, away_prob, dressed, True)
        sd = np.sqrt(self.p["sigma"] ** 2 + self.variance(h, a, home, away))
        return {"margin": self.hfa + h["strength"] - a["strength"],
                "sd": float(sd), "hfa": self.hfa,
                "home": {k: h[k] for k in ("strength", "box", "impact", "team")},
                "away": {k: a[k] for k in ("strength", "box", "impact", "team")}}

    # ── post-game ────────────────────────────────────────────────────────────
    def update(self, home, away, margin, jh, ja, min_h, min_a, pm_h, pm_a,
               st_h, st_a, day, unused_h=(), unused_a=()):
        """Learn from a finished game: add its margin row and on/off rows to
        the normal equations (solved at the next refit), then refresh box
        sums, expected minutes and counters.  jh / ja: rows of the players
        who played; min_*: minutes; pm_*: plus-minus; st_*: box lines
        (len x N_STATS, STATS order); unused_*: rows of players who dressed
        but did not play."""
        p = self.p
        if min_h.sum() <= 0 or min_a.sum() <= 0:
            return
        if self.ref_day is None:
            self.ref_day = day
        j = np.r_[jh, ja]
        nh, n = len(jh), len(jh) + len(ja)
        X = self.features(j)
        f = np.r_[5 * min_h / min_h.sum(), 5 * min_a / min_a.sum()]
        sgn = np.r_[np.ones(nh), -np.ones(n - nh)]
        th, ta = self.tcol(home), self.tcol(away)
        B = X @ self.theta + self.u[j]              # pre-game ratings
        raw = [self.weights(jh) @ B[:nh] + self.ceff[th - 1 - N_FEAT],
               self.weights(ja) @ B[nh:] + self.ceff[ta - 1 - N_FEAT]]

        # local columns: hfa, theta, team_h, team_a, players
        nc = 1 + N_FEAT + 2 + n
        Z = np.zeros((1 + n, nc))
        y = np.zeros(1 + n)
        Z[0, 0] = 1.0
        Z[0, 1:1 + N_FEAT] = (f * sgn) @ X
        Z[0, 1 + N_FEAT], Z[0, 2 + N_FEAT] = 1.0, -1.0
        Z[0, 3 + N_FEAT:] = f * sgn
        y[0] = np.clip(margin, -p["cap"], p["cap"])
        pm = np.r_[pm_h, pm_a]
        if p["w_pm"] > 0:
            for sl, mt in ((slice(0, nh), margin), (slice(nh, n), -margin)):
                fs = f[sl]
                m = len(fs)
                R = -fs[None, :] / (5 - fs)[:, None]
                R[np.diag_indices(m)] = 1.0
                fc = np.clip(fs, 0.02, 0.98)
                g = fc * (1 - fc)
                s = np.sqrt(p["w_pm"] * g)
                rr = slice(1 + sl.start, 1 + sl.stop)
                Z[rr, 1:1 + N_FEAT] = s[:, None] * (R @ X[sl])
                Z[rr, 3 + N_FEAT + sl.start:3 + N_FEAT + sl.stop] = s[:, None] * R
                d = np.clip(pm[sl] - fs * mt, -p["pm_cap"], p["pm_cap"])
                y[rr] = np.sqrt(p["w_pm"]) * d / np.sqrt(g)
        w = np.exp(self.kappa * (day - self.ref_day)) / p["sigma"] ** 2
        gc = np.r_[np.arange(1 + N_FEAT), th, ta, self.K + j]
        self.A[np.ix_(gc, gc)] += w * (Z.T @ Z)
        self.b[gc] += w * (Z.T @ y)

        # box sums, league rates, expected minutes, counters
        mins = np.r_[min_h, min_a]
        st = np.r_[st_h, st_a]
        ok = mins > 0
        jj = j[ok]
        self.S[jj] = self._a_box * self.S[jj] + st[ok]
        self.D[jj] = self._a_box * self.D[jj] + mins[ok]
        self.n_lg += 1
        rate = 1 / min(self.n_lg, 500)
        self.L += rate * (48 * st.sum(0) / mins.sum() - self.L)
        self.sbar += rate * (sum(raw) / 2 - self.sbar)
        self.em[jj] += self._a_min * (mins[ok] - self.em[jj])
        self.ea[jj] += self._a_min * (mins[ok] - self.ea[jj])
        for t, uu in ((home, unused_h), (away, unused_a)):
            if len(uu):
                self.ea[uu] -= self._a_min * self.ea[uu]
                self.team[uu], self.last_day[uu] = t, day
        self.ngames[jj] += 1
        self.played[jj] = True
        self.team[jh], self.team[ja] = home, away
        self.last_day[j] = day

    def maybe_refit(self, day, with_sd=True) -> bool:
        """Refit if refit_days have passed since the last refit."""
        if self.last_refit is None or day - self.last_refit >= self.p["refit_days"]:
            self.refit(day, with_sd)
            return True
        return False

    def refit(self, day, with_sd=True):
        """Solve the ridge problem with every game played so far (all of them
        before `day`), decayed to `day`."""
        p = self.p
        self.last_refit = day
        if self.ref_day is None:
            return
        m = len(self.ids)
        act = np.flatnonzero(self.last_day[:m] >= day - p["horizon"])
        K = self.K
        cols = np.r_[np.arange(K), K + act]
        lam = np.r_[1 / p["hfa_tau"] ** 2, np.full(N_FEAT, 1 / p["theta_tau"] ** 2),
                    np.full(MAX_TEAMS, 1 / p["team_tau"] ** 2 if p["team_tau"] > 0
                            else 1e8),
                    np.full(len(act), 1 / p["tau"] ** 2)]
        mu = np.r_[p["hfa0"], np.zeros(N_FEAT + MAX_TEAMS), p["rho"] * self.u_prev[act]]
        c = np.exp(-self.kappa * (day - self.ref_day))
        M = c * self.A[np.ix_(cols, cols)]
        M[np.diag_indices(len(cols))] += lam
        cf = cho_factor(M, lower=True, check_finite=False)
        phi = cho_solve(cf, c * self.b[cols] + lam * mu, check_finite=False)
        self.hfa, self.theta = phi[0], phi[1:1 + N_FEAT]
        self.ceff = phi[1 + N_FEAT:K]
        self.u[:m] = p["rho"] * self.u_prev[:m]          # unsolved: prior mean
        self.u[act] = phi[K:]
        Dw = self.D[act]
        self.bbar = float(Dw @ self.beta(act) / Dw.sum()) if Dw.sum() > 0 else 0.0
        self.cols = cols
        self.pos = np.full(len(self.D) + K, -1)
        self.pos[cols] = np.arange(len(cols))
        if with_sd:
            inv, info = dpotri(cf[0], lower=1)
            self.Sigma = np.tril(inv) + np.tril(inv, -1).T

    def new_season(self, season, day=None):
        """Offseason: count seasons since debut, remember u as next season's
        prior mean, discount old games by season_carry."""
        p = self.p
        if self.season is not None:
            m = len(self.ids)
            self.ns[:m] += self.played[:m]
            self.played[:m] = False
            self.u_prev[:m] = self.u[:m]
            if day is not None and self.ref_day is not None:
                c = p["season_carry"] * np.exp(-self.kappa * (day - self.ref_day))
                self.A *= c
                self.b *= c
                self.ref_day = day
        self.season = season
        self.n_seasons += 1

    # ── inspection / persistence ─────────────────────────────────────────────
    def leaderboard(self, n=10, min_games=20, active_since=None) -> list:
        """Top players by impact (points per 48 above the league-average
        player), split into box and plus-minus parts."""
        m = len(self.ids)
        ok = self.ngames[:m] >= min_games
        if active_since is not None:
            ok &= self.last_day[:m] >= active_since
        j = np.flatnonzero(ok)
        X = self.features(j)
        box = X @ self.theta
        B = box + self.u[j] - self.bbar
        sd = np.full(len(j), self.p["tau"])
        if self.Sigma is not None:
            ps = self.pos[self.K + j]
            sd[ps >= 0] = np.sqrt(np.diag(self.Sigma)[ps[ps >= 0]])
        o = np.argsort(-B)[:n]
        return [{"player_id": self.ids[k], "name": self.names[k],
                 "rating": round(float(B[i]), 2),
                 "box": round(float(box[i] - self.bbar), 2),
                 "pm": round(float(self.u[k]), 2),
                 "sd_pm": round(float(sd[i]), 2),
                 "exp_min": round(float(self.em[k]), 1),
                 "games": int(self.ngames[k]), "team": int(self.team[k])}
                for i, k in zip(o, j[o])]

    def coefficients(self) -> dict:
        """Box-score weights theta (points / 48 per unit of each feature)."""
        return dict(zip(FEATURES, np.round(self.theta, 4)))

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
    """Pack date-sorted one-row-per-game `games` (GAME_ID, GAME_DATE, SEASON,
    TEAM_ID_H, TEAM_ID_A, MARGIN) and player box rows (GAME_ID, TEAM_ID,
    PLAYER_ID, PLAYER_NAME, MIN, PM, COMMENT + STATS) into flat arrays:
    players with minutes > 0, and players who dressed but did not play."""
    import pandas as pd
    games = games.reset_index(drop=True)
    b, ptr = _pack(games, box[box["MIN"] > 0])
    dnp = box["COMMENT"].astype(str).str.startswith("DNP - Coach") & (box["MIN"] <= 0)
    u, uptr = _pack(games, box[dnp])
    day = (pd.to_datetime(games["GAME_DATE"]) - pd.Timestamp("2000-01-01")).dt.days
    return {
        "games": games, "ptr": ptr, "day": day.values,
        "season": games["SEASON"].values,
        "home": games["TEAM_ID_H"].values, "away": games["TEAM_ID_A"].values,
        "margin": games["MARGIN"].values.astype(float),
        "pid": b["PLAYER_ID"].values, "name": b["PLAYER_NAME"].values,
        "min": b["MIN"].values.astype(float), "pm": b["PM"].values.astype(float),
        "stats": b[STATS].values.astype(float),
        "uptr": uptr, "upid": u["PLAYER_ID"].values, "uname": u["PLAYER_NAME"].values,
    }


def replay(prep: dict, roster_games: int = 10, reg_min: float = 20.0,
           with_sd: bool = True, until_season: str = None, **params):
    """Chronological pass over every game.  Each output row holds the state
    BEFORE that game; the ridge is refit every refit_days from earlier games
    only, and all games of a day are predicted before any is learned from.
    Recorded per side:
      RP_H / RP_A          strength of the players who play (minutes > 0),
                           weighted by their expected minutes,
      RP_LIVE_H / _A       strength of the active roster (players who play +
                           dressed-but-unused), weighted by minutes-if-dressed:
                           what the live app can compute from inactive lists,
      RP_FULL_H / _A       full strength: the players who play plus every
                           rotation player (expected minutes >= reg_min) of
                           the last `roster_games` games still on the roster
                           (RP_DIFF - RP_FULL_DIFF is the effect of absences).
    Returns (DataFrame aligned with prep['games'], final engine)."""
    import pandas as pd
    from collections import deque
    eng = PlayerRAPM(**params)
    ptr, uptr = prep["ptr"], prep["uptr"]
    day, season = prep["day"], prep["season"]
    n = len(day)
    if until_season is not None and (season > until_season).any():
        n = int(np.argmax(season > until_season))      # stop after that season
    names = ("RP_H", "RP_A", "RP_HFA", "RP_VAR", "RP_IMP", "RP_BOX", "RP_TEAM",
             "RP_FULL_H", "RP_FULL_A", "RP_LIVE_H", "RP_LIVE_A")
    cols = {k: np.full(len(day), np.nan) for k in names}
    recent = {}                                       # team -> rows of last games
    starts = np.flatnonzero(np.r_[True, day[1:n] != day[:n - 1]])
    for s0, s1 in zip(starts, np.r_[starts[1:], n]):
        if season[s0] != eng.season:
            eng.new_season(season[s0], day[s0])
            eng.refit(day[s0], with_sd)
        else:
            eng.maybe_refit(day[s0], with_sd)
        seg = []
        for g in range(s0, s1):                       # pre-game, whole day
            a, m, z = ptr[2 * g], ptr[2 * g + 1], ptr[2 * g + 2]
            j = eng.rows(prep["pid"][a:z], prep["name"][a:z])
            ua, um, uz = uptr[2 * g], uptr[2 * g + 1], uptr[2 * g + 2]
            u = eng.rows(prep["upid"][ua:uz], prep["uname"][ua:uz])
            sides = []
            for jj, uu, t, sfx in ((j[:m - a], u[:um - ua], prep["home"][g], "H"),
                                   (j[m - a:], u[um - ua:], prep["away"][g], "A")):
                eng.tcol(t)
                past = recent.setdefault(t, deque(maxlen=roster_games))
                pool = np.unique(np.concatenate([jj, *past])).astype(np.int64)
                full = np.union1d(jj, pool[(eng.team[pool] == t)
                                           & (eng.em[pool] >= reg_min)])
                if len(jj) == 0:                      # no box score: regulars
                    jj = full
                s = eng.side(jj, t, with_var=with_sd)
                cols["RP_" + sfx][g] = s["strength"]
                cols["RP_LIVE_" + sfx][g] = eng.side(np.r_[jj, uu], t,
                                                     dressed=True)["strength"]
                cols["RP_FULL_" + sfx][g] = eng.side(full, t)["strength"]
                sides.append(s)
            hs, aws = sides
            cols["RP_IMP"][g] = hs["impact"] - aws["impact"]
            cols["RP_BOX"][g] = hs["box"] - aws["box"]
            cols["RP_TEAM"][g] = hs["team"] - aws["team"]
            cols["RP_VAR"][g] = eng.variance(hs, aws, prep["home"][g],
                                             prep["away"][g]) if with_sd else 0.0
            cols["RP_HFA"][g] = eng.hfa
            seg.append((g, a, m, z, j[:m - a], j[m - a:], u, um - ua))
        for g, a, m, z, jh, ja, u, nu in seg:         # post-game learning
            eng.update(prep["home"][g], prep["away"][g], prep["margin"][g],
                       jh, ja, prep["min"][a:m], prep["min"][m:z],
                       prep["pm"][a:m], prep["pm"][m:z],
                       prep["stats"][a:m], prep["stats"][m:z], day[g],
                       u[:nu], u[nu:])
            recent[prep["home"][g]].append(jh)
            recent[prep["away"][g]].append(ja)
    out = pd.DataFrame(cols, index=prep["games"].index)
    out["RP_DIFF"] = out["RP_HFA"] + out["RP_H"] - out["RP_A"]
    out["RP_FULL_DIFF"] = out["RP_HFA"] + out["RP_FULL_H"] - out["RP_FULL_A"]
    out["RP_LIVE_DIFF"] = out["RP_HFA"] + out["RP_LIVE_H"] - out["RP_LIVE_A"]
    out["RP_SD"] = np.sqrt(eng.p["sigma"] ** 2 + out["RP_VAR"])
    return out, eng
