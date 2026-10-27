"""
Rating engine for the kalman variant: team strength as a linear-Gaussian
state-space model, filtered with the exact (joint, full-covariance) Kalman
filter.

State x, one vector for the whole league:
  net mode   [r_1 .. r_N, home, b2b, lineup]
  od  mode   [o_1 .. o_N, d_1 .. d_N, home, b2b, lineup, mu]
r = points per game better than an average team (o = scoring, d = points
prevented, so r = o + d); home = home-court edge; b2b = points lost on the
2nd night of a back-to-back; lineup = points per unit of the lineup-strength
covariate L (who is active tonight, see kalman.py); mu = league points per
team-game (od only).

Dynamics.  Between game days (Δ days)       x ← x + w,  w ~ N(0, Δ·diag(q))
           At the first game of a season     r ← ρ·r,   P_rr ← ρ² P_rr + v_season·I
Observation (one game, scalar rows)
           margin = r_h − r_a + home·(1−neutral) + b2b·(B_a − B_h)
                    + lineup·(L_h − L_a) + ε_m,                 ε_m ~ N(0, σ²)
     od also total = 2mu + (o_h − d_h) + (o_a − d_a) + ε_t,    ε_t ~ N(0, σ_t²)
(margin and total are the rotation of the two team scores, so their noises
are independent and two sequential scalar updates are the exact update.)

Pre-game: margin ~ N(μ, s²) with μ = h'x, s² = h'Ph + σ², so
          P(home win) = Φ(μ / s).
All games of one day are predicted before any of them updates the state.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

VALID_PREFIXES = ("002", "004", "005")       # regular season, playoffs, play-in
EPOCH = pd.Timestamp("2000-01-01")

# Orlando bubble (no crowd, no travel) and the 2023 in-season-tournament
# semifinals in Las Vegas: neutral courts known before tip-off.
NEUTRAL_WINDOWS = [("2020-07-30", "2020-10-11"), ("2023-12-07", "2023-12-07")]

PARAMS = dict(
    mode="net",          # "net" or "od"
    sigma=12.0,          # sd of a game margin around its expectation (points)
    q=0.02,              # team-strength variance added per day (points²/day)
    v0=30.0,             # team variance at the very first game (2010-11)
    rho=0.75,            # share of a team's strength kept over the summer
    v_season=6.0,        # variance added to every team at a new season
    home0=3.0,           # prior home edge (points)
    v_home0=1.0,         # prior variance of the home edge
    q_home=2e-4,         # home-edge variance added per day
    b2b0=1.0, v_b2b0=1.0, q_b2b=0.0,          # back-to-back penalty
    lineup0=0.0, v_lineup0=0.0, q_lineup=0.0,  # lineup covariate coefficient
    clip=np.inf,         # innovations clipped at clip·s (robust update)
    sigma_po=1.0,        # playoff margin sd as a multiple of sigma
    v_team_home=0.0,     # prior variance of per-team home edges (0 = none)
    adapt=0.0,           # weight of each game in the online estimate of σ²
                         # (E[e²] = h'Ph + σ²; 0 keeps σ fixed)
    # od mode only
    sigma_t=17.0,        # sd of the game total (points)
    mu0=100.0, v_mu0=25.0, q_mu=0.01,         # league scoring level per team
    off_share=0.5,       # share of the net-strength variances given to offense
    rho_off=None, rho_def=None,               # per-side reversion (None → rho)
)


# ── games ─────────────────────────────────────────────────────────────────────

def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs (leaguegamefinder format) → one row per game in date
    order: teams, points, rest, neutral flag, result."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(VALID_PREFIXES)].dropna(subset=["PTS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    rest = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    df["B2B"] = (rest == 1).astype(int)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    cols = ["GAME_ID", "TEAM_ID", "PTS", "B2B"]
    h = df[df["IS_HOME"]][cols + ["GAME_DATE", "SEASON_YEAR"]]
    a = df[~df["IS_HOME"]][cols]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON", "TEAM_ID_H": "H", "TEAM_ID_A": "A"})
    g["MARGIN"] = (g["PTS_H"] - g["PTS_A"]).astype(float)
    g["RESULT"] = (g["MARGIN"] > 0).astype(int)
    g["PLAYOFF"] = (g["GAME_ID"].str[:3] == "004").astype(int)
    g["NEUTRAL"] = 0
    for lo, hi in NEUTRAL_WINDOWS:
        g.loc[g["GAME_DATE"].between(lo, hi), "NEUTRAL"] = 1
    g["DAY"] = (g["GAME_DATE"] - EPOCH).dt.days
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ── filter ────────────────────────────────────────────────────────────────────

class KalmanRatings:
    """Joint Kalman filter over every team's strength.

    Live use:  roll_season(season) when a new season starts, advance(day)
    before each game day, predict(...) for every game of the day, then
    update(...) with each final score.
    """

    def __init__(self, teams, n_lineup=1, **params):
        p = {**PARAMS, **params}
        self.p = p
        self.od = p["mode"] == "od"
        self.teams = list(teams)
        self.ix = {t: i for i, t in enumerate(self.teams)}
        n = self.n = len(self.teams)
        nt = 2 * n if self.od else n                     # team-block size
        self.g_home, self.g_b2b = nt, nt + 1
        self.g_lin = list(range(nt + 2, nt + 2 + n_lineup))
        self.g_mu = nt + 2 + n_lineup if self.od else None
        m = nt + 2 + n_lineup + int(self.od)
        so = p["off_share"]
        team_split = np.r_[np.full(n, so), np.full(n, 1 - so)] if self.od else np.ones(n)
        ones = np.ones(n_lineup)

        self.x = np.zeros(m)
        self.x[[self.g_home, self.g_b2b]] = p["home0"], p["b2b0"]
        self.x[self.g_lin] = np.broadcast_to(p["lineup0"], (n_lineup,))
        var0 = np.r_[p["v0"] * team_split, p["v_home0"], p["v_b2b0"],
                     np.broadcast_to(p["v_lineup0"], (n_lineup,))]
        qd = np.r_[p["q"] * team_split, p["q_home"], p["q_b2b"], p["q_lineup"] * ones]
        if self.od:
            self.x[self.g_mu] = p["mu0"]
            var0, qd = np.r_[var0, p["v_mu0"]], np.r_[qd, p["q_mu"]]
        self.g_th = []
        if p["v_team_home"] > 0:                 # team-specific home edges
            self.g_th = list(range(m, m + n))
            self.x = np.r_[self.x, np.zeros(n)]
            var0, qd = np.r_[var0, np.full(n, p["v_team_home"])], np.r_[qd, np.zeros(n)]
        self.P = np.diag(var0)
        self.qd = qd
        self.split = team_split
        self.nt = nt
        rho_o = p["rho"] if p["rho_off"] is None else p["rho_off"]
        rho_d = p["rho"] if p["rho_def"] is None else p["rho_def"]
        self.rho_vec = (np.r_[np.full(n, rho_o), np.full(n, rho_d)] if self.od
                        else np.full(n, p["rho"]))
        self.day = None
        self.season = None
        self.s2 = p["sigma"] ** 2               # current margin noise variance

    # ── time update ──
    def roll_season(self, season: str):
        """Summer: partial mean reversion of every team + extra variance."""
        if self.season is not None and season != self.season:
            k = self.nt
            R = np.ones(len(self.x))
            R[:k] = self.rho_vec
            self.x *= R
            self.P *= np.outer(R, R)
            self.P[np.arange(k), np.arange(k)] += self.p["v_season"] * self.split
            self.day = None                   # no daily drift over the summer
        self.season = season

    def advance(self, day: int):
        """Random-walk drift for the days since the last game day."""
        if self.day is not None and day > self.day:
            self.P[np.diag_indices_from(self.P)] += self.qd * (day - self.day)
        self.day = day if self.day is None else max(self.day, day)

    # ── observation rows ──
    def _margin_row(self, h, a, neutral=0, b2b_h=0, b2b_a=0, lin_h=0.0, lin_a=0.0):
        """lin_h / lin_a: lineup covariate(s) of each side (scalar or array)."""
        i, j, n = self.ix[h], self.ix[a], self.n
        idx = [i, j]
        cf = [1.0, -1.0]
        if self.od:
            idx += [n + i, n + j]
            cf += [1.0, -1.0]
        idx += [self.g_home, self.g_b2b] + self.g_lin
        dl = np.broadcast_to(np.asarray(lin_h, float) - np.asarray(lin_a, float),
                             (len(self.g_lin),))
        cf = np.r_[cf, 1.0 - neutral, float(b2b_a - b2b_h), dl]
        if self.g_th:
            idx.append(self.g_th[i])
            cf = np.r_[cf, 1.0 - neutral]
        return np.array(idx), cf

    def _total_row(self, h, a):
        i, j, n = self.ix[h], self.ix[a], self.n
        return (np.array([i, n + i, j, n + j, self.g_mu]),
                np.array([1.0, -1.0, 1.0, -1.0, 2.0]))

    def _moments(self, idx, cf):
        mu = cf @ self.x[idx]
        var = cf @ self.P[np.ix_(idx, idx)] @ cf
        return mu, var

    # ── public API ──
    def _sigma2(self, playoff):
        return self.s2 * (self.p["sigma_po"] ** 2 if playoff else 1.0)

    def predict(self, h, a, playoff=0, **cov):
        """Pre-game (mean margin, predictive sd, sd of the state part,
        expected total or nan)."""
        mu, var = self._moments(*self._margin_row(h, a, **cov))
        s = np.sqrt(var + self._sigma2(playoff))
        tot = self._moments(*self._total_row(h, a))[0] if self.od else np.nan
        return mu, s, np.sqrt(var), tot

    def win_prob(self, h, a, **cov) -> float:
        mu, s, _, _ = self.predict(h, a, **cov)
        return float(ndtr(mu / s))

    def _update(self, idx, cf, y, r2):
        """Scalar measurement update; returns (innovation, its variance)."""
        Ph = self.P[:, idx] @ cf
        S = cf @ Ph[idx] + r2
        e = y - cf @ self.x[idx]
        c = self.p["clip"] * np.sqrt(S)
        k = Ph / S
        self.x += k * min(max(e, -c), c)
        self.P -= np.outer(k, Ph)
        return e, S

    def update(self, h, a, margin, total=None, playoff=0, **cov):
        """Measurement update with a final score (after the pre-game read)."""
        r2 = self._sigma2(playoff)
        e, S = self._update(*self._margin_row(h, a, **cov), margin, r2)
        a_ = self.p["adapt"]
        if a_ > 0:        # moment-matching estimate of the margin noise
            scale = r2 / self.s2
            self.s2 += a_ * (e * e / scale - (S - r2) / scale - self.s2)
            self.s2 = max(self.s2, 25.0)
        if self.od and total is not None:
            self._update(*self._total_row(h, a), total, self.p["sigma_t"] ** 2)

    def ratings(self) -> pd.DataFrame:
        """Current team ratings with 1-sd bands (points vs average team)."""
        n = self.n
        if self.od:
            o, d = self.x[:n], self.x[n:2 * n]
            Po, Pd = np.diag(self.P)[:n], np.diag(self.P)[n:2 * n]
            Pod = self.P[np.arange(n), n + np.arange(n)]
            r, v = o + d, Po + Pd + 2 * Pod
        else:
            r, v = self.x[:n], np.diag(self.P)[:n]
        out = pd.DataFrame({"TEAM_ID": self.teams, "RATING": r, "SD": np.sqrt(v)})
        if self.od:
            out["OFF"], out["DEF"] = o, d
        return out.sort_values("RATING", ascending=False).reset_index(drop=True)


# ── replay over history ───────────────────────────────────────────────────────

def replay(games: pd.DataFrame, lineup=None, return_filter=False, **params):
    """Chronological pass over game_table() rows.  Each row gets the
    PRE-game prediction; the result only enters the state afterwards, and
    games of the same day never see each other.

    lineup: optional (L_H, L_A) arrays aligned with games, shape (n,) or
    (n, k) for k covariates (pre-game lineup strength).  Returns a DataFrame with MU, SD, SD_STATE, R_H, R_A, SD_H,
    SD_A, HOME, TOTAL (and the filter if return_filter)."""
    teams = sorted(set(games["H"]) | set(games["A"]))
    n = len(games)
    n_lin = 1 if lineup is None or np.ndim(lineup[0]) == 1 else lineup[0].shape[1]
    kf = KalmanRatings(teams, n_lineup=n_lin, **params)
    H, A = games["H"].values, games["A"].values
    M = games["MARGIN"].values
    T = (games["PTS_H"] + games["PTS_A"]).values.astype(float)
    NEU, BH, BA = games["NEUTRAL"].values, games["B2B_H"].values, games["B2B_A"].values
    PO = games["PLAYOFF"].values
    LH, LA = (np.zeros(n), np.zeros(n)) if lineup is None else lineup
    day, seas = games["DAY"].values, games["SEASON"].values
    out = np.full((n, 9), np.nan)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, n]
    for s, e in zip(starts[:-1], starts[1:]):
        kf.roll_season(seas[s])
        kf.advance(day[s])
        covs = [dict(neutral=NEU[k], b2b_h=BH[k], b2b_a=BA[k], lin_h=LH[k], lin_a=LA[k],
                     playoff=PO[k])
                for k in range(s, e)]
        for k, cov in zip(range(s, e), covs):
            mu, sd, sds, tot = kf.predict(H[k], A[k], **cov)
            i, j = kf.ix[H[k]], kf.ix[A[k]]
            if kf.od:
                n_ = kf.n
                rh = kf.x[i] + kf.x[n_ + i]
                ra = kf.x[j] + kf.x[n_ + j]
                vh = kf.P[i, i] + kf.P[n_ + i, n_ + i] + 2 * kf.P[i, n_ + i]
                va = kf.P[j, j] + kf.P[n_ + j, n_ + j] + 2 * kf.P[j, n_ + j]
            else:
                rh, ra, vh, va = kf.x[i], kf.x[j], kf.P[i, i], kf.P[j, j]
            out[k] = [mu, sd, sds, rh, ra, np.sqrt(vh), np.sqrt(va),
                      kf.x[kf.g_home], tot]
        for k, cov in zip(range(s, e), covs):
            kf.update(H[k], A[k], M[k], T[k], **cov)
    res = pd.DataFrame(out, columns=["MU", "SD", "SD_STATE", "R_H", "R_A",
                                     "SD_H", "SD_A", "HOME", "TOTAL"],
                       index=games.index)
    return (res, kf) if return_filter else res


def margin_loglik(res: pd.DataFrame, games: pd.DataFrame, mask=None) -> float:
    """Mean Gaussian predictive log-likelihood of the margins (per game)."""
    m = np.ones(len(games), bool) if mask is None else np.asarray(mask)
    z = (games["MARGIN"].values[m] - res["MU"].values[m]) / res["SD"].values[m]
    return float(np.mean(-0.5 * z ** 2 - np.log(res["SD"].values[m]) - 0.5 * np.log(2 * np.pi)))


def win_logloss(res: pd.DataFrame, games: pd.DataFrame, mask=None) -> float:
    m = np.ones(len(games), bool) if mask is None else np.asarray(mask)
    p = np.clip(ndtr(res["MU"].values[m] / res["SD"].values[m]), 1e-6, 1 - 1e-6)
    y = games["RESULT"].values[m]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
