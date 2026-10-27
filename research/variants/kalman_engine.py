"""
Rating engine for the kalman variant: team strength as a linear-Gaussian
state-space model, filtered with the exact (joint, full-covariance) Kalman
filter, plus a lineup term that says who is available tonight.

State x, one vector for the whole league:
  net mode  [r_1..r_N,           home, b2b, β_1..β_K,     h_1..h_N]
  od  mode  [o_1..o_N, d_1..d_N, home, b2b, β_1..β_K, mu, h_1..h_N]
r = o + d = points per game better than an average team beyond what its
lineup explains (o = scoring, d = points prevented); home = league home edge;
h_i = team-specific home edge (altitude etc.); b2b = points lost on the 2nd
night of a back-to-back; β_k = points per unit of lineup covariate k;
mu = league points per team-game (od only).

Dynamics.  Between game days (Δ days)       x ← x + w,  w ~ N(0, Δ·diag(q))
           At the first game of a season     r ← ρ·r,   P_rr ← ρ² P_rr + v_season·I
Observation (one game, scalar rows)
           margin = r_h − r_a + (home + h_home)·(1 − neutral) + b2b·(B_a − B_h)
                    + Σ_k β_k·(L_hk − L_ak) + ε_m,             ε_m ~ N(0, σ²)
     od also total = 2mu + (o_h − d_h) + (o_a − d_a) + ε_t,    ε_t ~ N(0, σ_t²)
(margin and total are a rotation of the two team scores, so their noises are
independent and two sequential scalar updates are the exact joint update.)
σ² is re-estimated online by moment matching, E[e²] = h'Ph + σ², because
scoring (and so margin spread) drifts across eras.

Pre-game: margin ~ N(μ, s²) with μ = h'x, s² = h'Ph + σ², P(home win) = Φ(μ/s).
All games of one day are predicted before any of them updates the state.

Lineup covariates L (player_values / lineup_strength / PlayerBook): each
player's pre-game value is a time-decayed, shrunk per-minute rate from his
earlier box scores (Game Score, and on/off plus-minus); a team's L is the
expected-minutes-weighted average over its available players (played or
DNP-coach's decision; injured / rested / not-with-team are out), times 240.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

VALID_PREFIXES = ("002", "004", "005")       # regular season, playoffs, play-in
EPOCH = pd.Timestamp("2000-01-01")
T0 = 3900                                    # day offset keeping exp() small

# Orlando bubble (no crowd, no travel) and the 2023 in-season-tournament
# semifinals in Las Vegas: neutral courts known before tip-off.
NEUTRAL_WINDOWS = [("2020-07-30", "2020-10-11"), ("2023-12-07", "2023-12-07")]

PARAMS = dict(
    mode="od",           # "net" or "od"
    sigma=12.0,          # initial sd of a game margin around its expectation
    adapt=0.0011,        # weight of each game in the online estimate of σ²
    q=0.0061,            # team-strength variance added per day (points²/day)
    v0=30.0,             # team variance at the very first game (2010-11)
    rho=0.625,           # share of a team's strength kept over the summer
    v_season=2.29,       # variance added to every team at a new season
    home0=3.0, v_home0=1.0, q_home=8.8e-4,   # league home edge
    v_team_home=1.0,     # prior variance of per-team home edges (0 = none)
    b2b0=1.0, v_b2b0=1.0, q_b2b=0.0,          # back-to-back penalty
    lineup0=0.0, v_lineup0=1.0, q_lineup=0.0,  # lineup coefficients β
    clip=np.inf,         # innovations clipped at clip·s (robust update)
    sigma_po=1.0,        # playoff margin sd as a multiple of σ
    # od mode only
    sigma_t=18.0,        # sd of the game total (points)
    mu0=100.0, v_mu0=25.0, q_mu=0.01,         # league scoring level per team
    off_share=0.40,      # share of the strength variances given to offense
    rho_off=0.533, rho_def=0.637,             # per-side summer reversion
)

PLAYER_PARAMS = dict(
    h_rate=365.0,        # half-life (days) of a player's box-score history
    k=1000.0,            # prior minutes shrinking a Game Score rate ...
    shrink=0.2,          # ... toward shrink × league-average rate
    k_onoff=10000.0,     # prior minutes shrinking on/off rates toward 0
    h_min=15.0,          # half-life (days) of the expected-minutes average
    km=3.0, m0=10.0,     # prior games / minutes of the expected-minutes average
    top=10,              # players per team counted in the lineup
    gmsc_avg=0.3413,     # league Game Score per minute 2010-24
)
LINEUP_COLS = ["L_GMSC", "L_ONOFF"]


# ── games ─────────────────────────────────────────────────────────────────────

def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs (leaguegamefinder format) → one row per game in date
    order: teams, points, back-to-backs, neutral flag, result."""
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


# ── players ───────────────────────────────────────────────────────────────────

def game_score(df: pd.DataFrame) -> pd.Series:
    """Hollinger Game Score from nba.com box-score columns."""
    return (df["points"] + 0.4 * df["fieldGoalsMade"] - 0.7 * df["fieldGoalsAttempted"]
            - 0.4 * (df["freeThrowsAttempted"] - df["freeThrowsMade"])
            + 0.7 * df["reboundsOffensive"] + 0.3 * df["reboundsDefensive"]
            + df["steals"] + 0.7 * df["assists"] + 0.7 * df["blocks"]
            - 0.4 * df["foulsPersonal"] - df["turnovers"])


def box_table(box: pd.DataFrame) -> pd.DataFrame:
    """nba.com player box scores (boxscoretraditionalv3 columns) → GAME_ID,
    TEAM_ID, PID, MIN, PLAYED, AVAIL, GMSC, PM.  AVAIL = played or DNP-coach's
    decision, i.e. not on the injury / rest / not-with-team list."""
    b = box.copy()
    b["GAME_ID"] = b["gameId"].astype(str).str.zfill(10)
    b = b.drop_duplicates(["GAME_ID", "personId"])
    mm = b["minutes"].astype("string").str.split(":", expand=True)
    mins = mm[0].astype(float) + mm[1].astype(float) / 60
    com = b["comment"].fillna("").astype(str)
    out = pd.DataFrame({
        "GAME_ID": b["GAME_ID"].values, "TEAM_ID": b["teamId"].astype(int).values,
        "PID": b["personId"].astype(int).values, "MIN": mins.fillna(0.0).values,
        "PLAYED": mins.notna().values,
        "GMSC": game_score(b).astype(float).values,
        "PM": b["plusMinusPoints"].astype(float).values,
    })
    out["AVAIL"] = out["PLAYED"].values | (com.str.startswith("DNP")
                                           & com.str.contains("Coach")).values
    return out


def player_values(box: pd.DataFrame, games: pd.DataFrame, **pp) -> pd.DataFrame:
    """Pre-game value of every player-game row of box_table(): RATE_GMSC,
    RATE_ONOFF (per minute) and MHAT (expected minutes), each from that
    player's EARLIER games only (one game per player per day)."""
    p = {**PLAYER_PARAMS, **pp}
    gm = games[["GAME_ID", "DAY", "H", "A", "MARGIN"]]
    cols = ["GAME_ID", "DAY", "TEAM_ID", "MARGIN"]
    side = pd.concat([gm.rename(columns={"H": "TEAM_ID"})[cols],
                      gm.rename(columns={"A": "TEAM_ID"}).assign(MARGIN=-gm["MARGIN"])[cols]])
    b = box.merge(side, on=["GAME_ID", "TEAM_ID"], how="inner")
    b = b.sort_values(["PID", "DAY", "GAME_ID"]).reset_index(drop=True)
    pid = b["PID"].values
    t = b["DAY"].values - T0
    e = np.exp(np.log(2) / p["h_rate"] * t)
    em = np.exp(np.log(2) / p["h_min"] * t)
    pl = b["PLAYED"].values.astype(float)
    mn = b["MIN"].values * pl

    def before(v):                    # sum over the player's earlier rows
        return pd.Series(v).groupby(pid).cumsum().values - v

    s_min = before(e * mn)
    k, ko = p["k"], p["k_onoff"]
    b["RATE_GMSC"] = ((before(e * b["GMSC"].values * pl) + k * p["shrink"] * p["gmsc_avg"] * e)
                      / (s_min + k * e))
    on, off = b["PM"].values * pl, (b["MARGIN"].values - b["PM"].values) * pl
    off_min = np.maximum(48.0 - b["MIN"].values, 0.0) * pl
    b["RATE_ONOFF"] = (before(e * on) / (s_min + ko * e)
                       - before(e * off) / (before(e * off_min) + ko * e))
    b["MHAT"] = ((before(em * mn) + p["km"] * p["m0"] * em)
                 / (before(em * pl) + p["km"] * em))
    return b


def lineup_from_values(rates: np.ndarray, mhat: np.ndarray, top: int) -> np.ndarray:
    """One team's lineup covariates from its available players' pre-game
    rates (n×K) and expected minutes: top-`top` by minutes, weights rescaled
    to 240 minutes."""
    order = np.argsort(-mhat, kind="stable")[:top]
    w = mhat[order]
    return 240.0 * (w / w.sum()) @ rates[order]


def lineup_strength(vals: pd.DataFrame, games: pd.DataFrame, **pp):
    """(L_H, L_A) arrays aligned with games, shape (n, 2), from
    player_values(); games without box scores get 0 (no lineup signal)."""
    p = {**PLAYER_PARAMS, **pp}
    a = vals[vals["AVAIL"]].sort_values(["GAME_ID", "TEAM_ID", "MHAT"],
                                        ascending=[True, True, False], kind="stable")
    a = a[a.groupby(["GAME_ID", "TEAM_ID"]).cumcount() < p["top"]]
    w = 240.0 * a["MHAT"] / a.groupby(["GAME_ID", "TEAM_ID"])["MHAT"].transform("sum")
    L = (a[["RATE_GMSC", "RATE_ONOFF"]].mul(w, axis=0)
         .groupby([a["GAME_ID"], a["TEAM_ID"]]).sum())
    LH = L.reindex(pd.MultiIndex.from_arrays([games["GAME_ID"], games["H"]])).values
    LA = L.reindex(pd.MultiIndex.from_arrays([games["GAME_ID"], games["A"]])).values
    miss = np.isnan(LH).any(axis=1) | np.isnan(LA).any(axis=1)
    LH[miss], LA[miss] = 0.0, 0.0
    return LH, LA


class PlayerBook:
    """Live twin of player_values(): per-player time-decayed box-score sums.
    add_game() after a final box score, value()/lineup() before tip-off."""

    def __init__(self, **pp):
        self.p = {**PLAYER_PARAMS, **pp}
        self.lam = np.log(2) / self.p["h_rate"]
        self.lam_m = np.log(2) / self.p["h_min"]
        self.s = {}        # pid -> (day, [gmsc, min, on, off, off_min], [min, games])

    def _at(self, pid, day):
        if pid not in self.s:
            return np.zeros(5), np.zeros(2)
        d0, a, m = self.s[pid]
        return a * np.exp(-self.lam * (day - d0)), m * np.exp(-self.lam_m * (day - d0))

    def add_game(self, day: int, rows):
        """rows: iterable of (pid, minutes or None, game_score, plus_minus,
        team_margin) for every player of a finished game."""
        for pid, mins, gmsc, pm, margin in rows:
            if mins is None or np.isnan(mins):
                continue
            a, m = self._at(pid, day)
            a = a + [gmsc, mins, pm, margin - pm, max(48.0 - mins, 0.0)]
            m = m + [mins, 1.0]
            self.s[pid] = (day, a, m)

    def value(self, pid, day):
        """(Game Score rate, on/off rate, expected minutes) before `day`."""
        p = self.p
        a, m = self._at(pid, day)
        k, ko = p["k"], p["k_onoff"]
        r_g = (a[0] + k * p["shrink"] * p["gmsc_avg"]) / (a[1] + k)
        r_o = a[2] / (a[1] + ko) - a[3] / (a[4] + ko)
        mhat = (m[0] + p["km"] * p["m0"]) / (m[1] + p["km"])
        return r_g, r_o, mhat

    def lineup(self, pids, day, p_play=None) -> np.ndarray:
        """Lineup covariates for the players expected to be available.
        p_play: optional chance each one plays (injury report), which scales
        his expected minutes."""
        v = np.array([self.value(pid, day) for pid in pids], float).reshape(-1, 3)
        mhat = v[:, 2] * (1.0 if p_play is None else np.asarray(p_play, float))
        return lineup_from_values(v[:, :2], mhat, self.p["top"])


# ── filter ────────────────────────────────────────────────────────────────────

class KalmanRatings:
    """Joint Kalman filter over every team's strength.

    Live use:  roll_season(season) when a new season starts, advance(day)
    before each game day, predict(...) for every game of the day, then
    update(...) with each final score.
    """

    def __init__(self, teams, n_lineup=len(LINEUP_COLS), **params):
        p = {**PARAMS, **params}
        self.p = p
        self.od = p["mode"] == "od"
        self.teams = list(teams)
        self.ix = {t: i for i, t in enumerate(self.teams)}
        n = self.n = len(self.teams)
        nt = self.nt = 2 * n if self.od else n           # team-block size
        self.g_home, self.g_b2b = nt, nt + 1
        self.g_lin = list(range(nt + 2, nt + 2 + n_lineup))
        m = nt + 2 + n_lineup
        self.g_mu = m if self.od else None
        m += int(self.od)
        self.g_th = list(range(m, m + n)) if p["v_team_home"] > 0 else []
        m += len(self.g_th)

        so = p["off_share"] if self.od else 1.0
        self.split = np.r_[np.full(n, so), np.full(n, 1 - so)] if self.od else np.ones(n)
        self.x = np.zeros(m)
        var0, qd = np.zeros(m), np.zeros(m)
        var0[:nt], qd[:nt] = p["v0"] * self.split, p["q"] * self.split
        for g, (x0, v0, q) in {self.g_home: ("home0", "v_home0", "q_home"),
                               self.g_b2b: ("b2b0", "v_b2b0", "q_b2b")}.items():
            self.x[g], var0[g], qd[g] = p[x0], p[v0], p[q]
        self.x[self.g_lin], var0[self.g_lin], qd[self.g_lin] = p["lineup0"], p["v_lineup0"], p["q_lineup"]
        if self.od:
            self.x[self.g_mu], var0[self.g_mu], qd[self.g_mu] = p["mu0"], p["v_mu0"], p["q_mu"]
        var0[self.g_th] = p["v_team_home"]
        self.P = np.diag(var0)
        self.qd = qd
        rho_o = p["rho"] if p["rho_off"] is None else p["rho_off"]
        rho_d = p["rho"] if p["rho_def"] is None else p["rho_def"]
        self.rho_vec = (np.r_[np.full(n, rho_o), np.full(n, rho_d)] if self.od
                        else np.full(n, p["rho"]))
        self.s2 = p["sigma"] ** 2                 # current margin noise variance
        self.day = None
        self.season = None

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
            self.day = None                     # no daily drift over the summer
        self.season = season

    def advance(self, day: int):
        """Random-walk drift for the days since the last game day."""
        if self.day is not None and day > self.day:
            self.P[np.diag_indices_from(self.P)] += self.qd * (day - self.day)
        self.day = day if self.day is None else max(self.day, day)

    # ── observation rows ──
    def margin_row(self, h, a, neutral=0, b2b_h=0, b2b_a=0, lin_h=0.0, lin_a=0.0):
        """Sparse row (indices, coefficients) with margin = row·x + noise."""
        i, j, n = self.ix[h], self.ix[a], self.n
        idx, cf = [i, j], [1.0, -1.0]
        if self.od:
            idx += [n + i, n + j]
            cf += [1.0, -1.0]
        dl = np.broadcast_to(np.asarray(lin_h, float) - np.asarray(lin_a, float),
                             (len(self.g_lin),))
        idx += [self.g_home, self.g_b2b] + self.g_lin
        cf = np.r_[cf, 1.0 - neutral, float(b2b_a - b2b_h), dl]
        if self.g_th:
            idx.append(self.g_th[i])
            cf = np.r_[cf, 1.0 - neutral]
        return np.array(idx), cf

    def total_row(self, h, a):
        i, j, n = self.ix[h], self.ix[a], self.n
        return (np.array([i, n + i, j, n + j, self.g_mu]),
                np.array([1.0, -1.0, 1.0, -1.0, 2.0]))

    def team_row(self, t, lin=None, lin_ref=None):
        """Row for a team's neutral-court strength vs an average team: its
        rating, plus β·(its lineup − reference lineup) when lin is given."""
        i, n = self.ix[t], self.n
        idx, cf = ([i, n + i], [1.0, 1.0]) if self.od else ([i], [1.0])
        if lin is not None:
            idx += self.g_lin
            cf += list(np.asarray(lin, float) - np.asarray(lin_ref, float))
        return np.array(idx), np.array(cf)

    def moments(self, idx, cf):
        """Mean and variance of row·x under the current state."""
        return cf @ self.x[idx], cf @ self.P[np.ix_(idx, idx)] @ cf

    def _sigma2(self, playoff):
        return self.s2 * (self.p["sigma_po"] ** 2 if playoff else 1.0)

    # ── public API ──
    def predict(self, h, a, playoff=0, **cov):
        """Pre-game (mean margin, predictive sd, sd of the mean, expected
        total or nan)."""
        mu, var = self.moments(*self.margin_row(h, a, **cov))
        s = np.sqrt(var + self._sigma2(playoff))
        tot = self.moments(*self.total_row(h, a))[0] if self.od else np.nan
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
        e, S = self._update(*self.margin_row(h, a, **cov), margin, r2)
        if self.p["adapt"] > 0:          # moment-matching estimate of σ²
            scale = r2 / self.s2
            self.s2 += self.p["adapt"] * ((e * e - (S - r2)) / scale - self.s2)
            self.s2 = max(self.s2, 25.0)
        if self.od and total is not None:
            self._update(*self.total_row(h, a), total, self.p["sigma_t"] ** 2)

    def ratings(self, lineups: dict = None) -> pd.DataFrame:
        """Rankings table: points better than an average team on a neutral
        court, with a 1-sd band.  lineups: optional {team: lineup covariates}
        (e.g. each team's latest lineup) to include roster strength."""
        ref = np.mean(list(lineups.values()), axis=0) if lineups else None
        rows = []
        for t in self.teams:
            lin = lineups.get(t) if lineups else None
            mu, var = self.moments(*self.team_row(t, lin, ref))
            row = {"TEAM_ID": t, "RATING": mu, "SD": np.sqrt(var)}
            if self.od:
                i = self.ix[t]
                row["OFF"], row["DEF"] = self.x[i], self.x[self.n + i]
            if self.g_th:
                row["HOME_EDGE"] = self.x[self.g_home] + self.x[self.g_th[self.ix[t]]]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("RATING", ascending=False).reset_index(drop=True)


# ── replay over history ───────────────────────────────────────────────────────

SIGNAL_COLS = ["MU", "SD", "SD_MEAN", "R_H", "R_A", "SD_H", "SD_A", "HOME",
               "LINEUP", "TOTAL"]


def replay(games: pd.DataFrame, lineup=None, return_filter=False, **params):
    """Chronological pass over game_table() rows.  Each row gets the
    PRE-game prediction; the result only enters the state afterwards, and
    games of the same day never see each other.

    lineup: optional (L_H, L_A) arrays aligned with games, shape (n, K).
    Returns a DataFrame with SIGNAL_COLS (and the filter if return_filter):
    MU/SD mean and sd of the home margin, SD_MEAN sd of MU, R_H/R_A and
    SD_H/SD_A team ratings (beyond lineup), HOME home edge applied, LINEUP
    lineup part of MU, TOTAL expected total points (od)."""
    teams = sorted(set(games["H"]) | set(games["A"]))
    n = len(games)
    if lineup is None:
        lineup = (np.zeros((n, len(LINEUP_COLS))), np.zeros((n, len(LINEUP_COLS))))
    LH, LA = np.asarray(lineup[0], float), np.asarray(lineup[1], float)
    kf = KalmanRatings(teams, n_lineup=LH.shape[1], **params)
    H, A = games["H"].values, games["A"].values
    M = games["MARGIN"].values
    T = (games["PTS_H"] + games["PTS_A"]).values.astype(float)
    NEU, PO = games["NEUTRAL"].values, games["PLAYOFF"].values
    BH, BA = games["B2B_H"].values, games["B2B_A"].values
    day, seas = games["DAY"].values, games["SEASON"].values
    out = np.full((n, len(SIGNAL_COLS)), np.nan)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, n]
    for s, e in zip(starts[:-1], starts[1:]):
        kf.roll_season(seas[s])
        kf.advance(day[s])
        covs = [dict(neutral=NEU[k], b2b_h=BH[k], b2b_a=BA[k], lin_h=LH[k], lin_a=LA[k],
                     playoff=PO[k]) for k in range(s, e)]
        for k, cov in zip(range(s, e), covs):
            mu, sd, sdm, tot = kf.predict(H[k], A[k], **cov)
            (rh, vh), (ra, va) = kf.moments(*kf.team_row(H[k])), kf.moments(*kf.team_row(A[k]))
            th = kf.x[kf.g_th[kf.ix[H[k]]]] if kf.g_th else 0.0
            home = (1 - NEU[k]) * (kf.x[kf.g_home] + th)
            lin = kf.x[kf.g_lin] @ (LH[k] - LA[k])
            out[k] = [mu, sd, sdm, rh, ra, np.sqrt(vh), np.sqrt(va), home, lin, tot]
        for k, cov in zip(range(s, e), covs):
            kf.update(H[k], A[k], M[k], T[k], **cov)
    res = pd.DataFrame(out, columns=SIGNAL_COLS, index=games.index)
    return (res, kf) if return_filter else res


def margin_loglik(res: pd.DataFrame, games: pd.DataFrame, mask=None) -> float:
    """Mean Gaussian predictive log-likelihood of the margins (per game)."""
    m = np.ones(len(games), bool) if mask is None else np.asarray(mask)
    sd = res["SD"].values[m]
    z = (games["MARGIN"].values[m] - res["MU"].values[m]) / sd
    return float(np.mean(-0.5 * z ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)))


def win_logloss(res: pd.DataFrame, games: pd.DataFrame, mask=None) -> float:
    """Log loss of the uncalibrated P(home win) = Φ(MU/SD)."""
    m = np.ones(len(games), bool) if mask is None else np.asarray(mask)
    p = np.clip(ndtr(res["MU"].values[m] / res["SD"].values[m]), 1e-6, 1 - 1e-6)
    y = games["RESULT"].values[m]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
