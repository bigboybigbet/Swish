"""
Rest-of-season Monte Carlo for the incentives variant.

For every regular-season game-day d:
  1. pre-game team ratings = the adj_ratings margin ridge (same parameters,
     same decayed normal equations) solved on games BEFORE d, plus its
     long-memory home edge;
  2. standings = W-L from games before d; future FIXTURES (date >= d) come
     from the schedule, never their results;
  3. S simulations of every remaining game: P(home) = Phi((r_h - r_a + home
     + eta_h - eta_a) / SIGMA), eta ~ N(0, TAU^2) a per-simulation team
     strength shock (rating uncertainty, keeps outcomes correlated);
  4. final conference seeds by win% (random tiebreak), lottery positions;
  5. for every game played on d, its outcome is forced to home-win and to
     away-win inside the SAME simulations (common random numbers), so
     leverage = P(goal | win) - P(goal | lose) is cheap and low-noise.

2019-20: before the March 2020 suspension the schedule still had 82 games
per team; the games that were later cancelled are replaced by "phantom"
games against a league-average opponent at a neutral site.  After the
restart the bubble seeding-game schedule (known by then) is used as is.

Approximations: random tiebreaks (no head-to-head / division rules), no
pre-2015 division-winner seed guarantee, play-in outcome as fixed weights.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr

from research.benchmark import CACHE_DIR
from research.variants import adj_ratings
from research.variants.incentives_standings import (
    IS_EAST, TEAM_IX, TEAMS, conference_ranks, has_play_in, lottery_odds,
    postseason_weight, regular_season_games)

MC = dict(
    n_sims=2000,
    sigma=10.0,        # game-level probit scale (dev-train fit of ADJ_PRED: 10.5)
    tau=2.0,           # per-sim team strength shock (points)
    bracket_days=15,   # compute bracket value only in the last N days
    seed=2024,
)
VERSION = 1
SUSPENSION = pd.Timestamp("2020-03-12")       # first day without games, 2019-20
SERIES_PATTERN = (1, 1, 0, 0, 1, 0, 1)        # 1 = better seed at home


# ── ratings ───────────────────────────────────────────────────────────────────

def _ridge_snapshots(games: pd.DataFrame, halflife, lam, carry, shrink, cap):
    """Margin-ridge coefficients before each game-day (adj_ratings logic)."""
    teams = sorted(set(games["H"]) | set(games["A"]))
    tix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = games["H"].map(tix).values
    A = games["A"].map(tix).values
    mm = adj_ratings.DecayedRidge(n + 1, np.r_[np.ones(n), 0], lam)
    margin = games["MARGIN"].clip(-cap, cap).values
    dates = games["GAME_DATE"].values
    seasons = games["SEASON"].values
    order = [TEAM_IX[t] for t in teams]
    snaps = {}
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(games)]
    for s, e in zip(starts[:-1], starts[1:]):
        if s > 0 and seasons[s] != seasons[s - 1]:
            mm.new_season(slice(0, n), carry, shrink)
        elif s > 0:
            mm.decay(0.5 ** ((dates[s] - dates[s - 1]) / np.timedelta64(1, "D") / halflife))
        b = mm.solve()
        r = np.zeros(30)
        r[order] = b[:n]
        snaps[pd.Timestamp(dates[s])] = (r, b[n])
        for k in range(s, e):
            mm.add([H[k], A[k], n], [1, -1, 1], margin[k])
    return snaps


def daily_ratings(raw: pd.DataFrame) -> dict:
    """GAME_DATE -> (ratings[30] in TEAMS order, home edge), pre-game."""
    games = adj_ratings.game_table(raw)
    p = adj_ratings.PARAMS
    team = _ridge_snapshots(games, **p)
    home = _ridge_snapshots(games, **{**p, "halflife": adj_ratings.HOME_HALFLIFE,
                                      "carry": 1.0, "shrink": 1.0})
    return {d: (team[d][0], home[d][1]) for d in team}


# ── playoff series ────────────────────────────────────────────────────────────

def series_prob(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    """P(better seed wins a best-of-7, 2-2-1-1-1) from its single-game win
    probabilities at home / away (elementwise)."""
    probs = {(0, 0): np.ones_like(p_home)}
    win = np.zeros_like(p_home)
    for home in SERIES_PATTERN:
        p = p_home if home else p_away
        nxt = {}
        for (a, b), q in probs.items():
            for (da, db, pq) in ((1, 0, p), (0, 1, 1 - p)):
                st = (a + da, b + db)
                if st[0] == 4:
                    win = win + q * pq
                elif st[1] < 4:
                    nxt[st] = nxt.get(st, 0) + q * pq
        probs = nxt
    return win


def series_matrix(r: np.ndarray, home: float, sigma: float) -> np.ndarray:
    """M[i, j] = P(team i beats team j in a series, i holding home court)."""
    d = r[:, None] - r[None, :]
    return series_prob(ndtr((d + home) / sigma), ndtr((d - home) / sigma))


def bracket_value(seed: np.ndarray, SM: np.ndarray, season: str) -> np.ndarray:
    """Expected series wins in rounds 1-2 for every team (S x 30), from the
    final conference seeds (S x 30) and the series matrix."""
    S = seed.shape[0]
    val = np.zeros(seed.shape)
    for conf in (IS_EAST, ~IS_EAST):
        idx = np.flatnonzero(conf)
        sc = seed[:, idx]
        slot = idx[np.argsort(sc, axis=1)[:, :8]]          # S x 8 team ids by seed
        q = np.zeros((S, 8))
        for hi, lo in ((0, 7), (3, 4), (1, 6), (2, 5)):
            q[:, hi] = SM[slot[:, hi], slot[:, lo]]
            q[:, lo] = 1 - q[:, hi]
        r2 = np.zeros((S, 8))
        for side_a, side_b in (((0, 7), (3, 4)), ((1, 6), (2, 5))):
            for k in side_a + side_b:
                opp = side_b if k in side_a else side_a
                for m in opp:
                    pk = np.where(k < m, SM[slot[:, k], slot[:, m]],
                                  1 - SM[slot[:, m], slot[:, k]])
                    r2[:, k] += q[:, k] * q[:, m] * pk
        slot_val = q + r2
        rows = np.arange(S)[:, None]
        s_cl = np.minimum(sc, 8) - 1
        w = postseason_weight(sc, season)
        val[:, idx] = w * slot_val[rows, s_cl]
    return val


# ── one season ────────────────────────────────────────────────────────────────

def _goals(Wf: np.ndarray, G: np.ndarray, tb: np.ndarray, season: str) -> dict:
    """Per-sim goal indicators / values (S x 30) from final wins."""
    key = Wf / G + 1e-7 * tb
    seed = conference_ranks(Wf / G, tb)
    lot = seed > 8
    below = key[:, None, :] < key[:, :, None]                 # [s, t, j]: j worse than t
    pos = 1 + (below & lot[:, None, :]).sum(axis=2)
    odds = np.where(lot, lottery_odds(season)[np.clip(pos, 1, 14) - 1], 0.0)
    rank_from_bottom = 1 + below.sum(axis=2)
    return {
        "seed": seed,
        "POST": postseason_weight(seed, season),
        "TOP1": (seed == 1).astype(float),
        "TOP4": (seed <= 4).astype(float),
        "TOP6": (seed <= 6).astype(float),
        "TOP8": (seed <= 8).astype(float),
        "TOP10": (seed <= 10).astype(float),
        "BOT3": (rank_from_bottom <= 3).astype(float),
        "ODDS": odds,
    }


def _clinch_flags(W, L, G, season):
    """Deterministic (conservative) clinch / elimination per team."""
    rem = G - W - L
    lo, hi = W / G, (W + rem) / G
    cut_po = 6 if has_play_in(season) else 8
    cut_post = 10 if has_play_in(season) else 8
    out = {}
    for name, k in (("CLINCH_PO", cut_po), ("CLINCH_POST", cut_post), ("CLINCH_1", 1)):
        f = np.zeros(30, bool)
        for conf in (IS_EAST, ~IS_EAST):
            idx = np.flatnonzero(conf)
            threats = (hi[idx][None, :] >= lo[idx][:, None]).sum(1) - 1
            f[idx] = threats < k
        out[name] = f
    for name, k in (("ELIM_POST", cut_post), ("ELIM_PO", cut_po)):
        f = np.zeros(30, bool)
        for conf in (IS_EAST, ~IS_EAST):
            idx = np.flatnonzero(conf)
            ahead = (lo[idx][None, :] > hi[idx][:, None]).sum(1)
            f[idx] = ahead >= k
        out[name] = f
    return out


GOALS = ["POST", "TOP1", "TOP4", "TOP6", "TOP8", "TOP10", "BOT3", "ODDS"]


def simulate_season(g: pd.DataFrame, ratings: dict, p: dict) -> pd.DataFrame:
    """Incentive features for every game of one regular season (rows of
    regular_season_games for that season)."""
    season = g["SEASON"].iloc[0]
    rng = np.random.default_rng(p["seed"] + int(season[:4]))
    S, sigma, tau = p["n_sims"], p["sigma"], p["tau"]
    HI, AI = g["HI"].values, g["AI"].values
    res = g["RESULT"].values
    dates = g["GAME_DATE"].values
    total = (np.bincount(HI, minlength=30) + np.bincount(AI, minlength=30)).astype(float)
    last_day = pd.Timestamp(dates[-1])
    pre_susp = (np.bincount(HI[dates < SUSPENSION], minlength=30)
                + np.bincount(AI[dates < SUSPENSION], minlength=30))

    W = np.zeros(30)
    L = np.zeros(30)
    cols = {}
    def put(name, i, side, v):
        cols.setdefault(f"{name}_{side}", np.full(len(g), np.nan))[i] = v

    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(g)]
    for s, e in zip(starts[:-1], starts[1:]):
        day = pd.Timestamp(dates[s])
        r, home = ratings[day]
        phantom = np.zeros(30)
        G = total.copy()
        if season == "2019-20" and day < SUSPENSION:
            phantom = 82 - pre_susp
            G = np.full(30, 82.0)
            rem_ix = np.arange(s, len(g))[dates[s:] < SUSPENSION]
        else:
            rem_ix = np.arange(s, len(g))
        rh, ra = HI[rem_ix], AI[rem_ix]

        eta = rng.normal(0, tau, (S, 30))
        x = (r[rh] - r[ra] + home)[None, :] + eta[:, rh] - eta[:, ra]
        O = (rng.random(x.shape) < ndtr(x / sigma)).astype(np.float32)
        IH = np.zeros((len(rem_ix), 30), np.float32)
        IA = np.zeros((len(rem_ix), 30), np.float32)
        IH[np.arange(len(rem_ix)), rh] = 1
        IA[np.arange(len(rem_ix)), ra] = 1
        Wf = W[None, :] + O @ IH + (1 - O) @ IA
        if phantom.any():
            Wf = Wf + rng.binomial(phantom.astype(int)[None, :].repeat(S, 0),
                                   ndtr((r[None, :] + eta) / sigma))
        tb = rng.random((S, 30))
        base = _goals(Wf, G, tb, season)
        flags = _clinch_flags(W, L, G, season)
        seed_dist = np.stack([(base["seed"] == k).mean(0) for k in range(1, 16)])
        lock = seed_dist.max(0) >= 0.99
        brk = (last_day - day).days < p["bracket_days"]
        SM = series_matrix(r, home, sigma) if brk else None
        bval = bracket_value(base["seed"], SM, season) if brk else None

        for j, i in enumerate(range(s, e)):            # today's games: rem_ix[j] == i
            h, a = HI[i], AI[i]
            o = O[:, j]
            Wh = Wf.copy(); Wh[:, h] += 1 - o; Wh[:, a] -= 1 - o
            Wa = Wf.copy(); Wa[:, h] -= o;     Wa[:, a] += o
            gh, ga = _goals(Wh, G, tb, season), _goals(Wa, G, tb, season)
            if brk:
                bh, ba = bracket_value(gh["seed"], SM, season), bracket_value(ga["seed"], SM, season)
            for side, t, win, lose in (("H", h, gh, ga), ("A", a, ga, gh)):
                for k in GOALS:
                    put("P_" + k, i, side, base[k][:, t].mean())
                    put("LEV_" + k, i, side, win[k][:, t].mean() - lose[k][:, t].mean())
                put("E_SEED", i, side, base["seed"][:, t].mean())
                put("LEV_SEED", i, side, lose["seed"][:, t].mean() - win["seed"][:, t].mean())
                put("SEED_LOCK", i, side, float(lock[t]))
                for f, v in flags.items():
                    put(f, i, side, float(v[t]))
                if brk:
                    bw, bl = (bh, ba) if side == "H" else (ba, bh)
                    put("BVAL", i, side, bval[:, t].mean())
                    put("BVAL_W", i, side, bw[:, t].mean())
                    put("BVAL_L", i, side, bl[:, t].mean())
        np.add.at(W, HI[s:e], res[s:e]); np.add.at(L, HI[s:e], 1 - res[s:e])
        np.add.at(W, AI[s:e], 1 - res[s:e]); np.add.at(L, AI[s:e], res[s:e])

    out = pd.DataFrame(cols)
    out.insert(0, "GAME_ID", g["GAME_ID"].values)
    return out


def season_features(raw: pd.DataFrame, seasons=None, **params) -> pd.DataFrame:
    """MC incentive features for every regular-season game, cached per season
    under research/cache/incentives_mc/."""
    p = {**MC, **params}
    key = hashlib.md5(json.dumps([VERSION, p, adj_ratings.PARAMS], sort_keys=True)
                      .encode()).hexdigest()[:10]
    folder = os.path.join(CACHE_DIR, "incentives_mc")
    os.makedirs(folder, exist_ok=True)
    games = regular_season_games(raw)
    seasons = seasons or sorted(games["SEASON"].unique())
    ratings = None
    parts = []
    for season in seasons:
        path = os.path.join(folder, f"{season}_{key}.parquet")
        if not os.path.exists(path):
            ratings = ratings or daily_ratings(raw)
            simulate_season(games[games["SEASON"] == season].reset_index(drop=True),
                            ratings, p).to_parquet(path, index=False)
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)
