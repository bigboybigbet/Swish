import numpy as np

from config import HOME_ADV, MC_CORR, N_SIMS


def run_monte_carlo(h: dict, a: dict, sims: int = N_SIMS) -> dict:
    """
    Correlated bivariate-normal score simulation.

    The two teams' scores share game pace, so they are positively
    correlated (rho = MC_CORR).  This makes margins tighter and totals
    wider than independent sampling — matching real NBA score behaviour.
    Home advantage (HOME_ADV points) is split symmetrically.
    """
    h_mean = (h["PTS"] + a["PTS_ALLOWED"]) / 2 + HOME_ADV / 2
    a_mean = (a["PTS"] + h["PTS_ALLOWED"]) / 2 - HOME_ADV / 2
    h_std  = max(np.sqrt(h["PTS_STD"] ** 2 + a["PTS_ALLOWED_STD"] ** 2) / np.sqrt(2), 4.0)
    a_std  = max(np.sqrt(a["PTS_STD"] ** 2 + h["PTS_ALLOWED_STD"] ** 2) / np.sqrt(2), 4.0)

    rng = np.random.default_rng()
    cov = MC_CORR * h_std * a_std
    scores = rng.multivariate_normal(
        [h_mean, a_mean],
        [[h_std ** 2, cov], [cov, a_std ** 2]],
        size=sims,
    )
    hs, as_ = scores[:, 0], scores[:, 1]
    margins = hs - as_
    totals  = hs + as_
    return {
        "mc_home_prob":  float(np.sum(margins > 0) / sims),
        "spread_margin": float(np.mean(margins)),
        "median_total":  float(np.median(totals)),
        "spread_lo":     float(np.percentile(margins, 25)),
        "spread_hi":     float(np.percentile(margins, 75)),
        "total_lo":      float(np.percentile(totals, 25)),
        "total_hi":      float(np.percentile(totals, 75)),
        "ot_prob":       float(np.mean(np.abs(margins) <= 3)),
        "_margins": margins, "_totals": totals, "_h_sims": hs, "_a_sims": as_,
    }


def run_poisson_model(h: dict, a: dict) -> dict:
    h_lam = (h["PTS"] / 4 + a["PTS_ALLOWED"] / 4) / 2 + HOME_ADV / 8
    a_lam = (a["PTS"] / 4 + h["PTS_ALLOWED"] / 4) / 2 - HOME_ADV / 8
    a_lam = max(a_lam, 1.0)
    qtrs, ht, at = [], 0, 0
    for q in range(1, 5):
        hq = int(np.random.poisson(h_lam))
        aq = int(np.random.poisson(a_lam))
        ht += hq
        at += aq
        qtrs.append({"Q": q, "Home": hq, "Away": aq, "H_Cum": ht, "A_Cum": at})
    rng = np.random.default_rng()
    hs2 = rng.poisson(h_lam, (10_000, 4)).sum(axis=1)
    as2 = rng.poisson(a_lam, (10_000, 4)).sum(axis=1)
    tots = hs2 + as2
    return {
        "quarters":       qtrs,
        "proj_h":         ht,
        "proj_a":         at,
        "poisson_h_prob": float(np.mean(hs2 > as2)),
        "median_total":   float(np.median(tots)),
        "ou_80":          float(np.percentile(tots, 80)),
        "ou_20":          float(np.percentile(tots, 20)),
        "lambda_h":       float(h_lam),   # expected points per quarter
        "lambda_a":       float(a_lam),
    }
