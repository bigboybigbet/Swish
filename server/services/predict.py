"""
predict_core(): one game's full forecast for POST /api/predict.

Wraps engine.models.predict_detail (XGBoost + Monte Carlo + Elo through the
blend calibrator, all on the same injury-adjusted rows) and adds the MC
percentiles and histograms, the market comparison (de-vig, edge, cover and
over probabilities), Kelly at a real price, the ensemble explanation ladder,
team form, availability, the radar, Poisson and SHAP.

Modes: "blend_v5" with a trained model; "elo_only" without one (the README
benchmark has Elo ahead of v5 XGBoost on log loss). "stacked_v6" arrives
with engine.ratings (M6) and fills the same schema.

_load_elo and _cache are data sources that demo mode swaps (demo.py).
"""
import math
import time
from datetime import date

import numpy as np

from config import ELO_DEFAULT, ELO_HOME_ADV, STATUS_WEIGHT
from data import cache
from data.elo import elo_expected, load_elo
from data.injuries import _ascii, calc_strength_factor, load_player_impact
from engine import features
from engine.models import get_calibrator, predict_detail
from engine.simulate import run_poisson_model
from server import clock, config_store, model_store
from server.errors import ApiError
from server.jsonutil import clean, r1, r4
from server.services import espn, market, teams

LOGS_TTL = cache.TTL_DEFAULT          # engine.features caches team logs for 6 h

# Data sources (replaced by demo.install()).
_load_elo = load_elo
_cache    = cache

ORDER  = ("elo", "mc", "xgb")
LABELS = {"elo": "Elo", "mc": "Monte Carlo", "xgb": "XGBoost"}
X0     = {"elo": 0.0, "mc": 0.5, "xgb": 0.5}      # neutral game: coin-flip engines, equal Elo
INCLUDE = ("hist", "shap", "poisson")
RADAR_AXES = ["eFG%", "TS%", "AST/TOV", "OREB%", "Reb", "+/-"]


# ── Elo ──────────────────────────────────────────────────────────────────────

def elo_table() -> dict:
    """{team_id: Elo} for all 30 teams; teams missing from the Elo file get ELO_DEFAULT."""
    raw = _load_elo() or {}
    return {t["id"]: float(teams.lookup(raw, t, ELO_DEFAULT)) for t in teams.all_teams()}


def elo_rank(table: dict, team_id: int) -> int | None:
    """1 = highest Elo. None when every team has the same rating (no Elo file)."""
    if len(set(table.values())) <= 1:
        return None
    return 1 + sum(v > table[team_id] for v in table.values())


# ── team form ────────────────────────────────────────────────────────────────

def team_stats(team: dict, d: date, elo: dict | None = None) -> dict:
    """
    get_team_advanced_stats for a team before ET date d: {stats, as_of, stale}.

    Team logs are cached for 6 h by engine.features. When stats.nba.com is
    unreachable an expired copy is used and flagged stale; with no copy it
    raises upstream_unavailable. Elo comes from the Elo table by team id
    (engine.features' substring lookup misses "LA Clippers").
    """
    key = f"lgf_{team['id']}"                      # engine.features' cache key
    stats = features.get_team_advanced_stats(team["name"], target_date=d.isoformat())
    hit = _cache.get_with_age(key, allow_stale=True)
    if stats is None:
        if hit is None:
            raise ApiError(
                "upstream_unavailable",
                f"Could not download {team['name']} game logs from stats.nba.com, and none are cached.",
                hint="Check the internet connection and retry. The first download can take ~10 s.",
                source="nba_api")
        if _cache.get(key) is None:
            # Only an expired copy exists: re-arm it briefly so engine.features reads it.
            _cache.set(key, hit[0], 60, stored_at=hit[1])
            stats = features.get_team_advanced_stats(team["name"], target_date=d.isoformat())
        if stats is None:
            raise ApiError(
                "not_found",
                f"{team['name']} has fewer than 3 games before {d.isoformat()}: not enough to compute form.",
                hint="Pick a later date.")
    stored_at = hit[1] if hit else None
    stale = bool(stored_at and time.time() - stored_at > LOGS_TTL)
    stats = dict(stats)
    stats["ELO"] = (elo or elo_table())[team["id"]]
    return {"stats": stats, "as_of": espn.iso_ts(stored_at) or clock.iso_utc(), "stale": stale}


def form_public(stats: dict) -> dict:
    """Engine stat dict → API keys (lower snake case), floats rounded to 4 dp."""
    out = {}
    for k, v in stats.items():
        if k in ("name", "team_id"):
            continue
        if isinstance(v, (bool, np.bool_)):
            out[k.lower()] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k.lower()] = int(v)
        else:
            out[k.lower()] = r4(v)
    return out


def form_response(team: dict, d: date) -> dict:
    """GET /api/teams/{id}/form."""
    ts = team_stats(team, d)
    return clean({"team": teams.ref(team), "date": d.isoformat(), **form_public(ts["stats"]),
                  "as_of": ts["as_of"], "stale": ts["stale"]})


# ── availability ─────────────────────────────────────────────────────────────

def live_feed(d: date) -> dict:
    """The injury feed for a game date. Past dates have no report ("unavailable")."""
    if d < clock.today_et():
        return {"feed": None, "status": "unavailable", "fetched_at": None}
    return espn.injury_feed()


def availability(team: dict, injured: list, status: str, fetched_at, source: str) -> dict:
    """
    v5 availability: healthy = 1.0 and current = the strength factor, so
    deficit = current − healthy (e.g. −0.06 shows as 94%). play_prob and
    margin_impact stay null until the v6 player engine.
    """
    impact = load_player_impact()
    players = []
    for p in injured:
        key = _ascii(p["name"])
        weight = STATUS_WEIGHT.get(p["status"], 0.2)
        players.append({
            "name": p["name"], "status": p["status"], "detail": p.get("detail") or "",
            "impact": r4(impact.get(key, 0.05)), "status_weight": weight,
            "play_prob": None, "margin_impact": None, "matched": key in impact,
        })
    players.sort(key=lambda x: x["impact"] * x["status_weight"], reverse=True)
    current = calc_strength_factor(injured) if injured else 1.0
    return {
        "team_id": team["id"], "status": status, "source": source, "fetched_at": fetched_at,
        "units": "factor", "strength": r4(current), "healthy": 1.0, "current": r4(current),
        "deficit": r4(current - 1.0), "players": players,
        "unmatched": sum(not p["matched"] for p in players),
    }


def availability_response(team: dict, d: date) -> dict:
    """GET /api/teams/{id}/availability (live report)."""
    info = live_feed(d)
    return availability(team, espn.team_injuries(team, info["feed"]), info["status"],
                        info["fetched_at"], "live")


# ── ensemble ladder ──────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))   # same clip as BlendCalibrator


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _ladder(method, base_logit, bias, contrib, coefs, probs, order, n_updates) -> dict:
    """base + Σ contributions = final (logits). step_pp: sequential probability step in pp."""
    final = base_logit + sum(contrib[k] for k in order)
    total_abs = sum(abs(contrib[k]) for k in order)
    running, prev = base_logit, _sigmoid(base_logit)
    comps = []
    for k in order:
        running += contrib[k]
        p = _sigmoid(running)
        comps.append({
            "key": k, "label": LABELS[k], "prob": r4(probs[k]), "coef": r4(coefs[k]),
            "contribution_logit": r4(contrib[k]), "step_pp": round((p - prev) * 100, 2),
            "share": r4(abs(contrib[k]) / total_abs) if total_abs else None,
        })
        prev = p
    return {
        "method": method, "n_updates": n_updates, "order": list(order),
        "base": {"logit": r4(base_logit), "prob": r4(_sigmoid(base_logit))},
        "components": comps, "bias": r4(bias),
        "final": {"logit": r4(final), "prob": r4(_sigmoid(final))},
    }


def blend_ladder(weights: dict, bias: float, x: dict, probs: dict, n_updates=None) -> dict:
    """Logistic blend sigmoid(w·x + b) with x = [xgb, mc, elo_diff/400], explained
    against a neutral game x0 = (0.5, 0.5, 0): contribution_i = w_i (x_i − x0_i)."""
    base = sum(weights[k] * X0[k] for k in ORDER) + bias
    contrib = {k: weights[k] * (x[k] - X0[k]) for k in ORDER}
    return _ladder("blend_calibrator", base, bias, contrib, weights, probs, ORDER, n_updates)


def elo_ladder(elo_p: float) -> dict:
    """elo_only: a single Elo component with weight 1 on its own log-odds."""
    return _ladder("elo", 0.0, 0.0, {"elo": _logit(elo_p)}, {"elo": 1.0}, {"elo": elo_p},
                   ("elo",), None)


# ── distributions ────────────────────────────────────────────────────────────

def _intervals(a) -> dict:
    q = np.percentile(a, [25, 75, 10, 90, 2.5, 97.5])
    return {"p50": [r1(q[0]), r1(q[1])], "p80": [r1(q[2]), r1(q[3])],
            "p95": [r1(q[4]), r1(q[5])]}


def _hist(a, bins: int = 60) -> dict:
    counts, edges = np.histogram(a, bins=bins)
    return {"edges": [round(float(e), 2) for e in edges], "counts": [int(c) for c in counts]}


def distributions(mc: dict, with_hist: bool) -> dict:
    """margin / total / scores blocks from the MC samples (raw arrays never leave)."""
    m, t, hs, as_ = mc["_margins"], mc["_totals"], mc["_h_sims"], mc["_a_sims"]
    margin = {"mean": r1(np.mean(m)), "sd": r1(np.std(m)), "home_spread": r1(-np.mean(m)),
              "p_within_3": r4(mc["ot_prob"]), "intervals": _intervals(m)}
    total = {"mean": r1(np.mean(t)), "median": r1(np.median(t)), "sd": r1(np.std(t)),
             "intervals": _intervals(t)}
    scores = {side: {"mean": r1(np.mean(s)), "p25": r1(np.percentile(s, 25)),
                     "p75": r1(np.percentile(s, 75))}
              for side, s in (("home", hs), ("away", as_))}
    if with_hist:
        margin["hist"], total["hist"] = _hist(m), _hist(t)
        scores["home"]["hist"], scores["away"]["hist"] = _hist(hs), _hist(as_)
    return {"margin": margin, "total": total, "scores": scores}


def radar_values(s: dict) -> list:
    """Six 0–1 axes (same scaling as the Streamlit radar)."""
    vals = [s["eFG_PCT"], s["TS_PCT"], min(s["AST_TOV"], 5) / 5, s["OREB_PCT"],
            s["REB"] / 50, (s["PLUS_MINUS"] + 15) / 30]
    return [r4(max(0.0, min(1.0, v))) for v in vals]


def poisson_block(hs: dict, as_: dict) -> dict:
    """Legacy Poisson engine: λ-based means are the headline, quarters are one draw."""
    pr = run_poisson_model(hs, as_)
    return {
        "lambda_h": r4(pr["lambda_h"]), "lambda_a": r4(pr["lambda_a"]),
        "mean_h": r1(4 * pr["lambda_h"]), "mean_a": r1(4 * pr["lambda_a"]),
        "home_prob": r4(pr["poisson_h_prob"]), "median_total": r1(pr["median_total"]),
        "ou_80": r1(pr["ou_80"]), "ou_20": r1(pr["ou_20"]),
        "quarters": [{"q": q["Q"], "home": q["Home"], "away": q["Away"],
                      "h_cum": q["H_Cum"], "a_cum": q["A_Cum"]} for q in pr["quarters"]],
    }


def tier(p: float, thresholds: dict) -> str:
    if p >= thresholds["safe"]:
        return "high"
    return "medium" if p >= thresholds["pick"] else "low"


# ── the forecast ─────────────────────────────────────────────────────────────

def _injuries(team, d, mode, custom) -> dict:
    if mode == "none":
        return availability(team, [], "ok", None, "none")
    if mode == "custom":
        return availability(team, custom, "ok", None, "custom")
    info = live_feed(d)
    return availability(team, espn.team_injuries(team, info["feed"]), info["status"],
                        info["fetched_at"], "live")


def _game_block(d, home, away, event) -> dict:
    ev = event or {}
    done = ev.get("completed") and ev["home"].get("score") is not None
    return {
        "date": d.isoformat(), "event_id": ev.get("event_id"),
        "tipoff_utc": ev.get("tipoff_utc"), "status": ev.get("status"),
        "home": teams.ref(home), "away": teams.ref(away),
        "result": {"home_score": ev["home"]["score"], "away_score": ev["away"]["score"]}
        if done else None,
    }


def predict_core(home_id: int, away_id: int, d: date | None = None, odds: dict | None = None,
                 injuries="live", include=("hist",)) -> dict:
    """
    The /api/predict response (§2.3 of docs/web_spec.md).

    odds: {home_ml, away_ml} manual prices, or None for the market lookup.
    injuries: "live" | "none" | {"home": [{name, status, detail?}], "away": [...]}.
    """
    home, away = teams.require(home_id), teams.require(away_id)
    if home["id"] == away["id"]:
        raise ApiError("validation", "Home and away must be different teams.")
    d = d or clock.today_et()
    include = set(include or ())
    settings = config_store.load_settings()
    thresholds, kelly = settings["thresholds"], settings["kelly"]
    caveats = []

    # Form (and Elo) as of the game date.
    elo = elo_table()
    h_form, a_form = team_stats(home, d, elo), team_stats(away, d, elo)
    hd, ad = h_form["stats"], a_form["stats"]

    # Injuries.
    if isinstance(injuries, dict):
        h_av = _injuries(home, d, "custom", injuries.get("home") or [])
        a_av = _injuries(away, d, "custom", injuries.get("away") or [])
    else:
        h_av, a_av = _injuries(home, d, injuries, None), _injuries(away, d, injuries, None)
    statuses = {h_av["status"], a_av["status"]}
    if "unavailable" in statuses:
        caveats.append("injuries_unavailable")
    elif "stale" in statuses:
        caveats.append("injuries_stale")
    h_inj = [{"name": p["name"], "status": p["status"]} for p in h_av["players"]]
    a_inj = [{"name": p["name"], "status": p["status"]} for p in a_av["players"]]

    # Engines, all on the same injury-adjusted rows (E7).
    model = model_store.get_model()
    calib = get_calibrator()
    det = predict_detail(hd, ad, model, h_inj, a_inj, calibrator=calib)
    hs, as_, mc = det["hs"], det["as"], det["mc"]
    elo_p = elo_expected(hs["ELO"] + ELO_HOME_ADV, as_["ELO"])
    mc_p, xgb_p = det["mc_p"], det["xgb_p"]

    if model is not None:
        mode = "blend_v5"
        w = dict(zip(("xgb", "mc", "elo"), (float(v) for v in calib.weights)))
        ensemble = blend_ladder(w, float(calib.bias),
                                {"xgb": xgb_p, "mc": mc_p, "elo": det["elo_diff"] / 400.0},
                                {"xgb": xgb_p, "mc": mc_p, "elo": elo_p}, calib.n_updates)
        p_home, conflict = float(det["fp"]), bool(det["conflict"])
    else:
        mode = "elo_only"
        caveats.append("elo_only_no_model")
        ensemble = elo_ladder(elo_p)
        p_home, conflict = float(elo_p), bool((elo_p > 0.5) != (mc_p > 0.5))

    # Pick and flags.
    side = "home" if p_home > 0.5 else "away"
    prob = max(p_home, 1 - p_home)
    pick = {"side": side, "team_id": (home if side == "home" else away)["id"],
            "prob": r4(prob), "tier": tier(prob, thresholds)}

    dist = distributions(mc, "hist" in include)

    # Market: manual prices, or the cached market lookup (spread/total come from it either way).
    found = None
    try:
        found = market.lookup(home, away, d)
    except ApiError:
        pass
    if odds:
        src = {"status": "manual", "source": "manual", "fetched_at": None, "stale": False,
               "home_ml": int(odds["home_ml"]), "away_ml": int(odds["away_ml"]),
               "spread": (found or {}).get("spread"), "total": (found or {}).get("total")}
    else:
        src = found
    mkt = market.market_block(src, p_home, mc["_margins"], mc["_totals"],
                              float(thresholds["value_edge"])) if src else None
    if found and found.get("stale"):
        caveats.append("odds_stale")

    flags = {
        "conflict": conflict,
        "show_pick": not conflict and prob >= thresholds["pick"],
        "safe": not conflict and prob >= thresholds["safe"],
        "vip": not conflict and prob >= thresholds["vip"],
        "value": bool(mkt and mkt["value_side"]),
    }

    if h_form["stale"] or a_form["stale"]:
        caveats.append("stats_stale")
    if d < clock.today_et():
        caveats.append("past_date_current_elo")
    if clock.is_demo():
        caveats.append("demo_mode")

    shap = None
    if "shap" in include and model is not None:
        shap = model_store.shap_explain(model, det["row"])

    from server.routers.meta import model_info          # late: routers import services
    info = model_info()
    out = {
        "as_of": clock.iso_utc(),
        "mode": mode,
        "model": {k: info[k] for k in ("version", "trained_at", "age_days")} if model is not None else None,
        "game": _game_block(d, home, away, (found or {}).get("event")),
        "win_prob": {"home": r4(p_home), "away": r4(1 - p_home)},
        "pick": pick,
        "flags": flags,
        "engines": {"xgb": r4(xgb_p), "mc": r4(mc_p), "elo": r4(elo_p), "poisson": None},
        "ensemble": ensemble,
        **dist,
        "market": mkt,
        "fair_line": market.fair_line(p_home),
        "kelly": market.kelly_block(p_home, mkt, kelly),
        "teams": {
            "home": {**form_public(hd), "adjusted": form_public(hs),
                     "as_of": h_form["as_of"], "stale": h_form["stale"]},
            "away": {**form_public(ad), "adjusted": form_public(as_),
                     "as_of": a_form["as_of"], "stale": a_form["stale"]},
        },
        "ratings": {
            side_: {"elo": r1(s["ELO"]), "elo_rank": elo_rank(elo, t["id"]),
                    "kalman": None, "adjusted": None, "player_sum": None}
            for side_, s, t in (("home", hd, home), ("away", ad, away))
        },
        "availability": {"home": h_av, "away": a_av},
        "radar": {"axes": RADAR_AXES, "home": radar_values(hd), "away": radar_values(ad)},
        "feature_row": {k: r4(v) for k, v in det["row"].iloc[0].items()},
        "caveats": caveats,
    }
    if "poisson" in include:
        out["poisson"] = poisson_block(hs, as_)
        out["engines"]["poisson"] = out["poisson"]["home_prob"]
    if "shap" in include:
        out["shap"] = shap
    return clean(out)
