import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from joblib import load
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

try:                                   # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:                    # older releases still accept cv="prefit"
    FrozenEstimator = None

from config import (BLEND_FILE, DATASET_FILE, ELO_DEFAULT, FEATURE_COLS,
                    MODEL_FILE, N_JOBS, OPTUNA_PARAMS_FILE, RECENCY_HALFLIFE_DAYS,
                    TRAIN_SEASONS)
from data.atomic import atomic_joblib_dump, atomic_path, atomic_write_json
from data.elo import save_elo, update_elo_game
from data.injuries import apply_injury_adjustment, calc_strength_factor
from engine.simulate import run_monte_carlo

# Search space — algorithm picks the best combination, no hardcoded values
_PARAM_GRID = {
    "n_estimators":     [200, 300, 400, 500, 600],
    "learning_rate":    [0.02, 0.04, 0.06, 0.08, 0.10],
    "max_depth":        [3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 5],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "gamma":            [0.0, 0.05, 0.1, 0.2],
    "reg_alpha":        [0.0, 0.05, 0.1, 0.2],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
}

# Regular season (002), playoffs (004), play-in (005).
# Excludes preseason (001) and All-Star (003) noise.
_VALID_GAME_PREFIXES = ("002", "004", "005")


# ── BlendCalibrator ────────────────────────────────────────────────────────────

class BlendCalibrator:
    """
    Online logistic regression over (xgb_prob, mc_prob, elo_diff).

    After every observed game outcome the weights are updated via gradient
    descent on log-loss — so "the numbers change after backtest" in a
    principled, data-driven way rather than by hand-tuning.

    Initial weights are neutral; they converge toward whatever combination
    of XGBoost + Monte Carlo + ELO has actually been working on your data.
    """

    def __init__(self):
        # coefficients for [xgb_p, mc_p, elo_diff_normalised]
        self.weights   = np.array([2.0, 1.5, 0.3])
        # bias initialised so that a balanced game (xgb=mc=0.5, elo=0) → 0.5
        # sigmoid(2.0*0.5 + 1.5*0.5 + bias) = 0.5  →  bias = -1.75
        self.bias      = -1.75
        self.lr        = 0.05
        self.n_updates = 0

    # ── core ──────────────────────────────────────────────────────────────

    def _feats(self, xgb_p: float, mc_p: float, elo_diff: float) -> np.ndarray:
        return np.array([xgb_p, mc_p, elo_diff / 400.0])

    def predict(self, xgb_p: float, mc_p: float, elo_diff: float = 0.0) -> float:
        logit = float(self._feats(xgb_p, mc_p, elo_diff) @ self.weights + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -15, 15)))

    def update(self, xgb_p: float, mc_p: float, elo_diff: float, actual: int):
        """One online gradient-descent step — call after every game result."""
        feats  = self._feats(xgb_p, mc_p, elo_diff)
        err    = self.predict(xgb_p, mc_p, elo_diff) - actual
        # Learning rate decays as more data accumulates
        lr_eff = self.lr / (1.0 + 0.001 * self.n_updates)
        l2     = 0.01
        self.weights -= lr_eff * (err * feats + l2 * self.weights)
        self.bias    -= lr_eff * err
        self.weights  = np.clip(self.weights, -6.0, 6.0)
        self.bias     = float(np.clip(self.bias, -3.0, 3.0))
        self.n_updates += 1

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str = None):
        atomic_write_json(path or BLEND_FILE, {
            "weights":   self.weights.tolist(),
            "bias":      self.bias,
            "n_updates": self.n_updates,
            "lr":        self.lr,
        }, indent=2)

    @classmethod
    def load(cls, path: str = None) -> "BlendCalibrator":
        bc = cls()
        p  = path or BLEND_FILE
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            bc.weights   = np.array(d["weights"])
            bc.bias      = float(d["bias"])
            bc.n_updates = int(d.get("n_updates", 0))
            bc.lr        = float(d.get("lr", 0.05))
        return bc

    def summary(self) -> dict:
        labels = ["XGBoost", "Monte Carlo", "ELO diff"]
        return {
            "weights":   dict(zip(labels, self.weights.round(4).tolist())),
            "bias":      round(self.bias, 4),
            "n_updates": self.n_updates,
        }


# Module-level cache so we don't re-read the JSON on every prediction call
_calib_cache:  "BlendCalibrator | None" = None
_calib_mtime:  float = 0.0


def get_calibrator() -> BlendCalibrator:
    global _calib_cache, _calib_mtime
    mtime = os.path.getmtime(BLEND_FILE) if os.path.exists(BLEND_FILE) else 0.0
    if _calib_cache is None or mtime > _calib_mtime:
        _calib_cache = BlendCalibrator.load()
        _calib_mtime = mtime
    return _calib_cache


def load_model(path: str = None):
    """Load the saved model as-is (no refit); None if the file doesn't exist."""
    p = path or MODEL_FILE
    return load(p) if os.path.exists(p) else None


# ── helpers ───────────────────────────────────────────────────────────────────

def get_feature_importance(model) -> dict:
    """Works with raw XGBClassifier or CalibratedClassifierCV."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        ests = [_base_estimator(getattr(cc, "estimator", None))
                for cc in model.calibrated_classifiers_]
        imps = [e.feature_importances_ for e in ests
                if hasattr(e, "feature_importances_")]
        imp = np.mean(imps, axis=0) if imps else np.zeros(len(FEATURE_COLS))
    else:
        imp = np.zeros(len(FEATURE_COLS))
    return dict(zip(FEATURE_COLS, imp.tolist()))


def _calibrate_prefit(est, X_cal, y_cal, method: str = "sigmoid"):
    """Calibrate an already-fitted classifier on a held-out split."""
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(est), method=method)
    else:
        model = CalibratedClassifierCV(est, cv="prefit", method=method)
    return model.fit(X_cal, y_cal)


def _base_estimator(est):
    """Strip a FrozenEstimator wrapper, if any."""
    return getattr(est, "estimator", est) if FrozenEstimator is not None \
        and isinstance(est, FrozenEstimator) else est


def unwrap_xgb(model):
    """Extract raw XGBClassifier from a CalibratedClassifierCV (needed by SHAP)."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return _base_estimator(model.calibrated_classifiers_[0].estimator)
    return model


def _recency_weights(df: pd.DataFrame,
                     halflife_days: int = RECENCY_HALFLIFE_DAYS) -> np.ndarray:
    """Exponential decay so recent games carry more weight during training."""
    dates    = pd.to_datetime(df["GAME_DATE"])
    days_ago = (dates.max() - dates).dt.days.values
    w = np.exp(-np.log(2) * days_ago / halflife_days)
    return w / w.mean()


def _search_params(X: pd.DataFrame, y: pd.Series, report) -> dict:
    """
    RandomizedSearchCV with TimeSeriesSplit — returns the best hyperparameter
    dict found across 20 random candidates × 3 time-ordered folds.
    No single number here is hardcoded; they all come from your data.
    """
    report("Running RandomizedSearchCV (20 candidates × 3 time-series folds)…")
    tscv   = TimeSeriesSplit(n_splits=3)
    # Parallelism comes from the search (N_JOBS fits at once, one thread each);
    # all cores on both levels oversubscribed the CPU.
    base   = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)
    search = RandomizedSearchCV(
        base, _PARAM_GRID,
        n_iter=20, scoring="neg_log_loss",
        cv=tscv, n_jobs=N_JOBS, random_state=42, verbose=0,
        refit=True,
    )
    search.fit(X, y)
    bp = search.best_params_
    report(
        f"Best params — depth={bp['max_depth']} "
        f"lr={bp['learning_rate']:.3f} "
        f"n_est={bp['n_estimators']} "
        f"cv_loss={-search.best_score_:.4f}"
    )
    return bp


# ── dataset ───────────────────────────────────────────────────────────────────

def _calc_streak(s: pd.Series) -> pd.Series:
    out, v = np.zeros(len(s)), 0
    for i, x in enumerate(s):
        v = (v + 1 if v >= 0 else 1) if x == 1 else (v - 1 if v <= 0 else -1)
        out[i] = v
    return pd.Series(out, index=s.index)


def fetch_game_logs(seasons=None, report=print) -> pd.DataFrame | None:
    """Team game logs (one row per team per game) from the NBA API."""
    if seasons is None:
        seasons = TRAIN_SEASONS
    all_games = []
    for s in seasons:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=s, league_id_nullable="00"
            )
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1.0)
        except Exception as e:
            report(f"Could not fetch {s}: {e}")
    if not all_games:
        return None
    return pd.concat(all_games, ignore_index=True)


def build_master_dataset(seasons=None, status_cb=None):
    """
    v5 — MATCHUP-LEVEL dataset, downloaded from the NBA API.

    Side effect: saves up-to-date Elo ratings to ELO_FILE so live
    predictions use real ratings instead of the 1500 default.
    """
    def report(msg):
        print(msg) if status_cb is None else status_cb(msg)

    raw = fetch_game_logs(seasons, report)
    if raw is None:
        return None
    out, elo = build_matchup_dataset(raw, report)
    save_elo(elo)
    report("Elo ratings replayed chronologically and saved (live ratings updated)")
    with atomic_path(DATASET_FILE) as tmp:
        out.to_csv(tmp, index=False)
    report(f"Dataset built: {len(out):,} matchup rows saved to {DATASET_FILE}")
    return out


def build_matchup_dataset(raw: pd.DataFrame, report=print) -> tuple:
    """
    Pure feature builder: team game logs (leaguegamefinder format) →
    (matchup rows, final Elo dict keyed by team name).

    Every row is one game: home team's rolling form vs away team's rolling
    form, expressed as differentials, plus a chronologically-computed
    pre-game MOV-Elo difference.  All rolling stats use shift(1) so nothing
    from the game itself leaks into its own features.  No I/O, so the
    research benchmark runs the exact same code on historical data.
    """
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(_VALID_GAME_PREFIXES)]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"]   = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["RESULT"]    = (df["WL"] == "W").astype(int)
    needed = ["PLUS_MINUS", "PTS", "FGM", "FGA", "FG3M",
              "FTA", "AST", "TOV", "REB", "OREB", "DREB", "WL"]
    df = df.dropna(subset=needed)
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # ── per-game raw rates ────────────────────────────────────────────────
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["EFG"]  = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, 1)
    df["TOVR"] = df["TOV"] / df["POSS"].replace(0, 1)
    df["ORBR"] = df["OREB"] / (df["OREB"] + df["DREB"]).replace(0, 1)
    df["FTR"]  = df["FTA"] / df["FGA"].replace(0, 1)
    df["PA"]   = df["PTS"] - df["PLUS_MINUS"]

    # ── rolling form (shift(1) = only past games) ─────────────────────────
    g = df.groupby("TEAM_ID")

    def roll(col, w):
        return g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())

    df["NET5"]   = roll("PLUS_MINUS", 5)
    df["NET15"]  = roll("PLUS_MINUS", 15)
    df["PTS10"]  = roll("PTS", 10)
    df["PA10"]   = roll("PA", 10)
    df["EFG10"]  = roll("EFG", 10)
    df["TOV10"]  = roll("TOVR", 10)
    df["ORB10"]  = roll("ORBR", 10)
    df["FTR10"]  = roll("FTR", 10)
    df["PACE10"] = roll("POSS", 10)
    df["WPCT10"] = roll("RESULT", 10)

    df["WIN_STREAK"] = g["RESULT"].transform(_calc_streak)
    df["WIN_STREAK"] = df.groupby("TEAM_ID")["WIN_STREAK"].shift(1)
    df["PREV_DATE"]  = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["DAYS_REST"]  = (df["GAME_DATE"] - df["PREV_DATE"]).dt.days.fillna(3).clip(1, 10)
    df["IS_B2B"]     = (df["DAYS_REST"] <= 1).astype(int)

    # venue-specific form: win% over last 10 games at the SAME venue type
    df["VENUE_WPCT"] = df.groupby(["TEAM_ID", "IS_HOME"])["RESULT"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # ── pair home/away rows into one matchup row per game ─────────────────
    team_feats = ["NET5", "NET15", "PTS10", "PA10", "EFG10", "TOV10",
                  "ORB10", "FTR10", "PACE10", "WPCT10", "WIN_STREAK",
                  "DAYS_REST", "IS_B2B", "VENUE_WPCT"]
    keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "PLUS_MINUS", "RESULT"] + team_feats

    home  = df[df["IS_HOME"] == 1][keep]
    away  = df[df["IS_HOME"] == 0][keep]
    games = home.merge(away, on="GAME_ID", suffixes=("_H", "_A"))
    games = games.sort_values("GAME_DATE_H").reset_index(drop=True)
    report(f"Paired {len(games):,} matchup rows from {len(df):,} team rows")

    # ── chronological pre-game Elo (no leakage) ───────────────────────────
    # Keyed by TEAM_ID so franchise renames (Bobcats → Hornets) keep their
    # rating; exported by latest team name for the live lookup.
    elo: dict = {}
    names: dict = {}
    elo_h, elo_a = [], []
    for r in games.itertuples(index=False):
        h, a = r.TEAM_ID_H, r.TEAM_ID_A
        names[h], names[a] = r.TEAM_NAME_H, r.TEAM_NAME_A
        elo_h.append(elo.get(h, ELO_DEFAULT))
        elo_a.append(elo.get(a, ELO_DEFAULT))
        update_elo_game(h, a, r.RESULT_H == 1, float(r.PLUS_MINUS_H), elo)
    games["ELO_H"], games["ELO_A"] = elo_h, elo_a
    elo_by_name = {names[t]: v for t, v in elo.items()}

    # ── matchup differential features ─────────────────────────────────────
    out = pd.DataFrame({
        "GAME_DATE":   games["GAME_DATE_H"],
        "GAME_ID":     games["GAME_ID"],
        "ELO_DIFF":    games["ELO_H"]       - games["ELO_A"],
        "NET5_DIFF":   games["NET5_H"]      - games["NET5_A"],
        "NET15_DIFF":  games["NET15_H"]     - games["NET15_A"],
        "PTS10_DIFF":  games["PTS10_H"]     - games["PTS10_A"],
        "PA10_DIFF":   games["PA10_H"]      - games["PA10_A"],
        "EFG10_DIFF":  games["EFG10_H"]     - games["EFG10_A"],
        "TOV10_DIFF":  games["TOV10_H"]     - games["TOV10_A"],
        "ORB10_DIFF":  games["ORB10_H"]     - games["ORB10_A"],
        "FTR10_DIFF":  games["FTR10_H"]     - games["FTR10_A"],
        "PACE10_MEAN": (games["PACE10_H"]   + games["PACE10_A"]) / 2,
        "WPCT10_DIFF": games["WPCT10_H"]    - games["WPCT10_A"],
        "STREAK_DIFF": games["WIN_STREAK_H"] - games["WIN_STREAK_A"],
        "REST_DIFF":   games["DAYS_REST_H"] - games["DAYS_REST_A"],
        "H_B2B":       games["IS_B2B_H"],
        "A_B2B":       games["IS_B2B_A"],
        "H_HOME_WPCT": games["VENUE_WPCT_H"],
        "A_AWAY_WPCT": games["VENUE_WPCT_A"],
        "RESULT":      games["RESULT_H"],
    })
    out["MARGIN"] = games["PLUS_MINUS_H"]
    out = out.dropna(subset=FEATURE_COLS + ["RESULT"]).reset_index(drop=True)
    return out, elo_by_name


# ── training ──────────────────────────────────────────────────────────────────

def train_model(force_rebuild: bool = False, use_optuna: bool = False,
                status_cb=None) -> tuple:
    """
    Build or reload the model using a proper ML pipeline:
      1. Temporal 85/7/8 split (respects time ordering)
      2. RandomizedSearchCV or Optuna to find best hyperparameters
      3. XGBoost trained with exponential recency weighting
      4. CalibratedClassifierCV(sigmoid) for calibrated probabilities
    Returns (model, row_count, metrics) or (None, 0, None) on failure.
    """
    def report(msg):
        print(msg) if status_cb is None else status_cb(msg)

    needs_rebuild = force_rebuild or not os.path.exists(DATASET_FILE)
    df = None

    if not needs_rebuild:
        df = pd.read_csv(DATASET_FILE)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            report(f"Dataset missing columns {missing} — rebuilding")
            needs_rebuild = True
            df = None

    if needs_rebuild:
        report(f"Downloading {len(TRAIN_SEASONS)} seasons of game data…")
        df = build_master_dataset(status_cb=status_cb)

    if df is None or df.empty:
        return None, 0, None

    from engine import team_strength
    report("Rebuilding Kalman-Elo team strength…")
    team_strength.refresh(status_cb=report)

    model, metrics = fit_matchup_model(df, use_optuna=use_optuna, report=report)
    atomic_joblib_dump(model, MODEL_FILE)
    report(
        f"Model saved — val_acc={metrics['val_acc']:.3f} "
        f"val_loss={metrics['val_loss']:.4f} "
        f"n_train={metrics['n_train']:,}"
    )
    return model, len(df), metrics


def fit_matchup_model(df: pd.DataFrame, use_optuna: bool = False,
                      report=print) -> tuple:
    """Fit + calibrate on a matchup dataset; returns (model, metrics). No I/O."""
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    X  = df[FEATURE_COLS]
    y  = df["RESULT"]

    # Temporal 85 / 7 / 8 split — small cal set is enough for sigmoid
    # (Platt) calibration, which only fits 2 parameters
    n       = len(df)
    tr_end  = int(n * 0.85)
    cal_end = int(n * 0.92)

    X_tr,  y_tr  = X.iloc[:tr_end],        y.iloc[:tr_end]
    X_cal, y_cal = X.iloc[tr_end:cal_end],  y.iloc[tr_end:cal_end]
    X_val, y_val = X.iloc[cal_end:],        y.iloc[cal_end:]

    # Find best hyperparameters
    if use_optuna:
        _tuner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optuna_tuner.py")
        report("Running Optuna HPO (30 trials)…")
        proc = subprocess.run(
            [sys.executable, _tuner, "30"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode == 0 and os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE) as f:
                best_params = json.load(f)
            report("Optuna params loaded.")
        else:
            report("Optuna failed — falling back to RandomizedSearchCV")
            best_params = _search_params(X_tr, y_tr, report)
    else:
        best_params = _search_params(X_tr, y_tr, report)

    # Sanitise params (remove any keys not in our grid, e.g. eval_metric)
    valid_keys = set(_PARAM_GRID.keys())
    clean_params = {k: v for k, v in best_params.items() if k in valid_keys}

    # Train with exponential recency weighting
    report("Training XGBoost with recency-weighted samples…")
    sw  = _recency_weights(df.iloc[:tr_end])
    xgb = XGBClassifier(**clean_params, eval_metric="logloss", random_state=42, n_jobs=N_JOBS)
    xgb.fit(X_tr, y_tr, sample_weight=sw, verbose=False)

    # Calibrate on the cal split — sigmoid (Platt scaling) only fits 2
    # parameters so it stays stable even with a small calibration set
    if len(X_cal) >= 20:
        report("Calibrating probabilities (Platt sigmoid)…")
        model = _calibrate_prefit(xgb, X_cal, y_cal)
    else:
        report("Not enough calibration data — skipping calibration step")
        model = xgb

    vproba = model.predict_proba(X_val)[:, 1]
    vp     = model.predict(X_val)
    metrics = {
        "val_acc":    float(accuracy_score(y_val, vp)),
        "val_loss":   float(log_loss(y_val, vproba)),
        "n_train":    len(X_tr),
        "n_val":      len(X_val),
        "best_iter":  clean_params.get("n_estimators", "N/A"),
        "best_params": clean_params,
        "feat_imp":   get_feature_importance(model),
        "val_proba":  vproba.tolist(),
        "val_y":      y_val.tolist(),
    }
    return model, metrics


# ── prediction ────────────────────────────────────────────────────────────────

def build_matchup_row(hd: dict, ad: dict) -> pd.DataFrame:
    """One model-ready row of matchup differentials from two team dicts."""
    vals = [
        hd["ELO"]        - ad["ELO"],
        hd["NET5"]       - ad["NET5"],
        hd["NET15"]      - ad["NET15"],
        hd["PTS10"]      - ad["PTS10"],
        hd["PA10"]       - ad["PA10"],
        hd["EFG10"]      - ad["EFG10"],
        hd["TOV10"]      - ad["TOV10"],
        hd["ORB10"]      - ad["ORB10"],
        hd["FTR10"]      - ad["FTR10"],
        (hd["PACE10"]    + ad["PACE10"]) / 2,
        hd["WPCT10"]     - ad["WPCT10"],
        hd["WIN_STREAK"] - ad["WIN_STREAK"],
        hd["DAYS_REST"]  - ad["DAYS_REST"],
        hd["IS_B2B"],
        ad["IS_B2B"],
        hd["HOME_WPCT"],
        ad["AWAY_WPCT"],
    ]
    return pd.DataFrame([vals], columns=FEATURE_COLS)


def injury_adjusted(hd: dict, ad: dict, h_inj=None, a_inj=None) -> tuple:
    """
    (hs, as_, hf, af): both team dicts after the injury adjustment, plus
    their strength factors.  This is the one place the adjustment happens:
    XGBoost, Monte Carlo, Elo and SHAP all read these same rows.
    """
    hf = calc_strength_factor(h_inj) if h_inj else 1.0
    af = calc_strength_factor(a_inj) if a_inj else 1.0
    return apply_injury_adjustment(hd, hf), apply_injury_adjustment(ad, af), hf, af


def team_strength_detail(hd: dict, ad: dict, hf: float, af: float,
                         game_date=None) -> dict | None:
    """Kalman-Elo prediction (engine.team_strength) or None without a state.
    Injury factors become points: a star out (factor ~0.7) ≈ −4.5 points."""
    from engine import team_strength
    state = team_strength.get_state()
    if state is None:
        return None
    date = game_date or hd.get("GAME_DATE")
    try:
        return team_strength.predict(
            state, hd.get("team_id") or hd["name"], ad.get("team_id") or ad["name"],
            date, home_pts_lost=(1.0 - hf) * 15.0, away_pts_lost=(1.0 - af) * 15.0)
    except KeyError:
        return None


def _recentre_margins(mc: dict, margin: float):
    """Shift the simulated scores so the margin distribution is centred on
    the team-strength margin; its spread and the totals are unchanged.
    mc_home_prob keeps the raw simulation as a reference engine."""
    delta = margin - float(np.mean(mc["_margins"]))
    mc["_h_sims"] = mc["_h_sims"] + delta / 2
    mc["_a_sims"] = mc["_a_sims"] - delta / 2
    mc["_margins"] = mc["_h_sims"] - mc["_a_sims"]
    m = mc["_margins"]
    mc["spread_margin"] = float(np.mean(m))
    mc["spread_lo"], mc["spread_hi"] = (float(x) for x in np.percentile(m, [25, 75]))
    mc["ot_prob"] = float(np.mean(np.abs(m) <= 3))


def predict_detail(hd: dict, ad: dict, model, h_inj=None, a_inj=None,
                   calibrator: "BlendCalibrator | None" = None, game_date=None) -> dict:
    """
    predict_hybrid's numbers as a dict, plus the injury-adjusted rows they
    came from (hs, as, row) so explanations use exactly what the model saw.

    v6: when a team-strength state exists (engine.team_strength), fp and
    the spread come from the Kalman-Elo margin model; XGBoost and Monte
    Carlo stay as reference engines.  Without it, model=None skips XGBoost
    and the blend: xgb_p, fp and conflict are then None and the caller
    decides the final number.
    """
    hs, as_, hf, af = injury_adjusted(hd, ad, h_inj, a_inj)
    row = build_matchup_row(hs, as_)

    xgb_p = float(model.predict_proba(row)[0][1]) if model is not None else None

    mc       = run_monte_carlo(hs, as_)
    mc_p     = mc["mc_home_prob"]
    elo_diff = hs.get("ELO", ELO_DEFAULT) - as_.get("ELO", ELO_DEFAULT)

    ts = team_strength_detail(hd, ad, hf, af, game_date)
    fp = conflict = None
    spread = mc["spread_margin"]
    if ts is not None:
        fp       = ts["p_home"]
        spread   = ts["exp_margin"]
        conflict = (fp > 0.5) != (mc_p > 0.5)
        _recentre_margins(mc, spread)
    elif xgb_p is not None:
        calib    = calibrator or get_calibrator()
        fp       = calib.predict(xgb_p, mc_p, elo_diff)
        conflict = (xgb_p > 0.5) != (mc_p > 0.5)
    return {
        "fp": fp, "xgb_p": xgb_p, "mc_p": mc_p, "elo_diff": elo_diff,
        "conflict": conflict, "mc": mc, "hf": hf, "af": af,
        "hs": hs, "as": as_, "row": row, "ts": ts, "spread": spread,
    }


def predict_hybrid(hd: dict, ad: dict, model,
                   h_inj=None, a_inj=None) -> tuple:
    """
    Returns (fp, spread, total, xgb_p, mc_p, conflict, mc_dict, hf, af).

    v6: fp and spread come from the Kalman-Elo team strength model when its
    state exists (built by Train); otherwise from the v5 XGBoost blend.
    The total comes from the Monte Carlo simulation.
    """
    d  = predict_detail(hd, ad, model, h_inj, a_inj)
    mc = d["mc"]
    return (d["fp"], d["spread"], mc["median_total"], d["xgb_p"], d["mc_p"],
            d["conflict"], mc, d["hf"], d["af"])
