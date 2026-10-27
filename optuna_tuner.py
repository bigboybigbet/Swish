"""
Standalone Optuna hyperparameter tuner for the NBA XGBoost model.
Run directly:  python optuna_tuner.py [n_trials]
Called by the Train job (server) via subprocess when Optuna HPO is enabled.
Writes best params to optuna_best_params.json on success.
"""
import json
import sys
import os

import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Single source of truth — config.py lives next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_FILE, FEATURE_COLS  # noqa: E402

BEST_PARAMS_FILE = "optuna_best_params.json"


def run_search(X_tr, y_tr, X_val, y_val, n_trials: int) -> dict:
    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 7),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 6),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            gamma=trial.suggest_float('gamma', 0.0, 0.5),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 0.3),
            reg_lambda=trial.suggest_float('reg_lambda', 0.5, 2.0),
        )
        m = XGBClassifier(**p, eval_metric='logloss', early_stopping_rounds=30,
                          random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


if __name__ == '__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    # Resolve dataset path relative to this script's directory
    base = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base, DATASET_FILE)
    output_path = os.path.join(base, BEST_PARAMS_FILE)

    if not os.path.exists(dataset_path):
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[FEATURE_COLS]
    y = df['RESULT']
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    best = run_search(X_tr, y_tr, X_val, y_val, n_trials)

    with open(output_path, 'w') as f:
        json.dump(best, f, indent=2)

    print(json.dumps(best))
