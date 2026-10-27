"""
player_elo_live — player_elo scored on what the live app knows before
tip-off: the active roster (players who play + dressed-but-unused), weighted
by minutes-if-dressed, instead of the set of players who log minutes.
Same ratings and parameters as player_elo; nothing tuned here.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.variants.player_elo import build_dataset  # noqa: F401

FEATURES = ["PE_LIVE_DIFF"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(train[FEATURES].values, train["RESULT"].values)
    return model.predict_proba(test[FEATURES].values)[:, 1]
