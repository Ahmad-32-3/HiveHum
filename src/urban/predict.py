"""Fit on train hive-days, score test. Ridge on standardized features."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(Xtr, ytr, Xte, alpha: float = 10.0):
    """-> yhat for Xte. Ridge; alpha modest because N (hive-days) is small."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def mean_baseline(ytr, n: int) -> np.ndarray:
    """Predict the train-mean FoB. The bar any real model must clear."""
    return np.full(n, float(np.mean(ytr)))
