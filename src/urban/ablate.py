"""Slice 2: audio vs internal-temp vs both, on the SAME leave-one-hive-out split.

Reuses slice-1 features and splits. The point is the column comparison, so all
three run the identical same_hive / loho rows. Temp is NOT free signal: internal
temperature covaries with season and hive, so we print temp-only same_hive vs
loho too — if temp "wins" only by memorising the hive, its loho column collapses
the same way audio's does.
"""
from __future__ import annotations

import numpy as np

from .eval import metrics
from .predict import fit_predict, mean_baseline
from .split import loho, same_hive


def temp_matrix(hives, days, sensor_daily):
    """-> (T, mask): [mean_temp, mean_rh] per row; mask drops rows with no sensor."""
    rows, mask = [], []
    for h, d in zip(hives, days):
        v = sensor_daily.get((str(h), d))
        mask.append(v is not None)
        rows.append(v if v is not None else (0.0, 0.0))
    return np.array(rows, float), np.array(mask, bool)


def run_ablation(Xd, hives, y, days, sensor_daily, test_id):
    """-> dict feature_set -> {'same_hive': m, 'loho': m}. Same rows throughout."""
    T, mask = temp_matrix(hives, days, sensor_daily)
    Xd, hives, y, days, T = Xd[mask], hives[mask], y[mask], days[mask], T[mask]

    feats = {"audio": Xd, "temp": T, "both": np.hstack([Xd, T])}
    tr, _ = loho(hives, test_id)
    dtr, dte = same_hive(hives, days, test_id)

    out = {}
    for name, F in feats.items():
        out[name] = {
            "same_hive": metrics(y[dte], fit_predict(F[dtr], y[dtr], F[dte])),
            "loho": metrics(y[dte], fit_predict(F[tr], y[tr], F[dte])),
        }
    base = metrics(y[dte], mean_baseline(y[tr], len(dte)))  # naive: always the mean
    info = {"n_test": int(len(dte)), "n_train": int(len(tr)), "base_mae": base["mae"]}
    # skill vs baseline for the audio model (positive = better than guessing the mean)
    a = out["audio"]
    info["skill_same_pct"] = round((base["mae"] - a["same_hive"]["mae"]) / base["mae"] * 100)
    info["skill_loho_pct"] = round((base["mae"] - a["loho"]["mae"]) / base["mae"] * 100)
    info["overfit_gap_pct"] = round((a["loho"]["mae"] - a["same_hive"]["mae"]) / a["same_hive"]["mae"] * 100)
    return out, info
