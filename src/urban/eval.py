"""MAE / RMSE / Spearman with a bootstrap CI on MAE. N is small here on purpose,
so every metric ships with its uncertainty and n_test, per the evidence bar."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def metrics(y, yhat, n_boot: int = 2000, seed: int = 0) -> dict:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    n = len(y)
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    # Spearman is undefined if either side is constant (common with sparse FoB).
    if n > 2 and np.ptp(y) > 0 and np.ptp(yhat) > 0:
        rho = round(float(spearmanr(y, yhat).correlation), 2)
    else:
        rho = None

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boot.append(np.mean(np.abs(y[i] - yhat[i])))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # mean absolute % error, relative to the true count (FoB >= 1 here, no /0)
    mape = float(np.mean(np.abs(y - yhat) / np.clip(np.abs(y), 1e-6, None)) * 100)

    return {
        "mae": round(mae, 2),
        "mae_ci": [round(float(lo), 2), round(float(hi), 2)],
        "rmse": round(rmse, 2),
        "mape": round(mape, 1),
        "spearman": rho,  # None when a side is constant
        "n_test": n,
    }


def line(tag: str, m: dict) -> str:
    noisy = "  (!! n<15, noisy)" if m["n_test"] < 15 else ""
    rho = "n/a " if m["spearman"] is None else f"{m['spearman']:.2f}"
    return (
        f"{tag:>10}  MAE {m['mae']:.2f} CI{m['mae_ci']}  "
        f"RMSE {m['rmse']:.2f}  rho {rho}  n={m['n_test']}{noisy}"
    )
