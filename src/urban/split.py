"""Leave-one-hive-out and same-hive debug splits over hive-day rows.

The whole point of the project lives here: the test hive must NEVER appear in
train. That is asserted, not hoped. same_hive is a debug split on a DIFFERENT
hive (earlier days -> train, later days -> test) to show the optimistic ceiling.
"""
from __future__ import annotations

import numpy as np


def loho(hives: np.ndarray, test_id: str):
    """-> (train_idx, test_idx). Hard-fails if the test hive leaks into train."""
    test_id = str(test_id)
    test = np.where(hives == test_id)[0]
    train = np.where(hives != test_id)[0]
    if test.size == 0:
        raise ValueError(f"test hive {test_id!r} has no rows")
    if test_id in set(hives[train]):
        raise AssertionError("LEAK: test hive present in train")  # never
    return train, test


def same_hive(hives: np.ndarray, days: np.ndarray, debug_id: str, frac: float = 0.6):
    """Optimistic debug: one hive split by time. earlier frac -> train."""
    debug_id = str(debug_id)
    idx = np.where(hives == debug_id)[0]
    if idx.size < 4:
        raise ValueError(f"debug hive {debug_id!r} has too few rows ({idx.size})")
    order = idx[np.argsort(days[idx])]
    cut = max(1, int(len(order) * frac))
    return order[:cut], order[cut:]
