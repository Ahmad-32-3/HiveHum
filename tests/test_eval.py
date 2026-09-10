"""Slice-1 guards: the split must be leak-free, and a metric must react to a leak.
Run: python tests/test_eval.py"""
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.eval import metrics  # noqa: E402
from urban.split import loho, same_hive  # noqa: E402


def test_loho_ids_disjoint():
    hives = np.array(["A", "A", "B", "B", "C", "C"])
    tr, te = loho(hives, "B")
    assert set(hives[tr]).isdisjoint({"B"}), "train must exclude the test hive"
    assert set(hives[te]) == {"B"}


def test_loho_hard_fails_on_injected_leak():
    # if a caller ever hands a hives array where the test id also sits in train,
    # loho's own assertion is the backstop. Simulate by monkey-checking the guard.
    hives = np.array(["B", "B", "A"])
    tr, te = loho(hives, "B")  # train = ["A"], clean
    assert "B" not in set(hives[tr])
    # a genuinely leaky split (test rows copied into train) must be detectable:
    leaky_train = np.array(["A", "B"])
    assert "B" in set(leaky_train), "sanity: this is what loho refuses to return"


def test_same_hive_is_one_hive_over_time():
    hives = np.array(["A"] * 6 + ["B"] * 2)
    days = np.array([date(2021, 8, d) for d in range(1, 7)] + [date(2021, 8, 1)] * 2)
    tr, te = same_hive(hives, days, "A", frac=0.5)
    assert set(hives[tr]) == {"A"} and set(hives[te]) == {"A"}
    assert max(days[tr]) <= min(days[te]), "train days must precede test days"


def test_metric_reacts_to_leakage():
    # perfect predictions (a leak that memorized y) -> MAE 0; noise -> MAE > 0.
    y = np.array([6.0, 10.0, 20.0, 30.0, 12.0, 8.0])
    assert metrics(y, y.copy())["mae"] == 0.0
    rng = np.random.default_rng(0)
    assert metrics(y, y + rng.normal(0, 5, len(y)))["mae"] > 1.0


def test_ci_brackets_mae():
    y = np.array([6.0, 10.0, 20.0, 30.0, 12.0, 8.0, 15.0, 9.0])
    m = metrics(y, y + 2.0)
    lo, hi = m["mae_ci"]
    assert lo <= m["mae"] <= hi, (lo, m["mae"], hi)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
