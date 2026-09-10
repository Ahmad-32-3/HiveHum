"""Slice 1: leave-one-hive-out FoB regression, printed next to same-hive debug.

  python scripts/run.py --data data --year 2021 --test-id 3690 --max-hours 80

Writes out/metrics.json (the loho row) and out/preds.jsonl. Exit 0 only after the
loho vs same-hive table prints and the hours cap held.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.const import DEFAULT_YEAR, MAX_HOURS  # noqa: E402
from urban.eval import line, metrics  # noqa: E402
from urban.features import prepare  # noqa: E402
from urban.predict import fit_predict, mean_baseline  # noqa: E402
from urban.split import loho, same_hive  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--test-id", default=None, help="held-out hive; default = most-sampled")
    ap.add_argument("--max-hours", type=float, default=MAX_HOURS)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    data = Path(args.data)

    Xd, hives, y, days, labeled, used = prepare(data, args.year, args.max_hours)
    if not labeled:
        print("no labeled clips", file=sys.stderr)
        return 1

    present = Counter(h for h, _, _, _ in labeled)
    if len(present) < 2:
        print(f"need >=2 hives, have {dict(present)}", file=sys.stderr)
        return 1
    # Test the hive whose in-window FoB actually varies (else same-hive/Spearman
    # are degenerate). Tie-break on clip count.
    spread = {h: len({f for hh, _, _, f in labeled if hh == h}) for h in present}
    test_id = args.test_id or max(present, key=lambda h: (spread[h], present[h]))

    # One hive, two ways. same_hive: 3627's early days -> late days (optimistic,
    # sees this hive in train). loho: train on OTHER hives only, score the SAME
    # late days. The MAE gap on identical test rows is the hive-identity leak.
    tr, _te_full = loho(hives, test_id)          # asserts test hive not in train
    dtr, dte = same_hive(hives, days, test_id)
    m_same = metrics(y[dte], fit_predict(Xd[dtr], y[dtr], Xd[dte]))
    m_loho = metrics(y[dte], fit_predict(Xd[tr], y[tr], Xd[dte]))
    m_base = metrics(y[dte], mean_baseline(y[tr], len(dte)))
    yhat = fit_predict(Xd[tr], y[tr], Xd[dte])   # loho preds for the record
    te = dte

    # skill vs the naive mean baseline: positive = better than guessing the average
    skill_same = round((m_base["mae"] - m_same["mae"]) / m_base["mae"] * 100, 0)
    skill_loho = round((m_base["mae"] - m_loho["mae"]) / m_base["mae"] * 100, 0)
    overfit_gap = round((m_loho["mae"] - m_same["mae"]) / m_same["mae"] * 100, 0)

    print(f"hives={dict(present)}  hours_used={used:.1f}  test_id={test_id}  "
          f"fob_spread={sorted({f for hh,_,_,f in labeled if hh==test_id})}")
    print(line("same_hive", m_same))
    print(line("loho", m_loho))
    print(line("mean_base", m_base))
    print(f"skill vs baseline: same_hive {skill_same:+.0f}%  loho {skill_loho:+.0f}%  "
          f"| held-out error is {overfit_gap:+.0f}% vs same-hive")
    if not (m_same["mae"] <= m_loho["mae"] <= m_base["mae"]):
        print("note: expected same_hive <= loho <= mean_base; small n, read the CIs")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        **m_loho,
        "same_hive_mae": m_same["mae"],
        "same_hive_mape": m_same["mape"],
        "loho_mape": m_loho["mape"],
        "mean_base_mae": m_base["mae"],
        "skill_same_pct": skill_same,
        "skill_loho_pct": skill_loho,
        "overfit_gap_pct": overfit_gap,
        "n_train": int(len(tr)),
        "train_ids": sorted({str(h) for h in hives[tr]}),
        "test_id": str(test_id),
        "hours_used": round(used, 1),
        "year": args.year,
        "features": "audio",
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (out / "preds.jsonl").open("w", encoding="utf-8") as fh:
        for j, i in enumerate(te):
            fh.write(json.dumps({
                "hive_id": str(hives[i]), "day": str(days[i]),
                "y_fob": float(y[i]), "yhat": round(float(yhat[j]), 2),
                "split": "loho",
            }) + "\n")
    print(f"wrote {out}/metrics.json, {out}/preds.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
