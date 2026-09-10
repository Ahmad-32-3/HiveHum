"""Slice 2: audio vs temp vs both, same leave-one-hive-out split.

  python scripts/ablate.py --data data --year 2021 --test-id 3627

Writes out/ablation.json. Temp is not free signal — the temp-only loho column
shows whether it generalises across hives or just tracks the hive it trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.ablate import run_ablation  # noqa: E402
from urban.const import DEFAULT_YEAR, MAX_HOURS  # noqa: E402
from urban.eval import line  # noqa: E402
from urban.features import prepare  # noqa: E402
from urban.io import load_sensor_daily  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--test-id", default=None)
    ap.add_argument("--max-hours", type=float, default=MAX_HOURS)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    data = Path(args.data)

    Xd, hives, y, days, labeled, used = prepare(data, args.year, args.max_hours)
    present = Counter(h for h, _, _, _ in labeled)
    spread = {h: len({f for hh, _, _, f in labeled if hh == h}) for h in present}
    test_id = args.test_id or max(present, key=lambda h: (spread[h], present[h]))
    sensor = load_sensor_daily(data / f"sensor_{args.year}.csv")

    table, info = run_ablation(Xd, hives, y, days, sensor, test_id)
    print(f"test_id={test_id}  n_train={info['n_train']}  n_test={info['n_test']}  "
          f"hours_used={used:.1f}")
    for name in ("audio", "temp", "both"):
        print(f"-- {name} --")
        print(line("same_hive", table[name]["same_hive"]))
        print(line("loho", table[name]["loho"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(
        json.dumps({"test_id": str(test_id), "year": args.year, **info, "table": table}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}/ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
