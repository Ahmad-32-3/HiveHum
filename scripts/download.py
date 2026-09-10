"""Fetch UrBAN into data/ (gitignored).

Metadata (inspections, sensor, weather CSVs) lives on GitHub over plain HTTPS
and downloads here for real. The AUDIO is Globus-only (verified 2026-09-04):
no per-file HTTPS, "Download as Zip" disabled during a backup. So audio is
BLOCKED and we fail loud with the one manual step, per DESIGN.

  python scripts/download.py            # pull the 4 metadata CSVs
  python scripts/download.py --audio    # print the Globus step, exit non-zero
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.const import FRDR_DOI, GH_RAW, GLOBUS_ENDPOINT, META_CSVS  # noqa: E402

AUDIO_BLOCKER = f"""\
AUDIO BLOCKED: UrBAN audio is Globus-only, no per-file HTTPS.

  record : https://doi.org/{FRDR_DOI}
  globus : endpoint {GLOBUS_ENDPOINT}

One-time manual step:
  1. Install Globus Connect Personal, sign in (free).
  2. On the record -> "Download with Globus", pick a CAPPED subset:
     ~50-100 h of wav from 2 hives whose Tag numbers appear in inspections_2021.csv.
  3. Transfer the wav into  data/audio/.

The metadata (labels + sensors) already downloaded here without Globus.
"""


def fetch_meta(data: Path) -> int:
    data.mkdir(parents=True, exist_ok=True)
    for name, rel in META_CSVS.items():
        url = f"{GH_RAW}/{rel}"
        dst = data / name
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"FAILED {name}: {e}", file=sys.stderr)
            return 1
        print(f"ok  {name}  ({dst.stat().st_size} bytes)")
    print(f"\nmetadata in {data}/  (audio still needs Globus: --audio)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--audio", action="store_true", help="show the Globus audio step")
    args = ap.parse_args()
    if args.audio:
        print(AUDIO_BLOCKER, file=sys.stderr)
        return 1
    return fetch_meta(Path(args.data))


if __name__ == "__main__":
    raise SystemExit(main())
