"""Slice 0: prove the FoB<->clip join before any model touches audio.

Real path : --data <dir> with inspections_<year>.csv + wav files.
Blocked   : FRDR is Globus-only, so --synth builds one fake hive for a month so
            the JOIN LOGIC still ships an evidence scatter. Banner says which ran.

Writes out/slice0_scatter.svg and prints n_clips / n_labeled / n_dropped / FoB range.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.const import CLIP_EVERY_MIN, DEFAULT_YEAR  # noqa: E402
from urban.io import join_fob, list_clips, load_inspections  # noqa: E402


def clip_grid(inspections, every_min=CLIP_EVERY_MIN):
    """Real audio is Globus-blocked, but the recording cadence is documented (a
    clip every 30 min). Build the clip timestamp grid per hive across its real
    inspection span so the join runs on REAL FoB labels. Only the timestamps are
    synthetic here; the labels, hives, and dates are the dataset's own."""
    clips = []
    by_hive: dict[str, list] = {}
    for hive, ts, _ in inspections:
        by_hive.setdefault(hive, []).append(ts)
    for hive, tss in by_hive.items():
        # start one week BEFORE the first inspection so the drop is visible
        t, end = min(tss) - timedelta(days=7), max(tss) + timedelta(days=1)
        while t < end:
            clips.append((hive, t, f"{t:%d-%m-%Y_%Hh%M}_HIVE-{hive}.wav"))
            t += timedelta(minutes=every_min)
    return clips


def scatter_svg(rows, path: Path) -> None:
    """rows: [(hive, ts, path, fob)]. clip time (x) vs fob (y), one colour/hive."""
    W, H, pad = 900, 360, 50
    xs = [ts.timestamp() for _, ts, _, _ in rows]
    ys = [f for _, _, _, f in rows]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0, y1 = y0 - 1, y1 + 1
    hives = sorted({h for h, _, _, _ in rows})
    palette = ["#c8963c", "#3c78c8", "#57a05a", "#b0533f", "#7a5aa0"]
    colour = {h: palette[i % len(palette)] for i, h in enumerate(hives)}

    def px(x):
        return pad + (x - x0) / (x1 - x0 or 1) * (W - 2 * pad)

    def py(y):
        return H - pad - (y - y0) / (y1 - y0 or 1) * (H - 2 * pad)

    dots = "".join(
        f'<circle cx="{px(ts.timestamp()):.1f}" cy="{py(f):.1f}" r="2" '
        f'fill="{colour[h]}" opacity="0.7"/>'
        for h, ts, _, f in rows
    )
    axes = (
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#888"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#888"/>'
    )
    labels = (
        f'<text x="{W/2}" y="{H-12}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13">clip time (one month)</text>'
        f'<text x="16" y="{H/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(-90 16 {H/2})">frames of bees (forward-filled)</text>'
        f'<text x="{pad}" y="{pad-16}" font-family="sans-serif" font-size="13" '
        f'fill="#444">slice 0 join: {len(rows)} labeled clips, '
        f'{len(hives)} hive(s)</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>'
        f'{axes}{dots}{labels}</svg>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dir with inspections_<year>.csv")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--out", default="out/slice0_scatter.svg")
    args = ap.parse_args()

    data = Path(args.data)
    insp = load_inspections(data / f"inspections_{args.year}.csv")
    wavs = list(data.rglob("*.wav"))
    if wavs:
        clips = list_clips(data)
        banner = f"REAL labels + REAL audio timestamps from {data}"
    else:
        clips = clip_grid(insp)
        banner = (
            f"REAL inspections ({args.year}); clip timestamps from the documented "
            "30-min cadence; audio pending Globus"
        )

    rows = join_fob(clips, insp)
    if not rows:
        print("join produced 0 labeled clips", file=sys.stderr)
        return 1
    fobs = [f for _, _, _, f in rows]
    scatter_svg(rows, Path(args.out))

    print(f"[{banner}]")
    print(f"n_clips   = {len(clips)}")
    print(f"n_labeled = {len(rows)}")
    print(f"n_dropped = {len(clips) - len(rows)}  (before first inspection / unknown hive)")
    print(f"fob_range = {min(fobs):.1f} .. {max(fobs):.1f}")
    print(f"scatter   -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
