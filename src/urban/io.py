"""Parse UrBAN names, load inspections, forward-fill FoB onto clips.

The only non-trivial thing here is join_fob: each clip gets the LAST inspection
FoB for ITS hive at or before the clip's time. No future labels, no cross-hive
interpolation, drop clips with no prior inspection. Everything else is plumbing.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .const import AUDIO_RE, SR


def parse_audio_name(name: str) -> tuple[str, datetime]:
    """'12-06-2021_08h30_R2.wav' -> ('R2', datetime(2021,6,12,8,30))."""
    m = AUDIO_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"unknown audio layout: {name!r}")  # DESIGN: fail loud
    d, mo, y, hh, mm, hive = m.groups()
    return hive, datetime(int(y), int(mo), int(d), int(hh), int(mm))


def list_clips(audio_dir) -> list[tuple[str, datetime, str]]:
    """-> [(hive_id, ts, path)] for every parseable wav under audio_dir."""
    out = []
    for p in sorted(Path(audio_dir).rglob("*.wav")):
        hive, ts = parse_audio_name(p.name)
        out.append((hive, ts, str(p)))
    return out


def load_inspections(csv_path) -> list[tuple[str, datetime, float]]:
    """Real UrBAN inspections_YYYY.csv -> [(hive_id, ts, fob)] sorted.

    Schema: Date, Tag number, Colony Size, Fob 1st, Fob 2nd, Fob 3rd, ...
    fob = Fob 1st + Fob 2nd + Fob 3rd (NaN->0), matching the authors'
    feature_extraction.py. hive_id is the Tag number as a string (so it joins
    the HIVE-<tag> parsed from audio names). Rows with no Fob 1st are dropped.
    """
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty inspections: {csv_path}")
    need = ("Date", "Tag number", "Fob 1st")
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")  # fail loud
    out = []
    for r in rows:
        f1 = _num(r.get("Fob 1st"))
        if f1 is None:
            continue  # no primary-box count -> not a usable FoB label
        fob = f1 + (_num(r.get("Fob 2nd")) or 0.0) + (_num(r.get("Fob 3rd")) or 0.0)
        try:
            ts = _parse_date(r["Date"])
        except ValueError:
            continue
        out.append((str(r["Tag number"]).strip(), ts, fob))
    if not out:
        raise ValueError(f"no usable inspection rows in {csv_path}")
    return sorted(out)


def join_fob(clips, inspections):
    """Forward-fill FoB onto clips.

    clips:       [(hive_id, ts, path)]
    inspections: [(hive_id, ts, fob)]
    returns:     [(hive_id, ts, path, fob)]

    fob is the last inspection for THAT hive with insp_ts <= clip_ts. A clip
    before its hive's first inspection is dropped. Cross-hive never happens.
    """
    by_hive: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for hive in {h for h, _, _ in inspections}:
        ins = sorted((t, f) for h, t, f in inspections if h == hive)
        ts = np.array([t.timestamp() for t, _ in ins])
        fob = np.array([f for _, f in ins], dtype=float)
        by_hive[hive] = (ts, fob)

    out = []
    for hive, cts, path in clips:
        hp = by_hive.get(hive)
        if hp is None:
            continue  # no inspections for this hive at all
        ts, fob = hp
        i = int(np.searchsorted(ts, cts.timestamp(), side="right")) - 1
        if i < 0:
            continue  # clip precedes the first inspection -> drop
        out.append((hive, cts, path, float(fob[i])))
    return out


def load_sensor_daily(csv_path) -> dict:
    """UrBAN sensor_YYYY.csv (Date, Tag number, temperature, humidity)
    -> {(hive_id, date): (mean_temp, mean_humidity)}. Internal brood-box sensor."""
    from collections import defaultdict

    acc: dict = defaultdict(lambda: [0.0, 0.0, 0])  # temp_sum, rh_sum, n
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            t, h = _num(r.get("temperature")), _num(r.get("humidity"))
            if t is None or h is None:
                continue
            try:
                day = datetime.fromisoformat(r["Date"].strip()).date()
            except (ValueError, KeyError):
                continue
            k = (str(r["Tag number"]).strip(), day)
            acc[k][0] += t
            acc[k][1] += h
            acc[k][2] += 1
    return {k: (ts / n, hs / n) for k, (ts, hs, n) in acc.items() if n}


def load_clip(path, sr: int = SR, max_sec: float | None = None) -> np.ndarray:
    """Mono float32 in [-1,1] via stdlib wave (UrBAN wavs are 16-bit PCM mono
    16 kHz — no soundfile/FFmpeg needed). max_sec reads only a leading segment."""
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != sr:
            raise ValueError(f"{path}: {w.getframerate()} Hz, expected {sr}")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: {w.getsampwidth()*8}-bit, expected 16-bit PCM")
        n = w.getnframes() if max_sec is None else min(w.getnframes(), int(max_sec * sr))
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    ch = w.getnchannels()
    return x if ch == 1 else x.reshape(-1, ch).mean(1)


def _num(s) -> float | None:
    if s is None or str(s).strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str) -> datetime:
    s = s.strip()
    # ISO first, then D-M-Y to match the audio names' DD-MM-YYYY convention.
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparseable date: {s!r}")
