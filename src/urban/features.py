"""Clip -> log-mel summary vector, then aggregate to one row per hive-day.

Minimal, no librosa: scipy STFT + a numpy mel filterbank. Per clip we keep the
per-band log-mel mean/std plus the authors' hive-band (122-515 Hz) power. Daily
aggregation (mean over a hive's clips that day) cuts pseudo-replication so the
leave-one-hive-out N is honest.

# ponytail: reads only the leading FEAT_SEC of each clip. The colony hum is
# stationary; 120 s is plenty for a summary vector. Raise to the full clip if a
# feature turns out to need the tail.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import stft

from .const import HIVE_BAND_HZ, SR
from .io import load_clip

N_FFT = 1024
HOP = 512
N_MELS = 32
FEAT_SEC = 120.0


def _mel_fb(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=50.0, fmax=None):
    fmax = fmax or sr / 2

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
        ce = max(ce, lo + 1)
        hi = max(hi, ce + 1)
        fb[m - 1, lo:ce] = (np.arange(lo, ce) - lo) / max(ce - lo, 1)
        fb[m - 1, ce:hi] = (hi - np.arange(ce, hi)) / max(hi - ce, 1)
    return fb


_MEL = _mel_fb()


def clip_vector(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """-> (2*N_MELS + 1,) : log-mel mean, log-mel std, hive-band log power."""
    f, _, Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    power = np.abs(Z) ** 2  # [freq, time]
    logmel = np.log(_MEL @ power + 1e-10)  # [n_mels, time]
    band = (f >= HIVE_BAND_HZ[0]) & (f <= HIVE_BAND_HZ[1])
    hive_band = np.log(power[band].mean() + 1e-10)
    return np.concatenate([logmel.mean(1), logmel.std(1), [hive_band]])


def clip_features(labeled_clips):
    """labeled_clips: [(hive, ts, path, fob)] -> (X, meta) one row per clip.
    meta rows: (hive, date, fob)."""
    X, meta = [], []
    for hive, ts, path, fob in labeled_clips:
        x = load_clip(path, max_sec=FEAT_SEC)
        X.append(clip_vector(x))
        meta.append((hive, ts.date(), float(fob)))
    return np.array(X), meta


def prepare(data_dir, year: int, max_hours: float):
    """inspections + capped audio -> (Xd, hives, y, days, labeled_kept, hours_used).
    Shared by run.py (slice 1) and ablate.py (slice 2) so the split is identical."""
    import wave
    from pathlib import Path

    from .io import join_fob, list_clips, load_inspections

    insp = load_inspections(Path(data_dir) / f"inspections_{year}.csv")
    labeled = join_fob(list_clips(data_dir), insp)
    kept, used = [], 0.0
    for c in labeled:
        with wave.open(str(c[2]), "rb") as w:
            h = w.getnframes() / w.getframerate() / 3600.0
        if used + h > max_hours:
            continue
        kept.append(c)
        used += h
    X, meta = clip_features(kept)
    Xd, hives, y, days = daily(X, meta)
    return Xd, hives, y, days, kept, used


def daily(X: np.ndarray, meta):
    """Mean clip vector per (hive, day). -> (Xd, hives, y, days)."""
    groups: dict = defaultdict(list)
    fob: dict = {}
    for row, (hive, day, y) in zip(X, meta):
        groups[(hive, day)].append(row)
        fob[(hive, day)] = y
    keys = sorted(groups)
    Xd = np.array([np.mean(groups[k], axis=0) for k in keys])
    hives = np.array([k[0] for k in keys])
    days = np.array([k[1] for k in keys])
    y = np.array([fob[k] for k in keys])
    return Xd, hives, y, days
