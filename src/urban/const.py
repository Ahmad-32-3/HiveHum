"""Frozen constants. One place, per DESIGN. Change a value here, not inline."""
import re

SR = 16000                       # UrBAN audio is released at 16 kHz
DEFAULT_YEAR = 2021
MAX_HOURS = 80                   # slice-1 audio cap; run.py hard-fails past it
HIVE_BAND_HZ = (122, 515)        # bee-band power, optional slice-1 feature

# UrBAN audio filename: DD-MM-YYYY_HHhMM_HIVE-<tag>.wav  (tag = integer hive id,
# matches inspections "Tag number"). Verified against MuSAELab/UrBAN
# feature_extraction.py:data_ls_to_string. groups: day, month, year, hr, min, tag
AUDIO_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})_(\d{2})h(\d{2})_HIVE-(\d+)\.wav$", re.IGNORECASE)

# Recording cadence UrBAN documents: a 15-min clip every 30 min.
CLIP_EVERY_MIN = 30

# Metadata (inspections, sensor, weather) is on GitHub over plain HTTPS.
GH_RAW = "https://raw.githubusercontent.com/MuSAELab/UrBAN/main"
META_CSVS = {
    "inspections_2021.csv": "data/annotations/inspections_2021.csv",
    "inspections_2022.csv": "data/annotations/inspections_2022.csv",
    "sensor_2021.csv": "data/temperature_humidity/sensor_2021.csv",
    "weather_2021_2022.csv": "data/weather_info/weather_2021_2022.csv",
}

# Only the AUDIO is Globus-only (verified 2026-09-04): no per-file HTTPS, and
# the "Download as Zip" is disabled during a backup. download.py fails loud on it.
FRDR_DOI = "10.20383/103.0972"
GLOBUS_ENDPOINT = "f163c1b3-9c88-42f6-a7bb-5839ed6c4063"
