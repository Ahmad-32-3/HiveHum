# HiveHum

Ten rooftop hives recorded a year of sound and temperature. I predict colony strength (frames of bees from inspections), then test on a hive left out so the model cannot cheat by recognizing which box it is hearing.

Same-hive error is the debug column. The number I trust is MAE, RMSE, and Spearman correlation under leave-one-hive-out. Clip accuracy is not the headline.

Audio is dense. Inspections are sparse. A clip-shuffled model can look strong while only memorizing hive identity, hour, and city noise. That gap is already reported in the UrBAN follow-up work. This repo measures it under a small hour budget and adds internal temperature as a covariate, with the same held-out-hive check on the temp-only column.

## Data

[UrBAN](https://doi.org/10.20383/103.0972) (CC BY 4.0). Paper: [Abdollahi et al. 2025](https://www.nature.com/articles/s41597-025-04869-1). 10 Langstroth hives, Montreal rooftop, 2021-2022. The working cap is a frozen 50-100 hour subset.

## Run

```bash
python -m pytest tests/ -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev
```

If the download is blocked, leak tests and the walkthrough still run. The CLI prints `ILLUSTRATIVE` plus a one-line blocker.

## Layout

- `src/` features, model, eval
- `scripts/run.py`
- `tests/` hive leakage checks
- `web/` case-study page
