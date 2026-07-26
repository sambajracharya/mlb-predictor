# Operating guide

Everything runs from the project folder:

```powershell
cd C:\Users\samba\cc1\mlb-predictor
```

Every command below starts with `.venv\Scripts\python` - that is the project's own Python,
with all the dependencies. Using plain `python` will fail.

---

## Daily: get today's predictions

```powershell
.venv\Scripts\python -m mlbpred.predict --date today
```

Takes ~30 seconds. Writes `reports\predictions_<today>.csv` and prints the slate.

**When to run it:** mid-to-late afternoon, roughly 3-4 hours before first pitch. Lineups get
posted progressively through the day, and the `lineups` column tells you which games are
using tonight's actual batting order:

| Value | Meaning |
|---|---|
| `posted` | real lineup - the more reliable prediction |
| `carried` | lineup not out yet, using the team's last posted nine |

Running at 4-5pm ET typically gets most of the slate `posted`. Running at noon gets almost
none. **Re-running later the same day is free and always improves the file** - it overwrites
the same filename, so there is no mess to clean up.

Also worth knowing:

- A `NaN` starting pitcher means no probable was announced. Those rows fall back to
  team-level features and are the weakest on the slate.
- In the first ~2 weeks of a season, rolling features are built on very few games. Early
  April predictions are the least trustworthy of the year.

---

## Next morning: score yesterday's predictions

Games must be final, so do this the day after.

```powershell
.venv\Scripts\python -m mlbpred.score --date 2026-07-25
```

Prints a per-game table (predicted vs actual, hit/miss) plus accuracy, log loss and run MAE
against the same baselines the backtest uses.

**Do not read anything into a single day.** 15 games carries a +/-25% confidence interval -
a coin flip returns 9/15 or better about 30% of the time. Single days are a bug check: if
predictions are wildly off or the file fails to join, something broke.

### Monthly: the number that actually means something

```powershell
.venv\Scripts\python -m mlbpred.score --since 2026-07-01 --quiet
```

Pools every saved day from that date on. `--quiet` skips the per-game table.

**What to expect** (from the walk-forward backtest over 8,100+ games):

| Metric | Healthy | Investigate if |
|---|---|---|
| accuracy | 54-57% | below 52% over 500+ games |
| log loss | 0.680-0.690 | above 0.6918 (the home-field baseline) |
| runs MAE | 2.35-2.65 | above 2.75 sustained |
| runs bias | flips sign season to season | same sign, >0.3, over 500+ games |

Roughly 500 games (~5 weeks of daily slates) before live results say anything the backtest
did not already tell you. Below that, the backtest is the better estimate of true skill.

---

## Annually: retrain after the season ends

Do this once, in the offseason (November-ish). **Not weekly, and not mid-season** - adding an
in-progress season to training measurably *hurt* on both folds tested (see README negative
results). Completed seasons are fine; partial ones are not.

Replace `2026` with whatever season just ended, and keep extending the season lists:

```powershell
# 1. make sure the finished season is fully downloaded
.venv\Scripts\python -m mlbpred.ingest   --seasons 2026 --refresh
.venv\Scripts\python -m mlbpred.statcast --seasons 2026

# 2. rebuild the feature table with every season
.venv\Scripts\python -m mlbpred.build_dataset --seasons 2021 2022 2023 2024 2025 2026

# 3. confirm it still beats the baselines BEFORE trusting new models
.venv\Scripts\python -m mlbpred.backtest --test-seasons 2024 2025 2026

# 4. train on everything complete
.venv\Scripts\python -m mlbpred.train --train-seasons 2021 2022 2023 2024 2025 2026

# 5. sanity check
.venv\Scripts\python -m pytest -q
```

**Step 3 is the gate.** If the model no longer beats `baseline:home_field` on log loss, do
not proceed to step 4 - something has broken and the old models are better than whatever you
would replace them with.

### When a new season starts

Before the first predictions of the year, download the new season so rolling features and the
schedule exist:

```powershell
.venv\Scripts\python -m mlbpred.ingest   --seasons 2027
.venv\Scripts\python -m mlbpred.statcast --seasons 2027
```

After that, `predict` refreshes the current season automatically on every run.

---

## Saving changes to GitHub

```powershell
git add -A
git commit -m "what changed"
git push
```

Prediction CSVs and downloaded data are gitignored on purpose - they regenerate. Only code,
docs, and backtest reports are tracked.

---

## Quick reference

| When | Command |
|---|---|
| Every afternoon | `.venv\Scripts\python -m mlbpred.predict --date today` |
| Next morning | `.venv\Scripts\python -m mlbpred.score --date <yesterday>` |
| Monthly | `.venv\Scripts\python -m mlbpred.score --since <date> --quiet` |
| After any code change | `.venv\Scripts\python -m mlbpred.backtest --test-seasons 2023 2024 2025 2026` |
| Once a year | the retrain block above |
