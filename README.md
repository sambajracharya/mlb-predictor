# mlb-predictor

Pregame MLB prediction MVP: **win probability + projected runs for each side**, trained on
2021-2025 and validated with a walk-forward backtest. No app, no live deployment - just the
data pipeline, the features, the models, and honest evaluation.

Everything comes from the free public [MLB Stats API](https://statsapi.mlb.com) (no key, no
scraping, no account).

## Setup from a fresh clone

No data or trained models are committed (see `.gitignore`) - everything regenerates from the
public APIs. On a new machine:

```powershell
git clone https://github.com/<your-username>/mlb-predictor.git
cd mlb-predictor

python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"

# 1. game results, box scores, probable starters, posted lineups   (~1 min)
.venv\Scripts\python -m mlbpred.ingest        --seasons 2021 2022 2023 2024 2025 2026
# 2. Statcast per-game aggregates - the slow one                   (~20 min)
.venv\Scripts\python -m mlbpred.statcast      --seasons 2021 2022 2023 2024 2025 2026
# 3. build the leakage-free feature table                          (~30 s)
.venv\Scripts\python -m mlbpred.build_dataset --seasons 2021 2022 2023 2024 2025 2026
# 4. train and save models                                         (~1 min)
.venv\Scripts\python -m mlbpred.train         --train-seasons 2021 2022 2023 2024 2025

# check everything works
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m mlbpred.backtest --test-seasons 2023 2024 2025 2026
```

Total ~25 minutes, almost all of it step 2. Steps 1-2 are resumable: re-running skips
seasons already cached, so a interrupted download costs nothing. After that, day-to-day use
is the one command in the next section.

On macOS/Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Daily use

**[OPERATIONS.md](OPERATIONS.md) is the runbook** - what to run daily, how to score yesterday,
when to retrain, and what numbers should worry you. Short version:

```powershell
cd C:\Users\samba\cc1\mlb-predictor
.venv\Scripts\python -m mlbpred.predict --date today
```

That is the whole routine. It takes ~30 seconds and:

1. re-downloads today's schedule, probable starters, **posted lineups**, and team/pitcher
   stats for the current season;
2. tops up Statcast (only the dates it does not already have);
3. rebuilds pregame features and runs the saved models;
4. writes `reports\predictions_<date>.csv`.

Useful details:

- The file is **overwritten** per date - re-running today replaces today's file rather than
  making a new one. Tomorrow gets its own file.
- **Re-run a few hours before first pitch.** Lineups drop progressively, and rows flip from
  `carried` to `posted` as they do. `posted` rows are the trustworthy ones.
- `--date 2026-08-01` for another date; add `--no-refresh` to skip the download and re-read
  cached numbers (~2s).
- `predict.ps1` is a shorter wrapper for the same thing (`.\predict.ps1`,
  `.\predict.ps1 2026-08-01`, `.\predict.ps1 today -Fast`). Windows blocks unsigned scripts
  by default, so it needs `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` run once -
  entirely optional, the `python -m` command above never needs it.

**Scoring yourself against reality:**

```powershell
.venv\Scripts\python -m mlbpred.score --date 2026-07-25      # one day
.venv\Scripts\python -m mlbpred.score --since 2026-07-01 --quiet   # pooled
```

Prints accuracy, log loss and run MAE against the same baselines the backtest uses, with a
confidence interval next to the accuracy. One slate is 15 games (±25% at 95% confidence), so
treat single days as a bug check; ~500 games is where live results start to mean anything.

**Occasional maintenance** (not daily):

```powershell
# retrain, e.g. monthly or after a season ends - takes ~1 minute
.venv\Scripts\python -m mlbpred.ingest        --seasons 2026
.venv\Scripts\python -m mlbpred.statcast      --seasons 2026
.venv\Scripts\python -m mlbpred.build_dataset --seasons 2021 2022 2023 2024 2025 2026
.venv\Scripts\python -m mlbpred.train         --train-seasons 2021 2022 2023 2024 2025

# check it still beats the baselines after any change
.venv\Scripts\python -m mlbpred.backtest --test-seasons 2023 2024 2025 2026
```

At the start of a new season, add that season number to the `--seasons` lists.

## What it predicts

```
=== 2026-07-25 ===
        away_team          home_team           venue     away_sp    home_sp   home_win_prob  pred_away_runs  pred_home_runs
 Colorado Rockies  Milwaukee Brewers  Am. Family Field  R. Feltner  R. Gasser         0.683           3.754           5.113
    New York Mets  Los Angeles ...          Citi Field  Y. Yamamoto N. McLean         0.457           4.767           3.951
```

Full slate is written to `reports/predictions_<date>.csv`.

## Results (walk-forward backtest)

Each test season is predicted by a model trained **only on earlier seasons**. Metrics from
`python -m mlbpred.backtest --test-seasons 2023 2024 2025`:

| Task | Model | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|
| Win — log loss | **lgbm** | 0.6824 | **0.6806** | **0.6811** | **0.6890** |
| Win — log loss | logreg | **0.6816** | 0.6819 | 0.6814 | 0.6909 |
| Win — log loss | baseline: home field only | 0.6924 | 0.6917 | 0.6904 | 0.6927 |
| Win — log loss | baseline: log5 on records | 0.6914 | 0.6937 | 0.6967 | 0.7155 |
| Win — accuracy | lgbm | 56.5% | 56.5% | 54.7% | 53.6% |
| Home runs — MAE | lgbm | **2.443** | **2.336** | **2.402** | **2.469** |
| Home runs — MAE | baseline: league mean | 2.473 | 2.386 | 2.431 | 2.490 |
| Away runs — MAE | lgbm | **2.488** | **2.469** | 2.621 | 2.567 |
| Away runs — MAE | baseline: league mean | 2.547 | 2.531 | 2.666 | 2.611 |

\* 2026 is a partial season (games through July).

Read this honestly:

- The edge over "always pick the home team" is **~0.010 of log loss** and ~3 points of
  accuracy. That is small, and it is supposed to be small - baseball is the noisiest of the
  major sports and roughly 55-58% is the practical ceiling for pregame prediction. Published
  betting-market accuracy sits in the same neighbourhood.
- It does beat both baselines on every fold, and the win probabilities are well calibrated:
  in the bins that hold real volume, predicted and actual agree to within ~0.02
  (`reports/backtest_calibration.csv`).
- Runs MAE improves on the league-average guess by only ~0.03 runs. Individual game scores
  are close to irreducibly random; the value is in the distribution, not the point estimate.
- **If a change makes these numbers dramatically better, suspect leakage first.** 60%+
  accuracy on pregame MLB means something from after first pitch got into the features.

## Model vs Polymarket

`python -m mlbpred.polymarket --seasons 2025 2026` downloads every MLB moneyline market from
Polymarket's public Gamma/CLOB APIs and keeps the **last traded price before first pitch**.
`python -m mlbpred.market_eval --test-seasons 2025 2026` then scores model and market on the
identical games, walk-forward (model trained only on earlier seasons).

| Season | Games | Source | Log loss | Accuracy |
|---|---|---|---|---|
| 2025 | 2,213 | model | 0.6811 | 54.9% |
| 2025 | 2,213 | **market** | **0.6779** | 55.4% |
| 2025 | 2,213 | 50/50 blend | 0.6780 | 54.9% |
| 2026 | 1,331 | model | 0.6888 | 54.2% |
| 2026 | 1,331 | market | 0.6862 | 55.4% |
| 2026 | 1,331 | **50/50 blend** | **0.6860** | 55.8% |

**Does the model add signal on top of the market?** `market_eval` fits
`P(home win) = sigmoid(b0 + b1·logit(model) + b2·logit(market))`. If b1 > 0 significantly,
the model knows something the market doesn't:

| Season | b1 (model) | z | b2 (market) | z |
|---|---|---|---|---|
| 2025 | 0.164 | 0.77 | 0.716 | 3.66 |
| 2026 | 0.187 | 0.64 | 0.663 | 2.53 |

The market coefficient is strongly significant both seasons; the model's is not (z < 1 in
both). **The market subsumes the model** - even with lineup features, which improved the
model against every other yardstick. The 2026 blend does edge the market alone (0.6860 vs
0.6862), but by an amount indistinguishable from noise.

Interpretation:

- **The market wins on pure probability quality, both seasons.** Expected - it aggregates
  lineups, injuries, and sharp money the model has never seen. Corr(model, market) is
  0.72-0.80, mean absolute gap ~0.045.
- **The model is not redundant**: the 50/50 blend beat the market alone in 2025, meaning the
  model carries *some* information the market hadn't fully priced. In 2026 it didn't.
- The flat-stake simulation (`reports/market_eval_trades.csv`) shows positive ROI at every
  disagreement threshold, but only one cell clears two standard errors (2025, edge > 0.06:
  +10.6% ROI, ±4.9%), and that's one cell out of eight - exactly the kind of thing multiple
  comparisons produce by luck. The entry price is the last *trade*, not the ask you would
  actually pay, and thin markets make big "edges" look better than they fill.
- Honest conclusion: **aligned with the market, not beating it.** The realistic path to an
  edge is speed and information (posted lineups, late scratches, weather updates) rather
  than a better rolling-average model.

### Could a thinner market be beaten instead?

The natural follow-up: moneylines are efficient, so price the *other* markets - totals, run
lines, player props - where fewer sharp participants are looking. Measured across 7,714 MLB
markets on 428 games before building anything:

| Market | Markets | Zero volume | Median volume | Live spread | Bid depth |
|---|---|---|---|---|---|
| Moneyline | 146 | 26% | $160,140 | **1c** | $8,577 |
| Game total (O/U) | 531 | 8% | $3,773 | 15c | **$21** |
| Run line | 459 | 8% | $1,185 | 5c | $412 |
| 1st inning run | 146 | 43% | $312 | 6c | $2,255 |
| Player prop | 5,309 | **75%** | **$0** | 4c | **$0.05** |
| F5 winner / spread | 625 | 32% | $27 | no book | $0 |

All 5,309 player-prop markets together traded $417k - less than three moneyline games.

Holding to resolution costs about half the spread, so a game total needs the model to be
**7.5 percentage points** better calibrated than the market on that exact line. This one is
not measurably 0.5 points better on moneylines.

The structure is consistent and it closes the thesis: **the only market deep enough to trade
is the efficient one.** The thin markets are not mispriced-and-exploitable, they are
untradeable - $21 of bids on a game total means a position you cannot exit, and five cents
on a prop means no reliable mid price to trade against.

## How leakage is prevented

This is the part that decides whether the project is real.

1. Every rolling feature goes through `features._shift_roll`, which shifts each team's (or
   pitcher's) series by one game *before* the rolling window is applied. Game G's features
   never include game G.
2. Park factors are computed from **prior seasons only** (`features.park_factors`).
3. Season-to-date records use an expanding window that is also shifted.
4. Splits are chronological. A random split leaks the future through team form; the backtest
   only ever trains on earlier seasons.
5. `tests/test_leakage.py` proves it: rewriting the last third of the season into 30-0
   blowouts leaves every earlier row's features **bit-identical**. If someone adds a leaky
   feature, that test fails.

## Feature groups

~196 columns, built per game from pregame information only:

| Group | Examples |
|---|---|
| **Posted lineup** | slot-weighted xwOBA of tonight's nine (30/90-game player windows), lineup PA experience |
| Team offense (7/15/30 game windows) | runs/game, OPS, OBP, SLG, HR/game, K% |
| Team pitching (7/15/30) | ERA, WHIP, K/9, BB/9, HR/9, runs allowed/game |
| Form & schedule | rolling win%, season-to-date win%, prior-season win%, days rest, games in last 7 days, bullpen workload proxy |
| Starting pitcher (last 3/5/10 starts) | ERA, WHIP, K/9, BB/9, K%, BB%, outs per start, days rest, pitch count last start |
| Matchup cross-terms | this offense's OPS × that starter's WHIP, K% × K% |
| Park | shrunken run factor from prior seasons |
| Weather | temperature, wind speed, wind blowing out/in, dome, rain |
| Calendar | month, day/night, doubleheader, division/interleague |
| Differentials | home-minus-away version of the key columns (`d_*`) |

The win model uses the compact 39-column differential + context set; the run models use all
196. That split was chosen by the sweep in the backtest, not by taste - the full set
overfits for classification at this sample size.

## Models

| Task | Baseline model | Stronger model |
|---|---|---|
| `home_win` | L2 logistic regression | **LightGBM (binary)** ← default |
| `home_runs`, `away_runs` | Poisson GLM | **LightGBM (Poisson objective)** ← default |

Both GBMs are deliberately small (7 leaves, `min_child_samples` 200-250, heavy L2). The
first, roomier configuration scored *worse than the linear model* and was badly overconfident
(said 74%, reality was 62%). Capacity is not the bottleneck here; signal is - which is why
the lineup features, not a bigger model, were what finally moved the number.

Logistic `C` is 0.02, tuned on the walk-forward folds (0.02 beat 0.05 and 0.15 for every
feature set tried).

Trained models and a `metadata.json` recording the exact feature lists land in `models/`.

## Layout

```
src/mlbpred/
  config.py         paths, windows, constants
  mlb_api.py        polite HTTP client (retries, small thread pool)
  ingest.py         API -> data/raw/*.parquet          (schedule, team logs, pitcher logs)
  features.py       raw -> pregame features            (all leakage control lives here)
  build_dataset.py  raw -> data/processed/dataset.parquet
  models.py         model factories + metrics
  baselines.py      home-field, log5, league-mean, team-form baselines
  train.py          fit + save models
  backtest.py       walk-forward evaluation vs baselines
  score.py          saved predictions vs what actually happened
  statcast.py       Baseball Savant -> per-game xwOBA/barrel/whiff aggregates
  polymarket.py     Polymarket Gamma/CLOB -> pregame market prices
  market_eval.py    model vs market scoring + paper-trade simulation
  predict.py        upcoming slate -> reports/predictions_<date>.csv
tests/              leakage + feature-correctness tests
```

## Known limits

- Lineups are used, but only about a third of a slate has them posted a few hours out; the
  rest fall back to the team's last posted nine (flagged `carried` in predictions).
- No injuries, no umpire assignments, no travel distance/time zones.
- Bullpen fatigue is a proxy (staff outs over the previous three games), not real reliever
  availability.
- Predictions for games without an announced probable pitcher fall back to team-level
  features; those rows are the least reliable of the slate.
- Weather is whatever the API reported for that game, which for future games is a forecast
  and for domes is nominal.

## Posted lineups: the one addition that worked

`lineups` on the schedule endpoint gives the nine posted batters per side (public ~3h before
first pitch, so pregame). Each hitter's quality comes from per-batter Statcast xwOBA rolled
over his previous 30/90 games, weighted by batting-order slot and shrunk toward the mean for
hitters without much history. The nine are aggregated to one number per team per game.

| Win log loss | 2023 | 2024 | 2025 | 2026 | mean |
|---|---|---|---|---|---|
| lgbm **with** lineups | 0.6824 | 0.6806 | 0.6811 | 0.6890 | **0.6833** |
| lgbm without | 0.6843 | 0.6817 | 0.6815 | 0.6908 | 0.6845 |
| logreg with lineups | 0.6816 | 0.6819 | 0.6814 | 0.6909 | 0.6839 |
| logreg without | 0.6821 | 0.6817 | 0.6802 | 0.6918 | 0.6839 |

LightGBM improves on **all four folds**; logistic regression is a dead tie. That split is the
interesting part: lineup quality matters *in interaction* with the opposing starter, which a
tree can represent and a linear model cannot. Run MAE improves on 3 of 4 folds for both
targets. This flipped the best win model from logreg to lgbm, so `predict.py` and
`market_eval.py` now use lgbm.

It also justifies the Statcast download after all - just at the **player** level, which is
where expected stats have small enough samples to beat outcome stats, exactly as the failed
team-level attempt suggested.

**Serving caveat.** Historical games have ~100% lineup coverage, but a slate is only about a
third posted a few hours out. `carry_forward_lineups` fills the rest with each team's most
recent posted lineup, and predictions carry a `lineups` column reading `posted` or `carried`
so you can tell which rows use tonight's actual nine. The model has effectively never seen a
missing lineup in training, so treat `carried` rows as the weaker predictions - and note that
re-running once lineups drop is the closest thing here to a real edge over a slow market.

## Things that were tried and did not work

Kept here because negative results are the expensive part of this project, and re-running
them costs hours.

Kept because negative results are the expensive part of this project, and re-running them
costs hours. Logistic regression, mean over the 2023-25 folds:

| Idea | Result | Mean log loss |
|---|---|---|
| baseline | - | **0.6813** |
| + handedness splits (team OPS vs LHP/RHP, 25-game window) | slightly worse | 0.6816 |
| Statcast *replacing* the outcome stats it should improve on | worse | 0.6822 |
| + **team-level** Statcast xwOBA, barrel%, hard-hit%, whiff%, velo | worse | 0.6826 |
| both together | worst | 0.6830 |

Why they failed: over a 15-30 game window a team's offense is already ~1,200 PA, so the
luck-stripping advantage of expected stats has washed out - xwOBA's edge is biggest in
*small* samples, which is why the same data works at the player level (see above). Handedness
splits at the team level are swamped by which opponents happened to be scheduled.

Both are one flag away if you want to revisit:

```bash
python -m mlbpred.backtest --test-seasons 2023 2024 2025 2026 --include-statcast --include-hand
```

### Deriving win probability from a joint run distribution

The obvious "next architecture" is to stop predicting the winner directly: model each side's
runs as a distribution, then compute `P(home runs > away runs)`. A full plate-appearance
simulator is the heavyweight version of the same idea. Tested before building anything -
negative binomial around the existing LGBM Poisson means, dispersion fitted by MLE, ties
split 50/50 (they go to extra innings):

| Method | 2023 | 2024 | 2025 | 2026 | mean log loss |
|---|---|---|---|---|---|
| blend: 50/50 direct + NB | 0.6817 | 0.6798 | 0.6818 | 0.6874 | 0.6827 |
| **direct classifier (current)** | 0.6824 | 0.6806 | 0.6811 | 0.6890 | **0.6833** |
| derived: NB, independent | 0.6830 | 0.6808 | 0.6842 | 0.6874 | 0.6838 |
| derived: NB + Gaussian copula | 0.6837 | 0.6807 | 0.6843 | 0.6868 | 0.6839 |
| derived: Poisson | 0.6875 | 0.6825 | 0.6880 | 0.6902 | 0.6871 |

**Derived loses to the direct classifier.** The distributional layer adds no information - it
routes the same features through a lossier path. A simulator would be a heavier, more
assumption-laden version of the approach that already lost, so it was never built.

The 50/50 blend does edge the current model by 0.0006 on 3 of 4 folds. Left out: for
comparison the lineup features were worth 0.0014 on 4/4 folds, and this costs two extra
models plus a dispersion fit in the prediction path for an effect indistinguishable from
noise.

Two useful things did fall out of the experiment:

- **Runs are heavily overdispersed.** NB beat Poisson decisively (0.6838 vs 0.6871); fitted
  dispersion r ~ 3.6-4.6 implies a variance near 9.6 where Poisson assumes ~4.5. Point
  predictions are unaffected, but **any interval derived from the Poisson run models would be
  far too narrow.** Treat `pred_home_runs` as a center, not a forecast with known spread.
- **The two teams' scores are conditionally independent** given the features (residual
  correlation 0.007-0.018). Shared park and weather effects are already absorbed by the
  feature set, which is why the copula version added nothing.

## Sensible next steps

1. **Push the lineup idea further** now that it is the one thing that worked: per-hitter
   platoon splits (posted nine vs *this* starter's hand), and lineup xwOBA weighted by the
   starter's pitch mix. The player level is where expected stats pay.
2. **Train with lineup dropout** - randomly blank lineup features during training so the
   model learns a sensible fallback for the ~2/3 of a slate that has no posted lineup yet.
   Right now `carried` rows lean on a feature the model has never seen missing.
3. **Shrink early-season features toward priors** (`w = n/(n+K)` blending with last season)
   so April games are usable instead of dropped.
4. **Bullpen quality, not just workload** - rolling reliever-only ERA/K% from team pitching
   minus the starter's line.
5. **More targets** - `home_hits`, `away_hits`, `home_hr`, `away_hr`, `home_so`, `away_so`
   are already in the dataset; pass them to `--targets` to train those models.
6. **Then generalise** - the ingest/features/train/backtest split is sport-agnostic; a new
   sport means a new `ingest.py` and `features.py`, not a new architecture.

Deliberately *not* on this list: a game simulator, and chasing thinner Polymarket markets.
Both were tested and ruled out - see the negative results above and the liquidity note below.

Extending to other sports is a matter of swapping ingest + feature modules; the training,
backtest, and leakage-test structure carries over unchanged.

## License

MIT - see [LICENSE](LICENSE).

Data comes from the public MLB Stats API, Baseball Savant, and Polymarket's public APIs.
This is a personal research project: it is not affiliated with MLB or Polymarket, and
nothing here is betting or investment advice.
