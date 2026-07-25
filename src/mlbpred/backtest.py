"""Walk-forward backtest: for each test season, train only on earlier seasons.

This is the number that matters. A random train/test split on sports data leaks
future information through team form and will flatter the model badly.

Usage:
    python -m mlbpred.backtest --test-seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .baselines import (
    home_field_baseline,
    mean_runs_baseline,
    record_baseline,
    team_form_runs_baseline,
)
from .config import REPORT_DIR
from .models import calibration_table, classification_metrics, regression_metrics
from .train import DEFAULT_COUNT_TARGETS, evaluate, feature_sets, fit_models, load_dataset

log = logging.getLogger(__name__)


def baseline_rows(train: pd.DataFrame, test: pd.DataFrame, targets: list[str]) -> list[dict]:
    y = test["home_win"].astype(int).to_numpy()
    rows = [
        {"task": "home_win", "model": "baseline:home_field",
         **classification_metrics(y, home_field_baseline(train, test))},
        {"task": "home_win", "model": "baseline:record_log5",
         **classification_metrics(y, record_baseline(test))},
    ]
    for t in targets:
        mask = test[t].notna().to_numpy()
        yt = test.loc[mask, t].to_numpy()
        rows.append({"task": t, "model": "baseline:league_mean",
                     **regression_metrics(yt, mean_runs_baseline(train, test, t)[mask])})
        rows.append({"task": t, "model": "baseline:team_form",
                     **regression_metrics(yt, team_form_runs_baseline(test, t)[mask])})
    return rows


def run(df: pd.DataFrame, test_seasons: list[int], targets: list[str],
        min_train_seasons: int = 2,
        fsets: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fsets = fsets or feature_sets(df)
    all_metrics, all_cal = [], []

    for season in sorted(test_seasons):
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        n_train_seasons = train["season"].nunique()
        if test.empty or n_train_seasons < min_train_seasons:
            log.warning("skipping %s (train seasons=%s, test games=%s)",
                        season, n_train_seasons, len(test))
            continue
        log.info("fold %s: train %s games (%s seasons) -> test %s games",
                 season, len(train), n_train_seasons, len(test))

        fitted = fit_models(train, fsets, targets)
        m = evaluate(fitted, test, fsets)
        m = pd.concat([m, pd.DataFrame(baseline_rows(train, test, targets))], ignore_index=True)
        m.insert(0, "test_season", season)
        all_metrics.append(m)

        p = fitted["win__lgbm"].predict_proba(test[fsets["win"]])[:, 1]
        cal = calibration_table(test["home_win"].astype(int).to_numpy(), p)
        cal.insert(0, "test_season", season)
        all_cal.append(cal)

    if not all_metrics:
        raise SystemExit("no usable folds - ingest more seasons")
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_cal, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=int, nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_COUNT_TARGETS)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--min-train-seasons", type=int, default=2)
    ap.add_argument("--include-statcast", action="store_true",
                    help="add team/SP Statcast columns (backtests worse - see README)")
    ap.add_argument("--include-hand", action="store_true",
                    help="add handedness-split columns (backtests worse - see README)")
    ap.add_argument("--no-lineup", action="store_true",
                    help="drop posted-lineup columns")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_dataset(args.dataset)
    fsets = feature_sets(df, include_statcast=args.include_statcast,
                         include_hand=args.include_hand,
                         include_lineup=not args.no_lineup)
    log.info("features: %s win, %s runs", len(fsets["win"]), len(fsets["runs"]))
    metrics, cal = run(df, args.test_seasons, args.targets, args.min_train_seasons, fsets)

    metrics.to_csv(REPORT_DIR / "backtest_metrics.csv", index=False)
    cal.to_csv(REPORT_DIR / "backtest_calibration.csv", index=False)

    pd.set_option("display.width", 160)
    for task, part in metrics.groupby("task", sort=False):
        cols = ["test_season", "model", "n", "log_loss", "brier", "auc", "accuracy"] \
            if task == "home_win" else ["test_season", "model", "n", "mae", "rmse", "bias"]
        cols = [c for c in cols if c in part.columns]
        print(f"\n=== {task} ===")
        print(part[cols].to_string(index=False, na_rep="-"))

    print("\n=== win probability calibration (lgbm, pooled across folds) ===")
    pooled = cal.groupby("bin").apply(
        lambda d: pd.Series(
            {
                "n": d["n"].sum(),
                "pred": np.average(d["pred"], weights=d["n"]),
                "actual": np.average(d["actual"], weights=d["n"]),
            }
        ),
        include_groups=False,
    ).reset_index()
    pooled["gap"] = pooled["pred"] - pooled["actual"]
    print(pooled.to_string(index=False))
    log.info("reports written to %s", REPORT_DIR)


if __name__ == "__main__":
    main()
