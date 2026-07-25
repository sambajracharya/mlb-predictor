"""Fit the win-probability and run-scoring models and save them to models/.

Usage:
    python -m mlbpred.train --train-seasons 2019 2021 2022 2023 --test-season 2024
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, PROCESSED_DIR
from .features import (
    HAND_KEYS,
    LINEUP_KEYS,
    STATCAST_KEYS,
    _drop_keys,
    core_feature_columns,
    feature_columns,
)
from .models import (
    classification_metrics,
    make_runs_model,
    make_win_model,
    regression_metrics,
)

log = logging.getLogger(__name__)

DEFAULT_COUNT_TARGETS = ["home_runs", "away_runs"]


def load_dataset(path=None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "dataset.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def feature_sets(df: pd.DataFrame, include_statcast: bool = False,
                 include_hand: bool = False,
                 include_lineup: bool = True) -> dict[str, list[str]]:
    """Which columns each task gets. Win likes the compact differential set;
    run totals want the full level features (this offense vs that pitcher).

    Statcast/handedness columns are off by default for both - see
    `core_feature_columns` for the backtest numbers that settled it.
    """
    runs = feature_columns(df)
    lineup_cols = [c for c in runs if any(k in c for k in LINEUP_KEYS)]
    if not include_statcast:
        runs = _drop_keys(runs, STATCAST_KEYS)
    if not include_hand:
        runs = _drop_keys(runs, HAND_KEYS)
    runs = sorted(set(runs + lineup_cols)) if include_lineup else _drop_keys(runs, LINEUP_KEYS)
    return {
        "win": core_feature_columns(df, include_statcast, include_hand, include_lineup),
        "runs": runs,
    }


def _cols(fsets: dict[str, list[str]], task: str) -> list[str]:
    return fsets["win"] if task == "win" else fsets["runs"]


def fit_models(train: pd.DataFrame, fsets: dict[str, list[str]], count_targets: list[str],
               kinds=("logreg", "lgbm")) -> dict:
    """Train every (task, model kind) pair on `train`. Returns a flat dict of fitted estimators."""
    fitted: dict[str, object] = {}

    y = train["home_win"].astype(int)
    for kind in kinds:
        fitted[f"win__{kind}"] = make_win_model(kind).fit(train[fsets["win"]], y)

    runs_kinds = ["poisson" if k == "logreg" else k for k in kinds]
    for target in count_targets:
        mask = train[target].notna().to_numpy()
        X = train.loc[mask, fsets["runs"]]
        for kind in runs_kinds:
            fitted[f"{target}__{kind}"] = make_runs_model(kind).fit(X, train.loc[mask, target])
    return fitted


def evaluate(fitted: dict, test: pd.DataFrame, fsets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, model in fitted.items():
        task, kind = name.split("__")
        if task == "win":
            p = model.predict_proba(test[fsets["win"]])[:, 1]
            rows.append({"task": "home_win", "model": kind,
                         **classification_metrics(test["home_win"].astype(int).to_numpy(), p)})
        else:
            mask = test[task].notna().to_numpy()
            pred = np.asarray(model.predict(test.loc[mask, fsets["runs"]]))
            rows.append({"task": task, "model": kind,
                         **regression_metrics(test.loc[mask, task].to_numpy(), pred)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seasons", type=int, nargs="+", required=True)
    ap.add_argument("--test-season", type=int, help="held-out season for a quick sanity check")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_COUNT_TARGETS)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_dataset(args.dataset)
    fsets = feature_sets(df)

    train = df[df["season"].isin(args.train_seasons)]
    log.info("train: %s games over seasons %s | %s win features, %s runs features",
             len(train), args.train_seasons, len(fsets["win"]), len(fsets["runs"]))

    fitted = fit_models(train, fsets, args.targets)

    if args.test_season:
        test = df[df["season"] == args.test_season]
        if len(test):
            metrics = evaluate(fitted, test, fsets)
            log.info("held-out %s:\n%s", args.test_season, metrics.to_string(index=False))

    for name, model in fitted.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": args.train_seasons,
        "test_season": args.test_season,
        "targets": args.targets,
        "features": fsets,
        "models": sorted(fitted),
        "n_train_games": int(len(train)),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %s models -> %s", len(fitted), MODEL_DIR)


if __name__ == "__main__":
    main()
