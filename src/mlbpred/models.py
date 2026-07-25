"""Model factories and evaluation metrics.

Two models per task on purpose: a boring linear baseline that is hard to fool,
and a gradient-boosted version that should beat it. If the GBM does not beat the
linear model, the features are the problem, not the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_win_model(kind: str):
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                # C tuned on the walk-forward folds: 0.02 beat 0.05 and 0.15 for
                # every feature set tried. Small data, correlated features.
                ("clf", LogisticRegression(C=0.02, max_iter=2000)),
            ]
        )
    if kind == "lgbm":
        # Deliberately tiny trees and hard shrinkage. Game outcomes are close to
        # coin flips; anything with capacity memorises the training seasons and
        # comes out overconfident (see reports/backtest_calibration.csv).
        return LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.4,
            reg_lambda=20.0,
            verbose=-1,
        )
    raise ValueError(kind)


def make_runs_model(kind: str):
    if kind == "poisson":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("reg", PoissonRegressor(alpha=1.0, max_iter=2000)),
            ]
        )
    if kind == "lgbm":
        return LGBMRegressor(
            objective="poisson",
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=250,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.4,
            reg_lambda=20.0,
            verbose=-1,
        )
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, (p >= 0.5).astype(int))),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "bias": float(np.mean(pred - y_true)),
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y_true})
    out = df.groupby("bin").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    out["gap"] = out["pred"] - out["actual"]
    return out.reset_index()
