"""Dumb-but-honest baselines. A model that cannot beat these is not a model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def home_field_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Every home team wins with the historical home-field rate."""
    p = float(train["home_win"].mean())
    return np.full(len(test), p)


def record_baseline(test: pd.DataFrame, home_edge: float = 0.04) -> np.ndarray:
    """Log5 on season-to-date win pct, nudged for home field.

    This is the baseline that actually hurts - a lot of "predictive" models never
    beat it.
    """
    h = test["h_win_pct_std"].fillna(0.5).clip(0.25, 0.75).to_numpy()
    a = test["a_win_pct_std"].fillna(0.5).clip(0.25, 0.75).to_numpy()
    num = h * (1 - a)
    den = num + a * (1 - h)
    p = np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)
    return np.clip(p + home_edge, 0.01, 0.99)


def mean_runs_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    return np.full(len(test), float(train[target].mean()))


def team_form_runs_baseline(test: pd.DataFrame, target: str) -> np.ndarray:
    """Average of the offense's recent runs/game and the opponent defense's runs allowed."""
    side = "h" if target.startswith("home") else "a"
    other = "a" if side == "h" else "h"
    off = test[f"{side}_bat_rpg_30"]
    dfn = test[f"{other}_pit_rapg_30"]
    return ((off + dfn) / 2).fillna(off.mean()).to_numpy()
