"""Paths and project-wide constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

API_BASE = "https://statsapi.mlb.com/api/v1"

# Regular season only. Playoff baseball is a different distribution and there is
# not enough of it to matter for the MVP.
GAME_TYPE = "R"

# Rolling windows (in team games / pitcher starts) used for form features.
TEAM_WINDOWS = (7, 15, 30)
SP_WINDOWS = (3, 5, 10)

# A team needs this many games logged in the current season before we trust its
# rolling features. Rows below the threshold are dropped from the training set.
MIN_PRIOR_TEAM_GAMES = 10

# Targets the pipeline knows how to model.
WIN_TARGET = "home_win"
COUNT_TARGETS = (
    "home_runs",
    "away_runs",
    "home_hits",
    "away_hits",
    "home_hr",
    "away_hr",
    "home_so",  # batter strikeouts (times the home lineup struck out)
    "away_so",
)
