"""raw/ -> processed/dataset.parquet  (one row per game, pregame features only).

Usage:
    python -m mlbpred.build_dataset --seasons 2019 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import logging

from .config import MIN_PRIOR_TEAM_GAMES, PROCESSED_DIR
from .features import build_dataset, feature_columns
from .ingest import load_raw, load_raw_optional

log = logging.getLogger(__name__)


def build(seasons: list[int], min_prior_games: int = MIN_PRIOR_TEAM_GAMES):
    games = load_raw("games", seasons)
    hitting = load_raw("team_hitting", seasons)
    pitching = load_raw("team_pitching", seasons)
    sp_logs = load_raw("sp_gamelog", seasons)
    sp_hands = load_raw("sp_hands", seasons)
    statcast_team = load_raw_optional("statcast_team", seasons)
    statcast_sp = load_raw_optional("statcast_sp", seasons)
    lineups = load_raw_optional("lineups", seasons)
    statcast_batter = load_raw_optional("statcast_batter", seasons)
    return build_dataset(games, hitting, pitching, sp_logs, sp_hands,
                         statcast_team, statcast_sp, lineups, statcast_batter,
                         min_prior_games=min_prior_games)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--min-prior-games", type=int, default=MIN_PRIOR_TEAM_GAMES)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "dataset.parquet"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build(args.seasons, args.min_prior_games)
    df.to_parquet(args.out, index=False)
    feats = feature_columns(df)
    log.info("dataset: %s games, %s features -> %s", len(df), len(feats), args.out)
    log.info("seasons: %s", df.groupby("season").size().to_dict())
    log.info("home win rate: %.4f", df["home_win"].mean())


if __name__ == "__main__":
    main()
