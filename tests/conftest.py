"""Synthetic fixtures - fast, offline, and shaped exactly like the real raw files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbpred.ingest import TEAM_HITTING_STATS, TEAM_PITCHING_STATS

TEAMS = [108, 109, 110, 111]


def _make_games(seasons=(2022, 2023), games_per_season=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pk = 1000
    for season in seasons:
        start = pd.Timestamp(f"{season}-04-01")
        for i in range(games_per_season):
            day = start + pd.Timedelta(days=i)
            for home, away in ((TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])):
                pk += 1
                hs, as_ = int(rng.poisson(4.6)), int(rng.poisson(4.3))
                if hs == as_:
                    hs += 1
                rows.append(
                    {
                        "game_pk": pk, "date": day, "season": season, "game_type": "R",
                        "game_number": 1, "double_header": "N", "day_night": "night",
                        "series_game": 1, "status": "F", "detailed_status": "Final",
                        "venue_id": 3000 + home, "venue": f"Park {home}",
                        "wx_condition": "Clear", "wx_temp": 70.0, "wx_wind": "8 mph, Out To CF",
                        "home_team_id": home, "home_team": f"Team {home}", "home_abbr": str(home),
                        "home_league_id": 103, "home_div_id": 200,
                        "home_score": hs, "home_h": hs + 3, "home_e": 0,
                        "home_sp_id": 500 + home, "home_sp_name": f"SP {home}",
                        "home_record_w": 0, "home_record_l": 0,
                        "away_team_id": away, "away_team": f"Team {away}", "away_abbr": str(away),
                        "away_league_id": 103, "away_div_id": 200,
                        "away_score": as_, "away_h": as_ + 3, "away_e": 1,
                        "away_sp_id": 500 + away, "away_sp_name": f"SP {away}",
                        "away_record_w": 0, "away_record_l": 0,
                    }
                )
    return pd.DataFrame(rows)


def _make_team_logs(games: pd.DataFrame, group: str) -> pd.DataFrame:
    cols = TEAM_HITTING_STATS if group == "hitting" else TEAM_PITCHING_STATS
    rng = np.random.default_rng(7)
    rows = []
    for _, g in games.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            runs = g[f"{side}_score"] if group == "hitting" else g[f"{opp}_score"]
            row = {
                "date": g["date"], "game_pk": g["game_pk"], "season": g["season"],
                "team_id": g[f"{side}_team_id"], "opponent_id": g[f"{opp}_team_id"],
                "is_home": side == "home",
                "is_win": bool(g["home_score"] > g["away_score"]) == (side == "home"),
            }
            for c in cols:
                row[c] = float(rng.integers(0, 12))
            row["runs"] = float(runs)
            if group == "hitting":
                row["atBats"], row["plateAppearances"] = 34.0, 38.0
            else:
                row["outs"], row["battersFaced"] = 27.0, 38.0
            rows.append(row)
    return pd.DataFrame(rows)


def _make_sp_logs(games: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for _, g in games.iterrows():
        for side in ("home", "away"):
            rows.append(
                {
                    "date": g["date"], "game_pk": g["game_pk"], "season": g["season"],
                    "player_id": int(g[f"{side}_sp_id"]), "team_id": g[f"{side}_team_id"],
                    "opponent_id": g["away_team_id" if side == "home" else "home_team_id"],
                    "is_home": side == "home", "is_win": None,
                    "gamesStarted": 1.0, "outs": float(rng.integers(9, 21)),
                    "runs": float(rng.integers(0, 6)), "earnedRuns": float(rng.integers(0, 5)),
                    "hits": float(rng.integers(2, 9)), "homeRuns": float(rng.integers(0, 3)),
                    "strikeOuts": float(rng.integers(2, 11)), "baseOnBalls": float(rng.integers(0, 5)),
                    "hitByPitch": 0.0, "battersFaced": float(rng.integers(18, 28)),
                    "numberOfPitches": float(rng.integers(70, 105)), "strikes": 60.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw():
    games = _make_games()
    return {
        "games": games,
        "hitting": _make_team_logs(games, "hitting"),
        "pitching": _make_team_logs(games, "pitching"),
        "sp": _make_sp_logs(games),
    }
