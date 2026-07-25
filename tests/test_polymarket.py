from __future__ import annotations

import pandas as pd

from mlbpred.polymarket import GAME_SLUG, match_to_games


def _games():
    return pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "date": pd.to_datetime(["2025-06-14"] * 3 + ["2025-06-15"]),
            "home_team": ["Boston Red Sox", "Chicago White Sox", "New York Mets",
                          "Boston Red Sox"],
            "away_team": ["New York Yankees", "Chicago Cubs", "Los Angeles Dodgers",
                          "New York Yankees"],
        }
    )


def _market(outcome_0, outcome_1, date_key="2025-06-14", price=0.6):
    return {
        "slug": f"mlb-x-y-{date_key}", "title": "t", "date_key": date_key,
        "outcome_0": outcome_0, "outcome_1": outcome_1,
        "token_0": "a", "token_1": "b", "final_0": 1.0,
        "price_0_pregame": price, "volume": 1000.0,
        "game_start": pd.Timestamp(date_key, tz="UTC"),
    }


def test_slug_filter_excludes_props():
    assert GAME_SLUG.match("mlb-lad-nym-2025-05-23")
    assert not GAME_SLUG.match("mlb-sd-mia-2026-07-24-player-props")
    assert not GAME_SLUG.match("mlb-nyy-bos-2025-06-14-game-2")


def test_match_orients_home_probability():
    pm = pd.DataFrame([
        _market("Red Sox", "Yankees", price=0.6),   # outcome_0 is the home side
        _market("Dodgers", "Mets", price=0.3),      # outcome_0 is the away side
    ])
    out = match_to_games(pm, _games()).set_index("game_pk")
    assert out.loc[1, "market_home_prob"] == 0.6
    assert out.loc[3, "market_home_prob"] == 0.7  # flipped to home terms


def test_sox_disambiguation():
    """'Red Sox' must not match 'White Sox' - suffix matching has to pair BOTH teams."""
    pm = pd.DataFrame([_market("White Sox", "Cubs", price=0.55)])
    out = match_to_games(pm, _games())
    assert out["game_pk"].tolist() == [2]


def test_same_matchup_next_day_not_confused():
    pm = pd.DataFrame([_market("Red Sox", "Yankees", date_key="2025-06-15", price=0.4)])
    out = match_to_games(pm, _games())
    assert out["game_pk"].tolist() == [4]
