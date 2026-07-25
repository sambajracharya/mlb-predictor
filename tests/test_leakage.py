"""The tests that matter: prove no future information reaches a game's features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbpred.features import (
    _shift_roll,
    build_dataset,
    feature_columns,
    park_factors,
    prepare_games,
)


def test_shift_roll_excludes_the_current_row():
    df = pd.DataFrame({"g": ["a"] * 5, "x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    got = _shift_roll(df, ["g"], ["x"], window=3, how="sum")["x"].tolist()
    # row i sees only rows < i
    assert np.isnan(got[0])
    assert got[1:] == [1.0, 3.0, 6.0, 9.0]  # last row never sees its own 100


def test_future_results_cannot_change_past_features(raw):
    base = build_dataset(raw["games"], raw["hitting"], raw["pitching"], raw["sp"],
                         min_prior_games=5)

    # Rewrite the final third of the schedule into absurd blowouts.
    games = raw["games"].copy()
    hitting = raw["hitting"].copy()
    cutoff = games["date"].quantile(0.66)
    fut_g = games["date"] > cutoff
    games.loc[fut_g, "home_score"] = 30
    games.loc[fut_g, "away_score"] = 0
    fut_pks = set(games.loc[fut_g, "game_pk"])
    fut_h = hitting["game_pk"].isin(fut_pks)
    hitting.loc[fut_h, ["runs", "hits", "homeRuns", "totalBases"]] = 99.0

    tampered = build_dataset(games, hitting, raw["pitching"], raw["sp"], min_prior_games=5)

    feats = feature_columns(base)
    past = base["date"] <= cutoff
    a = base.loc[past, ["game_pk"] + feats].set_index("game_pk").sort_index()
    b = tampered.set_index("game_pk").reindex(a.index)[feats]
    pd.testing.assert_frame_equal(a, b, check_like=True)


def test_park_factor_uses_only_prior_seasons(raw):
    games = prepare_games(raw["games"])
    pf = park_factors(games)
    # 2022 is the first season in the fixture, so it has no prior data at all.
    assert set(pf["season"]) == {2023}

    boosted = games.copy()
    mask = boosted["season"] == 2023
    boosted.loc[mask, "home_score"] = 40
    pf2 = park_factors(boosted)
    pd.testing.assert_frame_equal(
        pf.sort_values(["venue_id", "season"]).reset_index(drop=True),
        pf2.sort_values(["venue_id", "season"]).reset_index(drop=True),
    )


def test_batter_form_excludes_the_current_game(raw):
    """The lineup is pregame info, but a hitter's *quality* must not include today."""
    from mlbpred.features import batter_form, lineup_features

    dates = pd.to_datetime(["2023-04-01", "2023-04-02", "2023-04-03"])
    sc = pd.DataFrame({
        "player_id": [1, 1, 1], "game_pk": [10, 11, 12], "game_date": dates,
        "sc_pa": [4.0, 4.0, 4.0], "sc_xwoba": [0.100, 0.200, 0.900],
        "sc_woba": [0.100, 0.200, 0.900],
    })
    # batter_form is state *through* each date (causality is enforced downstream)
    bf = batter_form(sc).set_index("game_pk")
    assert bf.loc[10, "b_xwoba_30"] == pytest.approx(0.100)
    assert bf.loc[12, "b_xwoba_30"] == pytest.approx(0.400)  # (0.1+0.2+0.9)/3

    # the lineup aggregation is what must exclude today's game
    lu = pd.DataFrame({"game_pk": [10, 11, 12], "team_id": [108] * 3, "slot": [1] * 3,
                       "player_id": [1] * 3, "date": dates})
    out = lineup_features(lu, sc).set_index("game_pk")
    assert np.isnan(out.loc[10, "lineup_xwoba_30"])            # no prior history
    assert out.loc[11, "lineup_xwoba_30"] == pytest.approx(0.100)
    assert out.loc[12, "lineup_xwoba_30"] == pytest.approx(0.150)  # 0.9 excluded

    # an UNPLAYED game the next day still gets the hitter's real form
    upcoming = pd.DataFrame({"game_pk": [13], "team_id": [108], "slot": [1],
                             "player_id": [1], "date": [pd.Timestamp("2023-04-04")]})
    up = lineup_features(upcoming, sc)
    assert up["lineup_xwoba_30"].iloc[0] == pytest.approx(0.400)  # all 3 prior games


def test_no_target_column_leaks_into_features(raw):
    df = build_dataset(raw["games"], raw["hitting"], raw["pitching"], raw["sp"],
                       min_prior_games=5)
    feats = set(feature_columns(df))
    forbidden = {
        "home_win", "home_runs", "away_runs", "total_runs", "home_hits", "away_hits",
        "home_hr", "away_hr", "home_so", "away_so", "home_score", "away_score",
        "home_h", "away_h", "home_e", "away_e",
        "home_record_w", "home_record_l", "away_record_w", "away_record_l",
    }
    assert not (feats & forbidden)


def test_features_are_perfectly_correlated_with_nothing(raw):
    """A feature that correlates ~1.0 with the outcome is a leak, not a discovery."""
    df = build_dataset(raw["games"], raw["hitting"], raw["pitching"], raw["sp"],
                       min_prior_games=5)
    feats = feature_columns(df)
    corr = df[feats].corrwith(df["home_runs"]).abs()
    assert corr.max() < 0.9, corr.sort_values(ascending=False).head()
