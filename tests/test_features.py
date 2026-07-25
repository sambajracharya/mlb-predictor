from __future__ import annotations

import numpy as np
import pandas as pd

from mlbpred.features import (
    build_dataset,
    calendar_features,
    feature_columns,
    prepare_games,
    sp_form_features,
    weather_features,
)


def test_weather_parsing():
    g = pd.DataFrame(
        {
            "wx_wind": ["12 mph, Out To LF", "5 mph, In From CF", "3 mph, L To R", None],
            "wx_temp": ["81", 64.0, None, "72"],
            "wx_condition": ["Sunny", "Rain", "Dome", "Roof Closed"],
        }
    )
    w = weather_features(g)
    assert w["wind_mph"].tolist()[:2] == [12.0, 5.0]
    assert w["wind_mph"].tolist()[2] == 0.0  # indoors: reported wind is meaningless
    assert w["wind_out"].tolist() == [1, 0, 0, 0]
    assert w["wind_in"].tolist() == [0, 1, 0, 0]
    assert w["is_dome"].tolist() == [0, 0, 1, 1]
    assert w["is_rain"].tolist() == [0, 1, 0, 0]
    assert w["wx_temp"].tolist()[0] == 81.0


def test_sp_features_ignore_relief_outings():
    dates = pd.to_datetime(["2023-04-01", "2023-04-06", "2023-04-08", "2023-04-12"])
    logs = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1], "season": [2023] * 4, "date": dates,
            "game_pk": [1, 2, 3, 4],
            "gamesStarted": [1.0, 1.0, 0.0, 1.0],  # third outing is relief
            "outs": [18.0, 18.0, 3.0, 18.0], "earnedRuns": [1.0, 1.0, 9.0, 1.0],
            "hits": [4.0, 4.0, 9.0, 4.0], "homeRuns": [0.0, 0.0, 3.0, 0.0],
            "strikeOuts": [6.0, 6.0, 0.0, 6.0], "baseOnBalls": [1.0, 1.0, 4.0, 1.0],
            "battersFaced": [22.0, 22.0, 10.0, 22.0], "numberOfPitches": [90.0, 92.0, 40.0, 95.0],
        }
    )
    out = sp_form_features(logs).set_index("game_pk")
    assert 3 not in out.index                      # relief appearance dropped entirely
    assert out.loc[4, "sp_era_5"] == 1.5           # 2 ER over 12 IP, blowup excluded
    assert out.loc[4, "sp_days_rest"] == 6         # measured from the last *start*
    assert np.isnan(out.loc[1, "sp_era_5"])        # first start has no history


def test_calendar_features():
    g = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-04"]), "day_night": ["day"],
            "double_header": ["S"], "series_game": [2],
            "home_div_id": [200], "away_div_id": [201],
            "home_league_id": [103], "away_league_id": [104],
        }
    )
    c = calendar_features(g)
    assert c["month"][0] == 7 and c["is_night"][0] == 0
    assert c["is_doubleheader"][0] == 1
    assert c["same_division"][0] == 0 and c["interleague"][0] == 1


def test_dataset_shape_and_targets(raw):
    df = build_dataset(raw["games"], raw["hitting"], raw["pitching"], raw["sp"],
                       min_prior_games=5)
    assert len(df) > 0
    assert df["home_win"].isin([0.0, 1.0]).all()
    assert (df["total_runs"] == df["home_runs"] + df["away_runs"]).all()
    assert df["game_pk"].is_unique
    feats = feature_columns(df)
    assert len(feats) > 60
    # every feature must be numeric and not all-NaN
    assert df[feats].notna().any().all()


def test_upcoming_games_are_kept_when_requested(raw):
    games = raw["games"].copy()
    future = games.tail(2).copy()
    future["game_pk"] = future["game_pk"] + 90000
    future["date"] = future["date"] + pd.Timedelta(days=1)
    future["status"] = "S"
    future[["home_score", "away_score", "home_h", "away_h"]] = np.nan
    games = pd.concat([games, future], ignore_index=True)

    df = build_dataset(games, raw["hitting"], raw["pitching"], raw["sp"],
                       min_prior_games=5, keep_upcoming=True)
    upcoming = df[~df["is_final"]]
    assert len(upcoming) == 2
    assert upcoming["home_win"].isna().all()
    # the unplayed games still carry real pregame form
    assert upcoming["h_bat_rpg_15"].notna().all()
    assert upcoming["h_sp_era_5"].notna().all()


def test_vs_hand_features_roll_only_same_hand_history(raw):
    from mlbpred.features import build_team_game_frame, team_form_features

    games = prepare_games(raw["games"])
    tg = build_team_game_frame(games, raw["hitting"], raw["pitching"])
    # fixture starters: SP 608/609/610/611. Make one of each pair a lefty.
    hands = pd.Series({608: "R", 609: "L", 610: "R", 611: "L"})
    out = team_form_features(tg, hands)
    merged = out.merge(tg[["game_pk", "team_id", "opp_sp_id"]], on=["game_pk", "team_id"])

    # indicator matches the actual opposing starter's hand
    lhp_rows = merged["opp_sp_id"].map(hands).eq("L")
    assert (merged.loc[lhp_rows, "opp_sp_lhp"] == 1.0).all()
    assert (merged.loc[~lhp_rows, "opp_sp_lhp"] == 0.0).all()

    # first game against a given hand has no history -> NaN, second one doesn't
    team = merged[merged["team_id"] == 108].sort_values("game_pk")
    first_vs_l = team[team["opp_sp_lhp"] == 1.0].iloc[0]
    assert np.isnan(first_vs_l["bat_ops_vs_hand"])
    later_vs_l = team[team["opp_sp_lhp"] == 1.0].iloc[5]
    assert not np.isnan(later_vs_l["bat_ops_vs_hand"])

    # without hands data the columns simply don't exist
    out_no_hands = team_form_features(tg, None)
    assert "bat_ops_vs_hand" not in out_no_hands.columns


def test_prepare_games_drops_scores_for_unfinished(raw):
    games = raw["games"].copy()
    games.loc[games.index[:3], "status"] = "S"
    prepped = prepare_games(games)
    assert prepped.loc[~prepped["is_final"], "home_score"].isna().all()
