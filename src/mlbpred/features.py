"""Pregame feature engineering.

The one rule this module exists to enforce: **every feature for game G is
computed only from games that finished before G started.** That is done in
exactly one place - `_shift_roll` - which shifts each team's/pitcher's series by
one game before any rolling window is applied. If you add a feature, route it
through those helpers or you will leak the future into the past and your metrics
will look great and mean nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import MIN_PRIOR_TEAM_GAMES, SP_WINDOWS, TEAM_WINDOWS

META_COLS = [
    "game_pk", "date", "season", "game_type", "status", "venue_id", "venue",
    "home_team_id", "away_team_id", "home_team", "away_team", "home_abbr", "away_abbr",
    "home_sp_id", "away_sp_id", "home_sp_name", "away_sp_name", "is_final",
    "home_prior_games", "away_prior_games",
    "h_lineup_posted", "a_lineup_posted",
]

TARGET_COLS = [
    "home_win", "home_runs", "away_runs", "total_runs",
    "home_hits", "away_hits", "home_hr", "away_hr", "home_so", "away_so",
]

BAT_SUM_COLS = [
    "runs", "hits", "doubles", "triples", "homeRuns", "strikeOuts", "baseOnBalls",
    "hitByPitch", "atBats", "plateAppearances", "totalBases",
]
PIT_SUM_COLS = [
    "runs", "earnedRuns", "hits", "homeRuns", "strikeOuts", "baseOnBalls", "outs",
    "battersFaced", "numberOfPitches",
]
SP_SUM_COLS = [
    "outs", "earnedRuns", "hits", "homeRuns", "strikeOuts", "baseOnBalls",
    "battersFaced", "numberOfPitches",
]


# --------------------------------------------------------------------------- #
# leakage-safe rolling helpers
# --------------------------------------------------------------------------- #
def _shift_roll(df: pd.DataFrame, group: list[str], cols: list[str], window: int,
                how: str) -> pd.DataFrame:
    """Rolling `how` over the `window` rows *preceding* each row, per group.

    `df` must already be sorted chronologically within each group.
    """
    shifted = df.groupby(group, sort=False)[cols].shift(1)
    shifted[group] = df[group]
    grouped = shifted.groupby(group, sort=False)[cols]
    if how == "sum":
        out = grouped.transform(lambda s: s.rolling(window, min_periods=1).sum())
    elif how == "count":
        out = grouped.transform(lambda s: s.rolling(window, min_periods=1).count())
    elif how == "mean":
        out = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    else:
        raise ValueError(how)
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


# --------------------------------------------------------------------------- #
# raw -> tidy
# --------------------------------------------------------------------------- #
def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games[games["game_type"] == "R"].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.dropna(subset=["home_team_id", "away_team_id"])
    g["home_team_id"] = g["home_team_id"].astype(int)
    g["away_team_id"] = g["away_team_id"].astype(int)
    g["is_final"] = g["status"].eq("F")
    # A handful of games are suspended/cancelled with partial linescores.
    g.loc[~g["is_final"], ["home_score", "away_score", "home_h", "away_h"]] = np.nan
    g = g.sort_values(["date", "game_pk"]).drop_duplicates("game_pk").reset_index(drop=True)
    return g


def build_team_game_frame(games: pd.DataFrame, hitting: pd.DataFrame,
                          pitching: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team), chronological, with that game's box score.

    Scheduled-but-unplayed games are kept with NaN stats so that upcoming games
    flow through the same rolling machinery as historical ones.
    """
    long = []
    for side, opp in (("home", "away"), ("away", "home")):
        part = pd.DataFrame(
            {
                "game_pk": games["game_pk"],
                "date": games["date"],
                "season": games["season"],
                "team_id": games[f"{side}_team_id"],
                "opponent_id": games[f"{opp}_team_id"],
                "is_home": side == "home",
                "sp_id": games[f"{side}_sp_id"],
                "opp_sp_id": games[f"{opp}_sp_id"],
                "venue_id": games["venue_id"],
                "is_final": games["is_final"],
                "runs_for": games[f"{side}_score"],
                "runs_against": games[f"{opp}_score"],
            }
        )
        long.append(part)
    tg = pd.concat(long, ignore_index=True)
    tg["is_win"] = np.where(
        tg["is_final"], (tg["runs_for"] > tg["runs_against"]).astype(float), np.nan
    )

    bat = hitting.rename(columns={c: f"bat_{c}" for c in BAT_SUM_COLS})
    bat = bat[["game_pk", "team_id"] + [f"bat_{c}" for c in BAT_SUM_COLS]]
    pit = pitching.rename(columns={c: f"pit_{c}" for c in PIT_SUM_COLS})
    pit = pit[["game_pk", "team_id"] + [f"pit_{c}" for c in PIT_SUM_COLS]]

    tg = tg.merge(bat, on=["game_pk", "team_id"], how="left")
    tg = tg.merge(pit, on=["game_pk", "team_id"], how="left")
    return tg.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)


SC_SUM_COLS = ["sc_xwoba_pa", "sc_pa", "sc_hh", "sc_bip", "sc_barrels"]


def attach_statcast_team(tg: pd.DataFrame, statcast_team: pd.DataFrame) -> pd.DataFrame:
    """Add per-game Statcast offense (sc_*) and, via the opponent's row in the
    same game, quality-of-contact *allowed* by this team's staff (alw_*)."""
    sc = statcast_team.copy()
    sc["sc_xwoba_pa"] = sc["sc_xwoba"] * sc["sc_pa"]
    sc["sc_hh"] = sc["sc_hh_pct"].fillna(0) / 100.0 * sc["sc_bip"]
    sc = sc[["game_pk", "team_id"] + SC_SUM_COLS]

    tg = tg.merge(sc, on=["game_pk", "team_id"], how="left")
    opp = sc.rename(columns={"team_id": "opponent_id",
                             **{c: c.replace("sc_", "alw_") for c in SC_SUM_COLS}})
    tg = tg.merge(opp, on=["game_pk", "opponent_id"], how="left")
    return tg


# --------------------------------------------------------------------------- #
# team form features
# --------------------------------------------------------------------------- #
def team_form_features(tg: pd.DataFrame, sp_hands: pd.Series | None = None) -> pd.DataFrame:
    """Pregame rolling form, one row per (game_pk, team_id).

    `sp_hands` maps pitcher id -> "L"/"R". When given, adds handedness-split
    offense: this team's rolling production over its previous games against
    starters of the same hand as tonight's opposing starter.
    """
    df = tg.sort_values(["team_id", "season", "date", "game_pk"]).reset_index(drop=True)
    key = ["team_id", "season"]
    out = df[["game_pk", "team_id", "date", "season"]].copy()

    bat_cols = [f"bat_{c}" for c in BAT_SUM_COLS]
    pit_cols = [f"pit_{c}" for c in PIT_SUM_COLS]

    for w in TEAM_WINDOWS:
        s_bat = _shift_roll(df, key, bat_cols, w, "sum")
        n_bat = _shift_roll(df, key, ["bat_plateAppearances"], w, "count")["bat_plateAppearances"]
        s_pit = _shift_roll(df, key, pit_cols, w, "sum")
        n_pit = _shift_roll(df, key, ["pit_outs"], w, "count")["pit_outs"]

        out[f"bat_rpg_{w}"] = _safe_div(s_bat["bat_runs"], n_bat)
        out[f"bat_hpg_{w}"] = _safe_div(s_bat["bat_hits"], n_bat)
        out[f"bat_hrpg_{w}"] = _safe_div(s_bat["bat_homeRuns"], n_bat)
        out[f"bat_kpg_{w}"] = _safe_div(s_bat["bat_strikeOuts"], n_bat)
        out[f"bat_bbpg_{w}"] = _safe_div(s_bat["bat_baseOnBalls"], n_bat)
        obp = _safe_div(
            s_bat["bat_hits"] + s_bat["bat_baseOnBalls"] + s_bat["bat_hitByPitch"],
            s_bat["bat_plateAppearances"],
        )
        slg = _safe_div(s_bat["bat_totalBases"], s_bat["bat_atBats"])
        out[f"bat_obp_{w}"] = obp
        out[f"bat_slg_{w}"] = slg
        out[f"bat_ops_{w}"] = obp + slg
        out[f"bat_kpct_{w}"] = _safe_div(s_bat["bat_strikeOuts"], s_bat["bat_plateAppearances"])

        ip = s_pit["pit_outs"] / 3.0
        out[f"pit_rapg_{w}"] = _safe_div(s_pit["pit_runs"], n_pit)
        out[f"pit_era_{w}"] = _safe_div(9.0 * s_pit["pit_earnedRuns"], ip)
        out[f"pit_whip_{w}"] = _safe_div(s_pit["pit_hits"] + s_pit["pit_baseOnBalls"], ip)
        out[f"pit_k9_{w}"] = _safe_div(9.0 * s_pit["pit_strikeOuts"], ip)
        out[f"pit_bb9_{w}"] = _safe_div(9.0 * s_pit["pit_baseOnBalls"], ip)
        out[f"pit_hr9_{w}"] = _safe_div(9.0 * s_pit["pit_homeRuns"], ip)
        out[f"win_pct_{w}"] = _shift_roll(df, key, ["is_win"], w, "mean")["is_win"]

    # Statcast quality-of-contact, if the columns were attached. xwOBA and
    # barrel rate are *expected* stats - they strip batted-ball luck out and
    # stabilise much faster than the outcome stats above.
    if "sc_pa" in df.columns:
        alw_cols = [c.replace("sc_", "alw_") for c in
                    ("sc_xwoba_pa", "sc_pa", "sc_hh", "sc_bip", "sc_barrels")]
        for w in TEAM_WINDOWS:
            s = _shift_roll(df, key, ["sc_xwoba_pa", "sc_pa", "sc_hh", "sc_bip",
                                      "sc_barrels"], w, "sum")
            out[f"bat_xwoba_{w}"] = _safe_div(s["sc_xwoba_pa"], s["sc_pa"])
            out[f"bat_hh_{w}"] = _safe_div(s["sc_hh"], s["sc_bip"])
            out[f"bat_barrel_{w}"] = _safe_div(s["sc_barrels"], s["sc_bip"])
            a = _shift_roll(df, key, alw_cols, w, "sum")
            out[f"pit_xwoba_{w}"] = _safe_div(a["alw_xwoba_pa"], a["alw_pa"])
            out[f"pit_hh_{w}"] = _safe_div(a["alw_hh"], a["alw_bip"])
            out[f"pit_barrel_{w}"] = _safe_div(a["alw_barrels"], a["alw_bip"])

    # season-to-date (expanding, still shifted)
    prior_games = df.groupby(key, sort=False).cumcount()
    out["prior_games"] = prior_games
    wins_cum = df.groupby(key, sort=False)["is_win"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    out["win_pct_std"] = _safe_div(wins_cum, prior_games.replace(0, np.nan))

    # rest & schedule load
    last_date = df.groupby("team_id", sort=False)["date"].shift(1)
    out["days_rest"] = (df["date"] - last_date).dt.days.clip(upper=7)

    # games played in the previous 7 calendar days (excluding this one)
    counts = []
    for _, grp in df.groupby("team_id", sort=False):
        dates = grp["date"].to_numpy()
        prev = np.searchsorted(dates, dates - np.timedelta64(7, "D"), side="left")
        idx = np.arange(len(dates))
        counts.append(pd.Series(idx - prev, index=grp.index))
    out["games_last_7d"] = pd.concat(counts).sort_index()

    # bullpen workload proxy: staff outs recorded in the previous 3 games
    out["bp_outs_l3"] = _shift_roll(df, ["team_id"], ["pit_outs"], 3, "sum")["pit_outs"]
    out["is_home"] = df["is_home"].astype(int)

    if sp_hands is not None:
        # Rolling over the last 25 games this team faced a starter of the same
        # hand as tonight's. Grouped by (team, hand) across seasons - splits are
        # a roster trait that changes slowly, and per-season samples vs LHP are
        # too thin to roll on. Window > a season's LHP starts keeps it current.
        df = df.copy()
        df["opp_hand"] = df["opp_sp_id"].map(sp_hands)
        key = ["team_id", "opp_hand"]
        cols = ["bat_hits", "bat_baseOnBalls", "bat_hitByPitch", "bat_plateAppearances",
                "bat_atBats", "bat_totalBases", "bat_strikeOuts", "bat_runs"]
        s = _shift_roll(df, key, cols, 25, "sum")
        n = _shift_roll(df, key, ["bat_plateAppearances"], 25, "count")["bat_plateAppearances"]
        obp = _safe_div(s["bat_hits"] + s["bat_baseOnBalls"] + s["bat_hitByPitch"],
                        s["bat_plateAppearances"])
        slg = _safe_div(s["bat_totalBases"], s["bat_atBats"])
        out["bat_ops_vs_hand"] = obp + slg
        out["bat_kpct_vs_hand"] = _safe_div(s["bat_strikeOuts"], s["bat_plateAppearances"])
        out["bat_rpg_vs_hand"] = _safe_div(s["bat_runs"], n)
        out["opp_sp_lhp"] = np.where(
            df["opp_hand"].isna(), np.nan, (df["opp_hand"] == "L").astype(float)
        )
    return out


# Rough share of a game's plate appearances by batting-order slot. The leadoff
# hitter gets ~12% of the team's PA, the 9-hole ~9.5%.
SLOT_WEIGHTS = np.array([1.13, 1.10, 1.07, 1.04, 1.00, 0.97, 0.94, 0.91, 0.88])
SLOT_WEIGHTS = SLOT_WEIGHTS / SLOT_WEIGHTS.sum()

LINEUP_WINDOWS = (30, 90)


def batter_form(statcast_batter: pd.DataFrame) -> pd.DataFrame:
    """Each batter's rolling xwOBA/wOBA *through* each game he played in.

    Rolling over games rather than PA, grouped by player across seasons: hitters
    carry their true talent over the winter, and a 90-game window (~370 PA) is
    where xwOBA starts to be meaningful.

    Note the window *includes* the row's own game - this is a "state as of the
    end of this date" table. `lineup_features` is what enforces causality, by
    as-of joining strictly earlier dates. Doing it that way (rather than
    shifting here) means an upcoming game and a historical one are looked up
    identically, so predictions do not silently fall back to stale numbers.
    """
    df = statcast_batter.copy()
    df["date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    df["xwoba_pa"] = df["sc_xwoba"] * df["sc_pa"]
    df["woba_pa"] = df["sc_woba"] * df["sc_pa"]

    out = df[["game_pk", "player_id", "date"]].copy()
    grouped = df.groupby("player_id", sort=False)[["xwoba_pa", "woba_pa", "sc_pa"]]
    for w in LINEUP_WINDOWS:
        s = grouped.transform(lambda c: c.rolling(w, min_periods=1).sum())
        out[f"b_xwoba_{w}"] = _safe_div(s["xwoba_pa"], s["sc_pa"])
        out[f"b_woba_{w}"] = _safe_div(s["woba_pa"], s["sc_pa"])
        out[f"b_pa_{w}"] = s["sc_pa"]
    return out


def lineup_features(lineups: pd.DataFrame, batter: pd.DataFrame) -> pd.DataFrame:
    """PA-weighted quality of the nine posted starters, per (game_pk, team_id).

    The lineup is public ~3h before first pitch, so this is pregame information -
    but each hitter's *quality* is measured strictly from his earlier games.
    """
    if lineups.empty or batter.empty:
        return pd.DataFrame(columns=["game_pk", "team_id"])

    bf = batter_form(batter)
    lu = lineups.copy()
    lu["date"] = pd.to_datetime(lu["date"])
    lu["player_id"] = lu["player_id"].astype("int64")
    bf = bf.drop(columns="game_pk")
    bf["player_id"] = bf["player_id"].astype("int64")

    # As-of join on strictly earlier dates: a game sees each hitter's form
    # through his last game *before* today, whether or not today has been played.
    lu = pd.merge_asof(
        lu.sort_values("date"),
        bf.sort_values("date"),
        on="date", by="player_id",
        direction="backward", allow_exact_matches=False,
    )
    lu = lu[lu["slot"].between(1, 9)].copy()
    lu["weight"] = SLOT_WEIGHTS[lu["slot"].astype(int) - 1]

    rows = []
    for w in LINEUP_WINDOWS:
        col, pacol = f"b_xwoba_{w}", f"b_pa_{w}"
        # weight by slot AND by how much history the hitter has: a callup with
        # 12 career PA should not swing the lineup estimate
        credible = lu[pacol].fillna(0).clip(upper=200) / 200.0
        lu[f"_w{w}"] = lu["weight"] * credible
        lu[f"_num{w}"] = lu[col] * lu[f"_w{w}"]
    agg = lu.groupby(["game_pk", "team_id"]).agg(
        **{f"num_{w}": (f"_num{w}", "sum") for w in LINEUP_WINDOWS},
        **{f"den_{w}": (f"_w{w}", "sum") for w in LINEUP_WINDOWS},
        n_slots=("slot", "size"),
        n_known=("b_xwoba_90", lambda s: int(s.notna().sum())),
        lineup_pa=("b_pa_90", "sum"),
    ).reset_index()

    out = agg[["game_pk", "team_id", "n_slots", "n_known", "lineup_pa"]].copy()
    for w in LINEUP_WINDOWS:
        out[f"lineup_xwoba_{w}"] = _safe_div(agg[f"num_{w}"], agg[f"den_{w}"])
    # a full nine with real track records is the normal case; flag the rest
    out["lineup_complete"] = ((out["n_slots"] == 9) & (out["n_known"] >= 8)).astype(int)
    return out.drop(columns=["n_slots", "n_known"])


def carry_forward_lineups(team_feats: pd.DataFrame) -> pd.DataFrame:
    """Fill lineup columns for games whose lineup is not posted yet.

    Historical games have ~100% lineup coverage, but a slate is only ~1/3 posted
    a few hours out. Rather than hand the model NaN for a column it has never
    seen missing, carry each team's most recent posted lineup forward - day to
    day a lineup is mostly the same nine. `lineup_posted` records whether the
    row is real or carried, so predictions can be flagged (it is metadata, not a
    feature: it is ~constant in training and would teach the model nothing).
    """
    df = team_feats.sort_values(["team_id", "date"]).copy()
    cols = [c for c in df.columns if c.startswith("lineup_")]
    if not cols:
        return team_feats
    df["lineup_posted"] = df["lineup_xwoba_90"].notna().astype(int)
    df[cols] = df.groupby("team_id", sort=False)[cols].ffill()
    return df.sort_index()


def prev_season_win_pct(tg: pd.DataFrame) -> pd.DataFrame:
    played = tg[tg["is_final"]]
    by = played.groupby(["team_id", "season"])["is_win"].mean().reset_index()
    by["season"] = by["season"] + 1
    return by.rename(columns={"is_win": "prev_season_win_pct"})


# --------------------------------------------------------------------------- #
# starting pitcher features
# --------------------------------------------------------------------------- #
def sp_form_features(sp_logs: pd.DataFrame) -> pd.DataFrame:
    """Pregame rolling starter form, keyed by (game_pk, player_id).

    Only outings where the pitcher actually started are used, so relief
    appearances do not pollute the rate stats.
    """
    if sp_logs.empty:
        return pd.DataFrame(columns=["game_pk", "player_id"])
    df = sp_logs[sp_logs["gamesStarted"].fillna(0) > 0].copy()
    df = df.sort_values(["player_id", "season", "date", "game_pk"]).reset_index(drop=True)
    key = ["player_id", "season"]
    out = df[["game_pk", "player_id", "date", "season"]].copy()

    for w in SP_WINDOWS:
        s = _shift_roll(df, key, SP_SUM_COLS, w, "sum")
        n = _shift_roll(df, key, ["outs"], w, "count")["outs"]
        ip = s["outs"] / 3.0
        out[f"sp_era_{w}"] = _safe_div(9.0 * s["earnedRuns"], ip)
        out[f"sp_whip_{w}"] = _safe_div(s["hits"] + s["baseOnBalls"], ip)
        out[f"sp_k9_{w}"] = _safe_div(9.0 * s["strikeOuts"], ip)
        out[f"sp_bb9_{w}"] = _safe_div(9.0 * s["baseOnBalls"], ip)
        out[f"sp_hr9_{w}"] = _safe_div(9.0 * s["homeRuns"], ip)
        out[f"sp_kpct_{w}"] = _safe_div(s["strikeOuts"], s["battersFaced"])
        out[f"sp_bbpct_{w}"] = _safe_div(s["baseOnBalls"], s["battersFaced"])
        out[f"sp_outs_per_start_{w}"] = _safe_div(s["outs"], n)

    if "sc_pa" in df.columns:
        df["sc_xwoba_pa"] = df["sc_xwoba"] * df["sc_pa"]
        for w in SP_WINDOWS:
            s = _shift_roll(df, key, ["sc_xwoba_pa", "sc_pa", "sc_whiffs", "sc_swings"],
                            w, "sum")
            out[f"sp_xwoba_{w}"] = _safe_div(s["sc_xwoba_pa"], s["sc_pa"])
            out[f"sp_whiff_{w}"] = _safe_div(s["sc_whiffs"], s["sc_swings"])
        # fastball-velocity trend: recent velo vs longer baseline catches both
        # decline and injury recovery before ERA does
        out["sp_velo_3"] = _shift_roll(df, key, ["sc_velo"], 3, "mean")["sc_velo"]
        velo_10 = _shift_roll(df, key, ["sc_velo"], 10, "mean")["sc_velo"]
        out["sp_velo_delta"] = out["sp_velo_3"] - velo_10

    out["sp_starts_std"] = df.groupby(key, sort=False).cumcount()
    out["sp_pitches_last"] = df.groupby(key, sort=False)["numberOfPitches"].shift(1)
    last_start = df.groupby("player_id", sort=False)["date"].shift(1)
    out["sp_days_rest"] = (df["date"] - last_start).dt.days.clip(upper=15)
    return out


def sp_upcoming_state(sp_feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Carry each pitcher's latest form forward to games they have not started yet.

    Needed for prediction: an upcoming start has no row in the game-log-derived
    feature table, so we reuse the state as of their most recent completed start.
    """
    if sp_feats.empty:
        return sp_feats
    latest = sp_feats.sort_values(["player_id", "date"]).groupby("player_id").tail(1)
    latest = latest.drop(columns=["game_pk", "season"]).rename(columns={"date": "sp_last_start"})
    upcoming = games.loc[~games["is_final"], ["game_pk", "date", "home_sp_id", "away_sp_id"]]
    rows = []
    for side in ("home", "away"):
        part = upcoming[["game_pk", "date", f"{side}_sp_id"]].rename(
            columns={f"{side}_sp_id": "player_id"}
        )
        rows.append(part.dropna(subset=["player_id"]))
    if not rows:
        return sp_feats
    up = pd.concat(rows, ignore_index=True)
    up["player_id"] = up["player_id"].astype(int)
    up = up.merge(latest, on="player_id", how="inner")
    up["sp_days_rest"] = (up["date"] - up["sp_last_start"]).dt.days.clip(upper=15)
    up = up.drop(columns=["sp_last_start"])
    return pd.concat([sp_feats, up], ignore_index=True).drop_duplicates(
        ["game_pk", "player_id"], keep="first"
    )


# --------------------------------------------------------------------------- #
# park, weather, calendar
# --------------------------------------------------------------------------- #
def park_factors(games: pd.DataFrame, shrink: int = 150) -> pd.DataFrame:
    """Run/HR park factors for each (venue, season), built from *prior* seasons only."""
    played = games[games["is_final"]].copy()
    played["total_runs"] = played["home_score"] + played["away_score"]
    by = (
        played.groupby(["venue_id", "season"])
        .agg(runs=("total_runs", "sum"), n=("total_runs", "size"))
        .reset_index()
    )
    league = played.groupby("season")["total_runs"].mean().rename("lg_rpg")

    rows = []
    for season in sorted(games["season"].dropna().unique()):
        prior = by[by["season"] < season]
        lg_prior = league[league.index < season]
        if prior.empty or lg_prior.empty:
            continue
        lg_rpg = float(lg_prior.mean())
        agg = prior.groupby("venue_id").agg(runs=("runs", "sum"), n=("n", "sum")).reset_index()
        raw = (agg["runs"] / agg["n"]) / lg_rpg
        agg["pf_runs"] = (agg["n"] * raw + shrink * 1.0) / (agg["n"] + shrink)
        agg["season"] = season
        rows.append(agg[["venue_id", "season", "pf_runs"]])
    if not rows:
        return pd.DataFrame(columns=["venue_id", "season", "pf_runs"])
    return pd.concat(rows, ignore_index=True)


_WIND_RE = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)


def weather_features(games: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=games.index)
    wind = games["wx_wind"].fillna("")
    parsed = wind.str.extract(_WIND_RE)
    out["wind_mph"] = pd.to_numeric(parsed[0], errors="coerce")
    direction = parsed[1].fillna("").str.lower()
    out["wind_out"] = direction.str.contains("out to").astype(int)
    out["wind_in"] = direction.str.contains("in from").astype(int)
    out["wx_temp"] = pd.to_numeric(games["wx_temp"], errors="coerce")
    cond = games["wx_condition"].fillna("").str.lower()
    out["is_dome"] = cond.str.contains("dome|roof closed").astype(int)
    out["is_rain"] = cond.str.contains("rain|drizzle").astype(int)
    # Domes report a nominal 72F; keep the flag so the model can tell them apart.
    out.loc[out["is_dome"] == 1, "wind_mph"] = 0.0
    return out


def calendar_features(games: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=games.index)
    out["month"] = games["date"].dt.month
    out["day_of_week"] = games["date"].dt.dayofweek
    out["is_night"] = games["day_night"].eq("night").astype(int)
    out["is_doubleheader"] = games["double_header"].ne("N").astype(int)
    out["series_game"] = pd.to_numeric(games["series_game"], errors="coerce")
    out["same_division"] = (games["home_div_id"] == games["away_div_id"]).astype(int)
    out["interleague"] = (games["home_league_id"] != games["away_league_id"]).astype(int)
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
DIFF_BASES = [
    "bat_rpg_15", "bat_ops_15", "bat_ops_30", "bat_hrpg_30", "bat_kpct_30",
    "pit_era_15", "pit_era_30", "pit_whip_30", "pit_k9_30", "win_pct_30",
    "win_pct_std", "prev_season_win_pct",
    "sp_era_5", "sp_era_10", "sp_whip_10", "sp_k9_10", "sp_kpct_10",
    "sp_outs_per_start_10",
    "bat_ops_vs_hand", "bat_kpct_vs_hand", "bat_rpg_vs_hand",
    "bat_xwoba_15", "bat_xwoba_30", "bat_barrel_30", "bat_hh_30",
    "pit_xwoba_15", "pit_xwoba_30", "pit_barrel_30",
    "sp_xwoba_5", "sp_xwoba_10", "sp_whiff_10", "sp_velo_3", "sp_velo_delta",
    "lineup_xwoba_30", "lineup_xwoba_90", "lineup_pa",
]


def assemble(games: pd.DataFrame, team_feats: pd.DataFrame, sp_feats: pd.DataFrame,
             prev_wp: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    team_feats = team_feats.merge(prev_wp, on=["team_id", "season"], how="left")

    tf_cols = [c for c in team_feats.columns if c not in ("game_pk", "team_id", "date", "season")]
    sp_cols = [c for c in sp_feats.columns if c not in ("game_pk", "player_id", "date", "season")]

    for side in ("home", "away"):
        p = "h_" if side == "home" else "a_"
        tf = team_feats.rename(columns={c: p + c for c in tf_cols})
        g = g.merge(
            tf[["game_pk", "team_id"] + [p + c for c in tf_cols]],
            left_on=["game_pk", f"{side}_team_id"],
            right_on=["game_pk", "team_id"],
            how="left",
        ).drop(columns="team_id")

        if not sp_feats.empty:
            sf = sp_feats.rename(columns={c: p + c for c in sp_cols})
            g = g.merge(
                sf[["game_pk", "player_id"] + [p + c for c in sp_cols]],
                left_on=["game_pk", f"{side}_sp_id"],
                right_on=["game_pk", "player_id"],
                how="left",
            ).drop(columns="player_id")

    g = g.merge(pf, on=["venue_id", "season"], how="left")
    g["pf_runs"] = g["pf_runs"].fillna(1.0)

    derived: dict[str, pd.Series] = {}
    for base in DIFF_BASES:
        h, a = f"h_{base}", f"a_{base}"
        if h in g.columns and a in g.columns:
            derived[f"d_{base}"] = g[h] - g[a]

    # Each offense faces the *other* team's starter - hand the model that pairing
    # directly rather than making it discover the cross term.
    for off, opp_sp in (("h", "a"), ("a", "h")):
        if f"{off}_bat_ops_30" in g.columns and f"{opp_sp}_sp_whip_10" in g.columns:
            derived[f"mu_{off}_ops_vs_sp_whip"] = g[f"{off}_bat_ops_30"] * g[f"{opp_sp}_sp_whip_10"]
            derived[f"mu_{off}_kpct_vs_sp_kpct"] = g[f"{off}_bat_kpct_30"] * g[f"{opp_sp}_sp_kpct_10"]

    derived["home_prior_games"] = g["h_prior_games"]
    derived["away_prior_games"] = g["a_prior_games"]

    return pd.concat(
        [g, weather_features(g), calendar_features(g), pd.DataFrame(derived, index=g.index)],
        axis=1,
    )


def add_targets(g: pd.DataFrame, hitting: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DataFrame(
        {
            "home_runs": g["home_score"],
            "away_runs": g["away_score"],
            "total_runs": g["home_score"] + g["away_score"],
            "home_win": np.where(
                g["is_final"], (g["home_score"] > g["away_score"]).astype(float), np.nan
            ),
            "home_hits": g["home_h"],
            "away_hits": g["away_h"],
        },
        index=g.index,
    )
    g = pd.concat([g, targets], axis=1)

    box = hitting[["game_pk", "team_id", "homeRuns", "strikeOuts"]].rename(
        columns={"homeRuns": "hr", "strikeOuts": "so"}
    )
    for side in ("home", "away"):
        b = box.rename(columns={"hr": f"{side}_hr", "so": f"{side}_so"})
        g = g.merge(
            b, left_on=["game_pk", f"{side}_team_id"], right_on=["game_pk", "team_id"], how="left"
        ).drop(columns="team_id")
    return g


CORE_PREFIXES = ("d_", "pf_", "wind", "wx_", "is_", "mu_")
CORE_EXTRA = [
    "month", "same_division", "interleague",
    "h_days_rest", "a_days_rest", "h_bp_outs_l3", "a_bp_outs_l3",
    "h_games_last_7d", "a_games_last_7d",
]


STATCAST_KEYS = ("xwoba", "barrel", "_hh_", "whiff", "velo")
HAND_KEYS = ("vs_hand",)
LINEUP_KEYS = ("lineup_",)


def _drop_keys(cols: list[str], keys: tuple[str, ...]) -> list[str]:
    return [c for c in cols if not any(k in c for k in keys)]


def core_feature_columns(df: pd.DataFrame, include_statcast: bool = False,
                         include_hand: bool = False,
                         include_lineup: bool = True) -> list[str]:
    """Compact set: home-minus-away differentials plus game context.

    Backtesting says this beats the full 200+ column set for win probability -
    ~12k games is not enough data to stop that many correlated columns from
    overfitting, and win/loss is nearly symmetric in (home - away) anyway.

    Statcast and handedness columns are built into the dataset but excluded by
    default: both made the walk-forward log loss *worse*, whether added to the
    outcome stats or swapped in for them (mean over 2023-25 folds: 0.6813 base,
    0.6816 +hand, 0.6826 +statcast, 0.6830 +both). Flip the flags to experiment;
    they are most likely to earn their place at the player level (lineup
    features) rather than as team aggregates.
    """
    allowed = set(feature_columns(df))
    cols = [c for c in allowed if c.startswith(CORE_PREFIXES)]
    cols += [c for c in CORE_EXTRA if c in allowed]
    # lineup_* columns are statcast-derived but gated separately - they are the
    # one place expected stats earned their keep
    lineup_cols = [c for c in cols if any(k in c for k in LINEUP_KEYS)]
    if not include_statcast:
        cols = _drop_keys(cols, STATCAST_KEYS)
    if not include_hand:
        cols = _drop_keys(cols, HAND_KEYS)
    if include_lineup:
        cols += lineup_cols
    else:
        cols = _drop_keys(cols, LINEUP_KEYS)
    return sorted(set(cols))


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(META_COLS) | set(TARGET_COLS) | {
        "home_score", "away_score", "home_h", "away_h", "home_e", "away_e",
        "game_number", "double_header", "day_night", "detailed_status",
        "home_record_w", "home_record_l", "away_record_w", "away_record_l",
        "wx_condition", "wx_wind", "home_league_id", "away_league_id",
        "home_div_id", "away_div_id", "h_prior_games", "a_prior_games",
        # metadata, not signal: ~always 1 in training (see carry_forward_lineups)
        "h_lineup_posted", "a_lineup_posted", "d_lineup_posted",
    }
    cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)


def build_dataset(games_raw: pd.DataFrame, hitting: pd.DataFrame, pitching: pd.DataFrame,
                  sp_logs: pd.DataFrame, sp_hands: pd.DataFrame | None = None,
                  statcast_team: pd.DataFrame | None = None,
                  statcast_sp: pd.DataFrame | None = None,
                  lineups: pd.DataFrame | None = None,
                  statcast_batter: pd.DataFrame | None = None, *,
                  min_prior_games: int = MIN_PRIOR_TEAM_GAMES,
                  keep_upcoming: bool = False) -> pd.DataFrame:
    games = prepare_games(games_raw)
    tg = build_team_game_frame(games, hitting, pitching)
    if statcast_team is not None and not statcast_team.empty:
        tg = attach_statcast_team(tg, statcast_team)
    if statcast_sp is not None and not statcast_sp.empty:
        sc = statcast_sp[["game_pk", "player_id", "sc_pa", "sc_xwoba",
                          "sc_whiffs", "sc_swings", "sc_velo"]]
        sp_logs = sp_logs.merge(sc, on=["game_pk", "player_id"], how="left")
    hands = None
    if sp_hands is not None and not sp_hands.empty:
        hands = (sp_hands.dropna(subset=["sp_throws"])
                 .drop_duplicates("player_id").set_index("player_id")["sp_throws"])
    team_feats = team_form_features(tg, hands)
    if lineups is not None and statcast_batter is not None:
        lf = lineup_features(lineups, statcast_batter)
        if not lf.empty:
            team_feats = team_feats.merge(lf, on=["game_pk", "team_id"], how="left")
            team_feats = carry_forward_lineups(team_feats)
    prev_wp = prev_season_win_pct(tg)
    sp_feats = sp_form_features(sp_logs)
    if keep_upcoming:
        sp_feats = sp_upcoming_state(sp_feats, games)
    pf = park_factors(games)

    g = assemble(games, team_feats, sp_feats, prev_wp, pf)
    g = add_targets(g, hitting)

    if not keep_upcoming:
        g = g[g["is_final"]]
    enough = (g["home_prior_games"] >= min_prior_games) & (g["away_prior_games"] >= min_prior_games)
    if not keep_upcoming:
        g = g[enough]
    keep = [c for c in META_COLS if c in g.columns] + \
           [c for c in TARGET_COLS if c in g.columns] + feature_columns(g)
    return g[list(dict.fromkeys(keep))].sort_values(["date", "game_pk"]).reset_index(drop=True)
