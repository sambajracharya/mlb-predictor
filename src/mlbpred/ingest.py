"""Download raw MLB data and cache it under data/raw/.

Everything here is a faithful dump of what the API returned; no derived stats,
no filtering beyond "regular season". Feature logic lives in features.py.

Usage:
    python -m mlbpred.ingest --seasons 2019 2021 2022 2023 2024
    python -m mlbpred.ingest --seasons 2025 --refresh      # re-download
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

import pandas as pd

from .config import GAME_TYPE, RAW_DIR
from .mlb_api import chunked, get_json, map_parallel

log = logging.getLogger(__name__)

SCHEDULE_HYDRATE = "team,linescore,probablePitcher,venue,weather"

TEAM_HITTING_STATS = [
    "runs", "hits", "doubles", "triples", "homeRuns", "strikeOuts", "baseOnBalls",
    "hitByPitch", "atBats", "plateAppearances", "totalBases", "leftOnBase",
    "stolenBases", "groundIntoDoublePlay", "sacFlies", "numberOfPitches",
]
TEAM_PITCHING_STATS = [
    "runs", "earnedRuns", "hits", "homeRuns", "strikeOuts", "baseOnBalls",
    "hitByPitch", "outs", "battersFaced", "numberOfPitches", "strikes",
]
SP_STATS = [
    "gamesStarted", "outs", "runs", "earnedRuns", "hits", "homeRuns", "strikeOuts",
    "baseOnBalls", "hitByPitch", "battersFaced", "numberOfPitches", "strikes",
]


# --------------------------------------------------------------------------- #
# schedule / games
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_game(game: dict) -> dict:
    teams = game.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    line = (game.get("linescore") or {}).get("teams", {})
    line_home, line_away = line.get("home", {}), line.get("away", {})
    weather = game.get("weather") or {}
    status = game.get("status") or {}

    def side(prefix: str, node: dict, line_node: dict) -> dict:
        team = node.get("team") or {}
        sp = node.get("probablePitcher") or {}
        record = node.get("leagueRecord") or {}
        return {
            f"{prefix}_team_id": team.get("id"),
            f"{prefix}_team": team.get("name"),
            f"{prefix}_abbr": team.get("abbreviation"),
            f"{prefix}_league_id": (team.get("league") or {}).get("id"),
            f"{prefix}_div_id": (team.get("division") or {}).get("id"),
            f"{prefix}_score": line_node.get("runs", node.get("score")),
            f"{prefix}_h": line_node.get("hits"),
            f"{prefix}_e": line_node.get("errors"),
            f"{prefix}_sp_id": sp.get("id"),
            f"{prefix}_sp_name": sp.get("fullName"),
            # Record *entering* the game is not exposed; this is the record after
            # it. Only used for sanity checks, never as a feature.
            f"{prefix}_record_w": record.get("wins"),
            f"{prefix}_record_l": record.get("losses"),
        }

    row: dict[str, Any] = {
        "game_pk": game.get("gamePk"),
        "date": game.get("officialDate"),
        "season": int(game.get("season")) if game.get("season") else None,
        "game_type": game.get("gameType"),
        "game_number": game.get("gameNumber", 1),
        "double_header": game.get("doubleHeader"),
        "day_night": game.get("dayNight"),
        "series_game": game.get("seriesGameNumber"),
        "status": status.get("codedGameState"),
        "detailed_status": status.get("detailedState"),
        "venue_id": (game.get("venue") or {}).get("id"),
        "venue": (game.get("venue") or {}).get("name"),
        "wx_condition": weather.get("condition"),
        "wx_temp": _num(weather.get("temp")),
        "wx_wind": weather.get("wind"),
    }
    row.update(side("home", home, line_home))
    row.update(side("away", away, line_away))
    return row


def fetch_schedule(start: str, end: str, game_type: str | None = GAME_TYPE) -> pd.DataFrame:
    """One row per game between `start` and `end` (inclusive, YYYY-MM-DD)."""
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "hydrate": SCHEDULE_HYDRATE,
    }
    if game_type:
        params["gameType"] = game_type
    payload = get_json("schedule", params)
    rows = [_parse_game(g) for d in payload.get("dates", []) for g in d.get("games", [])]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_pk"]).reset_index(drop=True)


def fetch_season_games(season: int) -> pd.DataFrame:
    return fetch_schedule(f"{season}-01-01", f"{season}-12-31")


# --------------------------------------------------------------------------- #
# team game logs
# --------------------------------------------------------------------------- #
def _parse_splits(splits: Iterable[dict], stat_fields: list[str], id_key: str) -> list[dict]:
    out = []
    for sp in splits:
        stat = sp.get("stat") or {}
        game = sp.get("game") or {}
        row = {
            "date": sp.get("date"),
            "game_pk": game.get("gamePk"),
            "season": int(sp["season"]) if sp.get("season") else None,
            id_key: (sp.get("team") or {}).get("id") if id_key == "team_id" else (sp.get("player") or {}).get("id"),
            "team_id": (sp.get("team") or {}).get("id"),
            "opponent_id": (sp.get("opponent") or {}).get("id"),
            "is_home": sp.get("isHome"),
            "is_win": sp.get("isWin"),
        }
        for f in stat_fields:
            row[f] = _num(stat.get(f))
        out.append(row)
    return out


def fetch_team_gamelogs(season: int, group: str) -> pd.DataFrame:
    """Per-game team totals for `group` in {"hitting", "pitching"}, all 30 teams."""
    teams = get_json("teams", {"sportId": 1, "season": season})["teams"]
    fields = TEAM_HITTING_STATS if group == "hitting" else TEAM_PITCHING_STATS

    def one(team: dict) -> list[dict]:
        payload = get_json(
            f"teams/{team['id']}/stats",
            {"stats": "gameLog", "group": group, "season": season, "gameType": GAME_TYPE},
        )
        stats = payload.get("stats") or []
        splits = stats[0].get("splits", []) if stats else []
        return _parse_splits(splits, fields, "team_id")

    rows = [r for chunk in map_parallel(one, teams, workers=4) for r in chunk]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# starting pitcher game logs + handedness
# --------------------------------------------------------------------------- #
def fetch_pitcher_gamelogs(season: int, pitcher_ids: list[int]) -> pd.DataFrame:
    def one(pid: int) -> list[dict]:
        try:
            payload = get_json(
                f"people/{int(pid)}/stats",
                {"stats": "gameLog", "group": "pitching", "season": season, "gameType": GAME_TYPE},
            )
        except Exception as exc:  # a retired/missing id should not kill the run
            log.warning("pitcher %s season %s failed: %s", pid, season, exc)
            return []
        stats = payload.get("stats") or []
        splits = stats[0].get("splits", []) if stats else []
        rows = _parse_splits(splits, SP_STATS, "player_id")
        for r in rows:
            r["player_id"] = int(pid)
        return rows

    rows = [r for chunk in map_parallel(one, pitcher_ids, workers=6) for r in chunk]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)


def fetch_lineups(season: int) -> pd.DataFrame:
    """Posted batting orders: one row per (game_pk, team, slot).

    Lineups are public ~3 hours before first pitch, so they are legitimate
    pregame information - but only for games where they have actually been
    posted. `slot` is 1-9 in batting order.
    """
    rows = []
    months = pd.date_range(f"{season}-03-01", f"{season}-11-15", freq="MS")
    for start in months:
        end = start + pd.offsets.MonthEnd(1)
        payload = get_json(
            "schedule",
            {
                "sportId": 1, "gameType": GAME_TYPE, "hydrate": "lineups",
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
            },
        )
        for d in payload.get("dates", []):
            for g in d.get("games", []):
                lu = g.get("lineups") or {}
                for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                    players = lu.get(key) or []
                    team = ((g.get("teams") or {}).get(side) or {}).get("team") or {}
                    for slot, p in enumerate(players[:9], start=1):
                        rows.append(
                            {
                                "game_pk": g.get("gamePk"),
                                "date": g.get("officialDate"),
                                "season": season,
                                "team_id": team.get("id"),
                                "side": side,
                                "slot": slot,
                                "player_id": p.get("id"),
                                "position": (p.get("primaryPosition") or {}).get("abbreviation"),
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["player_id", "team_id"]).reset_index(drop=True)


def fetch_pitcher_hands(pitcher_ids: list[int]) -> pd.DataFrame:
    rows = []
    for batch in chunked([int(p) for p in pitcher_ids], 100):
        payload = get_json("people", {"personIds": ",".join(map(str, batch))})
        for person in payload.get("people", []):
            rows.append(
                {
                    "player_id": person.get("id"),
                    "sp_throws": (person.get("pitchHand") or {}).get("code"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def raw_path(kind: str, season: int) -> "object":
    return RAW_DIR / f"{kind}_{season}.parquet"


def ingest_season(season: int, refresh: bool = False) -> None:
    games_p = raw_path("games", season)
    if refresh or not games_p.exists():
        games = fetch_season_games(season)
        games.to_parquet(games_p, index=False)
        log.info("games %s -> %s rows", season, len(games))
    else:
        games = pd.read_parquet(games_p)

    for group in ("hitting", "pitching"):
        p = raw_path(f"team_{group}", season)
        if refresh or not p.exists():
            df = fetch_team_gamelogs(season, group)
            df.to_parquet(p, index=False)
            log.info("team %s %s -> %s rows", group, season, len(df))

    sp_ids = pd.unique(
        pd.concat([games["home_sp_id"], games["away_sp_id"]]).dropna().astype(int)
    ).tolist()

    p = raw_path("sp_gamelog", season)
    if refresh or not p.exists():
        df = fetch_pitcher_gamelogs(season, sp_ids)
        df.to_parquet(p, index=False)
        log.info("sp gamelogs %s -> %s rows (%s pitchers)", season, len(df), len(sp_ids))

    p = raw_path("sp_hands", season)
    if refresh or not p.exists():
        df = fetch_pitcher_hands(sp_ids)
        df.to_parquet(p, index=False)
        log.info("sp hands %s -> %s rows", season, len(df))

    p = raw_path("lineups", season)
    if refresh or not p.exists():
        df = fetch_lineups(season)
        df.to_parquet(p, index=False)
        log.info("lineups %s -> %s rows (%s games)", season, len(df),
                 df["game_pk"].nunique() if len(df) else 0)


def load_raw(kind: str, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = raw_path(kind, s)
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no raw {kind} files for seasons {seasons} - run ingest first")
    return pd.concat(frames, ignore_index=True)


def load_raw_optional(kind: str, seasons: list[int]) -> pd.DataFrame | None:
    """Like load_raw but returns None when nothing is cached (optional sources)."""
    try:
        return load_raw(kind, seasons)
    except FileNotFoundError:
        log.warning("no %s files cached - building without them", kind)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Download raw MLB data")
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for season in args.seasons:
        log.info("=== ingesting %s ===", season)
        ingest_season(season, refresh=args.refresh)


if __name__ == "__main__":
    main()
