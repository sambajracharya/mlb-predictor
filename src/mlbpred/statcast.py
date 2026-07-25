"""Download per-game Statcast aggregates from Baseball Savant.

Two tables per season, cached in data/raw/:
  statcast_team_<season>.parquet  - one row per (game_pk, team): offense quality
  statcast_sp_<season>.parquet    - one row per (game_pk, pitcher)

Savant's statcast_search CSV endpoint does the aggregation server-side
(group_by=team-date / name-date), so this is ~40 small requests per season, not
a pitch-level scrape. Ingest is incremental: re-running fetches only dates newer
than the cache.

Usage:
    python -m mlbpred.statcast --seasons 2021 2022 2023 2024 2025 2026
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from datetime import date

import pandas as pd
import requests

from .config import RAW_DIR

log = logging.getLogger(__name__)

BASE = "https://baseballsavant.mlb.com/statcast_search/csv"
WINDOW_DAYS = 14

TEAM_COLS = {
    "player_id": "team_id", "game_pk": "game_pk", "game_date": "game_date",
    "pa": "sc_pa", "bip": "sc_bip", "xwoba": "sc_xwoba", "xslg": "sc_xslg",
    "barrels_total": "sc_barrels", "hardhit_percent": "sc_hh_pct",
    "launch_speed": "sc_ev", "whiffs": "sc_whiffs", "swings": "sc_swings",
}
SP_COLS = {
    "player_id": "player_id", "game_pk": "game_pk", "game_date": "game_date",
    "pa": "sc_pa", "bip": "sc_bip", "xwoba": "sc_xwoba",
    "barrels_total": "sc_barrels", "hardhit_percent": "sc_hh_pct",
    "velocity": "sc_velo", "whiffs": "sc_whiffs", "swings": "sc_swings",
}
BATTER_COLS = {
    "player_id": "player_id", "game_pk": "game_pk", "game_date": "game_date",
    "pa": "sc_pa", "bip": "sc_bip", "xwoba": "sc_xwoba", "woba": "sc_woba",
    "barrels_total": "sc_barrels", "hardhit_percent": "sc_hh_pct",
    "whiffs": "sc_whiffs", "swings": "sc_swings",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (mlb-predictor research)"
    return s


GROUPING = {
    "team": ("batter", "team-date", TEAM_COLS),
    "sp": ("pitcher", "name-date", SP_COLS),
    "batter": ("batter", "name-date", BATTER_COLS),
}


def fetch_window(s: requests.Session, kind: str, season: int,
                 start: str, end: str) -> pd.DataFrame:
    """One aggregated CSV: kind is 'team', 'sp', or 'batter' (see GROUPING)."""
    player_type, group_by, _ = GROUPING[kind]
    params = {
        "all": "true", "hfGT": "R|", "hfSea": f"{season}|",
        "game_date_gt": start, "game_date_lt": end,
        "min_pitches": "0", "min_results": "0", "min_pas": "0",
        "sort_col": "pitches", "sort_order": "desc",
        "player_type": player_type,
        "group_by": group_by,
    }
    for attempt in range(3):
        try:
            resp = s.get(BASE, params=params, timeout=180)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            log.warning("savant %s %s..%s retry (%s)", kind, start, end, exc)
            time.sleep(5 * (attempt + 1))
    if not resp.text.strip():
        return pd.DataFrame()  # no games in this window (offseason edges)
    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    keep = GROUPING[kind][2]
    present = {k: v for k, v in keep.items() if k in df.columns}
    return df[list(present)].rename(columns=present)


def season_windows(season: int, start_after: str | None = None) -> list[tuple[str, str]]:
    lo = pd.Timestamp(start_after) if start_after else pd.Timestamp(f"{season}-03-15")
    hi = min(pd.Timestamp(f"{season}-11-10"), pd.Timestamp(date.today()))
    out = []
    cur = lo
    while cur < hi:
        nxt = min(cur + pd.Timedelta(days=WINDOW_DAYS), hi)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def ingest_statcast(season: int, refresh: bool = False,
                    kinds: tuple[str, ...] = ("team", "sp", "batter")) -> None:
    s = _session()
    for kind in kinds:
        path = RAW_DIR / f"statcast_{kind}_{season}.parquet"
        cached = None
        start_after = None
        if path.exists() and not refresh:
            cached = pd.read_parquet(path)
            if not cached.empty:
                # refetch the last cached day too - its window may have been partial
                start_after = (pd.Timestamp(cached["game_date"].max())
                               - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        windows = season_windows(season, start_after)
        if not windows:
            log.info("statcast %s %s: cache up to date", kind, season)
            continue
        frames = [cached] if cached is not None else []
        for lo, hi in windows:
            df = fetch_window(s, kind, season, lo, hi)
            log.info("statcast %s %s %s..%s -> %s rows", kind, season, lo, hi, len(df))
            frames.append(df)
            time.sleep(1.0)
        out = pd.concat([f for f in frames if f is not None and not f.empty],
                        ignore_index=True)
        if out.empty:
            log.warning("statcast %s %s: nothing fetched", kind, season)
            continue
        key = ["game_pk", "team_id"] if kind == "team" else ["game_pk", "player_id"]
        out = out.dropna(subset=key)
        out = out.drop_duplicates(key, keep="last").reset_index(drop=True)
        out.to_parquet(path, index=False)
        log.info("statcast %s %s: %s rows -> %s", kind, season, len(out), path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--refresh", action="store_true", help="full re-download")
    ap.add_argument("--kinds", nargs="+", default=["team", "sp", "batter"],
                    choices=["team", "sp", "batter"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for season in args.seasons:
        ingest_statcast(season, refresh=args.refresh, kinds=tuple(args.kinds))


if __name__ == "__main__":
    main()
