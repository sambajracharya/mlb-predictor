"""Download historical Polymarket MLB moneyline markets and pregame prices.

For each game we keep the last traded price *before* `gameStartTime` - the same
"only pregame information" rule the model lives under. Prices land in
data/raw/polymarket_<season>.parquet.

Usage:
    python -m mlbpred.polymarket --seasons 2025 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import timedelta

import pandas as pd
import requests

from .config import RAW_DIR
from .mlb_api import map_parallel

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# plain game slug: mlb-lad-nym-2025-05-23 (props etc. have suffixes)
GAME_SLUG = re.compile(r"^mlb-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d{2}-\d{2}$")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "mlb-predictor-research/0.1"
    return s


def fetch_mlb_events(season: int, session: requests.Session | None = None) -> pd.DataFrame:
    """All closed Polymarket MLB game (moneyline) events for one season.

    Gamma rejects offsets beyond ~2000 and props inflate the event count badly,
    so the season is fetched in one-week date windows.
    """
    s = session or _session()
    rows = []
    windows = pd.date_range(f"{season}-03-01", f"{season}-11-15", freq="7D")
    for win_start in windows:
        win_end = win_start + pd.Timedelta(days=7)
        offset = 0
        while True:
            resp = s.get(
                f"{GAMMA}/events",
                params={
                    "tag_slug": "mlb", "closed": "true", "limit": 100, "offset": offset,
                    "start_date_min": win_start.strftime("%Y-%m-%d"),
                    "start_date_max": win_end.strftime("%Y-%m-%d"),
                },
                timeout=60,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                break
            rows.extend(_parse_events(events))
            if len(events) < 100 or offset >= 1900:
                break
            offset += 100
            time.sleep(0.1)
    df = _finalize_events(rows)
    return df


def _parse_events(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
            slug = ev.get("slug", "")
            if not GAME_SLUG.match(slug):
                continue
            for m in ev.get("markets", []):
                try:
                    outcomes = json.loads(m.get("outcomes") or "[]")
                    tokens = json.loads(m.get("clobTokenIds") or "[]")
                    finals = json.loads(m.get("outcomePrices") or "[]")
                except json.JSONDecodeError:
                    continue
                if len(outcomes) != 2 or len(tokens) != 2:
                    continue
                rows.append(
                    {
                        "slug": slug,
                        "title": ev.get("title"),
                        "game_start": m.get("gameStartTime"),
                        "outcome_0": outcomes[0], "outcome_1": outcomes[1],
                        "token_0": tokens[0], "token_1": tokens[1],
                        "final_0": float(finals[0]) if finals else None,
                        "volume": float(m.get("volume") or 0),
                    }
                )
    return rows


def _finalize_events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("slug", keep="first")  # windows can overlap an event
    df["game_start"] = pd.to_datetime(df["game_start"], utc=True, format="mixed")
    df = df.dropna(subset=["game_start"])
    df["date_key"] = df["slug"].str.slice(-10)
    return df.reset_index(drop=True)


def fetch_pregame_price(row: pd.Series, session: requests.Session) -> float | None:
    """Last traded price of token_0 strictly before game start (48h lookback)."""
    start = int((row["game_start"] - timedelta(hours=48)).timestamp())
    end = int(row["game_start"].timestamp())
    try:
        resp = session.get(
            f"{CLOB}/prices-history",
            params={"market": row["token_0"], "startTs": start, "endTs": end, "fidelity": 60},
            timeout=60,
        )
        resp.raise_for_status()
        pts = resp.json().get("history", [])
    except requests.RequestException as exc:
        log.warning("history failed for %s: %s", row["slug"], exc)
        return None
    pts = [p for p in pts if p["t"] < end]
    return float(pts[-1]["p"]) if pts else None


def ingest_polymarket(season: int, refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / f"polymarket_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    s = _session()
    events = fetch_mlb_events(season, s)
    log.info("season %s: %s moneyline markets found", season, len(events))
    if events.empty:
        return events

    prices = map_parallel(
        lambda r: fetch_pregame_price(r[1], s), list(events.iterrows()), workers=8
    )
    events["price_0_pregame"] = prices
    events = events.dropna(subset=["price_0_pregame"]).reset_index(drop=True)
    # throw away dead markets: a pregame price pinned at the extremes or with no
    # volume is a stale quote, not a market opinion
    live = (events["price_0_pregame"].between(0.02, 0.98)) & (events["volume"] > 0)
    events = events[live].reset_index(drop=True)
    events.to_parquet(path, index=False)
    log.info("season %s: %s markets with usable pregame prices -> %s",
             season, len(events), path)
    return events


def match_to_games(pm: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach home-team market probability to schedule rows.

    Outcome strings are nicknames ("Red Sox"); our schedule has full names
    ("Boston Red Sox"), so match nickname-as-suffix within the same date.
    """
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    pm = pm.copy()
    pm["date"] = pd.to_datetime(pm["date_key"])

    rows = []
    for _, mk in pm.iterrows():
        day = g[g["date"] == mk["date"]]
        if day.empty:
            continue
        h = day["home_team"].str.endswith(mk["outcome_0"])
        a = day["away_team"].str.endswith(mk["outcome_0"])
        hit_h = day[h & day["away_team"].str.endswith(mk["outcome_1"])]
        hit_a = day[a & day["home_team"].str.endswith(mk["outcome_1"])]
        if len(hit_h) == 1 and hit_a.empty:
            rows.append({"game_pk": hit_h.iloc[0]["game_pk"],
                         "market_home_prob": mk["price_0_pregame"],
                         "market_volume": mk["volume"], "final_0": mk["final_0"],
                         "outcome_0_is_home": True})
        elif len(hit_a) == 1 and hit_h.empty:
            rows.append({"game_pk": hit_a.iloc[0]["game_pk"],
                         "market_home_prob": 1.0 - mk["price_0_pregame"],
                         "market_volume": mk["volume"], "final_0": mk["final_0"],
                         "outcome_0_is_home": False})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates("game_pk", keep=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for season in args.seasons:
        ingest_polymarket(season, refresh=args.refresh)


if __name__ == "__main__":
    main()
