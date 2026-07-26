"""Score saved predictions against what actually happened.

Reads reports/predictions_<date>.csv, fetches the final scores for that date, and
reports how the predictions did against the same baselines the backtest uses.

A single day is 15 games - far too small to conclude anything. Use `--since` to
pool days once you have a few weeks; the per-day numbers are for curiosity, the
pooled ones are the signal.

Usage:
    python -m mlbpred.score --date 2026-07-25
    python -m mlbpred.score --since 2026-07-01        # pool every saved day
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORT_DIR
from .ingest import fetch_schedule
from .models import classification_metrics, regression_metrics

log = logging.getLogger(__name__)


def load_actuals(date: str) -> pd.DataFrame:
    """Final scores for one date, keyed by game_pk."""
    games = fetch_schedule(date, date)
    if games.empty:
        return games
    final = games[games["status"].eq("F")].copy()
    return final[["game_pk", "home_score", "away_score", "home_team", "away_team"]].rename(
        columns={"home_score": "actual_home_runs", "away_score": "actual_away_runs"}
    )


def join_day(date: str) -> pd.DataFrame | None:
    path = REPORT_DIR / f"predictions_{date}.csv"
    if not path.exists():
        log.warning("no saved predictions for %s (%s)", date, path.name)
        return None
    preds = pd.read_csv(path)
    actuals = load_actuals(date)
    if actuals.empty:
        log.warning("%s: no completed games yet", date)
        return None
    df = preds.merge(actuals.drop(columns=["home_team", "away_team"]),
                     on="game_pk", how="inner")
    if df.empty:
        return None
    df["actual_home_win"] = (df["actual_home_runs"] > df["actual_away_runs"]).astype(int)
    df["predicted_winner_correct"] = (
        (df["home_win_prob"] >= 0.5) == (df["actual_home_win"] == 1)
    ).astype(int)
    df["date"] = date
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Model vs the two baselines it has to beat, on these games."""
    y = df["actual_home_win"].to_numpy()
    p = df["home_win_prob"].to_numpy()
    rows = [{"source": "model", **classification_metrics(y, p)}]
    # 53.1% is the long-run home-field rate in the training data
    rows.append({"source": "baseline:home_field",
                 **classification_metrics(y, np.full(len(y), 0.5312))})
    out = pd.DataFrame(rows)

    runs = []
    for side in ("home", "away"):
        a = df[f"actual_{side}_runs"].to_numpy()
        if f"pred_{side}_runs" in df.columns:
            runs.append({"target": f"{side}_runs", "source": "model",
                         **regression_metrics(a, df[f"pred_{side}_runs"].to_numpy())})
            runs.append({"target": f"{side}_runs", "source": "baseline:league_mean",
                         **regression_metrics(a, np.full(len(a), 4.5))})
    return out, pd.DataFrame(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="score one day (YYYY-MM-DD)")
    g.add_argument("--since", help="pool every saved predictions file from this date on")
    ap.add_argument("--quiet", action="store_true", help="skip the per-game table")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.date:
        dates = [args.date]
    else:
        saved = sorted(Path(REPORT_DIR).glob("predictions_*.csv"))
        dates = [p.stem.replace("predictions_", "") for p in saved]
        dates = [d for d in dates if d >= args.since]

    frames = [f for f in (join_day(d) for d in dates) if f is not None]
    if not frames:
        raise SystemExit("nothing to score")
    df = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 200)
    if not args.quiet:
        show = df[["date", "away_team", "home_team", "lineups", "home_win_prob",
                   "pred_away_runs", "pred_home_runs",
                   "actual_away_runs", "actual_home_runs", "predicted_winner_correct"]].copy()
        show = show.rename(columns={"home_win_prob": "p(home)",
                                    "predicted_winner_correct": "hit"})
        show["p(home)"] = show["p(home)"].round(3)
        for c in ("pred_away_runs", "pred_home_runs"):
            show[c] = show[c].round(1)
        show["hit"] = np.where(show["hit"] == 1, "yes", "NO")
        print(f"\n=== per game ({len(df)} games) ===")
        print(show.sort_values("p(home)", ascending=False).to_string(index=False))

    wins, runs = summarise(df)
    print("\n=== win prediction ===")
    print(wins.round(4).to_string(index=False))
    print("\n=== runs prediction ===")
    print(runs.round(3).to_string(index=False))

    n, hits = len(df), int(df["predicted_winner_correct"].sum())
    se = np.sqrt(0.25 / n)
    print(f"\npicked {hits}/{n} winners = {hits / n:.1%}  "
          f"(±{1.96 * se:.1%} at 95% confidence on this sample size)")
    if n < 200:
        print("Sample far too small to judge the model - the backtest over 8,000+ games "
              "is the number that means something. This is a spot check for bugs, "
              "not evidence.")


if __name__ == "__main__":
    main()
