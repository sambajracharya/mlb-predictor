"""Predict a slate of upcoming games.

Refreshes the current season's raw data, rebuilds pregame features (upcoming
games included), and runs the saved models.

Usage:
    python -m mlbpred.predict                 # today
    python -m mlbpred.predict --date 2025-04-15
    python -m mlbpred.predict --date today --no-refresh
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date as date_cls

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, RAW_DIR, REPORT_DIR
from .features import build_dataset
from .ingest import ingest_season, load_raw, load_raw_optional
from .statcast import ingest_statcast

log = logging.getLogger(__name__)


def available_seasons() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in RAW_DIR.glob("games_*.parquet"))


def load_models(kind: str = "lgbm") -> tuple[dict, dict]:
    """`kind` is "lgbm" or "linear" (logistic for the win model, Poisson for runs)."""
    wanted = {"win": "logreg" if kind == "linear" else kind,
              "runs": "poisson" if kind == "linear" else kind}
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {}
    for name in meta["models"]:
        task, k = name.split("__")
        if k != (wanted["win"] if task == "win" else wanted["runs"]):
            continue
        models[task] = joblib.load(MODEL_DIR / f"{name}.joblib")
    if "win" not in models:
        raise SystemExit(f"no '{kind}' win model in {MODEL_DIR} - run python -m mlbpred.train")
    return models, meta


def predict_slate(target: pd.Timestamp, kind: str = "lgbm", refresh: bool = True) -> pd.DataFrame:
    season = target.year
    if refresh:
        ingest_season(season, refresh=True)
        try:
            ingest_statcast(season)  # incremental: only fetches new dates
        except Exception as exc:
            log.warning("statcast refresh failed (%s) - using cached data", exc)
    seasons = available_seasons()
    if season not in seasons:
        raise SystemExit(f"no raw data for {season} - run python -m mlbpred.ingest --seasons {season}")

    df = build_dataset(
        load_raw("games", seasons),
        load_raw("team_hitting", seasons),
        load_raw("team_pitching", seasons),
        load_raw("sp_gamelog", seasons),
        load_raw("sp_hands", seasons),
        load_raw_optional("statcast_team", seasons),
        load_raw_optional("statcast_sp", seasons),
        load_raw_optional("lineups", seasons),
        load_raw_optional("statcast_batter", seasons),
        keep_upcoming=True,
    )
    slate = df[df["date"] == target]
    if slate.empty:
        raise SystemExit(f"no scheduled games found for {target.date()}")

    models, meta = load_models(kind)
    fsets = meta["features"]
    wanted = set(fsets["win"]) | set(fsets["runs"])
    missing = sorted(f for f in wanted if f not in slate.columns)
    if missing:
        slate = slate.assign(**{f: float("nan") for f in missing})
        log.warning("%s features missing from slate, filled with NaN: %s", len(missing), missing[:5])

    meta_cols = ["game_pk", "date", "away_team", "home_team", "venue",
                 "away_sp_name", "home_sp_name"]
    out = slate[meta_cols].copy()
    if {"h_lineup_posted", "a_lineup_posted"} <= set(slate.columns):
        # both lineups posted = the prediction is using tonight's actual nine
        out["lineups"] = np.where(
            (slate["h_lineup_posted"] == 1) & (slate["a_lineup_posted"] == 1),
            "posted", "carried",
        )
    out["home_win_prob"] = models["win"].predict_proba(slate[fsets["win"]])[:, 1]
    out["away_win_prob"] = 1 - out["home_win_prob"]
    for task, model in models.items():
        if task == "win":
            continue
        out[f"pred_{task}"] = model.predict(slate[fsets["runs"]])
    if {"pred_home_runs", "pred_away_runs"} <= set(out.columns):
        out["pred_total_runs"] = out["pred_home_runs"] + out["pred_away_runs"]
    return out.sort_values("home_win_prob", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "linear"])
    ap.add_argument("--no-refresh", action="store_true", help="use cached raw data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    target = pd.Timestamp(date_cls.today() if args.date == "today" else args.date)
    preds = predict_slate(target, kind=args.model, refresh=not args.no_refresh)

    show = preds.drop(columns=["game_pk", "date"]).copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    pd.set_option("display.width", 200)
    print(f"\n=== {target.date()} ===")
    print(show.to_string(index=False))

    path = REPORT_DIR / f"predictions_{target.date()}.csv"
    preds.to_csv(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
