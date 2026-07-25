"""Compare the model's win probabilities against Polymarket pregame prices.

For each test season the model is trained only on earlier seasons (same
walk-forward rule as backtest.py), then both model and market are scored on the
identical set of games. Also runs a paper-trading simulation: flat $1 stakes on
either side whenever the model's probability differs from the market price by
more than an edge threshold.

This is research tooling. It measures whether the model disagrees usefully with
the market on historical data; it is not trading advice, and live trading adds
spread, slippage, and size limits that are not modelled here.

Usage:
    python -m mlbpred.market_eval --test-seasons 2025 2026
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .config import REPORT_DIR
from .ingest import load_raw
from .models import classification_metrics
from .polymarket import ingest_polymarket, match_to_games
from .train import feature_sets, fit_models, load_dataset

log = logging.getLogger(__name__)

EDGES = (0.02, 0.04, 0.06, 0.08)


def paper_trade(df: pd.DataFrame, edge: float) -> dict:
    """Flat $1 stake whenever |model - market| > edge. Payout 1/price on a win.

    No fees are subtracted (Polymarket takes no fee on this market type), but
    the entry is the last *traded* price, not the ask - real fills are worse.
    """
    p_model, p_mkt = df["model_home_prob"], df["market_home_prob"]
    y = df["home_win"]

    bet_home = p_model - p_mkt > edge
    bet_away = p_mkt - p_model > edge
    n = int(bet_home.sum() + bet_away.sum())
    if n == 0:
        return {"edge": edge, "bets": 0, "roi": np.nan, "hit_rate": np.nan}

    pnl_home = np.where(y[bet_home] == 1, 1 / p_mkt[bet_home] - 1, -1.0)
    away_price = 1 - p_mkt[bet_away]
    pnl_away = np.where(y[bet_away] == 0, 1 / away_price - 1, -1.0)
    pnl = np.concatenate([pnl_home, pnl_away])
    wins = int((y[bet_home] == 1).sum() + (y[bet_away] == 0).sum())
    return {
        "edge": edge,
        "bets": n,
        "wins": wins,
        "hit_rate": wins / n,
        "total_pnl": float(pnl.sum()),
        "roi": float(pnl.mean()),
        # rough 1-sigma on ROI so we don't mistake noise for alpha
        "roi_se": float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibration_regression(test: pd.DataFrame) -> pd.DataFrame:
    """Does the model add information *on top of* the market?

    Fit  P(home win) = sigmoid(b0 + b1*logit(model) + b2*logit(market)).
    If b1 is significantly > 0, the model carries signal the market has not
    priced; the fitted equation is then your best fair-value estimate. Standard
    errors come from the Fisher information (unregularised fit).
    """
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([_logit(test["model_home_prob"].to_numpy()),
                         _logit(test["market_home_prob"].to_numpy())])
    y = test["home_win"].astype(int).to_numpy()
    clf = LogisticRegression(C=1e9, max_iter=5000).fit(X, y)

    Xd = np.column_stack([np.ones(len(X)), X])  # add intercept for the covariance
    p = clf.predict_proba(X)[:, 1]
    W = p * (1 - p)
    cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
    coefs = np.concatenate([clf.intercept_, clf.coef_[0]])
    ses = np.sqrt(np.diag(cov))
    return pd.DataFrame({
        "term": ["intercept", "logit(model)", "logit(market)"],
        "coef": coefs, "se": ses, "z": coefs / ses,
    })


def disagreement_table(test: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """When model and market disagree, which direction does reality move?

    Rows are buckets of (model - market). `actual_minus_market` > 0 in positive-
    gap buckets (and < 0 in negative ones) means outcomes drift toward the
    model's side of the disagreement - i.e. the disagreements contain alpha
    rather than noise.
    """
    t = test.copy()
    t["gap"] = t["model_home_prob"] - t["market_home_prob"]
    t["bucket"] = pd.qcut(t["gap"], bins, duplicates="drop")
    g = t.groupby("bucket", observed=True).agg(
        n=("home_win", "size"),
        mean_gap=("gap", "mean"),
        model=("model_home_prob", "mean"),
        market=("market_home_prob", "mean"),
        actual=("home_win", "mean"),
    )
    g["actual_minus_market"] = g["actual"] - g["market"]
    g["actual_minus_model"] = g["actual"] - g["model"]
    return g.reset_index()


def evaluate_season(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train = df[df["season"] < season]
    test = df[(df["season"] == season) & df["market_home_prob"].notna()].copy()
    if train["season"].nunique() < 2 or len(test) < 100:
        log.warning("season %s: not enough data (train seasons=%s, matched games=%s)",
                    season, train["season"].nunique(), len(test))
        return None

    fsets = feature_sets(df)
    # lgbm is the better win model once posted-lineup features are in play
    fitted = fit_models(train, fsets, count_targets=[], kinds=("lgbm",))
    test["model_home_prob"] = fitted["win__lgbm"].predict_proba(test[fsets["win"]])[:, 1]

    y = test["home_win"].astype(int).to_numpy()
    blend = 0.5 * test["model_home_prob"] + 0.5 * test["market_home_prob"]
    scores = pd.DataFrame(
        [
            {"source": "model", **classification_metrics(y, test["model_home_prob"].to_numpy())},
            {"source": "market", **classification_metrics(y, test["market_home_prob"].to_numpy())},
            {"source": "50/50 blend", **classification_metrics(y, blend.to_numpy())},
        ]
    )
    scores.insert(0, "season", season)

    trades = pd.DataFrame([paper_trade(test, e) for e in EDGES])
    trades.insert(0, "season", season)

    corr = test[["model_home_prob", "market_home_prob"]].corr().iloc[0, 1]
    gap = (test["model_home_prob"] - test["market_home_prob"]).abs()
    log.info("season %s: %s matched games | corr(model, market)=%.3f | "
             "mean |gap|=%.3f | games with |gap|>0.10: %s",
             season, len(test), corr, gap.mean(), int((gap > 0.10).sum()))

    print(f"\n=== season {season}: does the model add signal on top of the market? ===")
    print(calibration_regression(test).round(4).to_string(index=False))
    print(f"\n=== season {season}: where do outcomes land when model and market disagree? ===")
    print(disagreement_table(test).round(4).to_string(index=False))
    return scores, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=int, nargs="+", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_dataset()
    matches = []
    for season in args.test_seasons:
        pm = ingest_polymarket(season)
        if pm.empty:
            log.warning("no Polymarket data for %s", season)
            continue
        games = load_raw("games", [season])
        matches.append(match_to_games(pm, games))
    if not matches:
        raise SystemExit("no market data matched")
    market = pd.concat(matches, ignore_index=True)
    df = df.merge(market[["game_pk", "market_home_prob", "market_volume"]],
                  on="game_pk", how="left")

    all_scores, all_trades = [], []
    for season in sorted(args.test_seasons):
        result = evaluate_season(df, season)
        if result:
            all_scores.append(result[0])
            all_trades.append(result[1])

    scores = pd.concat(all_scores, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)
    scores.to_csv(REPORT_DIR / "market_eval_scores.csv", index=False)
    trades.to_csv(REPORT_DIR / "market_eval_trades.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== probability quality: model vs Polymarket (same games) ===")
    print(scores.to_string(index=False))
    print("\n=== paper trading: flat $1 when |model - market| > edge ===")
    print(trades.round(4).to_string(index=False))
    print("\nROI worth taking seriously only if it clears ~2x roi_se and survives "
          "the spread you actually pay to enter.")


if __name__ == "__main__":
    main()
