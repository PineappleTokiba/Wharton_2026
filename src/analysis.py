"""Portfolio backtesting, performance summary, and ranking analysis for weekly stock models.

Each snapshot's return covers the whole holding horizon (52 weeks by default) while snapshots
are monthly, so consecutive results overlap. They are therefore treated as separate cohorts and
averaged, never compounded: chaining overlapping one-year returns would hugely inflate results.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from src.preprocessing import Snapshot, path_risk


def predict_probabilities(
    model: torch.nn.Module, snapshots: Sequence[Snapshot], mean: np.ndarray, standard_deviation: np.ndarray
) -> List[np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for snapshot in snapshots:
            features = ((snapshot.features - mean) / standard_deviation).astype(np.float32)
            logits = model(torch.from_numpy(features), torch.from_numpy(snapshot.adjacency))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return predictions


def portfolio_weights(
    tickers: Sequence[str], probabilities: np.ndarray, sectors: Sequence[Optional[str]],
    top_k: int = 10, max_position: float = 0.10, max_sector: float = 0.25
) -> Dict[str, float]:
    """Fill top_k equal slots from the best score down, respecting position and sector caps.

    When a sector is full, its slot passes to the next-ranked stock in another sector, so the
    caps diversify the portfolio instead of leaving it in cash. Weight stays in cash only if
    the caps make full investment impossible, such as top_k * max_position below 1.
    """
    if len(tickers) == 0:
        return {}
    ranked = np.argsort(-probabilities, kind="stable")
    target_weight = min(1.0 / min(top_k, len(ranked)), max_position)
    budget = target_weight * min(top_k, len(ranked))
    weights: Dict[str, float] = {}
    sector_totals: Dict[str, float] = {}
    invested = 0.0
    for index in ranked:
        if invested >= budget - 1e-9:
            break
        ticker = tickers[int(index)]
        sector = sectors[int(index)]
        # Stocks with no sector share one bucket so they cannot each evade the sector cap.
        sector_key = sector if sector is not None else "__unknown__"
        sector_room = max_sector - sector_totals.get(sector_key, 0.0)
        weight = max(0.0, min(target_weight, sector_room, budget - invested))
        if weight > 1e-9:
            weights[ticker] = weight
            sector_totals[sector_key] = sector_totals.get(sector_key, 0.0) + weight
            invested += weight
    return weights


def _cohort_path_risk(prices: pd.DataFrame, snapshot: Snapshot, weights: Dict[str, float]) -> Dict[str, float]:
    """Realized volatility and drawdown of a buy-and-hold cohort, with its cash held flat."""
    window = prices.loc[snapshot.date : snapshot.target_end, list(weights)].ffill()
    invested = (window / window.iloc[0] * pd.Series(weights)).sum(axis=1)
    volatility, drawdown = path_risk((invested + 1.0 - sum(weights.values())).to_numpy())
    return {"path_volatility": volatility, "path_max_drawdown": drawdown}


def portfolio_backtest(
    snapshots: Sequence[Snapshot], predictions: Sequence[np.ndarray], strategy: str,
    top_k: int = 10, max_position: float = 0.10, max_sector: float = 0.25, cost_rate: float = 0.001,
    prices: Optional[pd.DataFrame] = None, independent_cohorts: bool = False,
) -> pd.DataFrame:
    """Backtest a strategy across snapshots.

    With independent_cohorts, each snapshot is bought and later sold on its own, so every
    cohort pays a full round trip. Otherwise costs follow turnover between consecutive
    snapshots. Passing prices adds each cohort's realized volatility and drawdown.
    """
    previous_weights: Dict[str, float] = {}
    records: List[dict] = []
    for snapshot, scores in zip(snapshots, predictions):
        if strategy == "equal_weight":
            weights = {ticker: 1.0 / len(snapshot.tickers) for ticker in snapshot.tickers}
        else:
            weights = portfolio_weights(snapshot.tickers, scores, snapshot.sectors, top_k, max_position, max_sector)

        return_by_ticker = dict(zip(snapshot.tickers, snapshot.stock_returns))
        gross_return = sum(weight * float(return_by_ticker[ticker]) for ticker, weight in weights.items())

        weights_sum = sum(weights.values())
        if independent_cohorts:
            turnover = 2.0 * weights_sum
        else:
            symbols = set(previous_weights) | set(weights)
            turnover = sum(abs(weights.get(s, 0.0) - previous_weights.get(s, 0.0)) for s in symbols)
        transaction_cost = turnover * cost_rate

        record = {
            "date": snapshot.date,
            "target_end": snapshot.target_end,
            "strategy": strategy,
            "gross_return": gross_return,
            "net_return": gross_return - transaction_cost,
            "benchmark_return": snapshot.spy_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "holdings": len(weights),
            "cash_weight": max(0.0, 1.0 - weights_sum),
        }
        if prices is not None and weights:
            record.update(_cohort_path_risk(prices, snapshot, weights))
        records.append(record)
        previous_weights = weights
    return pd.DataFrame(records)


def benchmark_backtest(snapshots: Sequence[Snapshot], benchmark_prices: pd.Series, name: str = "SPY") -> pd.DataFrame:
    """Benchmark held over the same cohorts as the strategies, with no costs."""
    records: List[dict] = []
    for snapshot in snapshots:
        volatility, drawdown = path_risk(benchmark_prices.loc[snapshot.date : snapshot.target_end].ffill().to_numpy())
        records.append({
            "date": snapshot.date,
            "target_end": snapshot.target_end,
            "strategy": name,
            "gross_return": snapshot.spy_return,
            "net_return": snapshot.spy_return,
            "benchmark_return": snapshot.spy_return,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "holdings": 1,
            "cash_weight": 0.0,
            "path_volatility": volatility,
            "path_max_drawdown": drawdown,
        })
    return pd.DataFrame(records)


def performance_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    """Average per-cohort results by strategy, with returns annualized from the holding period."""
    summaries: List[dict] = []
    for strategy, group in backtest.groupby("strategy", sort=True):
        returns = group["net_return"].astype(float).to_numpy()
        if len(returns) == 0:
            continue
        horizon_years = float((group["target_end"] - group["date"]).dt.days.mean() / 365.25)
        mean_return = float(returns.mean())
        annualized_return = float((1.0 + mean_return) ** (1.0 / horizon_years) - 1.0)
        volatility = float(group["path_volatility"].mean()) if "path_volatility" in group else float("nan")
        row = {
            "strategy": strategy,
            "cohorts": len(returns),
            "annualized_return": annualized_return,
            "worst_cohort_return": float(returns.min()),
            "best_cohort_return": float(returns.max()),
            "avg_volatility": volatility,
            "avg_max_drawdown": float(group["path_max_drawdown"].mean()) if "path_max_drawdown" in group else float("nan"),
            "return_per_volatility": annualized_return / volatility if volatility > 0 else float("nan"),
            "avg_holdings": float(group["holdings"].mean()),
            "avg_cash": float(group["cash_weight"].mean()),
        }
        if "benchmark_return" in group:
            row["cohorts_beating_spy"] = float((returns > group["benchmark_return"].astype(float).to_numpy()).mean())
        summaries.append(row)
    return pd.DataFrame(summaries)


def _cross_sectional_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def ranking_skill(snapshots: Sequence[Snapshot], predictions: Sequence[np.ndarray], top_k: int) -> Dict[str, float]:
    """How well scores rank each snapshot's stocks against the profile's own target.

    AUC of 0.5 is no skill. Precision is only meaningful relative to the base rate.
    """
    aucs, precisions, base_rates = [], [], []
    for snapshot, scores in zip(snapshots, predictions):
        selected = np.argsort(-scores, kind="stable")[:top_k]
        aucs.append(_cross_sectional_auc(snapshot.labels, scores))
        precisions.append(float(snapshot.labels[selected].mean()))
        base_rates.append(float(snapshot.labels.mean()))
    return {
        "mean_auc": float(np.nanmean(aucs)),
        "precision_at_k": float(np.mean(precisions)),
        "base_rate": float(np.mean(base_rates)),
    }


def get_snapshot_rankings(
    snapshot: Snapshot, probabilities: np.ndarray, top_n: int = 15
) -> pd.DataFrame:
    """Rank a snapshot's stocks by predicted probability of meeting the profile's target."""
    sorted_indices = np.argsort(-probabilities)
    rankings = []

    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        sector = snapshot.sectors[idx] if idx < len(snapshot.sectors) else None
        rankings.append({
            "rank": rank,
            "ticker": snapshot.tickers[idx],
            "sector": sector if sector else "Unknown",
            "probability": float(probabilities[idx]),
            "actual_return": float(snapshot.stock_returns[idx]),
            "target_met": bool(int(snapshot.labels[idx])),
        })

    return pd.DataFrame(rankings)
