
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from src.preprocessing import Snapshot

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
    selected = np.argsort(-probabilities, kind="stable")[:top_k]
    target_weight = 1.0 / len(selected)
    weights: Dict[str, float] = {}
    sector_totals: Dict[str, float] = {}
    for index in selected:
        ticker = tickers[int(index)]
        sector = sectors[int(index)]
        sector_key = sector if sector is not None else f"__unknown__:{ticker}"
        sector_room = max_sector - sector_totals.get(sector_key, 0.0)
        weight = max(0.0, min(target_weight, max_position, sector_room))
        if weight > 0:
            weights[ticker] = weight
            sector_totals[sector_key] = sector_totals.get(sector_key, 0.0) + weight
    return weights

def portfolio_backtest(
    snapshots: Sequence[Snapshot], predictions: Sequence[np.ndarray], strategy: str,
    top_k: int = 10, max_position: float = 0.10, max_sector: float = 0.25, cost_rate: float = 0.001
) -> pd.DataFrame:
    previous_weights: Dict[str, float] = {}
    records: List[dict] = []
    for snapshot, scores in zip(snapshots, predictions):
        if strategy == "equal_weight":
            weights = {ticker: 1.0 / len(snapshot.tickers) for ticker in snapshot.tickers}
        else:
            weights = portfolio_weights(snapshot.tickers, scores, snapshot.sectors, top_k, max_position, max_sector)

        return_by_ticker = dict(zip(snapshot.tickers, snapshot.stock_returns))
        gross_return = sum(weight * float(return_by_ticker[ticker]) for ticker, weight in weights.items())
        
        symbols = set(previous_weights) | set(weights)
        turnover = sum(abs(weights.get(s, 0.0) - previous_weights.get(s, 0.0)) for s in symbols)
        transaction_cost = turnover * cost_rate
        
        records.append({
            "date": snapshot.date, "target_end": snapshot.target_end, "strategy": strategy,
            "gross_return": gross_return, "net_return": gross_return - transaction_cost,
            "turnover": turnover, "transaction_cost": transaction_cost,
            "holdings": len(weights), "cash_weight": max(0.0, 1.0 - sum(weights.values()))
        })
        previous_weights = weights
    return pd.DataFrame(records)

def performance_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    summaries: List[dict] = []
    for strategy, group in backtest.groupby("strategy", sort=True):
        returns = group.sort_values("date")["net_return"].astype(float).to_numpy()
        if len(returns) == 0:
            continue
        equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
        peaks = np.maximum.accumulate(equity)
        drawdowns = equity / peaks - 1.0
        volatility = float(np.std(returns, ddof=1) * np.sqrt(12.0)) if len(returns) > 1 else 0.0
        annualized_return = float(equity[-1] ** (12.0 / len(returns)) - 1.0)
        sharpe = (float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(12.0)) if len(returns) > 1 and np.std(returns, ddof=1) > 0 else 0.0)
        
        summaries.append({
            "strategy": strategy, "periods": len(returns), "cumulative_return": float(equity[-1] - 1.0),
            "annualized_return": annualized_return, "annualized_volatility": volatility,
            "sharpe_rf_0": sharpe, "max_drawdown": float(drawdowns.min()),
            "average_turnover": float(group["turnover"].mean()),
            "total_transaction_cost": float(group["transaction_cost"].sum()),
            "average_holdings": float(group["holdings"].mean())
        })
    return pd.DataFrame(summaries)