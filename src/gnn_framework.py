"""Point-in-time GNN research pipeline for cached daily stock prices."""

#BY THE WAY THIS IS OLD REFERENCE CODE


from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURE_NAMES = ("return_21d", "volatility_63d", "volume_change")
REQUIRED_COLUMNS = {"date", "ticker", "adj_close", "volume"}


@dataclass
class Snapshot:
    """One monthly graph and its forward 21-session outcomes."""

    date: pd.Timestamp
    target_end: pd.Timestamp
    tickers: Tuple[str, ...]
    features: np.ndarray
    adjacency: np.ndarray
    labels: np.ndarray
    stock_returns: np.ndarray
    spy_return: float
    qqq_return: Optional[float]
    sectors: Tuple[Optional[str], ...]


class GraphConvolution(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(features)


class StockGNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_size, hidden_size)
        self.conv2 = GraphConvolution(hidden_size, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(features, adjacency))
        return self.conv2(hidden, adjacency).squeeze(-1)


class NodeMLP(nn.Module):
    """Non-graph comparator that sees exactly the same node features."""

    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        del adjacency
        return self.layers(features).squeeze(-1)


def load_price_cache(
    cache_dir: Path, benchmark_ticker: str = "SPY"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-ticker downloader CSVs into aligned price, volume, and sector panels."""
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Price-cache directory does not exist: {cache_dir}")

    files = sorted(cache_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No ticker CSV files found in {cache_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(None)
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        if frame[["adj_close", "volume"]].isna().any().any():
            raise ValueError(f"{path} contains missing or non-numeric prices/volume")
        if frame["date"].duplicated().any():
            raise ValueError(f"{path} contains duplicate dates")
        if (frame["adj_close"] <= 0).any() or (frame["volume"] < 0).any():
            raise ValueError(f"{path} contains nonpositive prices or negative volume")
        symbols = frame["ticker"].unique()
        if len(symbols) != 1 or symbols[0] != path.stem.upper():
            raise ValueError(
                f"{path} must contain exactly one ticker matching its filename; "
                f"found {symbols.tolist()}"
            )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("Cache contains duplicate ticker/date rows across files")

    benchmark_ticker = benchmark_ticker.upper()
    all_prices = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    all_volumes = panel.pivot(index="date", columns="ticker", values="volume").sort_index()
    all_sectors = (
        panel.pivot(index="date", columns="ticker", values="sector").sort_index()
        if "sector" in panel.columns
        else pd.DataFrame(index=all_prices.index)
    )
    if benchmark_ticker not in all_prices.columns:
        raise ValueError(
            f"Benchmark {benchmark_ticker} is not cached in {cache_dir}; "
            "download it or pass another --benchmark ticker."
        )
    stock_columns = [column for column in all_prices.columns if column != benchmark_ticker]
    if len(stock_columns) < 2:
        raise ValueError(
            "At least two non-benchmark stocks are required to build a correlation graph."
        )
    stock_prices = all_prices[stock_columns]
    stock_volumes = all_volumes[stock_columns]
    stock_sectors = (
        all_sectors.reindex(columns=stock_columns)
        if not all_sectors.empty
        else pd.DataFrame(index=all_prices.index, columns=stock_columns)
    )
    return stock_prices, stock_volumes, all_prices, stock_sectors


def build_snapshots(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    qqq_ticker: str = "QQQ",
    correlation_window: int = 60,
    edges_per_node: int = 5,
    minimum_correlation_observations: int = 40,
    horizon: int = 21,
    sectors: Optional[pd.DataFrame] = None,
) -> List[Snapshot]:
    """Build monthly features/graphs at close and labels over later sessions."""
    if correlation_window < minimum_correlation_observations:
        raise ValueError("Correlation window must cover the minimum observations")
    if edges_per_node < 1 or horizon < 1:
        raise ValueError("edges_per_node and horizon must be positive")
    if prices.empty or not prices.index.is_monotonic_increasing:
        raise ValueError("Stock prices must be nonempty and sorted by date")
    if prices.index.has_duplicates:
        raise ValueError("Stock prices contain duplicate dates")
    if not prices.index.equals(volumes.index) or list(prices.columns) != list(volumes.columns):
        raise ValueError("Price and volume panels must have identical dates and ticker columns")
    if benchmark_ticker not in benchmark_prices.columns:
        raise ValueError(f"Benchmark prices do not contain {benchmark_ticker}")
    if not prices.columns.is_unique or len(prices.columns) < 2:
        raise ValueError("At least two uniquely named stock columns are required")

    daily_returns = prices.pct_change(fill_method=None)
    dates = prices.index
    monthly_indices = pd.Series(np.arange(len(dates)), index=dates).groupby(
        dates.to_period("M")
    ).last()
    snapshots: List[Snapshot] = []
    for raw_index in monthly_indices:
        index = int(raw_index)
        end_index = index + horizon
        feature_index = index - 1
        if feature_index < 80 or end_index >= len(dates):
            continue
        snapshot_date, target_end = dates[index], dates[end_index]
        benchmark_start = benchmark_prices[benchmark_ticker].iloc[index]
        benchmark_end = benchmark_prices[benchmark_ticker].iloc[end_index]
        if pd.isna(benchmark_start) or pd.isna(benchmark_end) or benchmark_start <= 0:
            continue
        spy_return = float(benchmark_end / benchmark_start - 1.0)
        qqq_return: Optional[float] = None
        if qqq_ticker in benchmark_prices.columns:
            qqq_start = benchmark_prices[qqq_ticker].iloc[index]
            qqq_end = benchmark_prices[qqq_ticker].iloc[end_index]
            if pd.notna(qqq_start) and pd.notna(qqq_end) and qqq_start > 0:
                qqq_return = float(qqq_end / qqq_start - 1.0)

        feature_date = dates[feature_index]
        trailing_start = max(0, feature_index - correlation_window + 1)
        trailing_returns = daily_returns.iloc[trailing_start : feature_index + 1]
        features: List[List[float]] = []
        valid_tickers: List[str] = []
        valid_indices: List[int] = []
        stock_forward_returns: List[float] = []
        snapshot_sectors: List[Optional[str]] = []
        for ticker_index, ticker in enumerate(prices.columns):
            close = prices[ticker]
            volume = volumes[ticker]
            start_price = close.iloc[feature_index - horizon]
            current_price = close.iloc[feature_index]
            execution_price = close.iloc[index]
            end_price = close.iloc[end_index]
            recent_returns = daily_returns[ticker].iloc[feature_index - 62 : feature_index + 1]
            recent_volumes = volume.iloc[feature_index - 19 : feature_index + 1]
            prior_volumes = volume.iloc[feature_index - 79 : feature_index - 19]
            corr_values = trailing_returns[ticker]
            if (
                pd.isna(start_price)
                or pd.isna(current_price)
                or pd.isna(execution_price)
                or pd.isna(end_price)
                or start_price <= 0
                or current_price <= 0
                or execution_price <= 0
                or recent_returns.notna().sum() < 60
                or recent_volumes.isna().any()
                or prior_volumes.isna().any()
                or prior_volumes.mean() <= 0
                or corr_values.notna().sum() < minimum_correlation_observations
            ):
                continue
            recent_volume_mean = float(recent_volumes.mean())
            prior_volume_mean = float(prior_volumes.mean())
            node_features = [
                float(current_price / start_price - 1.0),
                float(recent_returns.std(ddof=1) * np.sqrt(252.0)),
                float(recent_volume_mean / prior_volume_mean - 1.0),
            ]
            if not np.isfinite(node_features).all():
                continue
            stock_return = float(end_price / execution_price - 1.0)
            if not np.isfinite(stock_return):
                continue
            valid_tickers.append(str(ticker))
            valid_indices.append(ticker_index)
            features.append(node_features)
            stock_forward_returns.append(stock_return)
            sector: Optional[str] = None
            if sectors is not None and ticker in sectors.columns and feature_date in sectors.index:
                raw_sector = sectors.at[feature_date, ticker]
                if pd.notna(raw_sector) and str(raw_sector).strip():
                    sector = str(raw_sector).strip()
            snapshot_sectors.append(sector)

        if len(valid_tickers) < 2:
            continue
        return_window = trailing_returns.iloc[:, valid_indices]
        correlations = return_window.corr(min_periods=minimum_correlation_observations).to_numpy()
        adjacency = correlation_adjacency(correlations, edges_per_node)
        labels = (np.asarray(stock_forward_returns) > spy_return).astype(np.float32)
        snapshots.append(
            Snapshot(
                date=snapshot_date,
                target_end=target_end,
                tickers=tuple(valid_tickers),
                features=np.asarray(features, dtype=np.float32),
                adjacency=adjacency,
                labels=labels,
                stock_returns=np.asarray(stock_forward_returns, dtype=np.float32),
                spy_return=spy_return,
                qqq_return=qqq_return,
                sectors=tuple(snapshot_sectors),
            )
        )
    if not snapshots:
        raise ValueError(
            "No monthly snapshots could be built. Check data history, missing values, "
            "and benchmark date coverage."
        )
    return snapshots


def correlation_adjacency(correlations: np.ndarray, edges_per_node: int) -> np.ndarray:
    """Keep positive top-k correlations, symmetrize, then add self-loops."""
    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("Correlation matrix must be square")
    node_count = correlations.shape[0]
    adjacency = np.zeros_like(correlations, dtype=np.float32)
    for node in range(node_count):
        candidates = [
            other
            for other in range(node_count)
            if other != node
            and np.isfinite(correlations[node, other])
            and correlations[node, other] > 0
        ]
        candidates.sort(key=lambda other: correlations[node, other], reverse=True)
        for other in candidates[:edges_per_node]:
            weight = float(correlations[node, other])
            adjacency[node, other] = max(adjacency[node, other], weight)
            adjacency[other, node] = max(adjacency[other, node], weight)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.power(degree, -0.5, where=degree > 0)
    normalized = inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    return normalized.astype(np.float32)


def chronological_split(
    snapshots: Sequence[Snapshot],
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> Tuple[List[Snapshot], List[Snapshot], List[Snapshot]]:
    """Split dates chronologically and purge labels crossing split boundaries."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.date)
    if len(ordered) < 10:
        raise ValueError(
            "At least 10 labeled monthly snapshots are required for chronological splits."
        )
    val_index = int(len(ordered) * train_fraction)
    test_index = int(len(ordered) * (train_fraction + validation_fraction))
    val_start, test_start = ordered[val_index].date, ordered[test_index].date
    train = [item for item in ordered[:val_index] if item.target_end < val_start]
    validation = [
        item
        for item in ordered[val_index:test_index]
        if item.date >= val_start and item.target_end < test_start
    ]
    test = [item for item in ordered[test_index:] if item.date >= test_start]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "A chronological split is empty after purging overlapping label windows; "
            "provide more history or adjust the split fractions."
        )
    return train, validation, test


def standardization_parameters(snapshots: Sequence[Snapshot]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([snapshot.features for snapshot in snapshots], axis=0)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def normalize_features(
    features: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    return ((features - mean) / standard_deviation).astype(np.float32)


def _tensor_snapshot(
    snapshot: Snapshot, mean: np.ndarray, standard_deviation: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = normalize_features(snapshot.features, mean, standard_deviation)
    return (
        torch.from_numpy(features),
        torch.from_numpy(snapshot.adjacency),
        torch.from_numpy(snapshot.labels),
    )


def _evaluate_loss(
    model: nn.Module,
    snapshots: Sequence[Snapshot],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> float:
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()
    losses: List[float] = []
    with torch.no_grad():
        for snapshot in snapshots:
            features, adjacency, labels = _tensor_snapshot(snapshot, mean, standard_deviation)
            logits = model(features, adjacency)
            losses.append(float(loss_function(logits, labels).item()))
    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train: Sequence[Snapshot],
    validation: Sequence[Snapshot],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    max_epochs: int = 100,
    patience: int = 12,
    learning_rate: float = 0.01,
) -> nn.Module:
    if max_epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        for snapshot in train:
            features, adjacency, labels = _tensor_snapshot(snapshot, mean, standard_deviation)
            optimizer.zero_grad()
            loss = loss_function(model(features, adjacency), labels)
            loss.backward()
            optimizer.step()

        validation_loss = _evaluate_loss(model, validation, mean, standard_deviation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    return model


def predict_probabilities(
    model: nn.Module,
    snapshots: Sequence[Snapshot],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> List[np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for snapshot in snapshots:
            features, adjacency, _ = _tensor_snapshot(snapshot, mean, standard_deviation)
            predictions.append(torch.sigmoid(model(features, adjacency)).cpu().numpy())
    return predictions


def portfolio_weights(
    tickers: Sequence[str],
    probabilities: np.ndarray,
    sectors: Sequence[Optional[str]],
    top_k: int = 10,
    max_position: float = 0.10,
    max_sector: float = 0.25,
) -> Dict[str, float]:
    """Select top scores and apply position/sector caps; unused weight is cash."""
    if len(tickers) != len(probabilities) or len(tickers) != len(sectors):
        raise ValueError("Tickers, scores, and sectors must have matching lengths")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Tickers must be unique within a portfolio")
    if top_k < 1 or not 0 < max_position <= 1 or not 0 < max_sector <= 1:
        raise ValueError("Portfolio settings are outside valid ranges")
    if not np.isfinite(probabilities).all():
        raise ValueError("Portfolio scores must all be finite")
    selected = np.argsort(-probabilities, kind="stable")[:top_k]
    target_weight = 1.0 / len(selected)
    weights: Dict[str, float] = {}
    sector_totals: Dict[str, float] = {}
    for index in selected:
        ticker = tickers[int(index)]
        sector = sectors[int(index)]
        sector_key = sector if sector is not None else "__unknown__"
        sector_room = max_sector - sector_totals.get(sector_key, 0.0)
        weight = max(0.0, min(target_weight, max_position, sector_room))
        if weight > 0:
            weights[ticker] = weight
            sector_totals[sector_key] = sector_totals.get(sector_key, 0.0) + weight
    return weights


def _portfolio_cost(
    previous_weights: Dict[str, float], new_weights: Dict[str, float], cost_rate: float
) -> Tuple[float, float]:
    symbols = set(previous_weights) | set(new_weights)
    turnover = sum(
        abs(new_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in symbols
    )
    return turnover, turnover * cost_rate


def portfolio_backtest(
    snapshots: Sequence[Snapshot],
    predictions: Sequence[np.ndarray],
    strategy: str,
    top_k: int = 10,
    max_position: float = 0.10,
    max_sector: float = 0.25,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Backtest one monthly strategy with turnover-based transaction costs."""
    if len(snapshots) != len(predictions):
        raise ValueError("Each test snapshot must have one prediction array")
    if not 0 <= cost_rate < 1:
        raise ValueError("Cost rate must be between zero and one")

    previous_weights: Dict[str, float] = {}
    records: List[dict] = []
    for snapshot, scores in zip(snapshots, predictions):
        if strategy == "equal_weight":
            weights = {
                ticker: 1.0 / len(snapshot.tickers)
                for ticker in snapshot.tickers
            }
        elif strategy in {"gnn", "no_graph"}:
            weights = portfolio_weights(
                snapshot.tickers,
                scores,
                snapshot.sectors,
                top_k=top_k,
                max_position=max_position,
                max_sector=max_sector,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return_by_ticker = dict(zip(snapshot.tickers, snapshot.stock_returns))
        gross_return = sum(
            weight * float(return_by_ticker[ticker])
            for ticker, weight in weights.items()
        )
        turnover, transaction_cost = _portfolio_cost(
            previous_weights, weights, cost_rate
        )
        records.append(
            {
                "date": snapshot.date,
                "target_end": snapshot.target_end,
                "strategy": strategy,
                "gross_return": gross_return,
                "net_return": gross_return - transaction_cost,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "holdings": len(weights),
                "cash_weight": max(0.0, 1.0 - sum(weights.values())),
            }
        )
        previous_weights = weights
    return pd.DataFrame(records)


def benchmark_backtest(snapshots: Sequence[Snapshot]) -> pd.DataFrame:
    records: List[dict] = []
    for snapshot in snapshots:
        records.append(
            {
                "date": snapshot.date,
                "target_end": snapshot.target_end,
                "strategy": "SPY",
                "gross_return": snapshot.spy_return,
                "net_return": snapshot.spy_return,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "holdings": 1,
                "cash_weight": 0.0,
            }
        )
        if snapshot.qqq_return is not None:
            records.append(
                {
                    "date": snapshot.date,
                    "target_end": snapshot.target_end,
                    "strategy": "QQQ",
                    "gross_return": snapshot.qqq_return,
                    "net_return": snapshot.qqq_return,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "holdings": 1,
                    "cash_weight": 0.0,
                }
            )
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
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(12.0))
            if len(returns) > 1 and np.std(returns, ddof=1) > 0
            else 0.0
        )
        summaries.append(
            {
                "strategy": strategy,
                "periods": len(returns),
                "cumulative_return": float(equity[-1] - 1.0),
                "annualized_return": annualized_return,
                "annualized_volatility": volatility,
                "sharpe_rf_0": sharpe,
                "max_drawdown": float(drawdowns.min()),
                "average_turnover": float(group["turnover"].mean()),
                "total_transaction_cost": float(group["transaction_cost"].sum()),
                "average_holdings": float(group["holdings"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def plot_backtest_results(backtest: pd.DataFrame, output_path: Path) -> None:
    """Save compounded snapshot returns and drawdowns as a two-panel chart."""
    required = {"date", "strategy", "net_return"}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(f"Backtest data is missing columns: {sorted(missing)}")
    if backtest.empty:
        raise ValueError("Cannot plot empty backtest results")

    chart_data = backtest.copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"], errors="raise")
    chart_data["net_return"] = pd.to_numeric(chart_data["net_return"], errors="raise")
    if not np.isfinite(chart_data["net_return"]).all() or (
        chart_data["net_return"] <= -1
    ).any():
        raise ValueError("Backtest returns must be finite and greater than -100%")

    figure, (equity_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for strategy, group in chart_data.groupby("strategy", sort=True):
        group = group.sort_values("date")
        returns = group["net_return"].to_numpy(dtype=float)
        equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
        dates = [group["date"].iloc[0] - pd.Timedelta(days=1)] + group["date"].tolist()
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        equity_axis.plot(dates, (equity - 1.0) * 100.0, label=strategy, linewidth=1.8)
        drawdown_axis.plot(dates, drawdown * 100.0, label=strategy, linewidth=1.4)

    equity_axis.set_title("Out-of-sample strategy comparison")
    equity_axis.set_ylabel("Compounded return (%)")
    equity_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    equity_axis.grid(True, alpha=0.25)
    equity_axis.legend(ncol=3, frameon=False)
    drawdown_axis.set_title("Drawdown")
    drawdown_axis.set_ylabel("From prior peak (%)")
    drawdown_axis.set_xlabel("Rebalance date")
    drawdown_axis.grid(True, alpha=0.25)
    drawdown_axis.legend(ncol=3, frameon=False)
    figure.text(
        0.5,
        0.01,
        "Caution: these values compound 21-session forward returns at monthly snapshots; "
        "windows may overlap or leave gaps and are not a contiguous investable return series.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def ranking_summary(
    snapshots: Sequence[Snapshot],
    predictions: Sequence[np.ndarray],
    top_k: int,
    model_name: str,
) -> dict:
    labels: List[float] = []
    scores: List[float] = []
    precision_values: List[float] = []
    selected_excess_returns: List[float] = []
    for snapshot, probabilities in zip(snapshots, predictions):
        labels.extend(snapshot.labels.tolist())
        scores.extend(probabilities.tolist())
        selected = np.argsort(-probabilities, kind="stable")[:top_k]
        precision_values.append(float(snapshot.labels[selected].mean()))
        selected_excess_returns.append(
            float(np.mean(snapshot.stock_returns[selected] - snapshot.spy_return))
        )

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    positive = score_array[label_array == 1]
    negative = score_array[label_array == 0]
    auc = (
        float(np.mean(positive[:, None] > negative[None, :])
              + 0.5 * np.mean(positive[:, None] == negative[None, :]))
        if len(positive) and len(negative)
        else float("nan")
    )
    return {
        "model": model_name,
        "out_of_sample_auc": auc,
        "precision_at_top_k": float(np.mean(precision_values)),
        "mean_selected_excess_return": float(np.mean(selected_excess_returns)),
        "test_date_count": len(snapshots),
    }


def _make_model(kind: str, feature_count: int) -> nn.Module:
    if kind == "gnn":
        return StockGNN(feature_count)
    if kind == "no_graph":
        return NodeMLP(feature_count)
    raise ValueError(f"Unknown model kind: {kind}")


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prices, volumes, benchmark_prices, sectors = load_price_cache(
        args.data_dir, args.benchmark
    )
    if len(prices.columns) < 50:
        warnings.warn(
            f"Only {len(prices.columns)} stocks are cached. This run is useful as a "
            "pipeline smoke test, not as evidence of strategy performance.",
            RuntimeWarning,
        )
    if sectors.isna().any().any():
        warnings.warn(
            "Some sector classifications are missing. Those stocks are grouped into a shared "
            "Unknown sector bucket; provide historical sector classifications for reliable "
            "sector diversification.",
            RuntimeWarning,
        )
    snapshots = build_snapshots(
        prices=prices,
        volumes=volumes,
        benchmark_prices=benchmark_prices,
        benchmark_ticker=args.benchmark,
        correlation_window=args.correlation_window,
        edges_per_node=args.edges_per_node,
        sectors=sectors,
    )
    train, validation, test = chronological_split(
        snapshots, args.train_fraction, args.validation_fraction
    )
    mean, standard_deviation = standardization_parameters(train)

    prediction_sets: Dict[str, List[np.ndarray]] = {}
    for model_kind in ("gnn", "no_graph"):
        model = train_model(
            _make_model(model_kind, len(FEATURE_NAMES)),
            train,
            validation,
            mean,
            standard_deviation,
            max_epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
        )
        prediction_sets[model_kind] = predict_probabilities(
            model, test, mean, standard_deviation
        )
        checkpoint = {
            "model_kind": model_kind,
            "model_state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": mean,
            "feature_std": standard_deviation,
            "seed": args.seed,
            "train_cutoff": str(max(item.date for item in train).date()),
        }
        torch.save(checkpoint, args.output_dir / f"{model_kind}.pt")

    ranking_metrics = pd.DataFrame(
        [
            ranking_summary(test, prediction_sets[kind], args.top_k, kind)
            for kind in ("gnn", "no_graph")
        ]
    )
    ranking_metrics.to_csv(args.output_dir / "ranking_metrics.csv", index=False)

    portfolio_frames = [
        portfolio_backtest(
            test,
            prediction_sets["gnn"],
            "gnn",
            top_k=args.top_k,
            max_position=args.max_position,
            max_sector=args.max_sector,
            cost_rate=args.cost_rate,
        ),
        portfolio_backtest(
            test,
            prediction_sets["no_graph"],
            "no_graph",
            top_k=args.top_k,
            max_position=args.max_position,
            max_sector=args.max_sector,
            cost_rate=args.cost_rate,
        ),
        portfolio_backtest(
            test,
            [np.zeros(len(item.tickers), dtype=np.float32) for item in test],
            "equal_weight",
            cost_rate=args.cost_rate,
        ),
        benchmark_backtest(test),
    ]
    results = pd.concat(portfolio_frames, ignore_index=True).sort_values(
        ["date", "strategy"]
    )
    summary = performance_summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "out_of_sample_returns.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_backtest_results(results, args.output_dir / "performance.png")

    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    config.update(
        {
            "feature_names": list(FEATURE_NAMES),
            "label": "stock next-21-session return > benchmark next-21-session return",
            "rf_assumption": "0%",
            "snapshot_count": len(snapshots),
            "train_snapshot_count": len(train),
            "validation_snapshot_count": len(validation),
            "test_snapshot_count": len(test),
            "test_start": str(min(item.date for item in test).date()),
            "test_end": str(max(item.date for item in test).date()),
        }
    )
    with (args.output_dir / "experiment.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)
    if "QQQ" not in benchmark_prices.columns:
        warnings.warn(
            "QQQ is not present in the cache. QQQ benchmark results will be omitted.",
            RuntimeWarning,
        )
    return results, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a graph model on cached stock prices."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/prices"))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gnn"))
    parser.add_argument("--correlation-window", type=int, default=60)
    parser.add_argument("--edges-per-node", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-position", type=float, default=0.10)
    parser.add_argument("--max-sector", type=float, default=0.25)
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.001,
        help="Transaction cost per dollar traded (0.001 means 10 bps, one-way).",
    )
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    args.benchmark = args.benchmark.strip().upper()
    if args.correlation_window < 60:
        parser.error("--correlation-window must be at least 60")
    if args.edges_per_node < 1 or args.top_k < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("--edges-per-node, --top-k, --epochs, and --patience must be positive")
    if not 0 < args.max_position <= 1 or not 0 < args.max_sector <= 1:
        parser.error("--max-position and --max-sector must be in (0, 1]")
    if not 0 <= args.cost_rate < 1:
        parser.error("--cost-rate must be in [0, 1)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, summary = run_experiment(args)
    print(summary.to_string(index=False))
    print(f"\nSaved out-of-sample results to {args.output_dir}")


if __name__ == "__main__":
    main()