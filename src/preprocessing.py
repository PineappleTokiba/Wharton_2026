"""Weekly preprocessing pipeline for GCN stock prediction with multi-factor fundamental and momentum features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import REQUIRED_COLUMNS, SNAPSHOT_CONFIG, get_profile

WEEKS_PER_YEAR = 52


@dataclass
class Snapshot:
    """One monthly/quarterly graph snapshot and its forward return outcomes."""
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


def load_price_cache(
    cache_dir: Path, benchmark_ticker: str = "SPY"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
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
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    benchmark_ticker = benchmark_ticker.upper()
    if benchmark_ticker not in set(panel["ticker"]):
        raise ValueError(f"Benchmark {benchmark_ticker} is not cached in {cache_dir}")

    all_prices = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    all_volumes = panel.pivot(index="date", columns="ticker", values="volume").sort_index()
    
    # Load optional/feature panels
    optional_panels = {}
    feature_keys = [
        "sector", "return_4w", "return_12w", "return_26w", "return_52w",
        "ma_10w_ratio", "ma_40w_ratio", "volatility_12w",
        "roe", "debt_to_equity", "revenue_growth", "profit_margin"
    ]
    for col in feature_keys:
        if col in panel.columns:
            optional_panels[col] = panel.pivot(index="date", columns="ticker", values=col).sort_index()
            
    stock_columns = [col for col in all_prices.columns if col not in {benchmark_ticker, "QQQ"}]
    sectors = optional_panels.get("sector", pd.DataFrame(index=all_prices.index)).reindex(columns=stock_columns)
    
    return (
        all_prices[stock_columns], 
        all_volumes[stock_columns], 
        all_prices, 
        sectors,
        optional_panels
    )


def correlation_adjacency(correlations: np.ndarray, edges_per_node: int) -> np.ndarray:
    node_count = correlations.shape[0]
    adjacency = np.zeros_like(correlations, dtype=np.float32)
    for node in range(node_count):
        candidates = [
            other for other in range(node_count)
            if other != node and np.isfinite(correlations[node, other]) and correlations[node, other] > 0
        ]
        candidates.sort(key=lambda other: correlations[node, other], reverse=True)
        for other in candidates[:edges_per_node]:
            weight = float(correlations[node, other])
            adjacency[node, other] = max(adjacency[node, other], weight)
            adjacency[other, node] = max(adjacency[other, node], weight)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.power(degree, -0.5, where=degree > 0, out=np.zeros_like(degree))
    normalized = inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    return normalized.astype(np.float32)


def path_risk(path: np.ndarray) -> Tuple[float, float]:
    """Annualized volatility and max drawdown (as a positive fraction) of a weekly value path."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3 or (path <= 0).any():
        return float("nan"), float("nan")
    volatility = float(np.std(path[1:] / path[:-1] - 1.0, ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    drawdown = float((1.0 - path / np.maximum.accumulate(path)).max())
    return volatility, drawdown


def enabled_features(profile: dict) -> List[str]:
    return [name for name, settings in profile["features"].items() if settings["enabled"]]


def feature_weights(profile: dict) -> np.ndarray:
    return np.array([profile["features"][name]["weight"] for name in enabled_features(profile)], dtype=np.float32)


def make_labels(
    target: dict, stock_returns: np.ndarray, benchmark_return: float,
    forward_volatility: np.ndarray, forward_drawdown: np.ndarray,
) -> np.ndarray:
    """Turn a snapshot's forward outcomes into the binary target a profile trains on."""
    if target["type"] == "beat_benchmark":
        labels = stock_returns > benchmark_return
    elif target["type"] == "steady_growth":
        labels = (
            (stock_returns >= target["min_return"])
            & (forward_volatility <= np.nanquantile(forward_volatility, target["max_volatility_quantile"]))
            & (forward_drawdown <= np.nanquantile(forward_drawdown, target["max_drawdown_quantile"]))
        )
    else:
        raise ValueError(f"Unknown target type: {target['type']!r}")
    return labels.astype(np.float32)


def chronological_split(
    snapshots: Sequence[Snapshot],
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> Tuple[List[Snapshot], List[Snapshot], List[Snapshot]]:
    """Splits snapshots chronologically into train, validation, and test sets,
    purging overlapping forward label horizons to prevent data leakage.
    """
    ordered = sorted(snapshots, key=lambda s: s.date)
    val_index = int(len(ordered) * train_fraction)
    test_index = int(len(ordered) * (train_fraction + validation_fraction))
    
    if val_index >= len(ordered) or test_index >= len(ordered):
        raise ValueError("Not enough snapshots to perform chronological split with current fractions.")
        
    val_start, test_start = ordered[val_index].date, ordered[test_index].date
    
    train = [item for item in ordered[:val_index] if item.target_end < val_start]
    validation = [item for item in ordered[val_index:test_index] if item.date >= val_start and item.target_end < test_start]
    test = [item for item in ordered[test_index:] if item.date >= test_start]

    if not train or not validation or not test:
        raise ValueError(
            f"A split is empty after purging overlapping labels (train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}, from {len(ordered)} snapshots). "
            "Add history, shorten the horizon, or adjust the split fractions."
        )
    return train, validation, test


def build_snapshots(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    optional_panels: Dict[str, pd.DataFrame] = None,
    benchmark_ticker: str = "SPY",
    qqq_ticker: str = "QQQ",
    correlation_window: int = SNAPSHOT_CONFIG["correlation_window"],
    edges_per_node: int = SNAPSHOT_CONFIG["edges_per_node"],
    minimum_correlation_observations: int = 80,
    horizon: int = SNAPSHOT_CONFIG["horizon"],
    sectors: Optional[pd.DataFrame] = None,
    profile: Optional[dict] = None,
) -> List[Snapshot]:
    """Build monthly graphs. `profile` selects features and target (default: DEFAULT_PROFILE).

    Features use data through the close before each rebalance; labels use later closes.
    """
    profile = profile if profile is not None else get_profile()
    active_features = enabled_features(profile)
    feature_scale = feature_weights(profile)
    weekly_returns = prices.pct_change(fill_method=None)
    benchmark_returns = benchmark_prices[benchmark_ticker].pct_change(fill_method=None)
    dates = prices.index
    
    # Create monthly snapshot cadence from weekly rows
    monthly_indices = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last()
    snapshots: List[Snapshot] = []
    
    for raw_index in monthly_indices:
        index = int(raw_index)
        end_index = index + horizon
        feature_index = index - 1

        # Ensure enough history for the correlation window and the forward horizon
        if feature_index < correlation_window or end_index >= len(dates):
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
        trailing_returns = weekly_returns.iloc[trailing_start : feature_index + 1]
        
        features, valid_tickers, valid_indices, stock_forward_returns, snapshot_sectors = [], [], [], [], []
        forward_volatility, forward_drawdown = [], []
        
        for ticker_index, ticker in enumerate(prices.columns):
            close = prices[ticker]
            execution_price = close.iloc[index]
            end_price = close.iloc[end_index]
            corr_values = trailing_returns[ticker]
            
            if (
                pd.isna(execution_price) or pd.isna(end_price) or execution_price <= 0
                or corr_values.notna().sum() < minimum_correlation_observations
            ):
                continue
                
            trailing_volatility, trailing_drawdown = path_risk(
                close.iloc[max(0, feature_index - WEEKS_PER_YEAR) : feature_index + 1].ffill().dropna().to_numpy()
            )
            derived = {"volatility_52w": trailing_volatility, "max_drawdown_52w": trailing_drawdown}

            def get_feat(name: str) -> float:
                """Feature value at the feature date; missing or non-finite values become 0."""
                if name in derived:
                    val = derived[name]
                elif optional_panels and name in optional_panels and ticker in optional_panels[name].columns:
                    val = optional_panels[name].at[feature_date, ticker] if feature_date in optional_panels[name].index else np.nan
                else:
                    val = np.nan
                return float(val) if pd.notna(val) and np.isfinite(float(val)) else 0.0

            node_features_arr = np.array([get_feat(name) for name in active_features], dtype=np.float32) * feature_scale

            stock_return = float(end_price / execution_price - 1.0)
            if not np.isfinite(stock_return):
                continue
                
            valid_tickers.append(str(ticker))
            valid_indices.append(ticker_index)
            features.append(node_features_arr)
            stock_forward_returns.append(stock_return)
            volatility, drawdown = path_risk(close.iloc[index : end_index + 1].ffill().to_numpy())
            forward_volatility.append(volatility)
            forward_drawdown.append(drawdown)
            
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
        labels = make_labels(
            profile["target"], np.asarray(stock_forward_returns), spy_return,
            np.asarray(forward_volatility), np.asarray(forward_drawdown),
        )
        
        snapshots.append(
            Snapshot(
                date=snapshot_date, target_end=target_end, tickers=tuple(valid_tickers),
                features=np.asarray(features, dtype=np.float32), adjacency=adjacency,
                labels=labels, stock_returns=np.asarray(stock_forward_returns, dtype=np.float32),
                spy_return=spy_return, qqq_return=qqq_return, sectors=tuple(snapshot_sectors)
            )
        )
    return snapshots