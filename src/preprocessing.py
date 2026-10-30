from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import FEATURE_CONFIG, REQUIRED_COLUMNS


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
    
    all_prices = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    all_volumes = panel.pivot(index="date", columns="ticker", values="volume").sort_index()
    
    # Optional columns check
    optional_panels = {}
    for col in ["sector", "beta", "pe_ratio"]:
        if col in panel.columns:
            optional_panels[col] = panel.pivot(index="date", columns="ticker", values=col).sort_index()
            
    stock_columns = [col for col in all_prices.columns if col != benchmark_ticker]
    
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
    inverse_sqrt_degree = np.power(degree, -0.5, where=degree > 0)
    normalized = inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    return normalized.astype(np.float32)

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
    
    return train, validation, test

def build_snapshots(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    optional_panels: Dict[str, pd.DataFrame] = None,
    benchmark_ticker: str = "SPY",
    qqq_ticker: str = "QQQ",
    correlation_window: int = 60,
    edges_per_node: int = 5,
    minimum_correlation_observations: int = 40,
    horizon: int = 21,
    sectors: Optional[pd.DataFrame] = None,
) -> List[Snapshot]:
    daily_returns = prices.pct_change(fill_method=None)
    benchmark_returns = benchmark_prices[benchmark_ticker].pct_change(fill_method=None)
    dates = prices.index
    monthly_indices = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last()
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
        
        features, valid_tickers, valid_indices, stock_forward_returns, snapshot_sectors = [], [], [], [], []
        for ticker_index, ticker in enumerate(prices.columns):
            close, volume = prices[ticker], volumes[ticker]
            start_price = close.iloc[feature_index - horizon]
            current_price = close.iloc[feature_index]
            execution_price = close.iloc[index]
            end_price = close.iloc[end_index]
            recent_returns = daily_returns[ticker].iloc[feature_index - 62 : feature_index + 1]
            recent_volumes = volume.iloc[feature_index - 19 : feature_index + 1]
            prior_volumes = volume.iloc[feature_index - 79 : feature_index - 19]
            corr_values = trailing_returns[ticker]
            
            if (
                pd.isna(start_price) or pd.isna(current_price) or pd.isna(execution_price) or pd.isna(end_price)
                or start_price <= 0 or current_price <= 0 or execution_price <= 0
                or recent_returns.notna().sum() < 60 or recent_volumes.isna().any()
                or prior_volumes.isna().any() or prior_volumes.mean() <= 0
                or corr_values.notna().sum() < minimum_correlation_observations
            ):
                continue
                
            # Compute base metrics
            ret_21d = float(current_price / start_price - 1.0)
            vol_63d = float(recent_returns.std(ddof=1) * np.sqrt(252.0))
            vol_change = float(recent_volumes.mean() / prior_volumes.mean() - 1.0)
            
            # Compute or lookup Beta dynamically if enabled and not in panel
            beta_val = 1.0
            if FEATURE_CONFIG["beta"]["enabled"]:
                if "beta" in optional_panels and ticker in optional_panels["beta"].columns and feature_date in optional_panels["beta"].index:
                    beta_val = float(optional_panels["beta"].at[feature_date, ticker])
                else:
                    # Fallback: calculate rolling beta against SPY
                    m_ret = benchmark_returns.iloc[feature_index - 62 : feature_index + 1]
                    valid_pair = pd.concat([recent_returns, m_ret], axis=1).dropna()
                    if len(valid_pair) > 30:
                        cov = np.cov(valid_pair.iloc[:, 0], valid_pair.iloc[:, 1])[0, 1]
                        m_var = valid_pair.iloc[:, 1].var()
                        beta_val = float(cov / m_var) if m_var > 0 else 1.0

            # Lookup P/E ratio if enabled and present in data
            pe_val = 0.0
            if FEATURE_CONFIG["pe_ratio"]["enabled"]:
                if "pe_ratio" in optional_panels and ticker in optional_panels["pe_ratio"].columns and feature_date in optional_panels["pe_ratio"].index:
                    pe_val = float(optional_panels["pe_ratio"].at[feature_date, ticker])

            # Assemble node feature vector based on active config and weights
            node_features = []
            if FEATURE_CONFIG["return_21d"]["enabled"]:
                node_features.append(ret_21d * FEATURE_CONFIG["return_21d"]["weight"])
            if FEATURE_CONFIG["volatility_63d"]["enabled"]:
                node_features.append(vol_63d * FEATURE_CONFIG["volatility_63d"]["weight"])
            if FEATURE_CONFIG["volume_change"]["enabled"]:
                node_features.append(vol_change * FEATURE_CONFIG["volume_change"]["weight"])
            if FEATURE_CONFIG["beta"]["enabled"]:
                node_features.append(beta_val * FEATURE_CONFIG["beta"]["weight"])
            if FEATURE_CONFIG["pe_ratio"]["enabled"]:
                node_features.append(pe_val * FEATURE_CONFIG["pe_ratio"]["weight"])

            node_features_arr = np.asarray(node_features, dtype=np.float32)
            if not np.isfinite(node_features_arr).all():
                continue
                
            stock_return = float(end_price / execution_price - 1.0)
            if not np.isfinite(stock_return):
                continue
                
            valid_tickers.append(str(ticker))
            valid_indices.append(ticker_index)
            features.append(node_features_arr)
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
                date=snapshot_date, target_end=target_end, tickers=tuple(valid_tickers),
                features=np.asarray(features, dtype=np.float32), adjacency=adjacency,
                labels=labels, stock_returns=np.asarray(stock_forward_returns, dtype=np.float32),
                spy_return=spy_return, qqq_return=qqq_return, sectors=tuple(snapshot_sectors)
            )
        )
    return snapshots