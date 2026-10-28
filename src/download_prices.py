"""Download yfinance price history into a per-ticker CSV cache."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

CACHE_COLUMNS = ["date", "ticker", "adj_close", "volume"]
TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")


def load_tickers(tickers_file: Path) -> list[str]:
    """Read one ticker per line or a comma-separated ticker list."""
    if not tickers_file.is_file():
        raise FileNotFoundError(f"Ticker list does not exist: {tickers_file}")

    tickers: list[str] = []
    with tickers_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
        header = [value.strip().upper() for value in rows[0]] if rows else []
        ticker_column = next(
            (index for index, value in enumerate(header) if value in {"TICKER", "SYMBOL"}),
            None,
        )
        for row in rows[1:] if ticker_column is not None else rows:
            if not row:
                continue
            values = [row[ticker_column]] if ticker_column is not None and len(row) > ticker_column else (
                [] if ticker_column is not None else row
            )
            for value in values:
                ticker = value.strip().upper()
                if not ticker or ticker.startswith("#"):
                    continue
                if not TICKER_PATTERN.fullmatch(ticker):
                    raise ValueError(f"Invalid ticker symbol in {tickers_file}: {value!r}")
                if ticker not in tickers:
                    tickers.append(ticker)

    if not tickers:
        raise ValueError(f"No ticker symbols found in {tickers_file}")
    return tickers


def _price_field(data: pd.DataFrame, field: str) -> pd.Series:
    """Extract a field from yfinance's flat or multi-level column formats."""
    matching_columns = [
        column
        for column in data.columns
        if any(str(part).casefold() == field.casefold()
               for part in (column if isinstance(column, tuple) else (column,)))
    ]
    if len(matching_columns) != 1:
        raise ValueError(
            f"Expected exactly one {field!r} column from yfinance; "
            f"received columns {list(data.columns)!r}"
        )

    values = data[matching_columns[0]]
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"Expected one column for {field!r}, received {values.shape[1]}")
        values = values.iloc[:, 0]
    return values


def _normalize_download(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"yfinance returned no price history for {ticker}")

    close = _price_field(data, "Close")
    volume = _price_field(data, "Volume")
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(data.index).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "adj_close": pd.to_numeric(close, errors="coerce").to_numpy(),
            "volume": pd.to_numeric(volume, errors="coerce").to_numpy(),
        }
    )
    if result[["adj_close", "volume"]].isna().any().any():
        raise ValueError(f"Downloaded history for {ticker} contains missing or non-numeric values")
    if (result["adj_close"] <= 0).any() or (result["volume"] < 0).any():
        raise ValueError(f"Downloaded history for {ticker} contains invalid prices or volume")
    if result["date"].duplicated().any():
        raise ValueError(f"Downloaded history for {ticker} contains duplicate dates")
    return result[CACHE_COLUMNS]


def cache_tickers(
    tickers: list[str],
    cache_dir: Path,
    start: str,
    end: Optional[str] = None,
    refresh: bool = False,
) -> None:
    """Cache adjusted daily closes and volume, downloading only missing tickers."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None
    if end_date is not None and end_date <= start_date:
        raise ValueError("The end date must be later than the start date")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        if cache_path.exists() and not refresh:
            cached = pd.read_csv(cache_path)
            if (
                cached.empty
                or not set(CACHE_COLUMNS).issubset(cached.columns)
                or not cached["ticker"].eq(ticker).all()
            ):
                raise ValueError(
                    f"Cache file is empty or has unexpected schema/content: {cache_path}. "
                    "Use --refresh to replace it."
                )
            print(f"Using cache: {cache_path}")
            continue

        print(f"Downloading {ticker}...")
        downloaded = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        normalized = _normalize_download(downloaded, ticker)
        temporary_path = cache_path.with_suffix(".csv.tmp")
        try:
            normalized.to_csv(temporary_path, index=False)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"Saved cache: {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download adjusted daily prices and volume into a local CSV cache."
    )
    parser.add_argument("tickers_file", type=Path, help="Text/CSV file containing ticker symbols")
    parser.add_argument("--start", default="2018-01-01", help="First date to download (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        help="Exclusive end date (YYYY-MM-DD); defaults to the latest available data",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/prices"))
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker to include in the cache (default: SPY)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download all tickers, replacing their cached CSV files",
    )
    args = parser.parse_args()

    tickers = load_tickers(args.tickers_file)
    benchmark = args.benchmark.strip().upper()
    if benchmark and not TICKER_PATTERN.fullmatch(benchmark):
        parser.error(f"Invalid benchmark ticker: {args.benchmark!r}")
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)

    cache_tickers(tickers, args.cache_dir, args.start, args.end, args.refresh)


if __name__ == "__main__":
    main()
