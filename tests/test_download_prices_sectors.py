import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.download_prices import cache_tickers


class DownloadPriceSectorsTests(unittest.TestCase):
    def test_existing_cache_adds_sector_without_redownloading_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "ticker": "AAPL",
                        "adj_close": 100.0,
                        "volume": 1000,
                    },
                    {
                        "date": "2024-01-03",
                        "ticker": "AAPL",
                        "adj_close": 101.0,
                        "volume": 1200,
                    },
                ]
            ).to_csv(cache_dir / "AAPL.csv", index=False)
            ticker = Mock()
            ticker.get_info.return_value = {"sector": "Technology"}

            with patch("src.download_prices.yf.download") as download:
                with patch("src.download_prices.yf.Ticker", return_value=ticker):
                    cache_tickers(["AAPL"], cache_dir, "2024-01-01")

            download.assert_not_called()
            cached = pd.read_csv(cache_dir / "AAPL.csv")
            self.assertEqual(cached["sector"].tolist(), ["Technology", "Technology"])

    def test_refresh_sectors_updates_metadata_without_price_download(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "ticker": "AAPL",
                        "adj_close": 100.0,
                        "volume": 1000,
                        "sector": "Old sector",
                    },
                ]
            ).to_csv(cache_dir / "AAPL.csv", index=False)
            ticker = Mock()
            ticker.get_info.return_value = {"sector": "Technology"}

            with patch("src.download_prices.yf.download") as download:
                with patch("src.download_prices.yf.Ticker", return_value=ticker):
                    cache_tickers(["AAPL"], cache_dir, "2024-01-01", refresh_sectors=True)

            download.assert_not_called()
            self.assertEqual(
                pd.read_csv(cache_dir / "AAPL.csv").loc[0, "sector"],
                "Technology",
            )

    def test_sector_metadata_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-02",
                        "ticker": "AAPL",
                        "adj_close": 100.0,
                        "volume": 1000,
                    },
                ]
            ).to_csv(cache_dir / "AAPL.csv", index=False)
            ticker = Mock()
            ticker.get_info.side_effect = OSError("network unavailable")

            with patch("src.download_prices.yf.download") as download:
                with patch("src.download_prices.yf.Ticker", return_value=ticker):
                    with self.assertRaisesRegex(RuntimeError, "metadata for AAPL"):
                        cache_tickers(["AAPL"], cache_dir, "2024-01-01")

            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
