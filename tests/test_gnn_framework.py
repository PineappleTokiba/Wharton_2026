import unittest

import numpy as np
import pandas as pd
import torch

from src.preprocessing import Snapshot, build_snapshots, chronological_split, correlation_adjacency
from src.training import StockGNN, standardization_parameters, train_model
from src.analysis import portfolio_backtest, portfolio_weights


def synthetic_panels(days=900):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-02", periods=days)
    market = rng.normal(0.0003, 0.01, size=days)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    prices = {}
    volumes = {}
    for index, ticker in enumerate(tickers):
        returns = 0.7 * market + rng.normal(0.0002 * index, 0.012, size=days)
        prices[ticker] = 50 * np.cumprod(1 + returns)
        volumes[ticker] = rng.integers(100_000, 1_000_000, size=days)
    prices["SPY"] = 400 * np.cumprod(1 + market)
    prices["QQQ"] = 300 * np.cumprod(1 + market + rng.normal(0, 0.003, size=days))
    price_panel = pd.DataFrame(prices, index=dates)
    volume_panel = pd.DataFrame(volumes, index=dates)
    benchmark_panel = price_panel[["SPY", "QQQ"]]
    return price_panel[tickers], volume_panel[tickers], benchmark_panel


class GNNFrameworkTests(unittest.TestCase):
    def test_graph_is_symmetric_weighted_and_has_self_loops(self):
        correlations = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 1.0, 0.6],
                [0.1, 0.6, 1.0],
            ]
        )

        adjacency = correlation_adjacency(correlations, edges_per_node=1)

        np.testing.assert_allclose(adjacency, adjacency.T)
        self.assertTrue(np.all(np.diag(adjacency) > 0))
        self.assertGreater(adjacency[0, 1], 0)
        self.assertGreater(adjacency[1, 2], 0)
        self.assertEqual(adjacency[0, 2], 0)

    def test_snapshots_use_only_pre_rebalance_feature_data(self):
        prices, volumes, benchmarks = synthetic_panels(days=900)
        initial = build_snapshots(prices, volumes, benchmarks)
        target_date = initial[10].date

        changed_prices = prices.copy()
        execution_index = changed_prices.index.get_loc(target_date)
        changed_prices.iloc[execution_index, 0] *= 3
        changed = build_snapshots(changed_prices, volumes, benchmarks)

        original_snapshot = next(item for item in initial if item.date == target_date)
        changed_snapshot = next(item for item in changed if item.date == target_date)
        np.testing.assert_allclose(original_snapshot.features, changed_snapshot.features)
        self.assertNotEqual(
            float(original_snapshot.stock_returns[0]),
            float(changed_snapshot.stock_returns[0]),
        )

    def test_chronological_split_purges_forward_labels(self):
        prices, volumes, benchmarks = synthetic_panels(days=900)
        snapshots = build_snapshots(prices, volumes, benchmarks)

        train, validation, test = chronological_split(snapshots)

        self.assertLess(max(item.target_end for item in train), min(item.date for item in validation))
        self.assertLess(
            max(item.target_end for item in validation),
            min(item.date for item in test),
        )

    def test_portfolio_respects_position_sector_caps_and_cash(self):
        weights = portfolio_weights(
            ["A", "B", "C", "D"],
            np.array([0.9, 0.8, 0.7, 0.6]),
            ["Technology", "Technology", "Health", "Health"],
            top_k=4,
            max_position=0.2,
            max_sector=0.5,
        )

        self.assertTrue(all(weight <= 0.2 for weight in weights.values()))
        self.assertLessEqual(weights["A"] + weights["B"], 0.5)
        self.assertLessEqual(weights["C"] + weights["D"], 0.5)
        self.assertLess(sum(weights.values()), 1.0)

    def test_portfolio_backtest_charges_cost_on_turnover(self):
        snapshot = Snapshot(
            date=pd.Timestamp("2024-01-31"),
            target_end=pd.Timestamp("2024-02-29"),
            tickers=("A", "B"),
            features=np.zeros((2, 3), dtype=np.float32),
            adjacency=np.eye(2, dtype=np.float32),
            labels=np.array([1, 0], dtype=np.float32),
            stock_returns=np.array([0.1, 0.0], dtype=np.float32),
            spy_return=0.02,
            qqq_return=None,
            sectors=(None, None),
        )

        result = portfolio_backtest(
            [snapshot],
            [np.array([0.9, 0.1])],
            "gnn",
            top_k=2,
            max_position=0.5,
            max_sector=0.5,
            cost_rate=0.001,
        )

        self.assertAlmostEqual(result.loc[0, "turnover"], 1.0)
        self.assertAlmostEqual(result.loc[0, "transaction_cost"], 0.001)
        self.assertAlmostEqual(result.loc[0, "net_return"], 0.049)

    def test_gnn_trains_on_synthetic_snapshots(self):
        snapshots = []
        for index in range(12):
            features = np.array(
                [[index / 10, 0.2, 0.1], [0.1, index / 12, 0.2]],
                dtype=np.float32,
            )
            labels = np.array([index % 2, (index + 1) % 2], dtype=np.float32)
            snapshots.append(
                Snapshot(
                    date=pd.Timestamp("2020-01-01") + pd.offsets.MonthEnd(index),
                    target_end=pd.Timestamp("2020-01-01") + pd.offsets.MonthEnd(index + 1),
                    tickers=("A", "B"),
                    features=features,
                    adjacency=np.eye(2, dtype=np.float32),
                    labels=labels,
                    stock_returns=np.array([0.02, 0.01], dtype=np.float32),
                    spy_return=0.0,
                    qqq_return=None,
                    sectors=(None, None),
                )
            )
        mean, standard_deviation = standardization_parameters(snapshots[:8])
        model = StockGNN(3)

        trained = train_model(
            model,
            snapshots[:8],
            snapshots[8:],
            mean,
            standard_deviation,
            max_epochs=2,
            patience=2,
        )

        self.assertIsInstance(trained, StockGNN)
        with torch.no_grad():
            logits = trained(
                torch.from_numpy(snapshots[0].features),
                torch.from_numpy(snapshots[0].adjacency),
            )
        self.assertEqual(tuple(logits.shape), (2,))


if __name__ == "__main__":
    unittest.main()
