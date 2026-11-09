import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import TRAINING_PROFILES, get_profile
from src.gnn_framework import plot_backtest_results
from src.preprocessing import (
    Snapshot,
    build_snapshots,
    chronological_split,
    correlation_adjacency,
    enabled_features,
    feature_weights,
    make_labels,
)
from src.training import StockGNN, normalize_features, standardization_parameters, train_model
from src.analysis import benchmark_backtest, performance_summary, portfolio_backtest, portfolio_weights


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
            sectors=("Technology", "Health Care"),
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

    def test_missing_sectors_share_unknown_sector_cap(self):
        weights = portfolio_weights(
            ["A", "B", "C"],
            np.array([0.9, 0.8, 0.7]),
            [None, None, "Health Care"],
            top_k=3,
            max_position=0.5,
            max_sector=0.3,
        )

        self.assertAlmostEqual(weights["A"], 0.3)
        self.assertNotIn("B", weights)
        self.assertAlmostEqual(weights["C"], 0.3)

    def test_full_sector_passes_its_slot_to_the_next_ranked_stock(self):
        weights = portfolio_weights(
            ["T1", "T2", "T3", "T4", "H1", "H2"],
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            ["Tech", "Tech", "Tech", "Tech", "Health", "Energy"],
            top_k=4,
            max_position=0.25,
            max_sector=0.5,
        )

        self.assertEqual(weights, {"T1": 0.25, "T2": 0.25, "H1": 0.25, "H2": 0.25})

    def test_plot_backtest_results_writes_png_chart(self):
        backtest = pd.DataFrame(
            [
                {"date": "2024-01-31", "strategy": "gnn", "net_return": 0.05},
                {"date": "2024-02-29", "strategy": "gnn", "net_return": -0.02},
                {"date": "2024-01-31", "strategy": "SPY", "net_return": 0.01},
                {"date": "2024-02-29", "strategy": "SPY", "net_return": 0.03},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "performance.png"

            plot_backtest_results(backtest, output_path)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 1000)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

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


def one_snapshot(stock_returns=(0.1, 0.0), date="2024-01-31", end="2025-01-31", spy_return=0.02):
    count = len(stock_returns)
    return Snapshot(
        date=pd.Timestamp(date),
        target_end=pd.Timestamp(end),
        tickers=tuple(f"S{i}" for i in range(count)),
        features=np.zeros((count, 3), dtype=np.float32),
        adjacency=np.eye(count, dtype=np.float32),
        labels=np.zeros(count, dtype=np.float32),
        stock_returns=np.array(stock_returns, dtype=np.float32),
        spy_return=spy_return,
        qqq_return=None,
        sectors=tuple("Tech" for _ in range(count)),
    )


class TrainingProfileTests(unittest.TestCase):
    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            get_profile("moonshot")

    def test_get_profile_returns_an_independent_copy(self):
        get_profile("stability")["features"]["roe"]["weight"] = 99
        self.assertNotEqual(TRAINING_PROFILES["stability"]["features"]["roe"]["weight"], 99)

    def test_no_fundamentals_disables_only_fundamentals(self):
        names = enabled_features(get_profile("stability", include_fundamentals=False))
        self.assertNotIn("roe", names)
        self.assertNotIn("revenue_growth", names)
        self.assertIn("volatility_52w", names)

    def test_potential_label_is_beating_the_benchmark(self):
        labels = make_labels(
            get_profile("potential")["target"],
            np.array([0.30, 0.05]), 0.10, np.array([0.9, 0.1]), np.array([0.8, 0.1]),
        )
        np.testing.assert_array_equal(labels, [1.0, 0.0])

    def test_stability_label_requires_gain_and_low_volatility_and_shallow_drawdown(self):
        target = get_profile("stability")["target"]
        labels = make_labels(
            target,
            #          steady  rocket  loser  bumpy
            np.array([0.12, 2.00, -0.05, 0.15]),
            0.10,
            np.array([0.15, 0.90, 0.15, 0.60]),
            np.array([0.10, 0.60, 0.10, 0.10]),
        )
        np.testing.assert_array_equal(labels, [1.0, 0.0, 0.0, 0.0])

    def test_profiles_produce_different_labels_on_the_same_data(self):
        prices, volumes, benchmarks = synthetic_panels(days=900)
        potential = build_snapshots(prices, volumes, benchmarks, profile=get_profile("potential"))
        stability = build_snapshots(prices, volumes, benchmarks, profile=get_profile("stability"))
        self.assertEqual([s.date for s in potential], [s.date for s in stability])
        self.assertTrue(any(not np.array_equal(p.labels, s.labels) for p, s in zip(potential, stability)))

    def test_derived_risk_features_ignore_the_execution_close_and_later_prices(self):
        prices, volumes, benchmarks = synthetic_panels(days=900)
        profile = get_profile("stability")
        initial = build_snapshots(prices, volumes, benchmarks, profile=profile)
        target = initial[10]
        position = enabled_features(profile).index("max_drawdown_52w")
        self.assertTrue(np.any(target.features[:, position] != 0))

        changed_prices = prices.copy()
        changed_prices.iloc[changed_prices.index.get_loc(target.date) :, 0] *= 3
        changed = build_snapshots(changed_prices, volumes, benchmarks, profile=profile)

        np.testing.assert_allclose(target.features, changed[10].features)

    def test_feature_weights_survive_standardization(self):
        rng = np.random.default_rng(0)
        profile = {"features": {"a": {"enabled": True, "weight": 3.0}, "b": {"enabled": True, "weight": 0.5}}}
        weights = feature_weights(profile)
        snapshot = one_snapshot(stock_returns=[0.0] * 50)
        snapshot.features = (rng.normal(size=(50, 2)) * weights).astype(np.float32)

        mean, std = standardization_parameters([snapshot], weights)
        normalized = normalize_features(snapshot.features, mean, std)

        np.testing.assert_allclose(normalized.std(axis=0), weights, rtol=1e-5)


class PerformanceSummaryTests(unittest.TestCase):
    def test_overlapping_cohorts_are_averaged_not_compounded(self):
        # Twelve overlapping one-year cohorts that each gain 10% are a 10% a year strategy.
        # Compounding them as if monthly would report a return of over 200%.
        snapshots = [
            one_snapshot(
                stock_returns=(0.1, 0.1),
                date=str(pd.Timestamp("2024-01-31") + pd.DateOffset(months=i))[:10],
                end=str(pd.Timestamp("2025-01-31") + pd.DateOffset(months=i))[:10],
            )
            for i in range(12)
        ]
        backtest = benchmark_backtest(snapshots, pd.Series([1.0, 1.1], index=pd.to_datetime(["2024-01-31", "2024-02-29"])))
        backtest["net_return"] = 0.10

        summary = performance_summary(backtest).iloc[0]

        self.assertAlmostEqual(summary["annualized_return"], 0.10, places=2)

    def test_independent_cohorts_each_pay_a_round_trip(self):
        snapshots = [one_snapshot(), one_snapshot(date="2024-02-29", end="2025-02-28")]
        result = portfolio_backtest(
            snapshots, [np.array([0.9, 0.1])] * 2, "model",
            top_k=2, max_position=0.5, max_sector=1.0, cost_rate=0.001, independent_cohorts=True,
        )
        self.assertAlmostEqual(result.loc[1, "turnover"], 2.0)
        self.assertAlmostEqual(result.loc[1, "transaction_cost"], 0.002)

    def test_portfolio_path_risk_tracks_a_drawdown(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="W-MON")
        prices = pd.DataFrame({"S0": [100.0, 120.0, 60.0, 90.0, 110.0], "S1": [10.0] * 5}, index=dates)
        snapshot = one_snapshot(date=str(dates[0])[:10], end=str(dates[-1])[:10])
        result = portfolio_backtest(
            [snapshot], [np.array([0.9, 0.1])], "model",
            top_k=1, max_position=1.0, max_sector=1.0, prices=prices,
        )
        self.assertAlmostEqual(result.loc[0, "path_max_drawdown"], 0.5)
        self.assertGreater(result.loc[0, "path_volatility"], 0)

    def test_empty_split_after_purging_is_reported(self):
        snapshots = [
            one_snapshot(
                date=str(pd.Timestamp("2024-01-31") + pd.DateOffset(months=i))[:10],
                end=str(pd.Timestamp("2025-01-31") + pd.DateOffset(months=i))[:10],
            )
            for i in range(20)
        ]
        with self.assertRaisesRegex(ValueError, "split is empty"):
            chronological_split(snapshots)


if __name__ == "__main__":
    unittest.main()
