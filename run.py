"""End-to-end workflow runner for the Weekly GCN Stock Prediction Model.

python run.py --profile potential    # high-upside growth (default)
python run.py --profile stability    # long, slow, steady growth
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_PROFILE, SNAPSHOT_CONFIG, SPLIT_CONFIG, TRAINING_PROFILES, get_profile
from src.preprocessing import (
    build_snapshots,
    chronological_split,
    enabled_features,
    feature_weights,
    load_price_cache,
)
from src.training import StockGNN, NodeMLP, standardization_parameters, train_model
from src.analysis import (
    benchmark_backtest,
    get_snapshot_rankings,
    performance_summary,
    portfolio_backtest,
    predict_probabilities,
    ranking_skill,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GCN stock prediction and portfolio backtest pipeline.")
    parser.add_argument("--profile", choices=sorted(TRAINING_PROFILES), default=DEFAULT_PROFILE, help="What to train for: high-upside growth (potential) or long, slow growth (stability)")
    parser.add_argument("--no-fundamentals", action="store_true", help="Drop fundamentals, which are current values repeated over history and so leak future information")
    parser.add_argument("--cache-dir", type=Path, default=Path("stock_data"), help="Directory containing cached ticker CSVs")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    parser.add_argument("--model-type", choices=["gcn", "mlp"], default="gcn", help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Where to save the trained model")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    profile = get_profile(args.profile, include_fundamentals=not args.no_fundamentals)
    portfolio = profile["portfolio"]
    print(f"Profile: {profile['name']} - {profile['description']}")
    print(f"Features: {', '.join(enabled_features(profile))}")

    print("=" * 60)
    print("STEP 1: Loading Weekly Price & Fundamental Cache...")
    print("=" * 60)
    prices, volumes, benchmark_prices, sectors, optional_panels = load_price_cache(args.cache_dir, args.benchmark)
    print(f"Loaded data for {prices.shape[1]} stocks across {prices.shape[0]} weekly timestamps.")

    print("\n" + "=" * 60)
    print("STEP 2: Building Graph Snapshots...")
    print("=" * 60)
    snapshots = build_snapshots(
        prices=prices,
        volumes=volumes,
        benchmark_prices=benchmark_prices,
        optional_panels=optional_panels,
        benchmark_ticker=args.benchmark,
        sectors=sectors,
        profile=profile,
        **SNAPSHOT_CONFIG,
    )
    print(f"Generated {len(snapshots)} valid monthly snapshots.")
    if not snapshots:
        raise RuntimeError("No snapshots generated. Check your date ranges or data filters.")

    print("\n" + "=" * 60)
    print("STEP 3: Chronological Train/Val/Test Split...")
    print("=" * 60)
    train_snapshots, val_snapshots, test_snapshots = chronological_split(snapshots, **SPLIT_CONFIG)
    print(f"Train snapshots: {len(train_snapshots)}")
    print(f"Validation snapshots: {len(val_snapshots)}")
    print(f"Test snapshots: {len(test_snapshots)}")
    print(f"Training positive rate: {np.mean([s.labels.mean() for s in train_snapshots]):.3f}")

    print("\n" + "=" * 60)
    print("STEP 4: Normalizing Features & Initializing Model...")
    print("=" * 60)
    mean, std = standardization_parameters(train_snapshots, feature_weights(profile))
    input_size = train_snapshots[0].features.shape[1]

    if args.model_type == "gcn":
        model = StockGNN(input_size=input_size, hidden_size=32, dropout=0.2)
        print("Initialized StockGNN model.")
    else:
        model = NodeMLP(input_size=input_size, hidden_size=32, dropout=0.2)
        print("Initialized NodeMLP model.")

    print("\n" + "=" * 60)
    print("STEP 5: Training Model with Early Stopping...")
    print("=" * 60)
    model = train_model(
        model=model,
        train=train_snapshots,
        validation=val_snapshots,
        mean=mean,
        standard_deviation=std,
        max_epochs=args.epochs,
        patience=15,
        learning_rate=args.lr,
    )
    print("Training complete! Best model checkpoint loaded.")

    # The scaler and settings are saved with the weights: the model is unusable without them.
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_no_fundamentals" if args.no_fundamentals else ""
    model_path = args.checkpoint_dir / f"{args.model_type}_{profile['name']}{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "profile": profile["name"],
            "include_fundamentals": not args.no_fundamentals,
            "feature_names": enabled_features(profile),
            "feature_mean": torch.from_numpy(mean),
            "feature_std": torch.from_numpy(std),
            "seed": args.seed,
            "train_cutoff": str(train_snapshots[-1].date.date()),
        },
        model_path,
    )
    print(f"Model successfully saved to {model_path}")

    print("\n" + "=" * 60)
    print("STEP 6: Running Portfolio Backtest on Test Set...")
    print("=" * 60)
    test_predictions = predict_probabilities(model, test_snapshots, mean, std)

    model_backtest = portfolio_backtest(
        snapshots=test_snapshots,
        predictions=test_predictions,
        strategy=f"{args.model_type}_{profile['name']}_top_{portfolio['top_k']}",
        cost_rate=0.001,
        prices=prices,
        independent_cohorts=True,
        **portfolio,
    )
    eq_backtest = portfolio_backtest(
        snapshots=test_snapshots,
        predictions=test_predictions,
        strategy="equal_weight",
        prices=prices,
        independent_cohorts=True,
    )
    spy_backtest = benchmark_backtest(test_snapshots, benchmark_prices[args.benchmark.upper()], args.benchmark.upper())

    summary_df = performance_summary(pd.concat([model_backtest, eq_backtest, spy_backtest], ignore_index=True))

    print("\n--- Performance Summary (overlapping cohorts, averaged; not an investable equity curve) ---")
    print(summary_df.round(3).to_string(index=False))

    skill = ranking_skill(test_snapshots, test_predictions, portfolio["top_k"])
    print(f"\n--- Ranking Skill vs '{profile['target']['type']}' target ---")
    print(
        f"Mean AUC: {skill['mean_auc']:.3f} (0.5 = no skill) | "
        f"Precision@{portfolio['top_k']}: {skill['precision_at_k']:.3f} | Base rate: {skill['base_rate']:.3f}"
    )

    print("\n" + "=" * 60)
    print("STEP 7: Top Picks (Latest Test Snapshot)...")
    print("=" * 60)
    latest_snapshot = test_snapshots[-1]
    latest_preds = test_predictions[-1]
    rankings_df = get_snapshot_rankings(latest_snapshot, latest_preds, top_n=10)
    print(f"Snapshot Date: {latest_snapshot.date.strftime('%Y-%m-%d')} (Holding Horizon End: {latest_snapshot.target_end.strftime('%Y-%m-%d')})")
    print(rankings_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
