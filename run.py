"""End-to-end workflow runner for the Weekly GCN Stock Prediction Model."""
#python run.py stock_symbols.txt --model-type gcn
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_price_cache, build_snapshots, chronological_split
from src.training import StockGNN, NodeMLP, standardization_parameters, train_model
from src.analysis import predict_probabilities, portfolio_backtest, performance_summary, get_snapshot_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GCN stock prediction and portfolio backtest pipeline.")
    parser.add_argument("--cache-dir", type=Path, default=Path("stock_data"), help="Directory containing cached ticker CSVs")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    parser.add_argument("--model-type", choices=["gcn", "mlp"], default="gcn", help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

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
    )
    print(f"Generated {len(snapshots)} valid monthly/quarterly snapshots.")
    if not snapshots:
        raise RuntimeError("No snapshots generated. Check your date ranges or data filters.")

    print("\n" + "=" * 60)
    print("STEP 3: Chronological Train/Val/Test Split...")
    print("=" * 60)
    train_snapshots, val_snapshots, test_snapshots = chronological_split(snapshots)
    print(f"Train snapshots: {len(train_snapshots)}")
    print(f"Validation snapshots: {len(val_snapshots)}")
    print(f"Test snapshots: {len(test_snapshots)}")

    print("\n" + "=" * 60)
    print("STEP 4: Normalizing Features & Initializing Model...")
    print("=" * 60)
    mean, std = standardization_parameters(train_snapshots)
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

    print("\n" + "=" * 60)
    print("STEP 6: Running Portfolio Backtest on Test Set...")
    print("=" * 60)
    test_predictions = predict_probabilities(model, test_snapshots, mean, std)
    # Save the trained model weights to disk
    model_path = Path("gcn_stock_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to {model_path}")
    # Backtest GCN / Model Strategy
    gcn_backtest = portfolio_backtest(
        snapshots=test_snapshots,
        predictions=test_predictions,
        strategy="gcn_top_10",
        top_k=10,
        max_position=0.10,
        max_sector=0.25,
        cost_rate=0.001,
    )

    # Backtest Equal-Weight Baseline Strategy
    eq_backtest = portfolio_backtest(
        snapshots=test_snapshots,
        predictions=test_predictions,
        strategy="equal_weight",
    )

    combined_backtest = pd.concat([gcn_backtest, eq_backtest], ignore_index=True)
    summary_df = performance_summary(combined_backtest)

    print("\n--- Performance Summary ---")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 7: Top Long-Term Picks (Latest Test Snapshot)...")
    print("=" * 60)
    latest_snapshot = test_snapshots[-1]
    latest_preds = test_predictions[-1]
    rankings_df = get_snapshot_rankings(latest_snapshot, latest_preds, top_n=10)
    print(f"Snapshot Date: {latest_snapshot.date.strftime('%Y-%m-%d')} (Holding Horizon End: {latest_snapshot.target_end.strftime('%Y-%m-%d')})")
    print(rankings_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()