"""Re-evaluate a checkpoint saved by run.py on the held-out test set.

python test.py --profile stability
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from config import DEFAULT_PROFILE, SNAPSHOT_CONFIG, SPLIT_CONFIG, TRAINING_PROFILES, get_profile
from src.training import StockGNN, NodeMLP
from src.preprocessing import load_price_cache, build_snapshots, chronological_split
from src.analysis import benchmark_backtest, predict_probabilities, portfolio_backtest, performance_summary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=sorted(TRAINING_PROFILES), default=DEFAULT_PROFILE)
parser.add_argument("--model-type", choices=["gcn", "mlp"], default="gcn")
parser.add_argument("--no-fundamentals", action="store_true")
args = parser.parse_args()

# 1. Load the checkpoint, including the scaler that was fit on the training period
suffix = "_no_fundamentals" if args.no_fundamentals else ""
checkpoint = torch.load(Path("checkpoints") / f"{args.model_type}_{args.profile}{suffix}.pt", weights_only=True)
mean, std = checkpoint["feature_mean"].numpy(), checkpoint["feature_std"].numpy()

# 2. Load data and extract test snapshots
profile = get_profile(args.profile, include_fundamentals=not args.no_fundamentals)
prices, volumes, benchmark_prices, sectors, optional_panels = load_price_cache(Path("stock_data"), "SPY")
snapshots = build_snapshots(prices, volumes, benchmark_prices, optional_panels, benchmark_ticker="SPY", sectors=sectors, profile=profile, **SNAPSHOT_CONFIG)
_, _, test_snapshots = chronological_split(snapshots, **SPLIT_CONFIG)

# 3. Rebuild the model
model_class = StockGNN if args.model_type == "gcn" else NodeMLP
model = model_class(input_size=len(checkpoint["feature_names"]), hidden_size=32, dropout=0.2)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 4. Compare the model with equal weight and SPY on unseen test data
test_predictions = predict_probabilities(model, test_snapshots, mean, std)
model_backtest = portfolio_backtest(test_snapshots, test_predictions, strategy=f"{args.model_type}_{args.profile}", prices=prices, independent_cohorts=True, **profile["portfolio"])
eq_backtest = portfolio_backtest(test_snapshots, test_predictions, strategy="equal_weight", prices=prices, independent_cohorts=True)
spy_backtest = benchmark_backtest(test_snapshots, benchmark_prices["SPY"])
summary_df = performance_summary(pd.concat([model_backtest, eq_backtest, spy_backtest], ignore_index=True))

print("--- Test Set Performance Summary (Overfitting Check) ---")
print(summary_df.round(3).to_string(index=False))
