"""Tuning parameters, including the training-profile toggle.

Pick a profile with `python run.py --profile potential|stability`, or change
DEFAULT_PROFILE below. A profile sets what the model is trained to predict
(target), what it sees (features), and how picks become a portfolio (portfolio).
"""

from __future__ import annotations

import copy
from typing import Optional


REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "adj_close",
    "volume",
    "sector",
    "return_4w",
    "return_12w",
    "return_26w",
    "return_52w",
    "ma_10w_ratio",
    "ma_40w_ratio",
    "volatility_12w",
    "roe",
    "debt_to_equity",
    "revenue_growth",
    "profit_margin",
}

# The cached fundamentals are today's values repeated across every historical row,
# so they leak future information into a backtest. --no-fundamentals removes them.
FUNDAMENTAL_FEATURES = {"roe", "debt_to_equity", "revenue_growth", "profit_margin"}

# Feature weights scale a feature's influence after standardization. They must be
# positive: they set emphasis only, and the model learns each feature's direction.
DEFAULT_PROFILE = "potential"

TRAINING_PROFILES = {
    "potential": {
        "description": "High-upside growth: stocks likely to outperform the benchmark.",
        "target": {"type": "beat_benchmark"},
        "features": {
            # Momentum & trend (reward strong continuation and breakout trends)
            "return_4w": {"enabled": True, "weight": 0.5},
            "return_12w": {"enabled": True, "weight": 1.0},
            "return_26w": {"enabled": True, "weight": 1.5},  # Medium-term structural trend
            "return_52w": {"enabled": True, "weight": 2.0},  # Multi-quarter macro leadership
            "ma_10w_ratio": {"enabled": True, "weight": 1.0},
            "ma_40w_ratio": {"enabled": True, "weight": 1.5},  # Trading well above long-term trend
            # Volatility (high-risk stocks have higher variance; embrace it)
            "volatility_12w": {"enabled": True, "weight": 1.2},
            # Fundamentals (reward growth; de-emphasize immediate profitability/debt)
            "roe": {"enabled": True, "weight": 0.5},  # Growth companies often reinvest earnings
            "debt_to_equity": {"enabled": True, "weight": 0.3},  # Growth companies often lever up
            "revenue_growth": {"enabled": True, "weight": 3.0},  # Top-line expansion
            "profit_margin": {"enabled": True, "weight": 0.2},  # Allows pre-profit companies
        },
        "portfolio": {"top_k": 10, "max_position": 0.10, "max_sector": 0.25},
    },
    "stability": {
        "description": "Long, slow growth: stocks likely to rise steadily with shallow drawdowns.",
        # A stock is a positive example only if, over the holding horizon, it gained at
        # least min_return AND its volatility and max drawdown were both no worse than
        # these quantiles of that snapshot's universe.
        "target": {
            "type": "steady_growth",
            "min_return": 0.05,
            "max_volatility_quantile": 0.5,
            "max_drawdown_quantile": 0.5,
        },
        "features": {
            # Long-term trend (short-term momentum is noise for this goal, so it is off)
            "return_4w": {"enabled": False, "weight": 0.0},
            "return_12w": {"enabled": True, "weight": 0.5},
            "return_26w": {"enabled": True, "weight": 1.0},
            "return_52w": {"enabled": True, "weight": 1.5},
            "ma_10w_ratio": {"enabled": False, "weight": 0.0},
            "ma_40w_ratio": {"enabled": True, "weight": 1.5},
            # Risk (the core signal: how smooth has the ride been?)
            "volatility_12w": {"enabled": True, "weight": 2.0},
            "volatility_52w": {"enabled": True, "weight": 3.0},  # Derived from prices
            "max_drawdown_52w": {"enabled": True, "weight": 3.0},  # Derived from prices
            # Fundamentals (reward durable, profitable, lightly levered businesses)
            "roe": {"enabled": True, "weight": 1.5},
            "debt_to_equity": {"enabled": True, "weight": 1.5},
            "revenue_growth": {"enabled": True, "weight": 0.5},
            "profit_margin": {"enabled": True, "weight": 2.0},
        },
        # More names and tighter caps than potential, to reduce single-stock risk.
        "portfolio": {"top_k": 20, "max_position": 0.05, "max_sector": 0.20},
    },
}

# Shared so both profiles are compared on identical dates and horizons.
SNAPSHOT_CONFIG = {
    "horizon": 52,  # Forward holding period, in weeks
    "correlation_window": 104,  # Trailing weeks used to build the correlation graph
    "edges_per_node": 7,
}

# A 52-week label must be fully observed before the next split starts, which purges
# about 12 snapshots at each boundary. Validation has to be large enough to survive.
SPLIT_CONFIG = {"train_fraction": 0.50, "validation_fraction": 0.30}


def get_profile(name: Optional[str] = None, include_fundamentals: bool = True) -> dict:
    """Return an independent copy of a training profile, with its settings validated."""
    name = name or DEFAULT_PROFILE
    if name not in TRAINING_PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Choose from: {sorted(TRAINING_PROFILES)}")
    profile = copy.deepcopy(TRAINING_PROFILES[name])
    profile["name"] = name
    for feature, settings in profile["features"].items():
        if feature in FUNDAMENTAL_FEATURES and not include_fundamentals:
            settings["enabled"] = False
        if settings["enabled"] and settings["weight"] <= 0:
            raise ValueError(f"Enabled feature {feature!r} in profile {name!r} needs a positive weight")
    return profile
