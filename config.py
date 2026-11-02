#some tuning parameters


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



FEATURE_CONFIG = {
    # Momentum & Trend (Reward strong continuation and breakout trends)
    "return_4w": {"enabled": True, "weight": 0.5},
    "return_12w": {"enabled": True, "weight": 1.0},
    "return_26w": {"enabled": True, "weight": 1.5},  # Medium-term structural trend
    "return_52w": {"enabled": True, "weight": 2.0},  # Multi-quarter macro leadership
    "ma_10w_ratio": {"enabled": True, "weight": 1.0},
    "ma_40w_ratio": {"enabled": True, "weight": 1.5},  # Reward stocks trading well above long-term trend
    
    # Volatility (High-risk stocks have higher variance; embrace it)
    "volatility_12w": {"enabled": True, "weight": 1.2},
    
    # Fundamentals (Aggressively reward growth; de-emphasize immediate profitability/debt)
    "roe": {"enabled": True, "weight": 0.5},          # Lower weight (growth companies often reinvest earnings rather than show high ROE yet)
    "debt_to_equity": {"enabled": True, "weight": 0.3}, # Low weight or penalize less (growth companies often leverage up to expand)
    "revenue_growth": {"enabled": True, "weight": 3.0}, # CRITICAL: Massive weight for top-line expansion
    "profit_margin": {"enabled": True, "weight": 0.2},   # Minimal weight (allows pre-profit tech/biotech rockets to qualify)
}