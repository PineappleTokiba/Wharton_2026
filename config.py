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
    "return_4w": {"enabled": True, "weight": 1.0},
    "return_12w": {"enabled": True, "weight": 1.0},
    "return_26w": {"enabled": True, "weight": 1.0},
    "return_52w": {"enabled": True, "weight": 1.0},
    "ma_10w_ratio": {"enabled": True, "weight": 1.0},
    "ma_40w_ratio": {"enabled": True, "weight": 1.0},
    "volatility_12w": {"enabled": True, "weight": 1.0},
    "roe": {"enabled": True, "weight": 1.0},
    "debt_to_equity": {"enabled": True, "weight": 1.0},
    "revenue_growth": {"enabled": True, "weight": 1.0},
    "profit_margin": {"enabled": True, "weight": 1.0},
}