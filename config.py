FEATURE_CONFIG = {
    "return_21d": {"enabled": False, "weight": 1.0},
    "volatility_63d": {"enabled": False, "weight": 1.0},
    "volume_change": {"enabled": False, "weight": 1.0},
    "beta": {"enabled": False, "weight": 1.5},  
    "pe_ratio": {"enabled": False, "weight": 0.8},  
}

REQUIRED_COLUMNS = {"date", "ticker", "adj_close", "volume"}