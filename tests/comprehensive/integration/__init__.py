"""
Integration tests for the Wagehood trading system.

This module contains tests that validate the integration with external APIs,
data sources, and services including Alpaca API, Redis cache, and real-time data feeds.
"""

# Integration test constants
API_TIMEOUT = 30.0  # seconds
CONNECTION_RETRY_COUNT = 3
TEST_SYMBOLS = ["SPY", "QQQ", "AAPL"]
TEST_TIMEFRAMES = ["1min", "5min", "1hour", "1day"]

# Test environment configurations
INTEGRATION_ENVIRONMENTS = {
    "sandbox": {
        "alpaca_base_url": "https://paper-api.alpaca.markets",
        "use_real_data": False,
        "rate_limit": 200,  # requests per minute
    },
    "live": {
        "alpaca_base_url": "https://api.alpaca.markets",
        "use_real_data": True,
        "rate_limit": 200,  # requests per minute
    },
}

# Expected data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "min_data_points": 100,
    "max_missing_percentage": 5.0,  # 5% missing data allowed
    "max_latency_ms": 1000,  # 1 second max latency
    "min_update_frequency": 0.9,  # 90% of expected updates
}
