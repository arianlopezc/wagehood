"""
End-to-end tests for the Wagehood trading system.

This module contains tests that validate complete trading workflows
from data ingestion through signal generation to portfolio management.
"""

# E2E test constants
WORKFLOW_TIMEOUT = 300.0  # 5 minutes for complete workflow
SCENARIO_TIMEOUT = 120.0  # 2 minutes for individual scenarios
DATA_WARMUP_PERIOD = 60.0  # 1 minute for data warmup
SIGNAL_GENERATION_TIMEOUT = 30.0  # 30 seconds for signal generation

# Test scenarios
TRADING_SCENARIOS = {
    "bull_market": {
        "description": "Strong uptrend with momentum signals",
        "duration_days": 30,
        "expected_signals": "buy_heavy",
        "expected_performance": "positive",
    },
    "bear_market": {
        "description": "Strong downtrend with risk-off signals",
        "duration_days": 30,
        "expected_signals": "sell_heavy",
        "expected_performance": "protected",
    },
    "sideways_market": {
        "description": "Range-bound market with limited signals",
        "duration_days": 60,
        "expected_signals": "minimal",
        "expected_performance": "stable",
    },
    "volatile_market": {
        "description": "High volatility with frequent signals",
        "duration_days": 20,
        "expected_signals": "frequent",
        "expected_performance": "risk_managed",
    },
}

# System integration points
INTEGRATION_POINTS = [
    "data_ingestion_to_storage",
    "storage_to_calculation",
    "calculation_to_signals",
    "signals_to_portfolio",
    "portfolio_to_reporting",
    "real_time_processing",
    "error_handling_and_recovery",
]

# Performance expectations for E2E tests
E2E_PERFORMANCE_TARGETS = {
    "data_latency_ms": 1000,
    "signal_generation_ms": 5000,
    "portfolio_update_ms": 2000,
    "report_generation_ms": 10000,
    "memory_usage_mb": 1024,
    "cpu_usage_percent": 85,
}
