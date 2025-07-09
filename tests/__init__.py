"""
Test suite for the trading system.

This test suite provides comprehensive coverage for all components of the trading system,
including unit tests, integration tests, and performance benchmarks.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_RESULTS_DIR = Path(__file__).parent / "results"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_RESULTS_DIR.mkdir(exist_ok=True)

# Common test constants
DEFAULT_SEED = 42
DEFAULT_PERIODS = 100
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_RATE = 0.001

# Test symbols
TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_execution_time": 5.0,  # seconds
    "max_memory_usage": 100,  # MB
    "min_test_coverage": 90.0,  # percentage
}

__version__ = "1.0.0"
