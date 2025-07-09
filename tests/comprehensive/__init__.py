"""
Comprehensive Test Framework for Wagehood Trading System

This module provides a comprehensive testing framework that validates all aspects
of the Wagehood trading system including mathematical accuracy, real-time data
integration, worker processes, and end-to-end system functionality.

Test Categories:
- Mathematical: Strategy and indicator validation
- Integration: API and data source integration
- Workers: Background process testing
- E2E: End-to-end system testing
- Performance: Stress and benchmark testing
"""

__version__ = "1.0.0"
__author__ = "Wagehood Team"

# Test execution order
TEST_EXECUTION_ORDER = ["mathematical", "integration", "workers", "e2e", "performance"]

# Test environment configurations
TEST_ENVIRONMENTS = {
    "unit": {
        "redis": False,
        "alpaca": False,
        "external_apis": False,
        "description": "Isolated unit tests with no external dependencies",
    },
    "integration": {
        "redis": True,
        "alpaca": "sandbox",
        "external_apis": "mock",
        "description": "Integration tests with mocked external services",
    },
    "e2e": {
        "redis": True,
        "alpaca": "sandbox",
        "external_apis": "live",
        "description": "End-to-end tests with live external services",
    },
    "performance": {
        "redis": True,
        "alpaca": "sandbox",
        "external_apis": "mock",
        "monitoring": True,
        "description": "Performance and stress testing with monitoring",
    },
}

# Test markers for pytest
PYTEST_MARKERS = {
    "mathematical": "Mathematical validation tests",
    "integration": "Integration tests with external services",
    "workers": "Worker process tests",
    "e2e": "End-to-end system tests",
    "performance": "Performance and benchmark tests",
    "stress": "Stress and load tests",
    "slow": "Slow-running tests",
    "redis": "Tests requiring Redis",
    "alpaca": "Tests requiring Alpaca API",
    "memory": "Memory usage tests",
}
