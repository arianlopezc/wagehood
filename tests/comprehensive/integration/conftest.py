"""
Integration Test Configuration and Fixtures

This module provides specialized fixtures and configuration for
comprehensive integration tests with real external dependencies.
"""

import pytest
import asyncio
import os
import logging
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def integration_event_loop():
    """Create event loop for integration tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def alpaca_credentials():
    """Get Alpaca credentials for integration tests."""
    credentials = {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'paper': os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true',
        'feed': os.getenv('ALPACA_DATA_FEED', 'iex')
    }
    
    if not credentials['api_key'] or not credentials['secret_key']:
        pytest.skip("Alpaca credentials not available for integration tests")
    
    return credentials


@pytest.fixture(scope="session")
def redis_available():
    """Check if Redis is available for testing."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        pytest.skip("Redis not available for integration tests")


@pytest.fixture(scope="session")
def test_environment_config():
    """Configuration for test environment."""
    return {
        'test_duration_short': 5.0,    # seconds
        'test_duration_medium': 15.0,  # seconds
        'test_duration_long': 30.0,    # seconds
        'max_memory_usage_mb': 100,    # MB
        'min_throughput_ops_sec': 100, # operations per second
        'max_latency_ms': 10.0,        # milliseconds
        'max_error_rate': 0.05,        # 5%
        'test_symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
        'crypto_symbols': ['BTC/USD', 'ETH/USD', 'LTC/USD']
    }


@pytest.fixture
def performance_requirements():
    """Performance requirements for integration tests."""
    return {
        'latency': {
            'avg_ms': 5.0,
            'p95_ms': 20.0,
            'p99_ms': 50.0,
            'max_ms': 100.0
        },
        'throughput': {
            'min_ops_per_sec': 500,
            'target_ops_per_sec': 1000
        },
        'reliability': {
            'max_error_rate': 0.01,
            'min_uptime': 0.99
        },
        'resources': {
            'max_memory_mb': 50,
            'max_cpu_percent': 80
        }
    }


@pytest.fixture
def integration_test_markers():
    """Markers for integration test categorization."""
    return {
        'slow': 'marks tests as slow (integration with external services)',
        'network': 'marks tests that require network connectivity',
        'alpaca': 'marks tests that require Alpaca API access',
        'redis': 'marks tests that require Redis',
        'performance': 'marks performance benchmark tests',
        'resilience': 'marks resilience and failure recovery tests'
    }


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data after each test."""
    yield
    # Add cleanup logic here if needed
    pass


@pytest.fixture
def mock_external_dependencies():
    """Mock external dependencies for offline testing."""
    mocks = {}
    
    # Mock Redis
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.xadd.return_value = b'1234567890-0'
        mock_redis.return_value.xgroup_create.return_value = True
        mock_redis.return_value.xinfo_stream.return_value = {
            'length': 0, 'groups': 1, 'last-generated-id': b'0-0'
        }
        mocks['redis'] = mock_redis
        
        yield mocks


def pytest_configure(config):
    """Configure pytest for integration tests."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network"
    )
    config.addinivalue_line(
        "markers", "alpaca: mark test as requiring Alpaca API"
    )
    config.addinivalue_line(
        "markers", "redis: mark test as requiring Redis"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "resilience: mark test as resilience test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for integration tests."""
    # Skip integration tests if running in CI without proper credentials
    skip_integration = pytest.mark.skip(reason="Integration tests require external dependencies")
    
    for item in items:
        if "integration" in item.keywords:
            # Add integration marker
            item.add_marker(pytest.mark.integration)
            
            # Skip if credentials not available and not using mocks
            if not os.getenv('ALPACA_API_KEY') and 'alpaca' in item.keywords:
                if 'mock' not in str(item.fspath):
                    item.add_marker(skip_integration)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['performance', 'load', 'stress']):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def integration_test_report():
    """Generate integration test report."""
    report = {
        'start_time': None,
        'end_time': None,
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'tests_skipped': 0,
        'performance_metrics': {},
        'error_summary': []
    }
    
    yield report
    
    # Generate final report
    if report['tests_run'] > 0:
        success_rate = report['tests_passed'] / report['tests_run']
        logging.info(f"Integration Test Summary:")
        logging.info(f"  Tests run: {report['tests_run']}")
        logging.info(f"  Success rate: {success_rate:.2%}")
        logging.info(f"  Failed: {report['tests_failed']}")
        logging.info(f"  Skipped: {report['tests_skipped']}")


# Pytest hooks for integration testing
def pytest_runtest_setup(item):
    """Setup for each integration test."""
    if "integration" in item.keywords:
        # Verify prerequisites
        pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown for each integration test."""
    if "integration" in item.keywords:
        # Cleanup test artifacts
        pass