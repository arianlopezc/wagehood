"""
Global pytest configuration for Wagehood tests.

IMPORTANT: Prevents real Discord API calls during testing.
"""

import pytest
import os
from unittest.mock import patch, Mock


@pytest.fixture(autouse=True)
def prevent_real_discord_calls():
    """
    Automatically mock urllib.request.urlopen for all tests to prevent real Discord API calls.
    
    This fixture runs automatically for all tests and ensures that no actual
    Discord webhooks are triggered during testing, which would confuse the
    audience with fake trading signals.
    """
    with patch('urllib.request.urlopen') as mock_urlopen:
        # Setup default mock response for Discord webhook calls
        mock_response = Mock()
        mock_response.read.return_value = b'{"message": "mocked response"}'
        mock_response.getcode.return_value = 200
        mock_response.geturl.return_value = "https://mocked.discord.webhook"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        yield mock_urlopen


@pytest.fixture
def mock_discord_environment():
    """
    Provide mock Discord environment variables for testing.
    """
    test_env = {
        'DISCORD_WEBHOOK_SWING_SIGNALS': 'https://mock.discord.webhook.url/test',
        'DISCORD_NOTIFICATIONS_ENABLED': 'true',
        'DISCORD_MAX_NOTIFICATIONS_PER_HOUR': '100',
        'DISCORD_MIN_CONFIDENCE_THRESHOLD': '0.0'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def test_signal_data():
    """
    Provide test signal data for notification testing.
    """
    from datetime import datetime
    
    return {
        'symbol': 'AAPL',
        'signal': 'BUY',
        'price': 150.0,
        'price_change': 2.5,
        'price_change_pct': 1.7,
        'strategy': 'MACD+RSI',
        'confidence': 0.85,
        'details': {
            'macd_signal': 'bullish_crossover',
            'rsi_value': 35.2
        },
        'timestamp': datetime(2024, 1, 1, 12, 0, 0),
        'company_name': 'Apple Inc.'
    }


# Ensure tests run in test mode
@pytest.fixture(autouse=True)
def test_environment_marker():
    """
    Set environment marker to indicate we're in test mode.
    """
    os.environ['WAGEHOOD_TEST_MODE'] = 'true'
    yield
    if 'WAGEHOOD_TEST_MODE' in os.environ:
        del os.environ['WAGEHOOD_TEST_MODE']