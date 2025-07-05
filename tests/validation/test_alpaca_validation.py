"""
Alpaca Integration Validation Tests

This module tests the Alpaca Markets integration to ensure all components
work correctly with both paper trading and live market data.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import Alpaca components
try:
    from src.data.providers.alpaca_provider import AlpacaProvider
    from src.trading.alpaca_client import AlpacaTradingClient
    from src.realtime.alpaca_ingestion import AlpacaIngestionService
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from src.core.models import OHLCV, TimeFrame


@pytest.mark.skipif(not ALPACA_AVAILABLE, reason="Alpaca components not available")
class TestAlpacaProviderValidation:
    """Test Alpaca data provider functionality."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client for testing."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock historical data response
            mock_bars = []
            base_time = datetime.now() - timedelta(days=30)
            
            for i in range(100):
                mock_bar = Mock()
                mock_bar.timestamp = base_time + timedelta(hours=i)
                mock_bar.open = 100.0 + i * 0.1
                mock_bar.high = 101.0 + i * 0.1
                mock_bar.low = 99.0 + i * 0.1
                mock_bar.close = 100.5 + i * 0.1
                mock_bar.volume = 1000 + i * 10
                mock_bars.append(mock_bar)
            
            mock_response = Mock()
            mock_response.__iter__ = lambda self: iter([("AAPL", mock_bars)])
            mock_instance.get_stock_bars.return_value = mock_response
            
            yield mock_instance
    
    def test_alpaca_provider_initialization(self):
        """Test Alpaca provider initialization."""
        # Test with mock credentials
        with patch('alpaca.data.StockHistoricalDataClient'):
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            assert provider is not None
            assert provider.paper is True
    
    def test_alpaca_provider_historical_data(self, mock_alpaca_client):
        """Test historical data retrieval."""
        with patch('alpaca.data.StockHistoricalDataClient', return_value=mock_alpaca_client):
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=start_date,
                end_date=end_date
            )
            
            assert len(data) == 100
            assert all(isinstance(bar, OHLCV) for bar in data)
            
            # Validate data structure
            for bar in data:
                assert bar.open > 0
                assert bar.high >= bar.open
                assert bar.low <= bar.open
                assert bar.volume > 0
                assert isinstance(bar.timestamp, datetime)
    
    def test_alpaca_provider_error_handling(self):
        """Test Alpaca provider error handling."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Simulate API error
            mock_instance.get_stock_bars.side_effect = Exception("API Error")
            
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Should handle errors gracefully
            data = provider.get_historical_data(
                symbol="INVALID",
                timeframe=TimeFrame.HOURLY,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            # Should return empty list on error
            assert data == []
    
    def test_alpaca_provider_rate_limiting(self, mock_alpaca_client):
        """Test rate limiting handling."""
        with patch('alpaca.data.StockHistoricalDataClient', return_value=mock_alpaca_client):
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Make multiple rapid requests
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            start_date = datetime.now() - timedelta(days=1)
            end_date = datetime.now()
            
            for symbol in symbols:
                data = provider.get_historical_data(
                    symbol=symbol,
                    timeframe=TimeFrame.HOURLY,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Should not crash due to rate limiting
                assert isinstance(data, list)


@pytest.mark.skipif(not ALPACA_AVAILABLE, reason="Alpaca components not available")
class TestAlpacaTradingClientValidation:
    """Test Alpaca trading client functionality."""
    
    @pytest.fixture
    def mock_trading_client(self):
        """Mock Alpaca trading client."""
        with patch('alpaca.trading.TradingClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock account response
            mock_account = Mock()
            mock_account.cash = "10000.00"
            mock_account.buying_power = "40000.00"
            mock_account.equity = "50000.00"
            mock_account.status = "ACTIVE"
            mock_instance.get_account.return_value = mock_account
            
            # Mock order response
            mock_order = Mock()
            mock_order.id = "order_123"
            mock_order.status = "accepted"
            mock_order.symbol = "AAPL"
            mock_order.qty = "10"
            mock_order.side = "buy"
            mock_instance.submit_order.return_value = mock_order
            
            # Mock positions response
            mock_position = Mock()
            mock_position.symbol = "AAPL"
            mock_position.qty = "100"
            mock_position.market_value = "15000.00"
            mock_position.unrealized_pl = "500.00"
            mock_instance.get_all_positions.return_value = [mock_position]
            
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_alpaca_trading_client_initialization(self):
        """Test trading client initialization."""
        with patch('alpaca.trading.TradingClient'):
            client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            assert client is not None
            assert client.paper is True
    
    @pytest.mark.asyncio
    async def test_alpaca_trading_client_account_info(self, mock_trading_client):
        """Test account information retrieval."""
        with patch('alpaca.trading.TradingClient', return_value=mock_trading_client):
            client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            account_info = await client.get_account_info()
            
            assert account_info is not None
            assert "cash" in account_info
            assert "buying_power" in account_info
            assert "equity" in account_info
            assert float(account_info["cash"]) > 0
    
    @pytest.mark.asyncio
    async def test_alpaca_trading_client_order_placement(self, mock_trading_client):
        """Test order placement."""
        with patch('alpaca.trading.TradingClient', return_value=mock_trading_client):
            client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            order_result = await client.place_market_order(
                symbol="AAPL",
                quantity=10.0,
                side="buy"
            )
            
            assert order_result is not None
            assert "id" in order_result
            assert "status" in order_result
            assert order_result["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_alpaca_trading_client_positions(self, mock_trading_client):
        """Test positions retrieval."""
        with patch('alpaca.trading.TradingClient', return_value=mock_trading_client):
            client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            positions = await client.get_positions()
            
            assert isinstance(positions, list)
            assert len(positions) > 0
            
            position = positions[0]
            assert "symbol" in position
            assert "quantity" in position
            assert "market_value" in position
    
    @pytest.mark.asyncio
    async def test_alpaca_trading_client_error_handling(self):
        """Test trading client error handling."""
        with patch('alpaca.trading.TradingClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Simulate API error
            mock_instance.submit_order.side_effect = Exception("Trading API Error")
            
            client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Should handle errors gracefully
            with pytest.raises(Exception):
                await client.place_market_order("AAPL", 10.0, "buy")


@pytest.mark.skipif(not ALPACA_AVAILABLE, reason="Alpaca components not available")
class TestAlpacaIngestionServiceValidation:
    """Test Alpaca real-time data ingestion."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        mock_redis = Mock()
        mock_redis.xadd.return_value = "123-0"
        mock_redis.ping.return_value = True
        return mock_redis
    
    @pytest.fixture
    def mock_stream_client(self):
        """Mock Alpaca stream client."""
        with patch('alpaca.data.StockDataStream') as mock_stream:
            mock_instance = Mock()
            mock_stream.return_value = mock_instance
            
            # Mock stream methods
            mock_instance.subscribe_bars = AsyncMock()
            mock_instance.run = AsyncMock()
            mock_instance.stop = AsyncMock()
            
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_alpaca_ingestion_service_initialization(self, mock_redis_client):
        """Test ingestion service initialization."""
        with patch('redis.Redis', return_value=mock_redis_client):
            service = AlpacaIngestionService(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            assert service is not None
    
    @pytest.mark.asyncio
    async def test_alpaca_ingestion_service_subscription(self, mock_redis_client, mock_stream_client):
        """Test symbol subscription."""
        with patch('redis.Redis', return_value=mock_redis_client), \
             patch('alpaca.data.StockDataStream', return_value=mock_stream_client):
            
            service = AlpacaIngestionService(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            await service.subscribe_symbols(["AAPL", "GOOGL"])
            
            # Should subscribe to bars
            mock_stream_client.subscribe_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alpaca_ingestion_service_data_processing(self, mock_redis_client):
        """Test data processing and Redis publishing."""
        with patch('redis.Redis', return_value=mock_redis_client):
            service = AlpacaIngestionService(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Simulate bar data
            mock_bar = Mock()
            mock_bar.symbol = "AAPL"
            mock_bar.timestamp = datetime.now()
            mock_bar.open = 150.0
            mock_bar.high = 151.0
            mock_bar.low = 149.0
            mock_bar.close = 150.5
            mock_bar.volume = 1000
            
            ohlcv = OHLCV(
                timestamp=mock_bar.timestamp,
                open=mock_bar.open,
                high=mock_bar.high,
                low=mock_bar.low,
                close=mock_bar.close,
                volume=mock_bar.volume
            )
            
            await service._handle_bar_data(ohlcv)
            
            # Should publish to Redis
            mock_redis_client.xadd.assert_called()


class TestAlpacaIntegrationValidation:
    """Test integration between Alpaca components."""
    
    @pytest.mark.asyncio
    async def test_provider_trading_client_integration(self):
        """Test integration between data provider and trading client."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_data_client, \
             patch('alpaca.trading.TradingClient') as mock_trading_client:
            
            # Setup mocks
            mock_data_instance = Mock()
            mock_data_client.return_value = mock_data_instance
            
            mock_trading_instance = Mock()
            mock_trading_client.return_value = mock_trading_instance
            
            # Mock responses
            mock_bars = []
            for i in range(10):
                mock_bar = Mock()
                mock_bar.timestamp = datetime.now() - timedelta(hours=i)
                mock_bar.close = 150.0 + i
                mock_bars.append(mock_bar)
            
            mock_response = Mock()
            mock_response.__iter__ = lambda self: iter([("AAPL", mock_bars)])
            mock_data_instance.get_stock_bars.return_value = mock_response
            
            mock_account = Mock()
            mock_account.cash = "10000.00"
            mock_trading_instance.get_account.return_value = mock_account
            
            # Test integration
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            trading_client = AlpacaTradingClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Get historical data
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            # Get account info
            account_info = await trading_client.get_account_info()
            
            # Both should work together
            assert len(data) == 10
            assert float(account_info["cash"]) > 0
    
    def test_alpaca_configuration_validation(self):
        """Test Alpaca configuration validation."""
        # Test invalid API keys
        with pytest.raises((ValueError, Exception)):
            AlpacaProvider(
                api_key="",
                secret_key="test_secret",
                paper=True
            )
        
        with pytest.raises((ValueError, Exception)):
            AlpacaProvider(
                api_key="test_key",
                secret_key="",
                paper=True
            )
    
    def test_alpaca_market_hours_handling(self):
        """Test market hours handling."""
        # Test requesting data during market hours vs. after hours
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock empty response for after-hours
            mock_response = Mock()
            mock_response.__iter__ = lambda self: iter([])
            mock_instance.get_stock_bars.return_value = mock_response
            
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Request data for a weekend (should handle gracefully)
            weekend_start = datetime(2023, 7, 1, 10, 0)  # Saturday
            weekend_end = datetime(2023, 7, 2, 16, 0)    # Sunday
            
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=weekend_start,
                end_date=weekend_end
            )
            
            # Should return empty data gracefully
            assert data == []


class TestAlpacaErrorHandlingValidation:
    """Test error handling in Alpaca integration."""
    
    def test_network_error_handling(self):
        """Test network error handling."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Simulate network error
            import requests
            mock_instance.get_stock_bars.side_effect = requests.exceptions.ConnectionError("Network error")
            
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Should handle network errors gracefully
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            assert data == []
    
    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Simulate authentication error
            mock_instance.get_stock_bars.side_effect = Exception("Authentication failed")
            
            provider = AlpacaProvider(
                api_key="invalid_key",
                secret_key="invalid_secret",
                paper=True
            )
            
            # Should handle auth errors gracefully
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            assert data == []
    
    def test_rate_limit_error_handling(self):
        """Test rate limit error handling."""
        with patch('alpaca.data.StockHistoricalDataClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Simulate rate limit error
            mock_instance.get_stock_bars.side_effect = Exception("Rate limit exceeded")
            
            provider = AlpacaProvider(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            
            # Should handle rate limit errors gracefully
            data = provider.get_historical_data(
                symbol="AAPL",
                timeframe=TimeFrame.HOURLY,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            assert data == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])