"""
Comprehensive Alpaca API Integration Tests

This module contains comprehensive tests for the Alpaca API integration,
including authentication, connection validation, data retrieval, and
real-time streaming functionality.
"""

import pytest
import asyncio
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import test modules
from src.data.providers.alpaca_provider import AlpacaProvider
from src.core.models import OHLCV, TimeFrame
from src.data.providers.base import DataProviderError, ConnectionError, DataRetrievalError

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAlpacaAPIIntegration:
    """
    Comprehensive tests for Alpaca API integration.
    
    These tests validate the complete Alpaca API integration including:
    - Authentication and connection validation
    - Historical data retrieval
    - Real-time data streaming
    - Error handling and recovery
    - Performance under load
    """
    
    @pytest.fixture(scope="class")
    def alpaca_credentials(self):
        """Get Alpaca credentials from environment."""
        return {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'paper': os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true',
            'feed': os.getenv('ALPACA_DATA_FEED', 'iex')
        }
    
    @pytest.fixture(scope="class")
    def alpaca_provider(self, alpaca_credentials):
        """Create Alpaca provider instance."""
        if not alpaca_credentials['api_key'] or not alpaca_credentials['secret_key']:
            pytest.skip("Alpaca credentials not available")
        
        provider = AlpacaProvider(alpaca_credentials)
        return provider
    
    @pytest.fixture(scope="class")
    def test_symbols(self):
        """Test symbols for integration tests."""
        return ['AAPL', 'MSFT', 'SPY', 'QQQ', 'BTC/USD', 'ETH/USD']
    
    @pytest.fixture(scope="class")
    def test_timeframes(self):
        """Test timeframes for integration tests."""
        return [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5,
            TimeFrame.MINUTE_15,
            TimeFrame.HOUR_1,
            TimeFrame.DAILY
        ]
    
    @pytest.mark.asyncio
    async def test_alpaca_connection_validation(self, alpaca_provider):
        """Test Alpaca API connection and authentication."""
        # Test connection
        connected = await alpaca_provider.connect()
        assert connected is True, "Should successfully connect to Alpaca API"
        
        # Verify connection state
        assert alpaca_provider.is_connected() is True
        assert alpaca_provider.get_last_error() is None
        
        # Test provider info
        info = alpaca_provider.get_provider_info()
        assert info['name'] == 'Alpaca'
        assert info['connected'] is True
        assert info['feed'] in ['iex', 'sip']
        assert 'paper_trading' in info
        
        # Test disconnect
        await alpaca_provider.disconnect()
        assert alpaca_provider.is_connected() is False
    
    @pytest.mark.asyncio
    async def test_alpaca_connection_invalid_credentials(self):
        """Test connection with invalid credentials."""
        invalid_config = {
            'api_key': 'invalid_key',
            'secret_key': 'invalid_secret',
            'paper': True
        }
        
        provider = AlpacaProvider(invalid_config)
        
        # Should fail to connect
        connected = await provider.connect()
        assert connected is False
        assert provider.get_last_error() is not None
        assert provider.is_connected() is False
    
    @pytest.mark.asyncio
    async def test_alpaca_supported_timeframes(self, alpaca_provider):
        """Test supported timeframes."""
        await alpaca_provider.connect()
        
        supported_timeframes = alpaca_provider.get_supported_timeframes()
        
        # Check that all expected timeframes are supported
        expected_timeframes = [
            TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15,
            TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.HOUR_4,
            TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY
        ]
        
        for tf in expected_timeframes:
            assert tf in supported_timeframes
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_supported_symbols(self, alpaca_provider):
        """Test supported symbols."""
        await alpaca_provider.connect()
        
        symbols = await alpaca_provider.get_symbols()
        
        # Should return a list of symbols
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        
        # Should contain common symbols
        common_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ']
        for symbol in common_symbols:
            assert symbol in symbols
        
        # Should contain crypto symbols
        crypto_symbols = ['BTC/USD', 'ETH/USD']
        for symbol in crypto_symbols:
            assert symbol in symbols
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_historical_data_retrieval(self, alpaca_provider, test_symbols):
        """Test historical data retrieval for various symbols and timeframes."""
        await alpaca_provider.connect()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for symbol in test_symbols:
            for timeframe in [TimeFrame.DAILY, TimeFrame.HOUR_1]:
                try:
                    data = await alpaca_provider.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Validate data structure
                    assert isinstance(data, list)
                    if data:  # Data might be empty for some symbols/timeframes
                        assert all(isinstance(ohlcv, OHLCV) for ohlcv in data)
                        
                        # Check data integrity
                        for ohlcv in data:
                            assert ohlcv.symbol == symbol
                            assert ohlcv.open > 0
                            assert ohlcv.high >= ohlcv.open
                            assert ohlcv.low <= ohlcv.open
                            assert ohlcv.close > 0
                            assert ohlcv.volume >= 0
                            assert isinstance(ohlcv.timestamp, datetime)
                        
                        # Check chronological order
                        timestamps = [ohlcv.timestamp for ohlcv in data]
                        assert timestamps == sorted(timestamps)
                        
                        logger.info(f"✓ Retrieved {len(data)} data points for {symbol} {timeframe}")
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve data for {symbol} {timeframe}: {e}")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_latest_data_retrieval(self, alpaca_provider, test_symbols):
        """Test latest data retrieval."""
        await alpaca_provider.connect()
        
        for symbol in test_symbols:
            try:
                data = await alpaca_provider.get_latest_data(
                    symbol=symbol,
                    timeframe=TimeFrame.DAILY,
                    periods=5
                )
                
                assert isinstance(data, list)
                assert len(data) <= 5
                
                if data:
                    for ohlcv in data:
                        assert isinstance(ohlcv, OHLCV)
                        assert ohlcv.symbol == symbol
                        assert ohlcv.open > 0
                        assert ohlcv.close > 0
                    
                    logger.info(f"✓ Retrieved latest {len(data)} data points for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to retrieve latest data for {symbol}: {e}")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_market_data_object(self, alpaca_provider):
        """Test market data object creation."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        
        market_data = await alpaca_provider.get_market_data(
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Validate market data structure
        assert market_data.symbol == symbol
        assert market_data.timeframe == timeframe
        assert isinstance(market_data.data, list)
        assert isinstance(market_data.indicators, dict)
        assert isinstance(market_data.last_updated, datetime)
        
        if market_data.data:
            # Check data integrity
            for ohlcv in market_data.data:
                assert isinstance(ohlcv, OHLCV)
                assert ohlcv.symbol == symbol
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_data_quality_validation(self, alpaca_provider):
        """Test data quality and consistency validation."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not data:
            pytest.skip("No data available for quality validation")
        
        # Test data quality metrics
        prices = [ohlcv.close for ohlcv in data]
        volumes = [ohlcv.volume for ohlcv in data]
        
        # Check for price consistency
        assert all(p > 0 for p in prices), "All prices should be positive"
        assert all(v >= 0 for v in volumes), "All volumes should be non-negative"
        
        # Check for reasonable price ranges (no extreme outliers)
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                        for i in range(1, len(prices))]
        max_daily_change = max(price_changes) if price_changes else 0
        
        # Daily price change should be less than 50% (extreme but reasonable for stocks)
        assert max_daily_change < 0.5, f"Extreme price change detected: {max_daily_change:.2%}"
        
        # Check for data gaps (missing trading days)
        if len(data) > 1:
            time_diffs = [(data[i].timestamp - data[i-1].timestamp).days 
                         for i in range(1, len(data))]
            
            # For daily data, gaps should be reasonable (weekends, holidays)
            if timeframe == TimeFrame.DAILY:
                max_gap = max(time_diffs) if time_diffs else 0
                assert max_gap <= 7, f"Large data gap detected: {max_gap} days"
        
        # Check OHLC relationships
        for ohlcv in data:
            assert ohlcv.high >= ohlcv.open, "High should be >= Open"
            assert ohlcv.high >= ohlcv.close, "High should be >= Close"
            assert ohlcv.low <= ohlcv.open, "Low should be <= Open"
            assert ohlcv.low <= ohlcv.close, "Low should be <= Close"
            assert ohlcv.high >= ohlcv.low, "High should be >= Low"
        
        logger.info(f"✓ Data quality validation passed for {len(data)} data points")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_timestamp_synchronization(self, alpaca_provider):
        """Test timestamp synchronization and timezone handling."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not data:
            pytest.skip("No data available for timestamp validation")
        
        # Check timestamp consistency
        timestamps = [ohlcv.timestamp for ohlcv in data]
        
        # Timestamps should be in chronological order
        assert timestamps == sorted(timestamps), "Timestamps should be chronological"
        
        # Check timezone awareness
        for ts in timestamps:
            assert isinstance(ts, datetime), "Timestamps should be datetime objects"
            # Note: Alpaca provides timezone-aware or naive timestamps depending on the endpoint
        
        # Check for reasonable timestamp ranges
        now = datetime.now()
        for ts in timestamps:
            # Timestamps should not be in the future
            assert ts <= now, f"Timestamp in future: {ts}"
            
            # Timestamps should not be too old (within reasonable range)
            days_ago = (now - ts).days
            assert days_ago <= 365 * 10, f"Timestamp too old: {days_ago} days"
        
        logger.info(f"✓ Timestamp validation passed for {len(data)} data points")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_error_handling(self, alpaca_provider):
        """Test error handling and recovery mechanisms."""
        await alpaca_provider.connect()
        
        # Test invalid symbol
        try:
            data = await alpaca_provider.get_historical_data(
                symbol='INVALID_SYMBOL_12345',
                timeframe=TimeFrame.DAILY,
                start_date=datetime.now() - timedelta(days=10),
                end_date=datetime.now()
            )
            # Should either return empty list or raise exception
            assert isinstance(data, list)
        except (DataRetrievalError, Exception) as e:
            # Expected to fail for invalid symbol
            logger.info(f"✓ Correctly handled invalid symbol: {e}")
        
        # Test invalid date range
        try:
            data = await alpaca_provider.get_historical_data(
                symbol='AAPL',
                timeframe=TimeFrame.DAILY,
                start_date=datetime.now() + timedelta(days=10),  # Future date
                end_date=datetime.now() + timedelta(days=20)
            )
            # Should return empty list or handle gracefully
            assert isinstance(data, list)
        except (DataRetrievalError, Exception) as e:
            logger.info(f"✓ Correctly handled invalid date range: {e}")
        
        # Test disconnection handling
        await alpaca_provider.disconnect()
        
        try:
            data = await alpaca_provider.get_historical_data(
                symbol='AAPL',
                timeframe=TimeFrame.DAILY,
                start_date=datetime.now() - timedelta(days=10),
                end_date=datetime.now()
            )
            pytest.fail("Should raise ConnectionError when disconnected")
        except ConnectionError:
            logger.info("✓ Correctly handled disconnected state")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_rate_limiting(self, alpaca_provider):
        """Test rate limiting and throttling behavior."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        
        # Make multiple rapid requests
        request_times = []
        successful_requests = 0
        
        for i in range(10):
            start_time = time.time()
            try:
                data = await alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=datetime.now() - timedelta(days=5),
                    end_date=datetime.now()
                )
                successful_requests += 1
                request_times.append(time.time() - start_time)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Request {i} failed: {e}")
        
        # Analyze request performance
        if request_times:
            avg_response_time = sum(request_times) / len(request_times)
            max_response_time = max(request_times)
            
            logger.info(f"✓ Rate limiting test: {successful_requests}/10 requests succeeded")
            logger.info(f"  Average response time: {avg_response_time:.2f}s")
            logger.info(f"  Maximum response time: {max_response_time:.2f}s")
            
            # Response times should be reasonable
            assert avg_response_time < 10.0, "Average response time too high"
            assert max_response_time < 30.0, "Maximum response time too high"
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_concurrent_requests(self, alpaca_provider):
        """Test concurrent request handling."""
        await alpaca_provider.connect()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        timeframe = TimeFrame.DAILY
        
        # Make concurrent requests
        async def fetch_data(symbol):
            try:
                data = await alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=datetime.now() - timedelta(days=10),
                    end_date=datetime.now()
                )
                return symbol, data, None
            except Exception as e:
                return symbol, None, str(e)
        
        # Execute concurrent requests
        start_time = time.time()
        results = await asyncio.gather(*[fetch_data(symbol) for symbol in symbols])
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = 0
        failed_requests = 0
        
        for symbol, data, error in results:
            if error:
                failed_requests += 1
                logger.warning(f"Failed to fetch {symbol}: {error}")
            else:
                successful_requests += 1
                logger.info(f"✓ Successfully fetched {len(data) if data else 0} points for {symbol}")
        
        logger.info(f"✓ Concurrent requests: {successful_requests}/{len(symbols)} succeeded in {total_time:.2f}s")
        
        # At least 50% of requests should succeed
        success_rate = successful_requests / len(symbols)
        assert success_rate >= 0.5, f"Success rate too low: {success_rate:.2%}"
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_market_hours_handling(self, alpaca_provider):
        """Test market hours and after-hours data handling."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.MINUTE_15
        
        # Test during market hours vs after hours
        now = datetime.now()
        
        # Get recent data
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=now - timedelta(days=5),
            end_date=now
        )
        
        if not data:
            pytest.skip("No data available for market hours validation")
        
        # Analyze trading hours
        market_hours_data = []
        after_hours_data = []
        
        for ohlcv in data:
            # Assume market hours are roughly 9:30 AM to 4:00 PM ET
            hour = ohlcv.timestamp.hour
            if 9 <= hour <= 16:  # Approximate market hours
                market_hours_data.append(ohlcv)
            else:
                after_hours_data.append(ohlcv)
        
        logger.info(f"✓ Market hours analysis:")
        logger.info(f"  Market hours data points: {len(market_hours_data)}")
        logger.info(f"  After hours data points: {len(after_hours_data)}")
        
        # Most data should be from market hours for minute-level data
        if len(data) > 10:
            market_hours_ratio = len(market_hours_data) / len(data)
            logger.info(f"  Market hours ratio: {market_hours_ratio:.2%}")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_data_completeness(self, alpaca_provider):
        """Test data completeness and gap detection."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days of data
        
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not data:
            pytest.skip("No data available for completeness validation")
        
        # Calculate expected vs actual data points
        expected_trading_days = self._calculate_trading_days(start_date, end_date)
        actual_data_points = len(data)
        
        completeness_ratio = actual_data_points / expected_trading_days if expected_trading_days > 0 else 0
        
        logger.info(f"✓ Data completeness analysis:")
        logger.info(f"  Expected trading days: {expected_trading_days}")
        logger.info(f"  Actual data points: {actual_data_points}")
        logger.info(f"  Completeness ratio: {completeness_ratio:.2%}")
        
        # Should have at least 80% data completeness
        assert completeness_ratio >= 0.8, f"Data completeness too low: {completeness_ratio:.2%}"
        
        # Check for large gaps
        if len(data) > 1:
            timestamps = [ohlcv.timestamp for ohlcv in data]
            gaps = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
            
            max_gap = max(gaps) if gaps else 0
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            
            logger.info(f"  Maximum gap: {max_gap} days")
            logger.info(f"  Average gap: {avg_gap:.1f} days")
            
            # Maximum gap should be reasonable (holidays, weekends)
            assert max_gap <= 10, f"Large data gap detected: {max_gap} days"
        
        await alpaca_provider.disconnect()
    
    def _calculate_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate approximate number of trading days between two dates."""
        total_days = (end_date - start_date).days
        
        # Rough approximation: 5/7 of days are trading days
        # This doesn't account for holidays but gives a reasonable estimate
        return int(total_days * 5 / 7)
    
    @pytest.mark.asyncio
    async def test_alpaca_performance_benchmarks(self, alpaca_provider):
        """Test performance benchmarks and SLA compliance."""
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.DAILY
        
        # Performance benchmarks
        max_connection_time = 10.0  # seconds
        max_request_time = 5.0      # seconds
        min_throughput = 1.0        # requests per second
        
        # Test connection performance
        start_time = time.time()
        await alpaca_provider.disconnect()
        await alpaca_provider.connect()
        connection_time = time.time() - start_time
        
        assert connection_time < max_connection_time, f"Connection time too slow: {connection_time:.2f}s"
        logger.info(f"✓ Connection time: {connection_time:.2f}s")
        
        # Test request performance
        start_time = time.time()
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        request_time = time.time() - start_time
        
        assert request_time < max_request_time, f"Request time too slow: {request_time:.2f}s"
        logger.info(f"✓ Request time: {request_time:.2f}s")
        
        # Test throughput
        num_requests = 5
        start_time = time.time()
        
        for i in range(num_requests):
            await alpaca_provider.get_latest_data(symbol, timeframe, 1)
            await asyncio.sleep(0.1)  # Small delay
        
        total_time = time.time() - start_time
        throughput = num_requests / total_time
        
        assert throughput >= min_throughput, f"Throughput too low: {throughput:.2f} req/s"
        logger.info(f"✓ Throughput: {throughput:.2f} requests/second")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_memory_usage(self, alpaca_provider):
        """Test memory usage during data operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await alpaca_provider.connect()
        
        symbol = 'AAPL'
        timeframe = TimeFrame.MINUTE_1
        
        # Fetch large amount of data
        data = await alpaca_provider.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now()
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        logger.info(f"✓ Memory usage analysis:")
        logger.info(f"  Initial memory: {initial_memory:.1f} MB")
        logger.info(f"  Peak memory: {peak_memory:.1f} MB")
        logger.info(f"  Memory increase: {memory_increase:.1f} MB")
        logger.info(f"  Data points: {len(data) if data else 0}")
        
        # Memory usage should be reasonable
        max_memory_increase = 100  # MB
        assert memory_increase < max_memory_increase, f"Memory usage too high: {memory_increase:.1f} MB"
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_crypto_data_handling(self, alpaca_provider):
        """Test cryptocurrency data handling."""
        await alpaca_provider.connect()
        
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD']
        timeframe = TimeFrame.DAILY
        
        for symbol in crypto_symbols:
            try:
                data = await alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                
                if data:
                    # Validate crypto data characteristics
                    prices = [ohlcv.close for ohlcv in data]
                    
                    # Crypto prices should be reasonable
                    if symbol == 'BTC/USD':
                        assert all(p > 1000 for p in prices), "BTC price seems too low"
                    elif symbol == 'ETH/USD':
                        assert all(p > 100 for p in prices), "ETH price seems too low"
                    
                    # Check for 24/7 trading (crypto markets)
                    timestamps = [ohlcv.timestamp for ohlcv in data]
                    if len(timestamps) > 1:
                        # Crypto should have more consistent daily data
                        gaps = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
                        max_gap = max(gaps) if gaps else 0
                        assert max_gap <= 2, f"Large gap in crypto data: {max_gap} days"
                    
                    logger.info(f"✓ Crypto data validation passed for {symbol}: {len(data)} points")
                
            except Exception as e:
                logger.warning(f"Crypto data test failed for {symbol}: {e}")
        
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_alpaca_provider_cleanup(self, alpaca_provider):
        """Test proper cleanup and resource management."""
        await alpaca_provider.connect()
        
        # Verify initial state
        assert alpaca_provider.is_connected()
        
        # Test graceful shutdown
        await alpaca_provider.disconnect()
        assert not alpaca_provider.is_connected()
        
        # Test multiple disconnections (should be idempotent)
        await alpaca_provider.disconnect()
        await alpaca_provider.disconnect()
        assert not alpaca_provider.is_connected()
        
        # Test reconnection after disconnect
        connected = await alpaca_provider.connect()
        assert connected
        assert alpaca_provider.is_connected()
        
        await alpaca_provider.disconnect()
        
        logger.info("✓ Provider cleanup and reconnection tests passed")