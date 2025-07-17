"""
Integration tests for AlpacaProvider using real Alpaca API connections.

These tests verify that the AlpacaProvider works with live market data from Alpaca.
NO MOCKING - All tests use real API calls and actual market data.
"""

import pytest
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

# Load environment at module level
load_env_file()

# Import the AlpacaProvider
from src.data.providers.alpaca_provider import AlpacaProvider


class TestAlpacaProviderIntegration:
    """Integration tests for Alpaca data provider with real API calls."""
    
    @pytest.fixture
    def alpaca_provider(self):
        """Create AlpacaProvider with real credentials from environment."""
        # These must be real credentials - no mocking allowed
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Skip tests if credentials not provided
        if not api_key or not secret_key:
            pytest.skip("Alpaca API credentials not found in environment")
        
        config = {
            'api_key': api_key,
            'secret_key': secret_key
        }
        
        return AlpacaProvider(config)
    
    @pytest.fixture
    def test_symbols(self):
        """Get test symbols from environment variable."""
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,MSFT,GOOGL')
        return symbols_str.split(',')[:5]  # Use first 5 symbols for faster tests
    
    @pytest.mark.asyncio
    async def test_alpaca_connection_with_real_credentials(self, alpaca_provider):
        """Test real connection to Alpaca Markets API."""
        # This makes an actual API call - no mocking
        connected = await alpaca_provider.connect()
        assert connected is True
        assert alpaca_provider.is_connected() is True
        
        # Clean up
        await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_historical_data_1d_timeframe(self, alpaca_provider, test_symbols):
        """Test retrieving real historical data with 1-day timeframe."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]  # Use first symbol
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Test 1-day timeframe with real API call
            data = await alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe='1d',
                start_date=start_date,
                end_date=end_date
            )
            
            # Verify we got real data
            assert isinstance(data, list)
            assert len(data) > 0
            print(f"Retrieved {len(data)} daily bars for {symbol}")
            
            # Verify data structure
            first_bar = data[0]
            assert isinstance(first_bar, dict)
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                assert field in first_bar
            
            # Verify data values are reasonable
            assert first_bar['open'] > 0
            assert first_bar['high'] >= first_bar['open']
            assert first_bar['low'] <= first_bar['open']
            assert first_bar['close'] > 0
            assert first_bar['volume'] >= 0
            
            # Verify timestamps are in order
            if len(data) > 1:
                assert data[0]['timestamp'] < data[-1]['timestamp']
            
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_historical_data_1h_timeframe(self, alpaca_provider, test_symbols):
        """Test retrieving real historical data with 1-hour timeframe."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]  # Use first symbol
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 1 week of hourly data
            
            # Test 1-hour timeframe with real API call
            data = await alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                start_date=start_date,
                end_date=end_date
            )
            
            # Verify we got real data
            assert isinstance(data, list)
            assert len(data) > 0
            print(f"Retrieved {len(data)} hourly bars for {symbol}")
            
            # Verify data structure
            first_bar = data[0]
            assert isinstance(first_bar, dict)
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                assert field in first_bar
            
            # Verify data values are reasonable
            assert first_bar['open'] > 0
            assert first_bar['high'] >= first_bar['low']
            assert first_bar['close'] > 0
            assert first_bar['volume'] >= 0
            
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_latest_data_retrieval(self, alpaca_provider, test_symbols):
        """Test retrieving latest real market data."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]
            
            # Get latest 5 periods of daily data
            data = await alpaca_provider.get_latest_data(
                symbol=symbol,
                timeframe='1d',
                periods=5
            )
            
            # Verify we got real recent data
            assert isinstance(data, list)
            assert len(data) <= 5
            assert len(data) > 0
            print(f"Retrieved {len(data)} latest daily bars for {symbol}")
            
            if len(data) > 0:
                latest_bar = data[-1]
                assert isinstance(latest_bar, dict)
                
                # Verify timestamp is recent (within last month)
                timestamp = latest_bar['timestamp']
                time_diff = datetime.now() - timestamp.replace(tzinfo=None)
                assert time_diff.days <= 30
                
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_symbols_real_data(self, alpaca_provider, test_symbols):
        """Test retrieving real data for multiple symbols."""
        await alpaca_provider.connect()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
            # Test with multiple real symbols
            for symbol in test_symbols[:3]:  # Test first 3 symbols
                print(f"Testing symbol: {symbol}")
                
                data = await alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe='1d',
                    start_date=start_date,
                    end_date=end_date
                )
                
                assert isinstance(data, list)
                assert len(data) > 0
                
                # Each symbol should have valid data
                for bar in data:
                    assert bar['open'] > 0
                    assert bar['high'] >= bar['low']
                    assert bar['volume'] >= 0
                    assert isinstance(bar['timestamp'], datetime)
                    
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio 
    async def test_all_supported_symbols_connection(self, alpaca_provider):
        """Test that all symbols from .env file can be processed."""
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', '')
        if not symbols_str:
            pytest.skip("No SUPPORTED_SYMBOLS found in environment")
        
        all_symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        print(f"Testing {len(all_symbols)} symbols from environment")
        
        await alpaca_provider.connect()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            successful_symbols = []
            failed_symbols = []
            
            # Test each symbol (limit to first 10 for speed)
            for symbol in all_symbols[:10]:
                try:
                    print(f"Testing symbol: {symbol}")
                    
                    data = await alpaca_provider.get_historical_data(
                        symbol=symbol,
                        timeframe='1d',
                        start_date=start_date,
                        end_date=end_date,
                        limit=5  # Just get a few bars to test
                    )
                    
                    if data and len(data) > 0:
                        successful_symbols.append(symbol)
                        print(f"✓ {symbol}: {len(data)} bars retrieved")
                    else:
                        failed_symbols.append(symbol)
                        print(f"✗ {symbol}: No data returned")
                    
                except Exception as e:
                    failed_symbols.append(symbol)
                    print(f"✗ {symbol}: Error - {str(e)}")
            
            print(f"Results: {len(successful_symbols)} successful, {len(failed_symbols)} failed")
            
            # We should have at least some successful symbols
            assert len(successful_symbols) > 0, f"No symbols worked. Failed: {failed_symbols}"
            
            # Most symbols should work (allow some failures for delisted/invalid symbols)
            success_rate = len(successful_symbols) / (len(successful_symbols) + len(failed_symbols))
            assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2%}"
            
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_streaming_functionality(self, alpaca_provider, test_symbols):
        """Test real-time streaming functionality."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]
            received_data = []
            
            # Set up callback to collect streamed data
            async def on_bar_received(bar_data):
                received_data.append(bar_data)
                print(f"Received bar for {bar_data['symbol']}: ${bar_data['close']}")
            
            # Start streaming
            await alpaca_provider.start_streaming(symbol, on_bar_received)
            assert alpaca_provider.is_streaming() is True
            
            # Wait a bit to see if we receive any data
            # Note: During market hours, we should receive data. After hours, we might not.
            print(f"Streaming {symbol} for 10 seconds...")
            await asyncio.sleep(10)
            
            # Stop streaming
            await alpaca_provider.stop_streaming()
            assert alpaca_provider.is_streaming() is False
            
            # During market hours, we should receive some data
            # After hours, we might not receive anything, which is expected
            print(f"Received {len(received_data)} bars during streaming test")
            
            # If we received data, verify its structure
            if received_data:
                bar = received_data[0]
                required_fields = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                for field in required_fields:
                    assert field in bar
                assert bar['symbol'] == symbol
            
        finally:
            # Ensure streaming is stopped
            if alpaca_provider.is_streaming():
                await alpaca_provider.stop_streaming()
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_symbol(self, alpaca_provider):
        """Test error handling with invalid symbol."""
        await alpaca_provider.connect()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Try to get data for an invalid symbol
            with pytest.raises(Exception):  # Should raise some kind of error
                await alpaca_provider.get_historical_data(
                    symbol="INVALID_SYMBOL_12345",
                    timeframe='1d',
                    start_date=start_date,
                    end_date=end_date
                )
                
        finally:
            await alpaca_provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_timeframe(self, alpaca_provider, test_symbols):
        """Test error handling with invalid timeframe."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Try to get data with invalid timeframe
            with pytest.raises(ValueError):  # Should raise ValueError
                await alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe='5m',  # Not supported
                    start_date=start_date,
                    end_date=end_date
                )
                
        finally:
            await alpaca_provider.disconnect()
    
    def test_supported_timeframes(self, alpaca_provider):
        """Test that provider returns correct supported timeframes."""
        timeframes = alpaca_provider.get_supported_timeframes()
        
        assert isinstance(timeframes, list)
        assert '1h' in timeframes
        assert '1d' in timeframes
        assert len(timeframes) == 2
    
    def test_supported_symbols_from_env(self, alpaca_provider):
        """Test that provider returns symbols from environment."""
        symbols = alpaca_provider.get_supported_symbols()
        
        assert isinstance(symbols, list)
        
        # Should match environment variable
        env_symbols = os.getenv('SUPPORTED_SYMBOLS', '')
        if env_symbols:
            expected_symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
            assert symbols == expected_symbols
    
    @pytest.mark.asyncio
    async def test_market_data_wrapper(self, alpaca_provider, test_symbols):
        """Test the get_market_data wrapper method."""
        await alpaca_provider.connect()
        
        try:
            symbol = test_symbols[0]
            
            market_data = await alpaca_provider.get_market_data(
                symbol=symbol,
                timeframe='1d'
            )
            
            # Verify structure
            assert isinstance(market_data, dict)
            assert market_data['symbol'] == symbol
            assert market_data['timeframe'] == '1d'
            assert 'data' in market_data
            assert 'last_updated' in market_data
            assert 'start_date' in market_data
            assert 'end_date' in market_data
            
            # Verify data
            data = market_data['data']
            assert isinstance(data, list)
            assert len(data) > 0
            
        finally:
            await alpaca_provider.disconnect()