"""
Integration tests for RSI Trend Analyzer using real Alpaca API connections.

These tests verify that the RSI Trend Analyzer works with live market data from Alpaca.
NO MOCKING - All tests use real API calls and actual market data.
"""

import pytest
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the RSI Trend Analyzer
from src.strategies.rsi_trend_analyzer import RSITrendAnalyzer


class TestRSITrendAnalyzerIntegration:
    """Integration tests for RSI Trend Analyzer with real API calls."""
    
    @pytest.fixture
    def rsi_analyzer(self):
        """Create RSI Trend Analyzer with real credentials from environment."""
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
        
        return RSITrendAnalyzer(config)
    
    @pytest.fixture
    def test_symbols(self):
        """Get test symbols from environment variable."""
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,MSFT,GOOGL')
        return symbols_str.split(',')[:5]  # Use first 5 symbols for faster tests
    
    @pytest.fixture
    def all_symbols(self):
        """Get all symbols from environment variable."""
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,MSFT,GOOGL')
        return [s.strip() for s in symbols_str.split(',') if s.strip()]
    
    @pytest.mark.asyncio
    async def test_ytd_signal_generation(self, rsi_analyzer):
        """Test YTD (Year-to-Date) signal generation with real market data."""
        # Calculate YTD date range
        current_date = datetime.now()
        ytd_start = datetime(current_date.year, 1, 1)
        ytd_end = current_date
        
        # Test symbols for YTD analysis
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            print(f"\nTesting YTD RSI Trend signals for {symbol} from {ytd_start.date()} to {ytd_end.date()}")
            
            signals = await rsi_analyzer.analyze_symbol(
                symbol=symbol,
                start_date=ytd_start,
                end_date=ytd_end,
                timeframe="1d"
            )
            
            # Validate results
            assert isinstance(signals, list)
            print(f"Generated {len(signals)} signals for {symbol}")
            
            # Validate signal structure and quality
            for signal in signals:
                assert "timestamp" in signal
                assert "symbol" in signal
                assert "signal_type" in signal
                assert "price" in signal
                assert "confidence" in signal
                assert "strategy_name" in signal
                assert "metadata" in signal
                
                # Validate signal content
                assert signal["symbol"] == symbol
                assert signal["strategy_name"] == "RSITrendFollowing"
                assert signal["signal_type"] in ["BUY", "SELL"]
                assert 0 <= signal["confidence"] <= 1
                assert signal["price"] > 0
                
                # Validate timestamp is within YTD range
                signal_timestamp = signal["timestamp"]
                if isinstance(signal_timestamp, str):
                    signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
                
                # Handle timezone-aware/naive comparison
                if hasattr(signal_timestamp, 'tzinfo') and signal_timestamp.tzinfo is not None:
                    # Signal timestamp is timezone-aware, convert YTD dates to aware
                    import pytz
                    ytd_start_aware = pytz.utc.localize(ytd_start) if ytd_start.tzinfo is None else ytd_start
                    ytd_end_aware = pytz.utc.localize(ytd_end) if ytd_end.tzinfo is None else ytd_end
                    assert ytd_start_aware <= signal_timestamp <= ytd_end_aware
                else:
                    # Signal timestamp is timezone-naive
                    assert ytd_start <= signal_timestamp <= ytd_end
                
                # Validate metadata
                metadata = signal["metadata"]
                assert isinstance(metadata, dict)
                assert "signal_name" in metadata
                
                # Print signal details for verification
                print(f"  Signal: {signal['signal_type']} at {signal['price']:.2f} "
                      f"(confidence: {signal['confidence']:.2f}) - {metadata['signal_name']}")

    @pytest.mark.asyncio
    async def test_analyze_symbol_daily_timeframe(self, rsi_analyzer, test_symbols):
        """Test RSI trend analysis with daily timeframe."""
        symbol = test_symbols[0]  # Use first symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)  # Use proper date range for RSI Trend
        
        # Analyze symbol for RSI trend signals
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )
        
        # Verify we got valid results (may be empty, that's OK)
        assert isinstance(signals, list)
        print(f"Generated {len(signals)} signals for {symbol} (daily)")
        
        # If we have signals, verify their structure
        for signal in signals:
            assert isinstance(signal, dict)
            required_fields = ['timestamp', 'symbol', 'signal_type', 'price', 'confidence', 'strategy_name']
            for field in required_fields:
                assert field in signal, f"Missing field {field} in signal"
            
            # Verify signal values
            assert signal['symbol'] == symbol
            assert signal['signal_type'] in ['BUY', 'SELL']
            assert signal['strategy_name'] == 'RSITrendFollowing'
            assert 0 <= signal['confidence'] <= 1
            assert signal['price'] > 0
            assert isinstance(signal['timestamp'], datetime)
    
    @pytest.mark.asyncio
    async def test_analyze_symbol_hourly_timeframe(self, rsi_analyzer, test_symbols):
        """Test RSI trend analysis with hourly timeframe."""
        symbol = test_symbols[0]  # Use first symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)  # 2 weeks of hourly data
        
        # Analyze symbol for RSI trend signals
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1h'
        )
        
        # Verify we got valid results (may be empty, that's OK)
        assert isinstance(signals, list)
        print(f"Generated {len(signals)} signals for {symbol} (hourly)")
        
        # If we have signals, verify their structure
        for signal in signals:
            assert isinstance(signal, dict)
            required_fields = ['timestamp', 'symbol', 'signal_type', 'price', 'confidence', 'strategy_name']
            for field in required_fields:
                assert field in signal, f"Missing field {field} in signal"
            
            # Verify signal values
            assert signal['symbol'] == symbol
            assert signal['signal_type'] in ['BUY', 'SELL']
            assert signal['strategy_name'] == 'RSITrendFollowing'
            assert 0 <= signal['confidence'] <= 1
            assert signal['price'] > 0
            assert isinstance(signal['timestamp'], datetime)
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_symbols_daily(self, rsi_analyzer, test_symbols):
        """Test RSI trend analysis for multiple symbols with daily data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1 month of data
        
        results = {}
        
        for symbol in test_symbols[:3]:  # Test first 3 symbols
            print(f"Testing RSI trend analysis for {symbol}")
            
            signals = await rsi_analyzer.analyze_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1d'
            )
            
            results[symbol] = signals
            
            # Verify results
            assert isinstance(signals, list)
            print(f"  {symbol}: {len(signals)} signals")
            
            # Verify signal structure if any signals found
            for signal in signals:
                assert signal['symbol'] == symbol
                assert signal['signal_type'] in ['BUY', 'SELL']
                assert 0 <= signal['confidence'] <= 1
                assert signal['price'] > 0
        
        print(f"Total signals across {len(test_symbols[:3])} symbols: {sum(len(s) for s in results.values())}")
    
    @pytest.mark.asyncio
    async def test_all_supported_symbols_analysis(self, rsi_analyzer, all_symbols):
        """Test RSI trend analysis for all symbols from .env file."""
        if not all_symbols:
            pytest.skip("No SUPPORTED_SYMBOLS found in environment")
        
        print(f"Testing RSI trend analysis for {len(all_symbols)} symbols from environment")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1 month of data
        
        successful_analyses = []
        failed_analyses = []
        total_signals = 0
        
        # Test each symbol (limit to first 10 for speed)
        for symbol in all_symbols[:10]:
            try:
                print(f"Analyzing {symbol} for RSI trend signals...")
                
                signals = await rsi_analyzer.analyze_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1d'
                )
                
                if isinstance(signals, list):
                    successful_analyses.append(symbol)
                    total_signals += len(signals)
                    print(f"  ✓ {symbol}: {len(signals)} signals")
                else:
                    failed_analyses.append(symbol)
                    print(f"  ✗ {symbol}: Invalid response type")
                
            except Exception as e:
                failed_analyses.append(symbol)
                print(f"  ✗ {symbol}: Error - {str(e)}")
        
        print(f"\\nResults: {len(successful_analyses)} successful, {len(failed_analyses)} failed")
        print(f"Total signals generated: {total_signals}")
        
        # We should have at least some successful analyses
        assert len(successful_analyses) > 0, f"No symbols worked. Failed: {failed_analyses}"
        
        # Most symbols should work (allow some failures for delisted/invalid symbols)
        success_rate = len(successful_analyses) / (len(successful_analyses) + len(failed_analyses))
        assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_input_validation_error_handling(self, rsi_analyzer):
        """Test input validation and error handling."""
        
        # Test invalid symbol
        signals = await rsi_analyzer.analyze_symbol(
            symbol="",  # Empty symbol
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            timeframe='1d'
        )
        assert signals == []
        
        # Test invalid date range
        signals = await rsi_analyzer.analyze_symbol(
            symbol="AAPL",
            start_date=datetime.now(),  # Start after end
            end_date=datetime.now() - timedelta(days=30),
            timeframe='1d'
        )
        assert signals == []
        
        # Test invalid timeframe
        signals = await rsi_analyzer.analyze_symbol(
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            timeframe='5m'  # Not supported
        )
        assert signals == []
        
        # Test insufficient data range
        signals = await rsi_analyzer.analyze_symbol(
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=2),  # Too short
            end_date=datetime.now(),
            timeframe='1d'
        )
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_custom_strategy_parameters(self, rsi_analyzer, test_symbols):
        """Test RSI trend analysis with custom strategy parameters."""
        symbol = test_symbols[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        # Test with custom parameters
        custom_params = {
            'rsi_period': 21,
            'min_confidence': 0.45,
            'uptrend_pullback_low': 35,
            'uptrend_pullback_high': 45,
            'trend_confirmation_periods': 15
        }
        
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            strategy_params=custom_params
        )
        
        # Verify we got valid results
        assert isinstance(signals, list)
        print(f"Generated {len(signals)} signals for {symbol} with custom parameters")
        
        # If we have signals, verify they meet custom confidence threshold
        for signal in signals:
            assert signal['confidence'] >= custom_params['min_confidence']
    
    @pytest.mark.asyncio
    async def test_signal_metadata_validation(self, rsi_analyzer, test_symbols):
        """Test that generated signals contain proper metadata."""
        symbol = test_symbols[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months for better signal chance
        
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )
        
        # Verify we got valid results
        assert isinstance(signals, list)
        
        # If we have signals, verify their metadata
        for signal in signals:
            assert 'metadata' in signal
            metadata = signal['metadata']
            
            # Check required metadata fields
            assert 'signal_name' in metadata
            assert 'entry_type' in metadata
            
            # Verify signal name is valid
            valid_signal_names = [
                'RSI Uptrend Pullback',
                'RSI Downtrend Rally',
                'RSI Bullish Divergence',
                'RSI Bearish Divergence'
            ]
            assert metadata['signal_name'] in valid_signal_names
            
            # Verify entry type is valid
            valid_entry_types = ['pullback', 'rally', 'divergence']
            assert metadata['entry_type'] in valid_entry_types
    
    @pytest.mark.asyncio
    async def test_timeframe_specific_validation(self, rsi_analyzer, test_symbols):
        """Test validation specific to different timeframes."""
        symbol = test_symbols[0]
        
        # Test hourly timeframe with date range too far in past
        old_end_date = datetime.now() - timedelta(days=60)
        old_start_date = old_end_date - timedelta(days=14)
        
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=old_start_date,
            end_date=old_end_date,
            timeframe='1h'  # Should fail for old dates
        )
        assert signals == []
        
        # Test daily timeframe with excessive range
        far_end_date = datetime.now()
        far_start_date = far_end_date - timedelta(days=800)  # Over 2 years
        
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=far_start_date,
            end_date=far_end_date,
            timeframe='1d'  # Should fail for excessive range
        )
        assert signals == []
    
    def test_supported_symbols_from_env(self, rsi_analyzer):
        """Test that analyzer returns symbols from environment."""
        symbols = rsi_analyzer.get_supported_symbols()
        
        assert isinstance(symbols, list)
        
        # Should match environment variable
        env_symbols = os.getenv('SUPPORTED_SYMBOLS', '')
        if env_symbols:
            expected_symbols = [s.strip() for s in env_symbols.split(',') if s.strip()]
            assert symbols == expected_symbols
    
    def test_supported_timeframes(self, rsi_analyzer):
        """Test that analyzer returns correct supported timeframes."""
        timeframes = rsi_analyzer.get_supported_timeframes()
        
        assert isinstance(timeframes, list)
        assert '1h' in timeframes
        assert '1d' in timeframes
        assert len(timeframes) == 2
    
    @pytest.mark.asyncio
    async def test_signal_chronological_order(self, rsi_analyzer, test_symbols):
        """Test that generated signals are in descending chronological order (newest first)."""
        symbol = test_symbols[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months
        
        signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )
        
        # If we have multiple signals, verify they're in descending chronological order
        if len(signals) > 1:
            timestamps = [signal['timestamp'] for signal in signals]
            assert timestamps == sorted(timestamps, reverse=True), "Signals should be in descending chronological order (newest first)"
    
    @pytest.mark.asyncio
    async def test_different_confidence_thresholds(self, rsi_analyzer, test_symbols):
        """Test RSI trend analysis with different confidence thresholds."""
        symbol = test_symbols[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Test with low confidence threshold
        low_conf_signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            strategy_params={'min_confidence': 0.3}
        )
        
        # Test with high confidence threshold
        high_conf_signals = await rsi_analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            strategy_params={'min_confidence': 0.8}
        )
        
        # Low confidence should have >= high confidence signals
        assert len(low_conf_signals) >= len(high_conf_signals)
        
        # Verify confidence thresholds
        for signal in low_conf_signals:
            assert signal['confidence'] >= 0.3
        
        for signal in high_conf_signals:
            assert signal['confidence'] >= 0.8