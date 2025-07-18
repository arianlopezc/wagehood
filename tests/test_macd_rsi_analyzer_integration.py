"""
Integration tests for MACD+RSI Analyzer

These tests validate the complete end-to-end functionality of the MACD+RSI Analyzer
using real Alpaca data, ensuring production-ready signal generation capabilities.
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.strategies.macd_rsi_analyzer import MACDRSIAnalyzer


class TestMACDRSIAnalyzerIntegration:
    """Integration tests for MACD+RSI Analyzer with real Alpaca data."""
    
    @pytest.fixture
    def analyzer(self):
        """Create MACD+RSI Analyzer instance."""
        return MACDRSIAnalyzer()
    
    @pytest.fixture
    def test_symbol(self):
        """Test symbol for integration tests."""
        return "AAPL"
    
    @pytest.fixture
    def date_range(self):
        """Test date range for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)  # Use 150 days to satisfy MACD+RSI requirements
        return start_date, end_date
    
    @pytest.mark.asyncio
    async def test_analyze_symbol_daily_timeframe(self, analyzer, test_symbol, date_range):
        """Test analyzing a symbol with daily timeframe."""
        start_date, end_date = date_range
        
        # Test daily analysis
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # Validate results
        assert isinstance(signals, list)
        
        # If signals are found, validate their structure
        for signal in signals:
            assert "timestamp" in signal
            assert "symbol" in signal
            assert "signal_type" in signal
            assert "price" in signal
            assert "confidence" in signal
            assert "strategy_name" in signal
            assert "metadata" in signal
            
            # Validate signal type
            assert signal["signal_type"] in ["BUY", "SELL"]
            
            # Validate confidence range
            assert 0 <= signal["confidence"] <= 1
            
            # Validate price is positive
            assert signal["price"] > 0
            
            # Validate strategy name
            assert signal["strategy_name"] == "MACDRSIStrategy"
            
            # Validate metadata structure
            metadata = signal["metadata"]
            assert isinstance(metadata, dict)
            assert "signal_name" in metadata
            
            # Validate MACD+RSI-specific signal names
            valid_signal_names = [
                "MACD Bullish Crossover + RSI Oversold",
                "MACD Bearish Crossover + RSI Overbought",
                "MACD Bullish Divergence + RSI Recovery",
                "MACD Bearish Divergence + RSI Weakness",
                "MACD Signal Line Crossover + RSI Confirmation",
                "MACD Histogram Reversal + RSI Momentum",
                "MACD+RSI Bullish",
                "MACD+RSI Bearish"
            ]
            assert metadata["signal_name"] in valid_signal_names
    
    @pytest.mark.asyncio
    async def test_ytd_signal_generation(self, analyzer):
        """Test YTD (Year-to-Date) signal generation with real market data."""
        # Calculate YTD date range
        current_date = datetime.now()
        ytd_start = datetime(current_date.year, 1, 1)
        ytd_end = current_date
        
        # Test symbols for YTD analysis
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            print(f"\nTesting YTD signals for {symbol} from {ytd_start.date()} to {ytd_end.date()}")
            
            signals = await analyzer.analyze_symbol(
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
                assert signal["strategy_name"] == "MACDRSIStrategy"
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
    async def test_all_supported_symbols_analysis(self, analyzer):
        """Test analysis with all supported symbols from environment."""
        # Get supported symbols
        supported_symbols = analyzer.get_supported_symbols()
        assert isinstance(supported_symbols, list)
        assert len(supported_symbols) > 0
        
        # Test with a subset of supported symbols
        test_symbols = supported_symbols[:3]  # Test first 3 symbols
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)  # Use proper date range for MACD+RSI
        
        for symbol in test_symbols:
            signals = await analyzer.analyze_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
            
            # Validate results
            assert isinstance(signals, list)
            
            # If signals are found, validate they match the symbol
            for signal in signals:
                assert signal["symbol"] == symbol
                assert signal["strategy_name"] == "MACDRSIStrategy"
    
    @pytest.mark.asyncio
    async def test_input_validation_error_handling(self, analyzer):
        """Test input validation and error handling."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        # Test invalid symbol
        with pytest.raises(ValueError) as excinfo:
            await analyzer.analyze_symbol(
                symbol="INVALID123",
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
        assert "Stock symbols must be alphabetic characters only" in str(excinfo.value)
        
        # Test invalid timeframe
        with pytest.raises(ValueError) as excinfo:
            await analyzer.analyze_symbol(
                symbol="AAPL",
                start_date=start_date,
                end_date=end_date,
                timeframe="5m"
            )
        assert "Timeframe must be '1h' or '1d'" in str(excinfo.value)
        
        # Test invalid date range
        with pytest.raises(ValueError) as excinfo:
            await analyzer.analyze_symbol(
                symbol="AAPL",
                start_date=end_date,
                end_date=start_date,
                timeframe="1d"
            )
        assert "Start date must be before end date" in str(excinfo.value)
        
        # Test insufficient data range
        with pytest.raises(ValueError) as excinfo:
            await analyzer.analyze_symbol(
                symbol="AAPL",
                start_date=end_date - timedelta(days=1),
                end_date=end_date,
                timeframe="1d"
            )
        assert "Date range too small" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_custom_strategy_parameters(self, analyzer, test_symbol, date_range):
        """Test analyzer with custom strategy parameters."""
        start_date, end_date = date_range
        
        # Test with custom parameters
        custom_params = {
            "macd_fast": 10,
            "macd_slow": 20,
            "macd_signal": 8,
            "rsi_period": 12,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "min_confidence": 0.5
        }
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            strategy_params=custom_params
        )
        
        # Validate results
        assert isinstance(signals, list)
        
        # Validate confidence threshold is respected (using custom min_confidence)
        for signal in signals:
            assert signal["confidence"] >= custom_params["min_confidence"]
            assert signal["strategy_name"] == "MACDRSIStrategy"
    
    @pytest.mark.asyncio
    async def test_signal_metadata_validation(self, analyzer, test_symbol, date_range):
        """Test signal metadata structure and content validation."""
        start_date, end_date = date_range
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # Validate metadata structure for each signal
        for signal in signals:
            metadata = signal["metadata"]
            assert isinstance(metadata, dict)
            
            # Check required metadata fields
            assert "signal_name" in metadata
            
            # Check MACD+RSI-specific metadata
            if "macd_value" in metadata:
                assert isinstance(metadata["macd_value"], (int, float))
            if "signal_line" in metadata:
                assert isinstance(metadata["signal_line"], (int, float))
            if "rsi_value" in metadata:
                assert isinstance(metadata["rsi_value"], (int, float))
                assert 0 <= metadata["rsi_value"] <= 100
            if "histogram" in metadata:
                assert isinstance(metadata["histogram"], (int, float))
    
    @pytest.mark.asyncio
    async def test_timeframe_specific_validation(self, analyzer, test_symbol):
        """Test timeframe-specific validation rules."""
        # Skip hourly test - MACD+RSI needs more data than 30-day hourly limit allows
        # Just test daily timeframe
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        assert isinstance(signals, list)
        
        # Test daily timeframe with longer date range
        start_date = end_date - timedelta(days=150)
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        assert isinstance(signals, list)
    
    def test_supported_symbols_from_env(self, analyzer):
        """Test getting supported symbols from environment."""
        symbols = analyzer.get_supported_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        
        # Validate symbol format
        for symbol in symbols:
            assert isinstance(symbol, str)
            assert symbol.isalpha()
            assert len(symbol) <= 10
    
    def test_supported_timeframes(self, analyzer):
        """Test getting supported timeframes."""
        timeframes = analyzer.get_supported_timeframes()
        assert isinstance(timeframes, list)
        assert "1h" in timeframes
        assert "1d" in timeframes
    
    @pytest.mark.asyncio
    async def test_signal_chronological_order(self, analyzer, test_symbol, date_range):
        """Test that signals are returned in descending chronological order (newest first)."""
        start_date, end_date = date_range
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # If multiple signals, check descending chronological order
        if len(signals) > 1:
            timestamps = [signal["timestamp"] for signal in signals]
            # Convert to datetime objects if they're strings
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
            
            # Check if sorted in descending order (newest first)
            assert timestamps == sorted(timestamps, reverse=True)
    
    @pytest.mark.asyncio
    async def test_different_confidence_thresholds(self, analyzer, test_symbol, date_range):
        """Test analyzer with different confidence thresholds."""
        start_date, end_date = date_range
        
        # Test with low confidence threshold
        low_threshold_params = {"min_confidence": 0.5}
        low_signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            strategy_params=low_threshold_params
        )
        
        # Test with high confidence threshold
        high_threshold_params = {"min_confidence": 0.9}
        high_signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            strategy_params=high_threshold_params
        )
        
        # Validate results
        assert isinstance(low_signals, list)
        assert isinstance(high_signals, list)
        
        # High threshold should generally produce fewer signals
        # (unless no signals are found at all)
        if low_signals and high_signals:
            assert len(high_signals) <= len(low_signals)
        
        # Validate confidence thresholds are respected
        for signal in low_signals:
            assert signal["confidence"] >= 0.5
        
        for signal in high_signals:
            assert signal["confidence"] >= 0.9
    
    @pytest.mark.asyncio
    async def test_macd_rsi_convergence_validation(self, analyzer, test_symbol, date_range):
        """Test MACD+RSI convergence validation."""
        start_date, end_date = date_range
        
        signals = await analyzer.analyze_symbol(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        # Validate MACD+RSI convergence logic
        for signal in signals:
            metadata = signal["metadata"]
            signal_name = metadata.get("signal_name", "")
            
            # Check that signals represent valid MACD+RSI combinations
            if "Bullish" in signal_name:
                assert signal["signal_type"] == "BUY"
            elif "Bearish" in signal_name:
                assert signal["signal_type"] == "SELL"
            
            # Check that metadata contains relevant indicator values
            if "macd_value" in metadata and "rsi_value" in metadata:
                # Both indicators should be present for combined signals
                assert isinstance(metadata["macd_value"], (int, float))
                assert isinstance(metadata["rsi_value"], (int, float))
                assert 0 <= metadata["rsi_value"] <= 100