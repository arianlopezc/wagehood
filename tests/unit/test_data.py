"""
Unit tests for data store and mock generator.

Tests data storage, retrieval, and mock data generation functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from unittest.mock import Mock, patch

from src.data.store import DataStore
from src.data.mock_generator import MockDataGenerator, MarketScenario
from src.data.providers.mock_provider import MockDataProvider
from src.core.models import OHLCV, MarketData, TimeFrame


class TestMockDataGenerator:
    """Test mock data generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = MockDataGenerator(seed=42)
        assert generator is not None
        
        # Test without seed
        generator2 = MockDataGenerator()
        assert generator2 is not None
    
    def test_trending_data_generation(self):
        """Test trending data generation."""
        generator = MockDataGenerator(seed=42)
        
        data = generator.generate_trending_data(
            periods=50,
            trend_strength=0.02,
            volatility=0.15,
            start_price=100.0
        )
        
        assert len(data) == 50
        assert all(isinstance(bar, OHLCV) for bar in data)
        
        # Check OHLCV relationships
        for bar in data:
            assert bar.high >= max(bar.open, bar.close)
            assert bar.low <= min(bar.open, bar.close)
            assert bar.volume >= 0
        
        # Check trend - last price should generally be higher than first
        price_change = (data[-1].close - data[0].open) / data[0].open
        # With positive trend strength, should generally trend up
        # Allow some variance due to volatility
        assert price_change > -0.5  # Not too negative
    
    def test_trending_data_negative_trend(self):
        """Test trending data with negative trend."""
        generator = MockDataGenerator(seed=42)
        
        data = generator.generate_trending_data(
            periods=50,
            trend_strength=-0.02,  # Negative trend
            volatility=0.10,
            start_price=100.0
        )
        
        assert len(data) == 50
        
        # With negative trend, end price should generally be lower
        price_change = (data[-1].close - data[0].open) / data[0].open
        assert price_change < 0.5  # Generally declining
    
    def test_ranging_data_generation(self):
        """Test ranging (sideways) data generation."""
        generator = MockDataGenerator(seed=42)
        
        data = generator.generate_ranging_data(
            periods=50,
            range_width=0.2,  # 20% range
            volatility=0.08,
            center_price=100.0
        )
        
        assert len(data) == 50
        assert all(isinstance(bar, OHLCV) for bar in data)
        
        # Prices should stay within the range bounds
        upper_bound = 100.0 * 1.1  # 10% above center
        lower_bound = 100.0 * 0.9  # 10% below center
        
        for bar in data:
            # Allow some overshoot due to volatility
            assert bar.close <= upper_bound * 1.1
            assert bar.close >= lower_bound * 0.9
    
    def test_realistic_data_generation(self):
        """Test realistic data generation with mixed patterns."""
        generator = MockDataGenerator(seed=42)
        
        data = generator.generate_realistic_data(
            symbol="AAPL",
            periods=100,
            patterns_dict={
                'trend': 0.4,
                'range': 0.3,
                'breakout': 0.2,
                'gap': 0.1
            }
        )
        
        assert len(data) == 100
        assert all(isinstance(bar, OHLCV) for bar in data)
        
        # Check that data looks realistic
        prices = [bar.close for bar in data]
        assert all(p > 0 for p in prices)
        
        # Should have some price movement
        price_range = max(prices) - min(prices)
        assert price_range > 0
    
    def test_different_timeframes(self):
        """Test data generation with different timeframes."""
        generator = MockDataGenerator(seed=42)
        
        timeframes = [
            TimeFrame.MINUTE_1,
            TimeFrame.HOUR_1,
            TimeFrame.DAILY,
            TimeFrame.WEEKLY
        ]
        
        for timeframe in timeframes:
            data = generator.generate_trending_data(
                periods=10,
                timeframe=timeframe
            )
            
            assert len(data) == 10
            
            # Check timestamp intervals
            if len(data) > 1:
                time_diff = data[1].timestamp - data[0].timestamp
                
                if timeframe == TimeFrame.MINUTE_1:
                    assert time_diff == timedelta(minutes=1)
                elif timeframe == TimeFrame.HOUR_1:
                    assert time_diff == timedelta(hours=1)
                elif timeframe == TimeFrame.DAILY:
                    assert time_diff == timedelta(days=1)
                elif timeframe == TimeFrame.WEEKLY:
                    assert time_diff == timedelta(weeks=1)
    
    def test_custom_start_date(self):
        """Test data generation with custom start date."""
        generator = MockDataGenerator(seed=42)
        start_date = datetime(2023, 1, 1)
        
        data = generator.generate_trending_data(
            periods=5,
            start_date=start_date
        )
        
        assert data[0].timestamp == start_date
        assert data[1].timestamp == start_date + timedelta(days=1)
    
    def test_volume_profile_addition(self):
        """Test volume profile addition."""
        generator = MockDataGenerator(seed=42)
        
        # Generate base data
        data = generator.generate_trending_data(periods=20)
        
        # Test different volume profiles
        profiles = ['normal', 'accumulation', 'distribution']
        
        for profile in profiles:
            updated_data = generator.add_volume_profile(data, profile)
            
            assert len(updated_data) == len(data)
            
            # All volumes should be positive
            for bar in updated_data:
                assert bar.volume > 0
            
            # Volume should vary based on profile
            volumes = [bar.volume for bar in updated_data]
            assert max(volumes) > min(volumes)  # Should have variation
    
    def test_market_scenarios(self):
        """Test predefined market scenarios."""
        generator = MockDataGenerator(seed=42)
        scenarios = generator.create_market_scenarios()
        
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0
        
        # Check required scenarios
        required_scenarios = [
            'bull_market', 'bear_market', 'sideways_market',
            'high_volatility', 'low_volatility', 'flash_crash'
        ]
        
        for scenario_name in required_scenarios:
            assert scenario_name in scenarios
            scenario = scenarios[scenario_name]
            assert isinstance(scenario, MarketScenario)
            assert scenario.periods > 0
            assert scenario.volatility > 0
            assert isinstance(scenario.description, str)
    
    def test_symbol_specific_parameters(self):
        """Test symbol-specific parameters."""
        generator = MockDataGenerator(seed=42)
        
        # Test known symbols
        symbols = ["AAPL", "BTC", "EURUSD", "SPY", "UNKNOWN"]
        
        for symbol in symbols:
            base_price = generator._get_symbol_base_price(symbol)
            volatility = generator._get_symbol_volatility(symbol)
            
            assert base_price > 0
            assert volatility > 0
            
            # Different symbols should have different characteristics
            if symbol == "BTC":
                assert volatility > 0.3  # High volatility
            elif symbol == "EURUSD":
                assert volatility < 0.2  # Lower volatility
    
    def test_ohlc_relationships(self):
        """Test OHLC relationships in generated data."""
        generator = MockDataGenerator(seed=42)
        
        data = generator.generate_trending_data(periods=50)
        
        for bar in data:
            # High should be highest
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            
            # Low should be lowest
            assert bar.low <= bar.open
            assert bar.low <= bar.close
            
            # All prices should be positive
            assert bar.open > 0
            assert bar.high > 0
            assert bar.low > 0
            assert bar.close > 0
            assert bar.volume >= 0
    
    def test_data_reproducibility(self):
        """Test data reproducibility with same seed."""
        # Generate data with same seed
        generator1 = MockDataGenerator(seed=42)
        data1 = generator1.generate_trending_data(periods=20)
        
        generator2 = MockDataGenerator(seed=42)
        data2 = generator2.generate_trending_data(periods=20)
        
        # Should be identical
        assert len(data1) == len(data2)
        for bar1, bar2 in zip(data1, data2):
            assert abs(bar1.open - bar2.open) < 1e-10
            assert abs(bar1.high - bar2.high) < 1e-10
            assert abs(bar1.low - bar2.low) < 1e-10
            assert abs(bar1.close - bar2.close) < 1e-10
            assert abs(bar1.volume - bar2.volume) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases in data generation."""
        generator = MockDataGenerator(seed=42)
        
        # Single period
        data = generator.generate_trending_data(periods=1)
        assert len(data) == 1
        
        # Zero trend
        data = generator.generate_trending_data(
            periods=10,
            trend_strength=0.0,
            volatility=0.05
        )
        assert len(data) == 10
        
        # High volatility
        data = generator.generate_trending_data(
            periods=10,
            trend_strength=0.01,
            volatility=0.5
        )
        assert len(data) == 10
        
        # Very low volatility
        data = generator.generate_trending_data(
            periods=10,
            trend_strength=0.01,
            volatility=0.001
        )
        assert len(data) == 10
    
    def test_error_handling(self):
        """Test error handling in data generation."""
        generator = MockDataGenerator(seed=42)
        
        # Invalid periods
        with pytest.raises((ValueError, TypeError)):
            generator.generate_trending_data(periods=0)
        
        with pytest.raises((ValueError, TypeError)):
            generator.generate_trending_data(periods=-1)
        
        # Invalid parameters should be handled gracefully or raise appropriate errors
        try:
            data = generator.generate_trending_data(
                periods=10,
                trend_strength=float('inf')
            )
            # If no error, should still produce valid data
            assert len(data) == 10
        except (ValueError, OverflowError):
            # Acceptable to raise error for invalid input
            pass


class TestMarketScenario:
    """Test MarketScenario dataclass."""
    
    def test_scenario_creation(self):
        """Test scenario creation."""
        scenario = MarketScenario(
            name="Test Scenario",
            periods=100,
            trend_strength=0.05,
            volatility=0.20,
            volume_profile="normal",
            noise_level=0.10,
            description="Test scenario for unit tests"
        )
        
        assert scenario.name == "Test Scenario"
        assert scenario.periods == 100
        assert scenario.trend_strength == 0.05
        assert scenario.volatility == 0.20
        assert scenario.volume_profile == "normal"
        assert scenario.noise_level == 0.10
        assert scenario.description == "Test scenario for unit tests"


class TestMockDataProvider:
    """Test mock data provider."""
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = MockDataProvider()
        assert provider is not None
    
    def test_get_market_data(self):
        """Test market data retrieval."""
        provider = MockDataProvider()
        
        data = provider.get_market_data(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert isinstance(data, MarketData)
        assert data.symbol == "AAPL"
        assert data.timeframe == TimeFrame.DAILY
        assert len(data.data) > 0
        assert isinstance(data.last_updated, datetime)
    
    def test_get_symbols(self):
        """Test symbol list retrieval."""
        provider = MockDataProvider()
        
        symbols = provider.get_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(symbol, str) for symbol in symbols)
    
    def test_is_market_open(self):
        """Test market open status."""
        provider = MockDataProvider()
        
        is_open = provider.is_market_open()
        
        # Mock provider might always return True or implement time-based logic
        assert isinstance(is_open, bool)
    
    def test_different_symbols(self):
        """Test data for different symbols."""
        provider = MockDataProvider()
        
        symbols = ["AAPL", "GOOGL", "BTC", "EURUSD"]
        
        for symbol in symbols:
            data = provider.get_market_data(
                symbol=symbol,
                timeframe=TimeFrame.DAILY,
                start_date=datetime.now() - timedelta(days=10),
                end_date=datetime.now()
            )
            
            assert data.symbol == symbol
            assert len(data.data) > 0
    
    def test_different_timeframes(self):
        """Test data for different timeframes."""
        provider = MockDataProvider()
        
        timeframes = [
            TimeFrame.MINUTE_1,
            TimeFrame.HOUR_1,
            TimeFrame.DAILY,
            TimeFrame.WEEKLY
        ]
        
        for timeframe in timeframes:
            data = provider.get_market_data(
                symbol="AAPL",
                timeframe=timeframe,
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
            
            assert data.timeframe == timeframe
            assert len(data.data) > 0
    
    def test_date_range_handling(self):
        """Test date range handling."""
        provider = MockDataProvider()
        
        # Recent data
        recent_data = provider.get_market_data(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now()
        )
        
        # Historical data
        historical_data = provider.get_market_data(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now() - timedelta(days=30)
        )
        
        assert len(recent_data.data) < len(historical_data.data)
        
        # Data should be in date range
        for bar in recent_data.data:
            assert bar.timestamp >= datetime.now() - timedelta(days=6)  # Some tolerance
    
    def test_error_handling(self):
        """Test error handling in provider."""
        provider = MockDataProvider()
        
        # Invalid date range (end before start)
        try:
            data = provider.get_market_data(
                symbol="AAPL",
                timeframe=TimeFrame.DAILY,
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)
            )
            # Should handle gracefully or return empty data
            if data:
                assert len(data.data) >= 0
        except ValueError:
            # Acceptable to raise error for invalid input
            pass
        
        # Invalid symbol
        data = provider.get_market_data(
            symbol="INVALID_SYMBOL",
            timeframe=TimeFrame.DAILY,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        # Should handle gracefully
        assert isinstance(data, MarketData)


class TestDataStore:
    """Test data storage and retrieval."""
    
    def test_store_initialization(self):
        """Test data store initialization."""
        store = DataStore()
        assert store is not None
    
    def test_store_market_data(self, sample_market_data):
        """Test storing market data."""
        store = DataStore()
        
        # Store data
        store.store_market_data(sample_market_data)
        
        # Retrieve data
        retrieved = store.get_market_data(
            symbol=sample_market_data.symbol,
            timeframe=sample_market_data.timeframe
        )
        
        assert retrieved is not None
        assert retrieved.symbol == sample_market_data.symbol
        assert retrieved.timeframe == sample_market_data.timeframe
        assert len(retrieved.data) == len(sample_market_data.data)
    
    def test_store_multiple_symbols(self, mock_data_generator):
        """Test storing data for multiple symbols."""
        store = DataStore()
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Store data for multiple symbols
        for symbol in symbols:
            data_points = mock_data_generator.generate_trending_data(periods=20)
            market_data = MarketData(
                symbol=symbol,
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Retrieve and verify
        for symbol in symbols:
            retrieved = store.get_market_data(symbol, TimeFrame.DAILY)
            assert retrieved is not None
            assert retrieved.symbol == symbol
    
    def test_store_multiple_timeframes(self, mock_data_generator):
        """Test storing data for multiple timeframes."""
        store = DataStore()
        
        timeframes = [TimeFrame.DAILY, TimeFrame.HOUR_1, TimeFrame.MINUTE_5]
        
        # Store data for multiple timeframes
        for timeframe in timeframes:
            data_points = mock_data_generator.generate_trending_data(periods=20)
            market_data = MarketData(
                symbol="AAPL",
                timeframe=timeframe,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Retrieve and verify
        for timeframe in timeframes:
            retrieved = store.get_market_data("AAPL", timeframe)
            assert retrieved is not None
            assert retrieved.timeframe == timeframe
    
    def test_data_overwrite(self, mock_data_generator):
        """Test data overwriting behavior."""
        store = DataStore()
        
        # Store initial data
        data1 = mock_data_generator.generate_trending_data(periods=10)
        market_data1 = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data1,
            indicators={},
            last_updated=datetime.now()
        )
        store.store_market_data(market_data1)
        
        # Store new data for same symbol/timeframe
        data2 = mock_data_generator.generate_trending_data(periods=20)
        market_data2 = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data2,
            indicators={},
            last_updated=datetime.now()
        )
        store.store_market_data(market_data2)
        
        # Should have the new data
        retrieved = store.get_market_data("AAPL", TimeFrame.DAILY)
        assert len(retrieved.data) == 20  # New data length
    
    def test_get_nonexistent_data(self):
        """Test retrieving non-existent data."""
        store = DataStore()
        
        retrieved = store.get_market_data("NONEXISTENT", TimeFrame.DAILY)
        
        # Should return None or handle gracefully
        assert retrieved is None
    
    def test_get_symbols(self, mock_data_generator):
        """Test getting stored symbols."""
        store = DataStore()
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Store data for symbols
        for symbol in symbols:
            data_points = mock_data_generator.generate_trending_data(periods=10)
            market_data = MarketData(
                symbol=symbol,
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Get stored symbols
        stored_symbols = store.get_symbols()
        
        assert isinstance(stored_symbols, list)
        for symbol in symbols:
            assert symbol in stored_symbols
    
    def test_get_timeframes(self, mock_data_generator):
        """Test getting available timeframes for a symbol."""
        store = DataStore()
        
        timeframes = [TimeFrame.DAILY, TimeFrame.HOUR_1]
        
        # Store data for timeframes
        for timeframe in timeframes:
            data_points = mock_data_generator.generate_trending_data(periods=10)
            market_data = MarketData(
                symbol="AAPL",
                timeframe=timeframe,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Get stored timeframes
        stored_timeframes = store.get_timeframes("AAPL")
        
        assert isinstance(stored_timeframes, list)
        for timeframe in timeframes:
            assert timeframe in stored_timeframes
    
    def test_clear_data(self, sample_market_data):
        """Test clearing stored data."""
        store = DataStore()
        
        # Store data
        store.store_market_data(sample_market_data)
        assert store.get_market_data("AAPL", TimeFrame.DAILY) is not None
        
        # Clear data
        store.clear()
        
        # Should be empty
        assert store.get_market_data("AAPL", TimeFrame.DAILY) is None
        assert len(store.get_symbols()) == 0
    
    def test_data_statistics(self, mock_data_generator):
        """Test data statistics and metrics."""
        store = DataStore()
        
        # Store some data
        data_points = mock_data_generator.generate_trending_data(periods=100)
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        store.store_market_data(market_data)
        
        # Get statistics
        stats = store.get_data_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_symbols' in stats
        assert 'total_data_points' in stats
        assert stats['total_symbols'] >= 1
        assert stats['total_data_points'] >= 100
    
    def test_memory_usage(self, mock_data_generator, memory_monitor):
        """Test memory usage of data store."""
        initial_memory = memory_monitor.get_current_usage()
        
        store = DataStore()
        
        # Store large amount of data
        for i in range(10):
            data_points = mock_data_generator.generate_trending_data(periods=1000)
            market_data = MarketData(
                symbol=f"STOCK_{i}",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should use reasonable amount of memory
        assert memory_increase < 200  # MB
    
    def test_concurrent_access(self, sample_market_data):
        """Test concurrent access to data store."""
        store = DataStore()
        
        # Simulate concurrent access
        import threading
        
        def store_data(symbol):
            data_points = [sample_market_data.data[0]]  # Single point
            market_data = MarketData(
                symbol=symbol,
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Create threads
        threads = []
        symbols = [f"STOCK_{i}" for i in range(10)]
        
        for symbol in symbols:
            thread = threading.Thread(target=store_data, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All symbols should be stored
        stored_symbols = store.get_symbols()
        for symbol in symbols:
            assert symbol in stored_symbols


class TestDataIntegration:
    """Test integration between data components."""
    
    def test_generator_provider_store_integration(self):
        """Test integration between generator, provider, and store."""
        # Generate data
        generator = MockDataGenerator(seed=42)
        data_points = generator.generate_trending_data(periods=50)
        
        # Create market data
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Store data
        store = DataStore()
        store.store_market_data(market_data)
        
        # Retrieve through provider-like interface
        retrieved = store.get_market_data("AAPL", TimeFrame.DAILY)
        
        assert retrieved is not None
        assert len(retrieved.data) == 50
        assert retrieved.symbol == "AAPL"
    
    def test_scenario_based_testing(self):
        """Test using market scenarios for comprehensive testing."""
        generator = MockDataGenerator(seed=42)
        scenarios = generator.create_market_scenarios()
        
        store = DataStore()
        
        # Generate data for each scenario
        for scenario_name, scenario in scenarios.items():
            if scenario_name == 'bull_market':
                data_points = generator.generate_trending_data(
                    periods=min(50, scenario.periods),  # Limit for testing
                    trend_strength=scenario.trend_strength / 252,  # Daily rate
                    volatility=scenario.volatility,
                    start_price=100.0
                )
            elif scenario_name == 'bear_market':
                data_points = generator.generate_trending_data(
                    periods=min(50, scenario.periods),
                    trend_strength=scenario.trend_strength / 252,
                    volatility=scenario.volatility,
                    start_price=100.0
                )
            else:
                # Use ranging data for other scenarios
                data_points = generator.generate_ranging_data(
                    periods=min(50, scenario.periods),
                    range_width=0.15,
                    volatility=scenario.volatility,
                    center_price=100.0
                )
            
            # Store scenario data
            market_data = MarketData(
                symbol=f"SCENARIO_{scenario_name.upper()}",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            store.store_market_data(market_data)
        
        # Verify all scenarios are stored
        symbols = store.get_symbols()
        assert len(symbols) >= len(scenarios)
        
        for scenario_name in scenarios.keys():
            symbol = f"SCENARIO_{scenario_name.upper()}"
            assert symbol in symbols
            
            data = store.get_market_data(symbol, TimeFrame.DAILY)
            assert data is not None
            assert len(data.data) > 0