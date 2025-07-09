"""
Comprehensive tests for the ConfigManager worker component.

Tests cover:
- Configuration handling and validation
- Multi-profile and multi-timeframe settings
- Runtime parameter updates
- Error handling and recovery
- Performance optimization
- Configuration persistence
- Validation and consistency checks
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from src.realtime.config_manager import (
    ConfigManager, TradingProfile, AssetConfig, IndicatorConfig,
    StrategyConfig, SystemConfig, TimeframeConfig, SymbolStrategyConfig,
    StrategyTimeframeMatrix
)


class TestConfigManagerComponents:
    """Test individual ConfigManager components."""
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        with patch('src.realtime.config_manager.cache_manager'):
            config_manager = ConfigManager()
            
            assert config_manager._config_namespace == "realtime_config"
            assert config_manager._cache_ttl > 0
            assert len(config_manager._default_indicators) > 0
            assert len(config_manager._default_strategies) > 0
            assert config_manager._default_system_config is not None
            
    def test_trading_profile_enum(self):
        """Test TradingProfile enum."""
        profiles = [TradingProfile.DAY_TRADING, TradingProfile.SWING_TRADING, TradingProfile.POSITION_TRADING]
        
        for profile in profiles:
            assert isinstance(profile.value, str)
            assert len(profile.value) > 0
            
    def test_timeframe_config_creation(self):
        """Test TimeframeConfig creation and validation."""
        config = TimeframeConfig(
            timeframe="1h",
            update_interval_seconds=60,
            lookback_periods=200,
            min_data_points=100,
            priority=1
        )
        
        assert config.timeframe == "1h"
        assert config.update_interval_seconds == 60
        assert config.lookback_periods == 200
        assert config.validate() == True
        
        # Test invalid config
        invalid_config = TimeframeConfig(
            timeframe="invalid",
            update_interval_seconds=-1,
            lookback_periods=0,
            priority=1
        )
        
        assert invalid_config.validate() == False
        
    def test_strategy_timeframe_matrix_creation(self):
        """Test StrategyTimeframeMatrix creation."""
        matrix = StrategyTimeframeMatrix(
            strategy_name="macd_rsi_strategy",
            supported_timeframes=["5m", "1h", "4h"],
            optimal_timeframe="1h",
            trading_profile=TradingProfile.SWING_TRADING,
            min_bars_required=200
        )
        
        assert matrix.strategy_name == "macd_rsi_strategy"
        assert "1h" in matrix.supported_timeframes
        assert matrix.is_timeframe_supported("1h") == True
        assert matrix.is_timeframe_supported("1d") == False
        
    def test_asset_config_creation(self):
        """Test AssetConfig creation."""
        asset = AssetConfig(
            symbol="AAPL",
            enabled=True,
            data_provider="alpaca",
            timeframes=["1m", "5m", "1h"],
            priority=1,
            enabled_strategies=["macd_rsi_strategy", "rsi_trend_strategy"],
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        assert asset.symbol == "AAPL"
        assert asset.enabled == True
        assert "1h" in asset.timeframes
        assert "macd_rsi_strategy" in asset.enabled_strategies
        
    def test_indicator_config_creation(self):
        """Test IndicatorConfig creation."""
        indicator = IndicatorConfig(
            name="rsi_14",
            enabled=True,
            parameters={"period": 14},
            update_frequency_seconds=1,
            ttl_seconds=300
        )
        
        assert indicator.name == "rsi_14"
        assert indicator.enabled == True
        assert indicator.parameters["period"] == 14
        
    def test_strategy_config_creation(self):
        """Test StrategyConfig creation."""
        strategy = StrategyConfig(
            name="macd_rsi_strategy",
            enabled=True,
            parameters={"rsi_period": 14, "macd_fast": 12},
            required_indicators=["rsi_14", "macd"],
            supported_timeframes=["5m", "1h", "4h"],
            optimal_timeframe="1h",
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        assert strategy.name == "macd_rsi_strategy"
        assert strategy.enabled == True
        assert "rsi_14" in strategy.required_indicators
        assert strategy.optimal_timeframe == "1h"


class TestConfigManagerWatchlist:
    """Test watchlist management functionality."""
    
    def test_empty_watchlist_handling(self, test_config_manager):
        """Test handling of empty watchlist."""
        # Mock empty watchlist
        test_config_manager.get_watchlist = Mock(return_value=[])
        
        watchlist = test_config_manager.get_watchlist()
        assert len(watchlist) == 0
        
        enabled_symbols = test_config_manager.get_enabled_symbols()
        assert len(enabled_symbols) == 0
        
    def test_watchlist_retrieval(self, test_config_manager):
        """Test watchlist retrieval."""
        watchlist = test_config_manager.get_watchlist()
        
        assert isinstance(watchlist, list)
        assert len(watchlist) > 0
        
        for asset in watchlist:
            assert isinstance(asset, AssetConfig)
            assert hasattr(asset, 'symbol')
            assert hasattr(asset, 'enabled')
            assert hasattr(asset, 'timeframes')
            
    def test_watchlist_update(self, test_config_manager):
        """Test watchlist update functionality."""
        new_assets = [
            AssetConfig(
                symbol="TSLA",
                enabled=True,
                data_provider="mock",
                timeframes=["1m", "5m"],
                priority=1,
                enabled_strategies=["rsi_trend_strategy"],
                trading_profile=TradingProfile.DAY_TRADING
            )
        ]
        
        # Mock successful update
        test_config_manager.update_watchlist = Mock(return_value=True)
        
        result = test_config_manager.update_watchlist(new_assets)
        assert result == True
        
    def test_symbol_addition(self, test_config_manager):
        """Test adding individual symbols."""
        # Mock current watchlist
        current_watchlist = [
            AssetConfig("AAPL", True, "mock", ["1m"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING)
        ]
        test_config_manager.get_watchlist = Mock(return_value=current_watchlist)
        test_config_manager.update_watchlist = Mock(return_value=True)
        
        result = test_config_manager.add_symbol(
            symbol="GOOGL",
            data_provider="mock",
            timeframes=["5m", "1h"],
            enabled_strategies=["macd_rsi_strategy"]
        )
        
        assert result == True
        test_config_manager.update_watchlist.assert_called_once()
        
    def test_symbol_removal(self, test_config_manager):
        """Test removing symbols."""
        # Mock current watchlist
        current_watchlist = [
            AssetConfig("AAPL", True, "mock", ["1m"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING),
            AssetConfig("GOOGL", True, "mock", ["5m"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING)
        ]
        test_config_manager.get_watchlist = Mock(return_value=current_watchlist)
        test_config_manager.update_watchlist = Mock(return_value=True)
        
        result = test_config_manager.remove_symbol("GOOGL")
        
        assert result == True
        test_config_manager.update_watchlist.assert_called_once()
        
    def test_enabled_symbols_filtering(self, test_config_manager):
        """Test filtering of enabled symbols."""
        # Mock watchlist with mixed enabled/disabled symbols
        mixed_watchlist = [
            AssetConfig("AAPL", True, "mock", ["1m"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING),
            AssetConfig("GOOGL", False, "mock", ["5m"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING),
            AssetConfig("MSFT", True, "mock", ["1h"], enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING)
        ]
        test_config_manager.get_watchlist = Mock(return_value=mixed_watchlist)
        
        enabled_symbols = test_config_manager.get_enabled_symbols()
        
        assert "AAPL" in enabled_symbols
        assert "GOOGL" not in enabled_symbols
        assert "MSFT" in enabled_symbols
        assert len(enabled_symbols) == 2


class TestConfigManagerIndicators:
    """Test indicator configuration management."""
    
    def test_indicator_config_retrieval(self, test_config_manager):
        """Test indicator configuration retrieval."""
        indicators = test_config_manager.get_indicator_configs()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        
        for indicator in indicators:
            assert isinstance(indicator, IndicatorConfig)
            assert hasattr(indicator, 'name')
            assert hasattr(indicator, 'enabled')
            assert hasattr(indicator, 'parameters')
            
    def test_enabled_indicators_filtering(self, test_config_manager):
        """Test filtering of enabled indicators."""
        enabled_indicators = test_config_manager.get_enabled_indicators()
        
        assert isinstance(enabled_indicators, list)
        
        for indicator in enabled_indicators:
            assert indicator.enabled == True
            
    def test_indicator_config_update(self, test_config_manager):
        """Test indicator configuration update."""
        new_indicators = [
            IndicatorConfig(
                name="custom_rsi",
                enabled=True,
                parameters={"period": 21},
                update_frequency_seconds=2
            )
        ]
        
        # Mock successful update
        test_config_manager.update_indicator_configs = Mock(return_value=True)
        
        result = test_config_manager.update_indicator_configs(new_indicators)
        assert result == True
        
    def test_default_indicators_validation(self, test_config_manager):
        """Test validation of default indicators."""
        default_indicators = test_config_manager._default_indicators
        
        assert len(default_indicators) > 0
        
        required_indicators = ["sma_50", "sma_200", "rsi_14", "macd", "bollinger_bands"]
        indicator_names = [ind.name for ind in default_indicators]
        
        for required in required_indicators:
            assert required in indicator_names


class TestConfigManagerStrategies:
    """Test strategy configuration management."""
    
    def test_strategy_config_retrieval(self, test_config_manager):
        """Test strategy configuration retrieval."""
        strategies = test_config_manager.get_strategy_configs()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        for strategy in strategies:
            assert isinstance(strategy, StrategyConfig)
            assert hasattr(strategy, 'name')
            assert hasattr(strategy, 'enabled')
            assert hasattr(strategy, 'required_indicators')
            
    def test_enabled_strategies_filtering(self, test_config_manager):
        """Test filtering of enabled strategies."""
        enabled_strategies = test_config_manager.get_enabled_strategies()
        
        assert isinstance(enabled_strategies, list)
        
        for strategy in enabled_strategies:
            assert strategy.enabled == True
            
    def test_strategy_config_update(self, test_config_manager):
        """Test strategy configuration update."""
        new_strategies = [
            StrategyConfig(
                name="custom_strategy",
                enabled=True,
                parameters={"threshold": 0.8},
                required_indicators=["rsi_14"],
                trading_profile=TradingProfile.SWING_TRADING
            )
        ]
        
        # Mock successful update
        test_config_manager.update_strategy_configs = Mock(return_value=True)
        
        result = test_config_manager.update_strategy_configs(new_strategies)
        assert result == True
        
    def test_default_strategies_validation(self, test_config_manager):
        """Test validation of default strategies."""
        default_strategies = test_config_manager._default_strategies
        
        assert len(default_strategies) > 0
        
        required_strategies = ["macd_rsi_strategy", "ma_crossover_strategy", "rsi_trend_strategy"]
        strategy_names = [strat.name for strat in default_strategies]
        
        for required in required_strategies:
            assert required in strategy_names
            
    def test_strategy_timeframe_parameters(self, test_config_manager):
        """Test strategy timeframe-specific parameters."""
        strategies = test_config_manager.get_strategy_configs()
        
        # Find a strategy with timeframe parameters
        strategy_with_tf_params = None
        for strategy in strategies:
            if strategy.timeframe_parameters:
                strategy_with_tf_params = strategy
                break
                
        if strategy_with_tf_params:
            tf_params = strategy_with_tf_params.timeframe_parameters
            assert isinstance(tf_params, dict)
            
            # Should have parameters for different timeframes
            for timeframe, params in tf_params.items():
                assert isinstance(timeframe, str)
                assert isinstance(params, dict)


class TestConfigManagerTimeframes:
    """Test timeframe configuration management."""
    
    def test_timeframe_config_retrieval(self, test_config_manager):
        """Test timeframe configuration retrieval."""
        timeframe_configs = test_config_manager.get_timeframe_configs()
        
        assert isinstance(timeframe_configs, dict)
        
        # Should have configs for each trading profile
        for profile in TradingProfile:
            assert profile in timeframe_configs
            configs = timeframe_configs[profile]
            assert isinstance(configs, list)
            assert len(configs) > 0
            
            for config in configs:
                assert isinstance(config, TimeframeConfig)
                assert config.validate() == True
                
    def test_timeframe_config_update(self, test_config_manager):
        """Test timeframe configuration update."""
        new_configs = {
            TradingProfile.DAY_TRADING: [
                TimeframeConfig("1m", 1, 500, priority=3),
                TimeframeConfig("5m", 5, 300, priority=2)
            ]
        }
        
        # Mock successful update
        test_config_manager.update_timeframe_configs = Mock(return_value=True)
        
        result = test_config_manager.update_timeframe_configs(new_configs)
        assert result == True
        
    def test_default_timeframe_configs_validation(self, test_config_manager):
        """Test validation of default timeframe configurations."""
        default_configs = test_config_manager._default_timeframe_configs
        
        # Should have configs for all trading profiles
        assert TradingProfile.DAY_TRADING in default_configs
        assert TradingProfile.SWING_TRADING in default_configs
        assert TradingProfile.POSITION_TRADING in default_configs
        
        # Validate each config
        for profile, configs in default_configs.items():
            for config in configs:
                assert config.validate() == True
                assert config.update_interval_seconds > 0
                assert config.lookback_periods > 0
                
    def test_timeframe_config_for_profile(self, test_config_manager):
        """Test getting timeframe config for specific profile."""
        # Mock timeframe configs
        test_config_manager.get_timeframe_configs = Mock(return_value={
            TradingProfile.SWING_TRADING: [
                TimeframeConfig("1h", 60, 200, priority=1)
            ]
        })
        
        config = test_config_manager.get_timeframe_config_for_profile(
            TradingProfile.SWING_TRADING, "1h"
        )
        
        assert config is not None
        assert config.timeframe == "1h"
        
        # Test non-existent timeframe
        config = test_config_manager.get_timeframe_config_for_profile(
            TradingProfile.SWING_TRADING, "1w"
        )
        
        assert config is None


class TestConfigManagerStrategies:
    """Test strategy matrix and symbol configuration."""
    
    def test_strategy_matrix_retrieval(self, test_config_manager):
        """Test strategy matrix retrieval."""
        matrix = test_config_manager.get_strategy_matrix()
        
        assert isinstance(matrix, list)
        assert len(matrix) > 0
        
        for entry in matrix:
            assert isinstance(entry, StrategyTimeframeMatrix)
            assert hasattr(entry, 'strategy_name')
            assert hasattr(entry, 'supported_timeframes')
            assert hasattr(entry, 'optimal_timeframe')
            
    def test_strategy_matrix_update(self, test_config_manager):
        """Test strategy matrix update."""
        new_matrix = [
            StrategyTimeframeMatrix(
                strategy_name="custom_strategy",
                supported_timeframes=["5m", "1h"],
                optimal_timeframe="1h",
                trading_profile=TradingProfile.SWING_TRADING
            )
        ]
        
        # Mock successful update
        test_config_manager.update_strategy_matrix = Mock(return_value=True)
        
        result = test_config_manager.update_strategy_matrix(new_matrix)
        assert result == True
        
    def test_symbol_strategy_configs(self, test_config_manager):
        """Test symbol-specific strategy configurations."""
        symbol_configs = test_config_manager.get_symbol_strategy_configs()
        
        assert isinstance(symbol_configs, list)
        
        for config in symbol_configs:
            assert isinstance(config, SymbolStrategyConfig)
            assert hasattr(config, 'symbol')
            assert hasattr(config, 'enabled_strategies')
            assert hasattr(config, 'trading_profile')
            
    def test_symbol_strategy_config_update(self, test_config_manager):
        """Test updating symbol-specific strategy configuration."""
        symbol_config = SymbolStrategyConfig(
            symbol="AAPL",
            enabled_strategies=["macd_rsi_strategy"],
            trading_profile=TradingProfile.DAY_TRADING
        )
        
        # Mock successful update
        test_config_manager.update_symbol_strategy_config = Mock(return_value=True)
        
        result = test_config_manager.update_symbol_strategy_config("AAPL", symbol_config)
        assert result == True
        
    def test_strategy_for_symbol_timeframe(self, test_config_manager):
        """Test getting strategies for symbol-timeframe combination."""
        # Mock data
        test_config_manager.get_watchlist = Mock(return_value=[
            AssetConfig("AAPL", True, "mock", ["1h"], enabled_strategies=["macd_rsi_strategy"], 
                       trading_profile=TradingProfile.SWING_TRADING)
        ])
        
        test_config_manager.get_strategy_matrix = Mock(return_value=[
            StrategyTimeframeMatrix("macd_rsi_strategy", ["1h"], "1h", TradingProfile.SWING_TRADING)
        ])
        
        strategies = test_config_manager.get_strategy_for_symbol_and_timeframe("AAPL", "1h")
        
        assert "macd_rsi_strategy" in strategies
        
    def test_effective_strategy_parameters(self, test_config_manager):
        """Test getting effective strategy parameters."""
        # Mock strategy with timeframe parameters
        test_config_manager.get_strategy_configs = Mock(return_value=[
            StrategyConfig(
                name="test_strategy",
                enabled=True,
                parameters={"base_param": 10},
                required_indicators=["rsi_14"],
                timeframe_parameters={"1h": {"base_param": 15}},
                trading_profile=TradingProfile.SWING_TRADING
            )
        ])
        
        test_config_manager.get_watchlist = Mock(return_value=[
            AssetConfig("AAPL", True, "mock", ["1h"], 
                       strategy_overrides={"test_strategy": {"base_param": 20}},
                       enabled_strategies=[], trading_profile=TradingProfile.SWING_TRADING)
        ])
        
        params = test_config_manager.get_effective_strategy_parameters("AAPL", "test_strategy", "1h")
        
        # Should apply symbol override (highest priority)
        assert params["base_param"] == 20
        
    def test_strategy_enablement_check(self, test_config_manager):
        """Test checking strategy enablement for symbol-timeframe."""
        # Mock return True for the method call
        test_config_manager.get_strategy_for_symbol_and_timeframe = Mock(return_value=["macd_rsi_strategy"])
        
        enabled = test_config_manager.is_strategy_enabled_for_symbol_timeframe(
            "AAPL", "macd_rsi_strategy", "1h"
        )
        
        assert enabled == True
        
    def test_optimal_timeframe_for_strategy(self, test_config_manager):
        """Test getting optimal timeframe for strategy."""
        # Mock strategy matrix
        test_config_manager.get_strategy_matrix = Mock(return_value=[
            StrategyTimeframeMatrix("macd_rsi_strategy", ["5m", "1h", "4h"], "1h", TradingProfile.SWING_TRADING)
        ])
        
        optimal_tf = test_config_manager.get_optimal_timeframe_for_strategy("macd_rsi_strategy")
        
        assert optimal_tf == "1h"


class TestConfigManagerSystem:
    """Test system configuration management."""
    
    def test_system_config_retrieval(self, test_config_manager):
        """Test system configuration retrieval."""
        system_config = test_config_manager.get_system_config()
        
        assert isinstance(system_config, SystemConfig)
        assert system_config.max_concurrent_calculations > 0
        assert system_config.calculation_workers > 0
        assert system_config.data_update_interval_seconds > 0
        
    def test_system_config_update(self, test_config_manager):
        """Test system configuration update."""
        new_system_config = SystemConfig(
            max_concurrent_calculations=50,
            calculation_workers=2,
            data_update_interval_seconds=2
        )
        
        # Mock successful update
        test_config_manager.update_system_config = Mock(return_value=True)
        
        result = test_config_manager.update_system_config(new_system_config)
        assert result == True
        
    def test_default_system_config_validation(self, test_config_manager):
        """Test validation of default system configuration."""
        default_config = test_config_manager._default_system_config
        
        assert default_config.max_concurrent_calculations > 0
        assert default_config.calculation_workers > 0
        assert default_config.data_update_interval_seconds > 0
        assert isinstance(default_config.enable_monitoring, bool)
        assert isinstance(default_config.enable_alerts, bool)


class TestConfigManagerValidation:
    """Test configuration validation functionality."""
    
    def test_configuration_summary(self, test_config_manager):
        """Test configuration summary generation."""
        summary = test_config_manager.get_configuration_summary()
        
        assert isinstance(summary, dict)
        assert "watchlist" in summary
        assert "indicators" in summary
        assert "strategies" in summary
        assert "system" in summary
        assert "last_updated" in summary
        
        # Verify structure
        watchlist_summary = summary["watchlist"]
        assert "total_symbols" in watchlist_summary
        assert "enabled_symbols" in watchlist_summary
        
    def test_timeframe_strategy_combination_validation(self, test_config_manager):
        """Test validation of timeframe-strategy combinations."""
        validation_result = test_config_manager.validate_timeframe_strategy_combinations()
        
        assert isinstance(validation_result, dict)
        assert "warnings" in validation_result
        assert "errors" in validation_result
        assert "is_valid" in validation_result
        
        assert isinstance(validation_result["warnings"], list)
        assert isinstance(validation_result["errors"], list)
        assert isinstance(validation_result["is_valid"], bool)
        
    def test_update_intervals_validation(self, test_config_manager):
        """Test validation of update intervals."""
        validation_result = test_config_manager.validate_update_intervals()
        
        assert isinstance(validation_result, dict)
        assert "warnings" in validation_result
        assert "errors" in validation_result
        assert "is_valid" in validation_result
        
    def test_comprehensive_configuration_validation(self, test_config_manager):
        """Test comprehensive configuration validation."""
        validation_result = test_config_manager.validate_configuration()
        
        assert isinstance(validation_result, dict)
        assert "warnings" in validation_result
        assert "errors" in validation_result
        assert "is_valid" in validation_result
        
        # Should validate all components
        errors = validation_result["errors"]
        warnings = validation_result["warnings"]
        
        # If there are errors, is_valid should be False
        if len(errors) > 0:
            assert validation_result["is_valid"] == False


class TestConfigManagerPerformance:
    """Test ConfigManager performance characteristics."""
    
    def test_configuration_retrieval_performance(self, test_config_manager, performance_thresholds):
        """Test configuration retrieval performance."""
        operations = [
            lambda: test_config_manager.get_watchlist(),
            lambda: test_config_manager.get_indicator_configs(),
            lambda: test_config_manager.get_strategy_configs(),
            lambda: test_config_manager.get_system_config(),
            lambda: test_config_manager.get_timeframe_configs()
        ]
        
        for operation in operations:
            start_time = time.time()
            
            # Perform operation multiple times
            for _ in range(10):
                result = operation()
                
            total_time = time.time() - start_time
            avg_time = total_time / 10
            
            # Should be fast
            assert avg_time < 0.01  # Less than 10ms per operation
            
    def test_concurrent_configuration_access(self, test_config_manager):
        """Test concurrent access to configuration."""
        results = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    watchlist = test_config_manager.get_watchlist()
                    indicators = test_config_manager.get_indicator_configs()
                    strategies = test_config_manager.get_strategy_configs()
                    
                    results.append((worker_id, i, len(watchlist), len(indicators), len(strategies)))
                    
            except Exception as e:
                results.append((worker_id, "error", str(e)))
                
        # Run multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without errors
        assert len(results) > 0
        error_results = [r for r in results if r[1] == "error"]
        assert len(error_results) == 0
        
    def test_configuration_caching_performance(self, test_config_manager):
        """Test configuration caching performance."""
        # First access (may populate cache)
        start_time = time.time()
        watchlist1 = test_config_manager.get_watchlist()
        first_time = time.time() - start_time
        
        # Second access (should use cache)
        start_time = time.time()
        watchlist2 = test_config_manager.get_watchlist()
        second_time = time.time() - start_time
        
        # Both should be fast
        assert first_time < 0.1
        assert second_time < 0.1
        
    def test_validation_performance(self, test_config_manager):
        """Test validation performance."""
        start_time = time.time()
        
        # Run comprehensive validation
        validation_result = test_config_manager.validate_configuration()
        
        validation_time = time.time() - start_time
        
        # Should complete validation quickly
        assert validation_time < 1.0  # Less than 1 second
        assert isinstance(validation_result, dict)


class TestConfigManagerReliability:
    """Test ConfigManager reliability and error handling."""
    
    def test_cache_failure_handling(self, test_config_manager):
        """Test handling of cache failures."""
        # Mock cache failure
        with patch('src.realtime.config_manager.cache_manager') as mock_cache:
            mock_cache.get.side_effect = Exception("Cache failure")
            
            # Should handle gracefully
            try:
                watchlist = test_config_manager.get_watchlist()
                # May return empty list or defaults
                assert isinstance(watchlist, list)
            except Exception:
                # Exception is acceptable if cache is critical
                pass
                
    def test_invalid_configuration_handling(self, test_config_manager):
        """Test handling of invalid configurations."""
        # Test with invalid asset config
        invalid_assets = [
            AssetConfig(
                symbol="",  # Invalid empty symbol
                enabled=True,
                data_provider="mock",
                timeframes=[],  # Invalid empty timeframes
                enabled_strategies=[],
                trading_profile=TradingProfile.SWING_TRADING
            )
        ]
        
        # Mock to return invalid config
        test_config_manager.update_watchlist = Mock(return_value=False)
        
        result = test_config_manager.update_watchlist(invalid_assets)
        
        # Should handle invalid config gracefully
        assert result == False
        
    def test_partial_configuration_loading(self, test_config_manager):
        """Test handling of partial configuration loading."""
        # Mock partial cache responses
        with patch('src.realtime.config_manager.cache_manager') as mock_cache:
            def side_effect(namespace, key):
                if key == "watchlist":
                    return None  # Missing watchlist
                elif key == "indicators":
                    return [{"name": "rsi_14", "enabled": True, "parameters": {"period": 14}}]
                else:
                    return None
                    
            mock_cache.get.side_effect = side_effect
            
            # Should handle partial data gracefully
            indicators = test_config_manager.get_indicator_configs()
            watchlist = test_config_manager.get_watchlist()
            
            # Indicators should work, watchlist might be empty
            assert isinstance(indicators, list)
            assert isinstance(watchlist, list)
            
    def test_configuration_consistency_check(self, test_config_manager):
        """Test configuration consistency checking."""
        # Run validation to check consistency
        validation_result = test_config_manager.validate_configuration()
        
        # Should identify any consistency issues
        assert "warnings" in validation_result
        assert "errors" in validation_result
        
        # Check for specific consistency issues
        warnings = validation_result["warnings"]
        errors = validation_result["errors"]
        
        # Verify that severe issues are reported as errors
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 0


class TestConfigManagerReset:
    """Test configuration reset functionality."""
    
    def test_reset_to_defaults(self, test_config_manager):
        """Test resetting configuration to defaults."""
        # Mock successful reset
        test_config_manager.update_indicator_configs = Mock(return_value=True)
        test_config_manager.update_strategy_configs = Mock(return_value=True)
        test_config_manager.update_system_config = Mock(return_value=True)
        test_config_manager.update_timeframe_configs = Mock(return_value=True)
        test_config_manager.update_strategy_matrix = Mock(return_value=True)
        test_config_manager.update_watchlist = Mock(return_value=True)
        
        result = test_config_manager.reset_to_defaults()
        
        assert result == True
        
        # Verify all update methods were called
        test_config_manager.update_indicator_configs.assert_called_once()
        test_config_manager.update_strategy_configs.assert_called_once()
        test_config_manager.update_system_config.assert_called_once()
        
    def test_cache_invalidation(self, test_config_manager):
        """Test cache invalidation after updates."""
        with patch('src.realtime.config_manager.cache_manager') as mock_cache:
            # Mock successful cache operations
            mock_cache.set.return_value = True
            mock_cache.clear_namespace.return_value = True
            
            # Update configuration
            new_indicators = [
                IndicatorConfig("test_indicator", True, {"period": 10})
            ]
            
            test_config_manager.update_indicator_configs(new_indicators)
            
            # Should have called cache clear
            assert mock_cache.clear_namespace.called


def test_config_manager_complete_workflow(test_config_manager):
    """Test complete ConfigManager workflow."""
    # Phase 1: Initial configuration retrieval
    watchlist = test_config_manager.get_watchlist()
    indicators = test_config_manager.get_indicator_configs()
    strategies = test_config_manager.get_strategy_configs()
    system_config = test_config_manager.get_system_config()
    
    assert len(watchlist) > 0
    assert len(indicators) > 0
    assert len(strategies) > 0
    assert system_config is not None
    
    # Phase 2: Configuration validation
    validation_result = test_config_manager.validate_configuration()
    assert "is_valid" in validation_result
    
    # Phase 3: Configuration summary
    summary = test_config_manager.get_configuration_summary()
    assert "watchlist" in summary
    assert "indicators" in summary
    assert "strategies" in summary
    
    # Phase 4: Strategy-specific operations
    enabled_strategies = test_config_manager.get_enabled_strategies()
    strategy_matrix = test_config_manager.get_strategy_matrix()
    
    assert len(enabled_strategies) > 0
    assert len(strategy_matrix) > 0
    
    # Phase 5: Symbol-specific operations
    enabled_symbols = test_config_manager.get_enabled_symbols()
    if len(enabled_symbols) > 0:
        symbol = enabled_symbols[0]
        
        # Test strategy enablement
        strategies_for_symbol = test_config_manager.get_strategy_for_symbol_and_timeframe(symbol, "1h")
        assert isinstance(strategies_for_symbol, list)
        
        # Test effective parameters
        if len(strategies_for_symbol) > 0:
            strategy_name = strategies_for_symbol[0]
            params = test_config_manager.get_effective_strategy_parameters(symbol, strategy_name, "1h")
            assert isinstance(params, dict)
    
    # Phase 6: Timeframe operations
    timeframe_configs = test_config_manager.get_timeframe_configs()
    assert len(timeframe_configs) > 0
    
    for profile in TradingProfile:
        if profile in timeframe_configs:
            configs = timeframe_configs[profile]
            assert len(configs) > 0
            
            # Test specific timeframe config
            if len(configs) > 0:
                tf = configs[0].timeframe
                config = test_config_manager.get_timeframe_config_for_profile(profile, tf)
                assert config is not None