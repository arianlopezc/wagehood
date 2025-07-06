"""
Configuration Manager for Real-time Market Data Processing

This module provides centralized configuration management for the real-time
market data system, including watchlist management, calculation parameters,
and runtime settings stored in Redis.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum

from src.storage.cache import cache_manager
from src.core.constants import SUPPORTED_SYMBOLS

logger = logging.getLogger(__name__)


class TradingProfile(Enum):
    """Trading profile that determines timeframe and update frequency settings."""
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: str
    update_interval_seconds: int
    lookback_periods: int
    min_data_points: int = 200
    priority: int = 1  # Higher priority = more frequent updates
    
    def validate(self) -> bool:
        """Validate timeframe configuration."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        return (
            self.timeframe in valid_timeframes and
            self.update_interval_seconds > 0 and
            self.lookback_periods > 0 and
            self.min_data_points > 0
        )


@dataclass
class StrategyTimeframeMatrix:
    """Maps strategies to their appropriate timeframes and trading profiles."""
    strategy_name: str
    supported_timeframes: List[str]
    optimal_timeframe: str
    trading_profile: TradingProfile
    min_bars_required: int = 200
    
    def is_timeframe_supported(self, timeframe: str) -> bool:
        """Check if a timeframe is supported for this strategy."""
        return timeframe in self.supported_timeframes


@dataclass
class SymbolStrategyConfig:
    """Per-symbol strategy enablement and configuration."""
    symbol: str
    enabled_strategies: List[str]
    strategy_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trading_profile: TradingProfile = TradingProfile.SWING_TRADING
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled for this symbol."""
        return strategy_name in self.enabled_strategies
    
    def get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy parameters with overrides applied."""
        return self.strategy_overrides.get(strategy_name, {})


@dataclass
class IndicatorConfig:
    """Configuration for a specific technical indicator."""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    update_frequency_seconds: int = 1
    ttl_seconds: int = 300


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    required_indicators: List[str]
    update_frequency_seconds: int = 1
    ttl_seconds: int = 600
    # Multi-timeframe support
    supported_timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '1h', '1d'])
    optimal_timeframe: str = '5m'
    trading_profile: TradingProfile = TradingProfile.SWING_TRADING
    timeframe_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AssetConfig:
    """Configuration for a specific asset/symbol."""
    symbol: str
    enabled: bool
    data_provider: str
    timeframes: List[str]
    priority: int = 1  # Higher priority = more frequent updates
    asset_type: str = "stock"  # stock, crypto, forex
    last_updated: Optional[datetime] = None
    # Multi-strategy support
    enabled_strategies: List[str] = field(default_factory=list)
    trading_profile: TradingProfile = TradingProfile.SWING_TRADING
    strategy_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    max_concurrent_calculations: int = 100
    batch_calculation_size: int = 10
    data_update_interval_seconds: int = 1
    calculation_workers: int = 4
    redis_streams_maxlen: int = 10000
    enable_monitoring: bool = True
    enable_alerts: bool = True


class ConfigManager:
    """
    Centralized configuration manager for real-time market data processing.
    
    This class manages all configuration aspects including:
    - Watchlist symbols and their settings
    - Technical indicator configurations
    - Trading strategy configurations  
    - System performance settings
    - Runtime parameter updates
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._config_namespace = "realtime_config"
        self._cache_ttl = 3600  # 1 hour cache for config
        
        # Default configurations
        self._default_indicators = self._get_default_indicators()
        self._default_strategies = self._get_default_strategies()
        self._default_system_config = self._get_default_system_config()
        self._default_timeframe_configs = self.get_default_timeframe_configs()
        self._default_strategy_matrix = self.get_default_strategy_matrix()
        
        # Initialize configuration if not exists
        self._initialize_config()
    
    def get_default_timeframe_configs(self) -> Dict[TradingProfile, List[TimeframeConfig]]:
        """Get default timeframe configurations for each trading profile."""
        return {
            TradingProfile.DAY_TRADING: [
                TimeframeConfig(
                    timeframe='1m',
                    update_interval_seconds=1,
                    lookback_periods=500,
                    min_data_points=200,
                    priority=3
                ),
                TimeframeConfig(
                    timeframe='5m',
                    update_interval_seconds=5,
                    lookback_periods=300,
                    min_data_points=100,
                    priority=2
                ),
                TimeframeConfig(
                    timeframe='15m',
                    update_interval_seconds=15,
                    lookback_periods=200,
                    min_data_points=100,
                    priority=1
                )
            ],
            TradingProfile.SWING_TRADING: [
                TimeframeConfig(
                    timeframe='5m',
                    update_interval_seconds=5,
                    lookback_periods=300,
                    min_data_points=200,
                    priority=2
                ),
                TimeframeConfig(
                    timeframe='1h',
                    update_interval_seconds=60,
                    lookback_periods=200,
                    min_data_points=100,
                    priority=3
                ),
                TimeframeConfig(
                    timeframe='4h',
                    update_interval_seconds=300,
                    lookback_periods=150,
                    min_data_points=50,
                    priority=2
                ),
                TimeframeConfig(
                    timeframe='1d',
                    update_interval_seconds=3600,
                    lookback_periods=100,
                    min_data_points=50,
                    priority=1
                )
            ],
            TradingProfile.POSITION_TRADING: [
                TimeframeConfig(
                    timeframe='1h',
                    update_interval_seconds=300,
                    lookback_periods=500,
                    min_data_points=200,
                    priority=1
                ),
                TimeframeConfig(
                    timeframe='4h',
                    update_interval_seconds=900,
                    lookback_periods=300,
                    min_data_points=100,
                    priority=2
                ),
                TimeframeConfig(
                    timeframe='1d',
                    update_interval_seconds=3600,
                    lookback_periods=200,
                    min_data_points=100,
                    priority=3
                ),
                TimeframeConfig(
                    timeframe='1w',
                    update_interval_seconds=86400,
                    lookback_periods=52,
                    min_data_points=52,
                    priority=1
                )
            ]
        }
    
    def get_default_strategy_matrix(self) -> List[StrategyTimeframeMatrix]:
        """Get default strategy-timeframe matrix configurations."""
        return [
            StrategyTimeframeMatrix(
                strategy_name="macd_rsi_strategy",
                supported_timeframes=['5m', '15m', '1h', '4h', '1d'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                min_bars_required=200
            ),
            StrategyTimeframeMatrix(
                strategy_name="ma_crossover_strategy",
                supported_timeframes=['15m', '1h', '4h', '1d', '1w'],
                optimal_timeframe='1d',
                trading_profile=TradingProfile.POSITION_TRADING,
                min_bars_required=250
            ),
            StrategyTimeframeMatrix(
                strategy_name="rsi_trend_strategy",
                supported_timeframes=['5m', '15m', '1h', '4h'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                min_bars_required=150
            ),
            StrategyTimeframeMatrix(
                strategy_name="bollinger_breakout_strategy",
                supported_timeframes=['1m', '5m', '15m', '1h'],
                optimal_timeframe='15m',
                trading_profile=TradingProfile.DAY_TRADING,
                min_bars_required=100
            ),
            StrategyTimeframeMatrix(
                strategy_name="sr_breakout_strategy",
                supported_timeframes=['5m', '15m', '1h', '4h'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                min_bars_required=200
            )
        ]
    
    def _get_default_indicators(self) -> List[IndicatorConfig]:
        """Get default indicator configurations."""
        return [
            IndicatorConfig(
                name="sma_50",
                enabled=True,
                parameters={"period": 50},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="sma_200", 
                enabled=True,
                parameters={"period": 200},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="ema_12",
                enabled=True,
                parameters={"period": 12},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="ema_26",
                enabled=True,
                parameters={"period": 26},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="rsi_14",
                enabled=True,
                parameters={"period": 14},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="macd",
                enabled=True,
                parameters={"fast": 12, "slow": 26, "signal": 9},
                update_frequency_seconds=1,
                ttl_seconds=300
            ),
            IndicatorConfig(
                name="bollinger_bands",
                enabled=True,
                parameters={"period": 20, "std_dev": 2.0},
                update_frequency_seconds=1,
                ttl_seconds=300
            )
        ]
    
    def _get_default_strategies(self) -> List[StrategyConfig]:
        """Get default strategy configurations."""
        return [
            StrategyConfig(
                name="macd_rsi_strategy",
                enabled=True,
                parameters={
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9
                },
                required_indicators=["rsi_14", "macd"],
                update_frequency_seconds=1,
                ttl_seconds=600,
                supported_timeframes=['5m', '15m', '1h', '4h', '1d'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                timeframe_parameters={
                    '5m': {'rsi_overbought': 75, 'rsi_oversold': 25},
                    '1h': {'rsi_overbought': 70, 'rsi_oversold': 30},
                    '1d': {'rsi_overbought': 65, 'rsi_oversold': 35}
                }
            ),
            StrategyConfig(
                name="ma_crossover_strategy",
                enabled=True,
                parameters={
                    "fast_period": 50,
                    "slow_period": 200
                },
                required_indicators=["sma_50", "sma_200"],
                update_frequency_seconds=1,
                ttl_seconds=600,
                supported_timeframes=['15m', '1h', '4h', '1d', '1w'],
                optimal_timeframe='1d',
                trading_profile=TradingProfile.POSITION_TRADING,
                timeframe_parameters={
                    '15m': {'fast_period': 20, 'slow_period': 50},
                    '1h': {'fast_period': 50, 'slow_period': 200},
                    '1d': {'fast_period': 50, 'slow_period': 200}
                }
            ),
            StrategyConfig(
                name="rsi_trend_strategy",
                enabled=True,
                parameters={
                    "rsi_period": 14,
                    "trend_sma_period": 50,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                },
                required_indicators=["rsi_14", "sma_50"],
                update_frequency_seconds=1,
                ttl_seconds=600,
                supported_timeframes=['5m', '15m', '1h', '4h'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                timeframe_parameters={
                    '5m': {'rsi_overbought': 75, 'rsi_oversold': 25},
                    '1h': {'rsi_overbought': 70, 'rsi_oversold': 30}
                }
            ),
            StrategyConfig(
                name="bollinger_breakout_strategy",
                enabled=True,
                parameters={
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "volume_threshold": 1.5
                },
                required_indicators=["bollinger_bands"],
                update_frequency_seconds=1,
                ttl_seconds=600,
                supported_timeframes=['1m', '5m', '15m', '1h'],
                optimal_timeframe='15m',
                trading_profile=TradingProfile.DAY_TRADING,
                timeframe_parameters={
                    '1m': {'bb_std_dev': 2.5, 'volume_threshold': 2.0},
                    '5m': {'bb_std_dev': 2.0, 'volume_threshold': 1.5},
                    '15m': {'bb_std_dev': 1.8, 'volume_threshold': 1.2}
                }
            ),
            StrategyConfig(
                name="sr_breakout_strategy",
                enabled=True,
                parameters={
                    "lookback_period": 20,
                    "min_touches": 3,
                    "breakout_threshold": 0.01
                },
                required_indicators=["sma_50"],
                update_frequency_seconds=1,
                ttl_seconds=600,
                supported_timeframes=['5m', '15m', '1h', '4h'],
                optimal_timeframe='1h',
                trading_profile=TradingProfile.SWING_TRADING,
                timeframe_parameters={
                    '5m': {'lookback_period': 10, 'breakout_threshold': 0.005},
                    '1h': {'lookback_period': 20, 'breakout_threshold': 0.01},
                    '4h': {'lookback_period': 30, 'breakout_threshold': 0.015}
                }
            )
        ]
    
    def _get_default_system_config(self) -> SystemConfig:
        """Get default system configuration."""
        return SystemConfig(
            max_concurrent_calculations=int(os.environ.get('MAX_CONCURRENT_CALCULATIONS', 100)),
            batch_calculation_size=int(os.environ.get('BATCH_CALCULATION_SIZE', 10)),
            data_update_interval_seconds=int(os.environ.get('DATA_UPDATE_INTERVAL', 1)),
            calculation_workers=int(os.environ.get('CALCULATION_WORKERS', 4)),
            redis_streams_maxlen=int(os.environ.get('REDIS_STREAMS_MAXLEN', 10000)),
            enable_monitoring=os.environ.get('ENABLE_MONITORING', 'true').lower() == 'true',
            enable_alerts=os.environ.get('ENABLE_ALERTS', 'true').lower() == 'true'
        )
    
    def _initialize_config(self):
        """Initialize configuration in Redis if not exists."""
        try:
            # Initialize watchlist if not exists
            if not self.get_watchlist():
                default_symbols = os.environ.get('WATCHLIST_SYMBOLS', 'SPY,QQQ,IWM').split(',')
                default_assets = [
                    AssetConfig(
                        symbol=symbol.strip(),
                        enabled=True,
                        data_provider=os.environ.get('DATA_PROVIDER', 'mock'),
                        timeframes=['1m', '5m', '1h', '1d'],
                        priority=1,
                        enabled_strategies=['macd_rsi_strategy', 'ma_crossover_strategy', 'rsi_trend_strategy'],
                        trading_profile=TradingProfile.SWING_TRADING
                    ) for symbol in default_symbols if symbol.strip()
                ]
                self.update_watchlist(default_assets)
            
            # Initialize indicators if not exists
            if not self.get_indicator_configs():
                self.update_indicator_configs(self._default_indicators)
            
            # Initialize strategies if not exists
            if not self.get_strategy_configs():
                self.update_strategy_configs(self._default_strategies)
            
            # Initialize system config if not exists
            if not self.get_system_config():
                self.update_system_config(self._default_system_config)
            
            # Initialize timeframe configs if not exists
            if not self.get_timeframe_configs():
                self.update_timeframe_configs(self._default_timeframe_configs)
            
            # Initialize strategy matrix if not exists
            if not self.get_strategy_matrix():
                self.update_strategy_matrix(self._default_strategy_matrix)
                
            logger.info("Configuration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise
    
    def get_watchlist(self) -> List[AssetConfig]:
        """
        Get the current watchlist of assets to monitor.
        
        Returns:
            List of AssetConfig objects for monitored assets
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "watchlist")
            if cached_data:
                assets = []
                for asset_data in cached_data:
                    # Handle backward compatibility for new fields
                    if 'enabled_strategies' not in asset_data:
                        asset_data['enabled_strategies'] = []
                    if 'trading_profile' not in asset_data:
                        asset_data['trading_profile'] = TradingProfile.SWING_TRADING
                    elif isinstance(asset_data['trading_profile'], str):
                        asset_data['trading_profile'] = TradingProfile(asset_data['trading_profile'])
                    if 'strategy_overrides' not in asset_data:
                        asset_data['strategy_overrides'] = {}
                    
                    assets.append(AssetConfig(**asset_data))
                return assets
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get watchlist: {e}")
            return []
    
    def update_watchlist(self, assets: List[AssetConfig]) -> bool:
        """
        Update the watchlist of assets to monitor.
        
        Args:
            assets: List of AssetConfig objects
            
        Returns:
            True if update was successful
        """
        try:
            # Validate symbols
            for asset in assets:
                if asset.symbol not in SUPPORTED_SYMBOLS:
                    logger.warning(f"Symbol {asset.symbol} not in supported symbols list")
            
            # Convert to serializable format
            asset_data = []
            for asset in assets:
                data = asdict(asset)
                # Convert enum to string for serialization
                if isinstance(data.get('trading_profile'), TradingProfile):
                    data['trading_profile'] = data['trading_profile'].value
                data['last_updated'] = datetime.now().isoformat()
                asset_data.append(data)
            
            # Store in cache
            success = cache_manager.set(
                self._config_namespace, 
                "watchlist", 
                asset_data, 
                self._cache_ttl
            )
            
            if success:
                logger.info(f"Updated watchlist with {len(assets)} assets")
                # Invalidate related caches
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update watchlist: {e}")
            return False
    
    def add_symbol(self, symbol: str, data_provider: str = None, 
                   timeframes: List[str] = None, priority: int = 1,
                   enabled_strategies: List[str] = None,
                   trading_profile: TradingProfile = None) -> bool:
        """
        Add a symbol to the watchlist.
        
        Args:
            symbol: Trading symbol to add
            data_provider: Data provider for this symbol
            timeframes: List of timeframes to monitor
            priority: Priority level (higher = more frequent updates)
            enabled_strategies: List of strategies to enable for this symbol
            trading_profile: Trading profile for this symbol
            
        Returns:
            True if symbol was added successfully
        """
        try:
            current_watchlist = self.get_watchlist()
            
            # Check if symbol already exists
            for asset in current_watchlist:
                if asset.symbol == symbol:
                    logger.warning(f"Symbol {symbol} already in watchlist")
                    return False
            
            # Create new asset config
            new_asset = AssetConfig(
                symbol=symbol,
                enabled=True,
                data_provider=data_provider or os.environ.get('DATA_PROVIDER', 'mock'),
                timeframes=timeframes or ['1m', '5m', '1h', '1d'],
                priority=priority,
                enabled_strategies=enabled_strategies or ['macd_rsi_strategy', 'rsi_trend_strategy'],
                trading_profile=trading_profile or TradingProfile.SWING_TRADING
            )
            
            # Add to watchlist
            current_watchlist.append(new_asset)
            return self.update_watchlist(current_watchlist)
            
        except Exception as e:
            logger.error(f"Failed to add symbol {symbol}: {e}")
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the watchlist.
        
        Args:
            symbol: Trading symbol to remove
            
        Returns:
            True if symbol was removed successfully
        """
        try:
            current_watchlist = self.get_watchlist()
            updated_watchlist = [asset for asset in current_watchlist if asset.symbol != symbol]
            
            if len(updated_watchlist) == len(current_watchlist):
                logger.warning(f"Symbol {symbol} not found in watchlist")
                return False
            
            return self.update_watchlist(updated_watchlist)
            
        except Exception as e:
            logger.error(f"Failed to remove symbol {symbol}: {e}")
            return False
    
    def get_enabled_symbols(self) -> List[str]:
        """
        Get list of enabled symbols from watchlist.
        
        Returns:
            List of enabled trading symbols
        """
        watchlist = self.get_watchlist()
        return [asset.symbol for asset in watchlist if asset.enabled]
    
    def get_indicator_configs(self) -> List[IndicatorConfig]:
        """
        Get current indicator configurations.
        
        Returns:
            List of IndicatorConfig objects
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "indicators")
            if cached_data:
                return [IndicatorConfig(**indicator_data) for indicator_data in cached_data]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get indicator configs: {e}")
            return []
    
    def update_indicator_configs(self, indicators: List[IndicatorConfig]) -> bool:
        """
        Update indicator configurations.
        
        Args:
            indicators: List of IndicatorConfig objects
            
        Returns:
            True if update was successful
        """
        try:
            indicator_data = [asdict(indicator) for indicator in indicators]
            
            success = cache_manager.set(
                self._config_namespace,
                "indicators",
                indicator_data,
                self._cache_ttl
            )
            
            if success:
                logger.info(f"Updated {len(indicators)} indicator configurations")
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update indicator configs: {e}")
            return False
    
    def get_enabled_indicators(self) -> List[IndicatorConfig]:
        """
        Get list of enabled indicators.
        
        Returns:
            List of enabled IndicatorConfig objects
        """
        indicators = self.get_indicator_configs()
        return [indicator for indicator in indicators if indicator.enabled]
    
    def get_strategy_configs(self) -> List[StrategyConfig]:
        """
        Get current strategy configurations.
        
        Returns:
            List of StrategyConfig objects
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "strategies")
            if cached_data:
                strategies = []
                for strategy_data in cached_data:
                    # Handle backward compatibility for new fields
                    if 'supported_timeframes' not in strategy_data:
                        strategy_data['supported_timeframes'] = ['1m', '5m', '1h', '1d']
                    if 'optimal_timeframe' not in strategy_data:
                        strategy_data['optimal_timeframe'] = '5m'
                    if 'trading_profile' not in strategy_data:
                        strategy_data['trading_profile'] = TradingProfile.SWING_TRADING
                    elif isinstance(strategy_data['trading_profile'], str):
                        strategy_data['trading_profile'] = TradingProfile(strategy_data['trading_profile'])
                    if 'timeframe_parameters' not in strategy_data:
                        strategy_data['timeframe_parameters'] = {}
                    
                    strategies.append(StrategyConfig(**strategy_data))
                return strategies
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get strategy configs: {e}")
            return []
    
    def update_strategy_configs(self, strategies: List[StrategyConfig]) -> bool:
        """
        Update strategy configurations.
        
        Args:
            strategies: List of StrategyConfig objects
            
        Returns:
            True if update was successful
        """
        try:
            strategy_data = []
            for strategy in strategies:
                data = asdict(strategy)
                # Convert enum to string for serialization
                if isinstance(data.get('trading_profile'), TradingProfile):
                    data['trading_profile'] = data['trading_profile'].value
                strategy_data.append(data)
            
            success = cache_manager.set(
                self._config_namespace,
                "strategies",
                strategy_data,
                self._cache_ttl
            )
            
            if success:
                logger.info(f"Updated {len(strategies)} strategy configurations")
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update strategy configs: {e}")
            return False
    
    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """
        Get list of enabled strategies.
        
        Returns:
            List of enabled StrategyConfig objects
        """
        strategies = self.get_strategy_configs()
        return [strategy for strategy in strategies if strategy.enabled]
    
    def get_timeframe_configs(self) -> Dict[TradingProfile, List[TimeframeConfig]]:
        """
        Get current timeframe configurations.
        
        Returns:
            Dictionary mapping trading profiles to their timeframe configs
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "timeframe_configs")
            if cached_data:
                result = {}
                for profile_str, timeframe_list in cached_data.items():
                    profile = TradingProfile(profile_str)
                    result[profile] = [TimeframeConfig(**tf_data) for tf_data in timeframe_list]
                return result
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get timeframe configs: {e}")
            return {}
    
    def update_timeframe_configs(self, configs: Dict[TradingProfile, List[TimeframeConfig]]) -> bool:
        """
        Update timeframe configurations.
        
        Args:
            configs: Dictionary mapping trading profiles to timeframe configs
            
        Returns:
            True if update was successful
        """
        try:
            # Convert to serializable format
            serialized_configs = {}
            for profile, timeframe_configs in configs.items():
                serialized_configs[profile.value] = [asdict(tf_config) for tf_config in timeframe_configs]
            
            success = cache_manager.set(
                self._config_namespace,
                "timeframe_configs",
                serialized_configs,
                self._cache_ttl
            )
            
            if success:
                logger.info(f"Updated timeframe configurations for {len(configs)} profiles")
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update timeframe configs: {e}")
            return False
    
    def get_strategy_matrix(self) -> List[StrategyTimeframeMatrix]:
        """
        Get current strategy-timeframe matrix.
        
        Returns:
            List of StrategyTimeframeMatrix objects
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "strategy_matrix")
            if cached_data:
                result = []
                for matrix_data in cached_data:
                    # Convert trading_profile string back to enum
                    matrix_data['trading_profile'] = TradingProfile(matrix_data['trading_profile'])
                    result.append(StrategyTimeframeMatrix(**matrix_data))
                return result
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get strategy matrix: {e}")
            return []
    
    def update_strategy_matrix(self, matrix: List[StrategyTimeframeMatrix]) -> bool:
        """
        Update strategy-timeframe matrix.
        
        Args:
            matrix: List of StrategyTimeframeMatrix objects
            
        Returns:
            True if update was successful
        """
        try:
            # Convert to serializable format
            matrix_data = []
            for item in matrix:
                item_dict = asdict(item)
                # Convert enum to string
                item_dict['trading_profile'] = item.trading_profile.value
                matrix_data.append(item_dict)
            
            success = cache_manager.set(
                self._config_namespace,
                "strategy_matrix",
                matrix_data,
                self._cache_ttl
            )
            
            if success:
                logger.info(f"Updated strategy matrix with {len(matrix)} entries")
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update strategy matrix: {e}")
            return False
    
    def get_symbol_strategy_configs(self) -> List[SymbolStrategyConfig]:
        """
        Get per-symbol strategy configurations.
        
        Returns:
            List of SymbolStrategyConfig objects
        """
        try:
            watchlist = self.get_watchlist()
            result = []
            
            for asset in watchlist:
                symbol_config = SymbolStrategyConfig(
                    symbol=asset.symbol,
                    enabled_strategies=asset.enabled_strategies,
                    strategy_overrides=asset.strategy_overrides,
                    trading_profile=asset.trading_profile
                )
                result.append(symbol_config)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get symbol strategy configs: {e}")
            return []
    
    def update_symbol_strategy_config(self, symbol: str, config: SymbolStrategyConfig) -> bool:
        """
        Update strategy configuration for a specific symbol.
        
        Args:
            symbol: Trading symbol
            config: SymbolStrategyConfig object
            
        Returns:
            True if update was successful
        """
        try:
            watchlist = self.get_watchlist()
            
            # Find and update the asset
            for asset in watchlist:
                if asset.symbol == symbol:
                    asset.enabled_strategies = config.enabled_strategies
                    asset.strategy_overrides = config.strategy_overrides
                    asset.trading_profile = config.trading_profile
                    break
            else:
                logger.warning(f"Symbol {symbol} not found in watchlist")
                return False
            
            return self.update_watchlist(watchlist)
            
        except Exception as e:
            logger.error(f"Failed to update symbol strategy config for {symbol}: {e}")
            return False
    
    def get_system_config(self) -> Optional[SystemConfig]:
        """
        Get current system configuration.
        
        Returns:
            SystemConfig object or None if not found
        """
        try:
            cached_data = cache_manager.get(self._config_namespace, "system")
            if cached_data:
                return SystemConfig(**cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get system config: {e}")
            return None
    
    def update_system_config(self, config: SystemConfig) -> bool:
        """
        Update system configuration.
        
        Args:
            config: SystemConfig object
            
        Returns:
            True if update was successful
        """
        try:
            config_data = asdict(config)
            
            success = cache_manager.set(
                self._config_namespace,
                "system",
                config_data,
                self._cache_ttl
            )
            
            if success:
                logger.info("Updated system configuration")
                self._invalidate_dependent_caches()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update system config: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all current configurations.
        
        Returns:
            Dictionary with configuration summary
        """
        try:
            watchlist = self.get_watchlist()
            indicators = self.get_indicator_configs()
            strategies = self.get_strategy_configs()
            system_config = self.get_system_config()
            
            return {
                "watchlist": {
                    "total_symbols": len(watchlist),
                    "enabled_symbols": len([a for a in watchlist if a.enabled]),
                    "symbols": [a.symbol for a in watchlist if a.enabled]
                },
                "indicators": {
                    "total_indicators": len(indicators),
                    "enabled_indicators": len([i for i in indicators if i.enabled]),
                    "indicator_names": [i.name for i in indicators if i.enabled]
                },
                "strategies": {
                    "total_strategies": len(strategies),
                    "enabled_strategies": len([s for s in strategies if s.enabled]),
                    "strategy_names": [s.name for s in strategies if s.enabled]
                },
                "system": asdict(system_config) if system_config else None,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get configuration summary: {e}")
            return {"error": str(e)}
    
    def validate_timeframe_strategy_combinations(self) -> Dict[str, List[str]]:
        """
        Validate that timeframe-strategy combinations are valid.
        
        Returns:
            Dictionary with validation results
        """
        warnings = []
        errors = []
        
        try:
            strategies = self.get_strategy_configs()
            matrix = self.get_strategy_matrix()
            watchlist = self.get_watchlist()
            
            # Create lookup for strategy matrix
            matrix_lookup = {item.strategy_name: item for item in matrix}
            
            # Validate each symbol's enabled strategies
            for asset in watchlist:
                if not asset.enabled:
                    continue
                    
                for strategy_name in asset.enabled_strategies:
                    # Check if strategy exists
                    strategy_config = next((s for s in strategies if s.name == strategy_name), None)
                    if not strategy_config:
                        errors.append(f"Strategy '{strategy_name}' not found for symbol {asset.symbol}")
                        continue
                    
                    # Check if strategy is enabled
                    if not strategy_config.enabled:
                        warnings.append(f"Strategy '{strategy_name}' is disabled for symbol {asset.symbol}")
                    
                    # Check timeframe compatibility
                    matrix_entry = matrix_lookup.get(strategy_name)
                    if matrix_entry:
                        for timeframe in asset.timeframes:
                            if timeframe not in matrix_entry.supported_timeframes:
                                warnings.append(
                                    f"Timeframe '{timeframe}' not supported by strategy '{strategy_name}' "
                                    f"for symbol {asset.symbol}"
                                )
                    
                    # Check trading profile compatibility
                    if matrix_entry and asset.trading_profile != matrix_entry.trading_profile:
                        warnings.append(
                            f"Trading profile mismatch for symbol {asset.symbol}: "
                            f"asset={asset.trading_profile.value}, "
                            f"strategy={matrix_entry.trading_profile.value}"
                        )
            
        except Exception as e:
            errors.append(f"Validation failed: {e}")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "is_valid": len(errors) == 0
        }
    
    def validate_update_intervals(self) -> Dict[str, List[str]]:
        """
        Validate that update intervals are appropriate for each trading profile.
        
        Returns:
            Dictionary with validation results
        """
        warnings = []
        errors = []
        
        try:
            timeframe_configs = self.get_timeframe_configs()
            
            for profile, configs in timeframe_configs.items():
                for config in configs:
                    if not config.validate():
                        errors.append(f"Invalid timeframe config for {profile.value}: {config.timeframe}")
                    
                    # Check reasonable intervals based on profile
                    if profile == TradingProfile.DAY_TRADING:
                        if config.timeframe in ['1m', '5m'] and config.update_interval_seconds > 10:
                            warnings.append(
                                f"Update interval too slow for day trading on {config.timeframe}: "
                                f"{config.update_interval_seconds}s"
                            )
                    elif profile == TradingProfile.SWING_TRADING:
                        if config.timeframe in ['1h', '4h'] and config.update_interval_seconds > 3600:
                            warnings.append(
                                f"Update interval too slow for swing trading on {config.timeframe}: "
                                f"{config.update_interval_seconds}s"
                            )
                    elif profile == TradingProfile.POSITION_TRADING:
                        if config.timeframe == '1d' and config.update_interval_seconds > 86400:
                            warnings.append(
                                f"Update interval too slow for position trading on {config.timeframe}: "
                                f"{config.update_interval_seconds}s"
                            )
            
        except Exception as e:
            errors.append(f"Update interval validation failed: {e}")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "is_valid": len(errors) == 0
        }
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate current configuration for potential issues.
        
        Returns:
            Dictionary with validation results (warnings and errors)
        """
        warnings = []
        errors = []
        
        try:
            # Validate watchlist
            watchlist = self.get_watchlist()
            if not watchlist:
                errors.append("No symbols in watchlist")
            
            enabled_symbols = self.get_enabled_symbols()
            if not enabled_symbols:
                errors.append("No enabled symbols in watchlist")
            
            # Check for unsupported symbols
            for symbol in enabled_symbols:
                if symbol not in SUPPORTED_SYMBOLS:
                    warnings.append(f"Symbol {symbol} not in supported symbols list")
            
            # Validate indicators
            indicators = self.get_enabled_indicators()
            if not indicators:
                warnings.append("No enabled indicators")
            
            # Validate strategies
            strategies = self.get_enabled_strategies()
            if not strategies:
                warnings.append("No enabled strategies")
            
            # Check strategy dependencies
            enabled_indicator_names = {indicator.name for indicator in indicators}
            for strategy in strategies:
                for required_indicator in strategy.required_indicators:
                    if required_indicator not in enabled_indicator_names:
                        errors.append(
                            f"Strategy '{strategy.name}' requires indicator '{required_indicator}' "
                            f"which is not enabled"
                        )
            
            # Validate system config
            system_config = self.get_system_config()
            if not system_config:
                errors.append("System configuration not found")
            elif system_config.calculation_workers < 1:
                errors.append("At least 1 calculation worker is required")
            
            # Validate timeframe-strategy combinations
            timeframe_validation = self.validate_timeframe_strategy_combinations()
            warnings.extend(timeframe_validation["warnings"])
            errors.extend(timeframe_validation["errors"])
            
            # Validate update intervals
            interval_validation = self.validate_update_intervals()
            warnings.extend(interval_validation["warnings"])
            errors.extend(interval_validation["errors"])
            
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "is_valid": len(errors) == 0
        }
    
    def _invalidate_dependent_caches(self):
        """Invalidate caches that depend on configuration."""
        try:
            # Clear calculation results when config changes
            cache_manager.clear_namespace("indicators")
            cache_manager.clear_namespace("strategies")
            logger.debug("Invalidated dependent caches after configuration update")
        except Exception as e:
            logger.warning(f"Failed to invalidate dependent caches: {e}")
    
    def get_strategy_for_symbol_and_timeframe(self, symbol: str, timeframe: str) -> List[str]:
        """
        Get enabled strategies for a specific symbol and timeframe combination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')
            
        Returns:
            List of enabled strategy names
        """
        try:
            watchlist = self.get_watchlist()
            matrix = self.get_strategy_matrix()
            
            # Find the asset
            asset = next((a for a in watchlist if a.symbol == symbol), None)
            if not asset or not asset.enabled:
                return []
            
            # Check if timeframe is supported by the asset
            if timeframe not in asset.timeframes:
                return []
            
            # Filter strategies by timeframe support
            matrix_lookup = {item.strategy_name: item for item in matrix}
            enabled_strategies = []
            
            for strategy_name in asset.enabled_strategies:
                matrix_entry = matrix_lookup.get(strategy_name)
                if matrix_entry and timeframe in matrix_entry.supported_timeframes:
                    enabled_strategies.append(strategy_name)
            
            return enabled_strategies
            
        except Exception as e:
            logger.error(f"Failed to get strategies for {symbol}@{timeframe}: {e}")
            return []
    
    def get_timeframe_config_for_profile(self, profile: TradingProfile, timeframe: str) -> Optional[TimeframeConfig]:
        """
        Get timeframe configuration for a specific trading profile and timeframe.
        
        Args:
            profile: Trading profile
            timeframe: Timeframe string
            
        Returns:
            TimeframeConfig object or None if not found
        """
        try:
            timeframe_configs = self.get_timeframe_configs()
            profile_configs = timeframe_configs.get(profile, [])
            
            for config in profile_configs:
                if config.timeframe == timeframe:
                    return config
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get timeframe config for {profile.value}@{timeframe}: {e}")
            return None
    
    def get_effective_strategy_parameters(self, symbol: str, strategy_name: str, timeframe: str) -> Dict[str, Any]:
        """
        Get effective strategy parameters with symbol and timeframe overrides applied.
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            timeframe: Timeframe string
            
        Returns:
            Dictionary of effective parameters
        """
        try:
            # Get base strategy parameters
            strategies = self.get_strategy_configs()
            strategy = next((s for s in strategies if s.name == strategy_name), None)
            if not strategy:
                return {}
            
            # Start with base parameters
            params = strategy.parameters.copy()
            
            # Apply timeframe-specific parameters
            if timeframe in strategy.timeframe_parameters:
                params.update(strategy.timeframe_parameters[timeframe])
            
            # Apply symbol-specific overrides
            watchlist = self.get_watchlist()
            asset = next((a for a in watchlist if a.symbol == symbol), None)
            if asset and strategy_name in asset.strategy_overrides:
                params.update(asset.strategy_overrides[strategy_name])
            
            return params
            
        except Exception as e:
            logger.error(f"Failed to get effective parameters for {symbol}@{strategy_name}@{timeframe}: {e}")
            return {}
    
    def is_strategy_enabled_for_symbol_timeframe(self, symbol: str, strategy_name: str, timeframe: str) -> bool:
        """
        Check if a strategy is enabled for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            timeframe: Timeframe string
            
        Returns:
            True if strategy is enabled and supported for this combination
        """
        try:
            enabled_strategies = self.get_strategy_for_symbol_and_timeframe(symbol, timeframe)
            return strategy_name in enabled_strategies
            
        except Exception as e:
            logger.error(f"Failed to check strategy enablement for {symbol}@{strategy_name}@{timeframe}: {e}")
            return False
    
    def get_optimal_timeframe_for_strategy(self, strategy_name: str) -> Optional[str]:
        """
        Get the optimal timeframe for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Optimal timeframe string or None if not found
        """
        try:
            matrix = self.get_strategy_matrix()
            matrix_entry = next((m for m in matrix if m.strategy_name == strategy_name), None)
            
            if matrix_entry:
                return matrix_entry.optimal_timeframe
            
            # Fallback to strategy config
            strategies = self.get_strategy_configs()
            strategy = next((s for s in strategies if s.name == strategy_name), None)
            if strategy:
                return strategy.optimal_timeframe
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get optimal timeframe for {strategy_name}: {e}")
            return None
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all configuration to defaults.
        
        Returns:
            True if reset was successful
        """
        try:
            # Reset to defaults
            success = (
                self.update_indicator_configs(self._default_indicators) and
                self.update_strategy_configs(self._default_strategies) and
                self.update_system_config(self._default_system_config) and
                self.update_timeframe_configs(self._default_timeframe_configs) and
                self.update_strategy_matrix(self._default_strategy_matrix)
            )
            
            # Reset watchlist to environment defaults
            default_symbols = os.environ.get('WATCHLIST_SYMBOLS', 'SPY,QQQ,IWM').split(',')
            default_assets = [
                AssetConfig(
                    symbol=symbol.strip(),
                    enabled=True,
                    data_provider=os.environ.get('DATA_PROVIDER', 'mock'),
                    timeframes=['1m', '5m', '1h', '1d'],
                    priority=1,
                    enabled_strategies=['macd_rsi_strategy', 'ma_crossover_strategy', 'rsi_trend_strategy'],
                    trading_profile=TradingProfile.SWING_TRADING
                ) for symbol in default_symbols if symbol.strip()
            ]
            success = success and self.update_watchlist(default_assets)
            
            if success:
                logger.info("Configuration reset to defaults successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to reset configuration to defaults: {e}")
            return False