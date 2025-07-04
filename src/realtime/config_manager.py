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
from dataclasses import dataclass, asdict

from ..storage.cache import cache_manager
from ..core.constants import SUPPORTED_SYMBOLS

logger = logging.getLogger(__name__)


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


@dataclass
class AssetConfig:
    """Configuration for a specific asset/symbol."""
    symbol: str
    enabled: bool
    data_provider: str
    timeframes: List[str]
    priority: int = 1  # Higher priority = more frequent updates
    last_updated: Optional[datetime] = None


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
        
        # Initialize configuration if not exists
        self._initialize_config()
    
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
                ttl_seconds=600
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
                ttl_seconds=600
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
                ttl_seconds=600
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
                ttl_seconds=600
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
                ttl_seconds=600
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
                        priority=1
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
                return [AssetConfig(**asset_data) for asset_data in cached_data]
            
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
            asset_data = [asdict(asset) for asset in assets]
            
            # Update timestamp for all assets
            for data in asset_data:
                data['last_updated'] = datetime.now().isoformat()
            
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
                   timeframes: List[str] = None, priority: int = 1) -> bool:
        """
        Add a symbol to the watchlist.
        
        Args:
            symbol: Trading symbol to add
            data_provider: Data provider for this symbol
            timeframes: List of timeframes to monitor
            priority: Priority level (higher = more frequent updates)
            
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
                priority=priority
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
                return [StrategyConfig(**strategy_data) for strategy_data in cached_data]
            
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
            strategy_data = [asdict(strategy) for strategy in strategies]
            
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
                self.update_system_config(self._default_system_config)
            )
            
            # Reset watchlist to environment defaults
            default_symbols = os.environ.get('WATCHLIST_SYMBOLS', 'SPY,QQQ,IWM').split(',')
            default_assets = [
                AssetConfig(
                    symbol=symbol.strip(),
                    enabled=True,
                    data_provider=os.environ.get('DATA_PROVIDER', 'mock'),
                    timeframes=['1m', '5m', '1h', '1d'],
                    priority=1
                ) for symbol in default_symbols if symbol.strip()
            ]
            success = success and self.update_watchlist(default_assets)
            
            if success:
                logger.info("Configuration reset to defaults successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to reset configuration to defaults: {e}")
            return False