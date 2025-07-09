"""
Multi-Channel Discord Notification Configuration

Manages strategy-specific Discord channels with individual webhook URLs,
rate limits, and configuration settings.
"""

import os
import logging
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .constants import STRATEGY_CONFIGS, STRATEGY_NAME_MAPPINGS, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class StrategyChannelConfig:
    """Configuration for a single strategy Discord channel."""

    strategy_name: str
    webhook_url: str
    enabled: bool = True
    max_notifications_per_hour: int = 5
    channel_color: int = 3447003  # Default blue
    emoji: str = "📊"
    allowed_timeframes: List[str] = field(
        default_factory=lambda: ["1d"]
    )  # Supported timeframes
    trading_profile: str = "swing_trading"  # day_trading or swing_trading

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enabled and not self.webhook_url:
            logger.warning(
                f"Strategy {self.strategy_name} enabled but no webhook URL configured"
            )
            self.enabled = False

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if strategy supports the given timeframe."""
        return timeframe in self.allowed_timeframes


@dataclass
class MultiChannelNotificationConfig:
    """Configuration for multi-channel Discord notifications."""

    # Global settings
    enabled: bool = True
    multi_channel_enabled: bool = True
    default_timeframes: List[str] = field(default_factory=lambda: ["1d"])

    # Strategy channels
    strategy_channels: Dict[str, StrategyChannelConfig] = field(default_factory=dict)

    # Fallback settings
    fallback_webhook_url: str = ""
    fallback_rate_limit: int = 10

    @classmethod
    def from_environment(cls) -> "MultiChannelNotificationConfig":
        """
        Load multi-channel configuration from environment variables.

        Returns:
            MultiChannelNotificationConfig: Loaded configuration
        """
        try:
            # Global settings
            enabled = (
                os.getenv("DISCORD_NOTIFICATIONS_ENABLED", "true").lower() == "true"
            )
            multi_channel_enabled = (
                os.getenv("DISCORD_MULTI_CHANNEL_ENABLED", "false").lower() == "true"
            )

            # Timeframe filtering
            timeframes_str = os.getenv("DISCORD_NOTIFY_TIMEFRAMES", "1d")
            timeframes = [t.strip() for t in timeframes_str.split(",") if t.strip()]

            # Fallback webhook
            fallback_webhook = os.getenv("DISCORD_WEBHOOK_SWING_SIGNALS", "")

            # Strategy channel configurations
            strategy_channels = cls._build_strategy_configs()

            config = cls(
                enabled=enabled,
                multi_channel_enabled=multi_channel_enabled,
                default_timeframes=timeframes,
                strategy_channels=strategy_channels,
                fallback_webhook_url=fallback_webhook,
                fallback_rate_limit=int(
                    os.getenv("DISCORD_MAX_NOTIFICATIONS_PER_HOUR", "10")
                ),
            )

            logger.info(
                f"Multi-channel Discord config loaded: {len(strategy_channels)} strategy channels configured"
            )
            return config

        except Exception as e:
            logger.error(f"Error loading multi-channel Discord configuration: {e}")
            # Return disabled configuration on error
            return cls(enabled=False, multi_channel_enabled=False)

    @classmethod
    def _build_strategy_configs(cls) -> Dict[str, StrategyChannelConfig]:
        """Build strategy configurations from environment variables and constants."""
        strategy_channels = {}

        # Parse strategy-timeframe mapping
        strategy_timeframe_mapping = cls._parse_strategy_timeframe_mapping()

        for strategy_key, strategy_info in STRATEGY_CONFIGS.items():
            # Get webhook URL
            webhook_key = f"DISCORD_WEBHOOK_{strategy_key.upper()}"
            webhook_url = os.getenv(webhook_key, "")

            # Get enabled status
            enabled_key = f"DISCORD_ENABLED_{strategy_key.upper()}"
            strategy_enabled = os.getenv(enabled_key, "true").lower() == "true"

            # Get rate limit
            rate_limit_key = f"DISCORD_RATE_LIMIT_{strategy_key.upper()}"
            rate_limit = int(
                os.getenv(rate_limit_key, str(strategy_info["default_rate_limit"]))
            )

            # Get allowed timeframes and trading profile for this strategy
            allowed_timeframes, trading_profile = (
                cls._get_strategy_timeframes_and_profile(
                    strategy_key, strategy_timeframe_mapping
                )
            )

            # Create strategy channel config
            if webhook_url and strategy_enabled:
                strategy_channels[strategy_key] = StrategyChannelConfig(
                    strategy_name=strategy_info["name"],
                    webhook_url=webhook_url,
                    enabled=strategy_enabled,
                    max_notifications_per_hour=rate_limit,
                    channel_color=strategy_info["color"],
                    emoji=strategy_info["emoji"],
                    allowed_timeframes=allowed_timeframes,
                    trading_profile=trading_profile,
                )
                logger.info(
                    f"Configured Discord channel for {strategy_info['name']} ({trading_profile}, {allowed_timeframes})"
                )
            else:
                logger.warning(
                    f"Strategy {strategy_key} not configured - missing webhook or disabled"
                )

        return strategy_channels

    @classmethod
    def _parse_strategy_timeframe_mapping(cls) -> Dict[str, List[str]]:
        """Parse strategy-timeframe mapping from environment."""
        mapping_str = os.getenv("DISCORD_STRATEGY_TIMEFRAME_MAPPING", "")
        mapping = {}

        if mapping_str:
            try:
                for item in mapping_str.split(","):
                    if ":" in item:
                        strategy, timeframe = item.strip().split(":", 1)
                        strategy = strategy.strip()
                        timeframe = timeframe.strip()
                        if strategy not in mapping:
                            mapping[strategy] = []
                        mapping[strategy].append(timeframe)
            except Exception as e:
                logger.warning(f"Error parsing strategy timeframe mapping: {e}")

        return mapping

    @classmethod
    def _get_strategy_timeframes_and_profile(
        cls, strategy_key: str, mapping: Dict[str, List[str]]
    ) -> tuple:
        """Get allowed timeframes and trading profile for a strategy."""
        # Default to swing trading with 1d timeframe
        default_timeframes = ["1d"]
        default_profile = "swing_trading"

        # Check if strategy has specific timeframe mapping
        if strategy_key in mapping:
            timeframes = mapping[strategy_key]

            # Determine trading profile based on timeframes
            if any(tf in ["1m", "5m", "15m", "30m", "1h"] for tf in timeframes):
                profile = "day_trading"
            else:
                profile = "swing_trading"

            return timeframes, profile

        # Fallback: determine by strategy type
        if strategy_key in ["rsi_trend", "bollinger_breakout"]:
            # These are configured for day trading (1h timeframe)
            return ["1h"], "day_trading"
        elif strategy_key in ["macd_rsi", "sr_breakout"]:
            # These are configured for swing trading (1d timeframe)
            return ["1d"], "swing_trading"

        return default_timeframes, default_profile

    def get_strategy_config(
        self, strategy_name: str
    ) -> Optional[StrategyChannelConfig]:
        """
        Get configuration for a specific strategy.

        Args:
            strategy_name: Strategy identifier

        Returns:
            StrategyChannelConfig or None if not found
        """
        # Normalize strategy name for lookup
        normalized_name = self._normalize_strategy_name(strategy_name)
        return self.strategy_channels.get(normalized_name)

    @staticmethod
    @lru_cache(maxsize=32)
    def _normalize_strategy_name(strategy_name: str) -> str:
        """
        Normalize strategy name for consistent lookup with caching.

        Args:
            strategy_name: Raw strategy name

        Returns:
            Normalized strategy name
        """
        normalized = strategy_name.lower().strip()
        return STRATEGY_NAME_MAPPINGS.get(normalized, normalized)

    def has_strategy_channel(self, strategy_name: str) -> bool:
        """
        Check if a strategy has a configured channel.

        Args:
            strategy_name: Strategy identifier

        Returns:
            True if strategy has configured channel
        """
        config = self.get_strategy_config(strategy_name)
        return config is not None and config.enabled

    def get_enabled_strategies(self) -> List[str]:
        """
        Get list of enabled strategy names.

        Returns:
            List of enabled strategy identifiers
        """
        return [
            strategy_key
            for strategy_key, config in self.strategy_channels.items()
            if config.enabled
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "enabled": self.enabled,
            "multi_channel_enabled": self.multi_channel_enabled,
            "default_timeframes": self.default_timeframes,
            "strategy_channels": {
                key: {
                    "strategy_name": config.strategy_name,
                    "enabled": config.enabled,
                    "max_notifications_per_hour": config.max_notifications_per_hour,
                    "channel_color": config.channel_color,
                    "emoji": config.emoji,
                    "webhook_configured": bool(config.webhook_url),
                }
                for key, config in self.strategy_channels.items()
            },
            "fallback_configured": bool(self.fallback_webhook_url),
            "total_channels": len(self.strategy_channels),
        }
