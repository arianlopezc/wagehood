"""
Channel routing and configuration management for notifications.
"""

import os
from typing import Dict, Optional, List
import logging
from pathlib import Path
from dotenv import load_dotenv

from .models import ChannelConfig, CHANNEL_ROUTING, NotificationMessage

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class ChannelConfigManager:
    """
    Manages Discord channel configurations and webhook URLs.
    
    Handles loading configurations from environment variables and
    validating channel routing rules.
    """
    
    def __init__(self):
        self.configs: Dict[str, ChannelConfig] = {}
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load channel configurations from environment variables."""
        # Expected environment variables
        env_mappings = {
            'infra': 'DISCORD_WEBHOOK_INFRA',
            'macd-rsi': 'DISCORD_WEBHOOK_MACD_RSI', 
            'support-resistance': 'DISCORD_WEBHOOK_SUPPORT_RESISTANCE',
            'rsi-trend-following': 'DISCORD_WEBHOOK_RSI_TREND',
            'bollinger-band-breakout': 'DISCORD_WEBHOOK_BOLLINGER',
            'crypto-bollinger-1d': 'DISCORD_WEBHOOK_CRYPTO_BOLLINGER_1D',
            'eod-summary': 'DISCORD_WEBHOOK_EOD_SUMMARY'
        }
        
        for channel_name, env_var in env_mappings.items():
            webhook_url = os.getenv(env_var)
            if webhook_url:
                config = ChannelConfig(
                    channel_name=channel_name,
                    webhook_url=webhook_url,
                    enabled=True
                )
                
                self.configs[channel_name] = config
                logger.info(f"Loaded config for channel: {channel_name}")
            else:
                logger.warning(f"Missing environment variable: {env_var}")
    
    def get_config(self, channel_name: str) -> Optional[ChannelConfig]:
        """Get configuration for a specific channel."""
        return self.configs.get(channel_name)
    
    def set_config(self, config: ChannelConfig):
        """Set configuration for a channel."""
        self.configs[config.channel_name] = config
        logger.info(f"Updated config for channel: {config.channel_name}")
    
    def get_all_configs(self) -> Dict[str, ChannelConfig]:
        """Get all channel configurations."""
        return self.configs.copy()
    
    def is_configured(self) -> bool:
        """Check if at least one channel is configured."""
        return len(self.configs) > 0
    
    def get_missing_channels(self) -> List[str]:
        """Get list of channels that are not configured."""
        expected_channels = set(CHANNEL_ROUTING.values())
        configured_channels = set(self.configs.keys())
        return list(expected_channels - configured_channels)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate channel configuration completeness.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        # Check for missing required channels
        missing = self.get_missing_channels()
        if missing:
            errors.extend([f"Missing configuration for channel: {ch}" for ch in missing])
        
        # Check for disabled channels
        disabled = [name for name, config in self.configs.items() if not config.enabled]
        if disabled:
            warnings.extend([f"Channel disabled: {ch}" for ch in disabled])
        
        # Check webhook URL format
        for name, config in self.configs.items():
            if not config.webhook_url.startswith('https://discord.com/api/webhooks/'):
                errors.append(f"Invalid webhook URL format for channel: {name}")
        
        return {'errors': errors, 'warnings': warnings}


class MessageRouter:
    """
    Routes notification messages to appropriate Discord channels.
    
    Implements the strategy + timeframe -> channel mapping logic
    defined in the notification requirements.
    """
    
    def __init__(self, config_manager: ChannelConfigManager):
        self.config_manager = config_manager
    
    def route_message(self, message: NotificationMessage) -> Optional[ChannelConfig]:
        """
        Route message to appropriate channel configuration.
        
        Args:
            message: Notification message to route
            
        Returns:
            ChannelConfig if routing successful, None otherwise
        """
        channel_name = message.channel
        
        # Validate channel exists and is enabled
        config = self.config_manager.get_config(channel_name)
        if not config:
            logger.error(f"No configuration found for channel: {channel_name}")
            return None
        
        if not config.enabled:
            logger.warning(f"Channel disabled: {channel_name}")
            return None
        
        # Additional validation for signal messages
        if message.type.value == "signal":
            if not self._validate_signal_routing(message, config):
                return None
        
        return config
    
    def _validate_signal_routing(self, message: NotificationMessage, 
                                config: ChannelConfig) -> bool:
        """
        Validate signal message routing against channel configuration.
        
        Args:
            message: Signal notification message
            config: Target channel configuration
            
        Returns:
            True if routing is valid
        """
        # Check strategy matching
        if config.strategy and message.strategy != config.strategy:
            logger.error(f"Strategy mismatch: message={message.strategy}, "
                        f"channel={config.strategy}")
            return False
        
        # Check timeframe matching  
        if config.timeframe and message.timeframe != config.timeframe:
            logger.error(f"Timeframe mismatch: message={message.timeframe}, "
                        f"channel={config.timeframe}")
            return False
        
        return True
    
    def get_channel_for_signal(self, strategy: str, timeframe: str) -> Optional[str]:
        """
        Get appropriate channel name for a signal.
        
        Args:
            strategy: Trading strategy name
            timeframe: Signal timeframe
            
        Returns:
            Channel name if routing exists, None otherwise
        """
        return CHANNEL_ROUTING.get((strategy, timeframe))
    
    def validate_routing_rules(self) -> List[str]:
        """
        Validate all routing rules against configured channels.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for (strategy, timeframe), channel_name in CHANNEL_ROUTING.items():
            if strategy == 'service':  # Skip service routing
                continue
                
            config = self.config_manager.get_config(channel_name)
            if not config:
                errors.append(f"No config for required channel: {channel_name}")
                continue
            
            if config.strategy and config.strategy != strategy:
                errors.append(f"Strategy mismatch in {channel_name}: "
                             f"expected {strategy}, got {config.strategy}")
            
            if config.timeframe and config.timeframe != timeframe:
                errors.append(f"Timeframe mismatch in {channel_name}: "
                             f"expected {timeframe}, got {config.timeframe}")
        
        return errors


class RoutingValidator:
    """
    Validates notification routing configuration and rules.
    """
    
    @staticmethod
    def validate_channel_routing() -> Dict[str, List[str]]:
        """
        Validate the CHANNEL_ROUTING configuration.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check for duplicate channels
        channels = list(CHANNEL_ROUTING.values())
        unique_channels = set(channels)
        if len(channels) != len(unique_channels):
            duplicates = [ch for ch in unique_channels if channels.count(ch) > 1]
            errors.extend([f"Duplicate channel mapping: {ch}" for ch in duplicates])
        
        # Check for missing timeframe combinations
        expected_combinations = [
            ('macd_rsi', '1d'),
            ('sr_breakout', '1d'), 
            ('rsi_trend', '1h'),
            ('bollinger_breakout', '1h'),
            ('service', None),
            ('eod_summary', None)
        ]
        
        actual_combinations = list(CHANNEL_ROUTING.keys())
        missing = [combo for combo in expected_combinations if combo not in actual_combinations]
        
        if missing:
            errors.extend([f"Missing routing for: {combo}" for combo in missing])
        
        # Check for unexpected combinations
        unexpected = [combo for combo in actual_combinations if combo not in expected_combinations]
        if unexpected:
            warnings.extend([f"Unexpected routing: {combo}" for combo in unexpected])
        
        return {'errors': errors, 'warnings': warnings}
    
    @staticmethod
    def validate_message_format(message: NotificationMessage) -> List[str]:
        """
        Validate message format for Discord compatibility.
        
        Args:
            message: Message to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check message content length (Discord limit is 2000 characters)
        if len(message.content) > 2000:
            errors.append(f"Message content too long: {len(message.content)} chars (max 2000)")
        
        # Check required fields for signal messages
        if message.type.value == "signal":
            if not message.symbol:
                errors.append("Signal message missing symbol")
            if not message.strategy:
                errors.append("Signal message missing strategy")
            if not message.timeframe:
                errors.append("Signal message missing timeframe")
        
        # Check channel name format
        if not message.channel:
            errors.append("Message missing channel")
        elif not isinstance(message.channel, str):
            errors.append("Channel must be a string")
        
        return errors


def create_default_config_manager() -> ChannelConfigManager:
    """
    Create and return a default configured ChannelConfigManager.
    
    Returns:
        Configured ChannelConfigManager instance
    """
    manager = ChannelConfigManager()
    
    # Log configuration status
    if manager.is_configured():
        logger.info(f"Loaded {len(manager.configs)} channel configurations")
        missing = manager.get_missing_channels()
        if missing:
            logger.warning(f"Missing configurations for: {missing}")
    else:
        logger.error("No channel configurations loaded from environment")
        logger.info("Required environment variables:")
        for channel, env_var in [
            ('infra', 'DISCORD_WEBHOOK_INFRA'),
            ('macd-rsi', 'DISCORD_WEBHOOK_MACD_RSI'),
            ('support-resistance', 'DISCORD_WEBHOOK_SUPPORT_RESISTANCE'),
            ('rsi-trend-following', 'DISCORD_WEBHOOK_RSI_TREND'),
            ('bollinger-band-breakout', 'DISCORD_WEBHOOK_BOLLINGER')
        ]:
            logger.info(f"  {env_var} -> {channel}")
    
    return manager