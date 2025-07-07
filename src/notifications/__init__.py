"""
Notification Service Module

Provides Discord webhook integration for sharing swing trading signals
with public channels. Uses existing Redis streams and requires no
additional dependencies beyond Python standard library.
"""

from .discord_notifier import DiscordNotifier
from .notification_service import NotificationService
from .message_formatter import MessageFormatter
from .config import NotificationConfig

__all__ = [
    'DiscordNotifier',
    'NotificationService', 
    'MessageFormatter',
    'NotificationConfig'
]