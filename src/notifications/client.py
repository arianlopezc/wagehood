"""
Client for sending notifications to the notification service.

This module provides a simple way to send notifications from other processes
by directly enqueuing messages to the notification database.
"""

import os
from typing import Optional
import logging

from .models import NotificationMessage, NotificationDatabase

logger = logging.getLogger(__name__)


class NotificationClient:
    """
    Client for sending notifications to the notification service.
    
    This client directly enqueues messages to the notification database,
    allowing different processes to send notifications without needing
    to share the NotificationService instance.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.expanduser('~/.wagehood/notifications.db')
        self.db = NotificationDatabase(db_path)
    
    async def send_signal(self, symbol: str, strategy: str, timeframe: str, content: str) -> bool:
        """
        Send a signal notification.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name
            timeframe: Signal timeframe
            content: Notification content
            
        Returns:
            True if notification enqueued successfully
        """
        try:
            message = NotificationMessage.create_signal_notification(
                symbol, strategy, timeframe, content
            )
            success = self.db.add_message(message)
            
            if success:
                logger.info(f"Signal notification enqueued: {symbol} {strategy} {timeframe}")
            else:
                logger.warning(f"Failed to enqueue signal notification for {symbol} {strategy} - likely duplicate within 5 minute window")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}", exc_info=True)
            return False
    
    async def send_service(self, content: str, priority: int = 2) -> bool:
        """
        Send a service notification.
        
        Args:
            content: Notification content
            priority: Message priority (1=high, 2=normal, 3=low)
            
        Returns:
            True if notification enqueued successfully
        """
        try:
            message = NotificationMessage.create_service_notification(content, priority)
            success = self.db.add_message(message)
            
            if success:
                logger.debug("Service notification enqueued")
            else:
                logger.warning("Failed to enqueue service notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending service notification: {e}")
            return False
    
    async def send_summary(self, content: str, priority: int = 1) -> bool:
        """
        Send an end-of-day summary notification.
        
        Args:
            content: Summary content (JSON string for rich formatting)
            priority: Message priority (1=high, 2=normal, 3=low)
            
        Returns:
            True if notification enqueued successfully
        """
        try:
            message = NotificationMessage.create_summary_notification(content, priority)
            success = self.db.add_message(message)
            
            if success:
                logger.info("EOD summary notification enqueued")
            else:
                logger.warning("Failed to enqueue EOD summary notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending summary notification: {e}")
            return False


# Global client instance
_notification_client: Optional[NotificationClient] = None


def get_notification_client() -> NotificationClient:
    """Get global notification client instance."""
    global _notification_client
    
    if _notification_client is None:
        _notification_client = NotificationClient()
    
    return _notification_client


async def send_signal_notification(symbol: str, strategy: str, timeframe: str, content: str) -> bool:
    """
    Send signal notification using the client.
    
    This function can be called from any process to send notifications.
    """
    client = get_notification_client()
    return await client.send_signal(symbol, strategy, timeframe, content)


async def send_service_notification(content: str, priority: int = 2) -> bool:
    """
    Send service notification using the client.
    
    This function can be called from any process to send notifications.
    """
    client = get_notification_client()
    return await client.send_service(content, priority)


async def send_summary_notification(content: str, priority: int = 1) -> bool:
    """
    Send end-of-day summary notification using the client.
    
    This function can be called from any process to send summary notifications.
    """
    client = get_notification_client()
    return await client.send_summary(content, priority)