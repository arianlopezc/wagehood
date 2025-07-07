"""
Notification Service

Main service that listens to Redis streams for signal events and
sends Discord notifications for swing trading signals.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .discord_notifier import DiscordNotifier
from .message_formatter import MessageFormatter
from .config import NotificationConfig, NotificationRateLimiter

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Main notification service that monitors Redis streams and sends Discord notifications.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notification service.
        
        Args:
            config: Notification configuration (loads from env if None)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package required for notifications. Install with: pip install redis")
        
        self.config = config or NotificationConfig.from_environment()
        self.running = False
        self.tasks = []
        
        # Initialize components
        if self.config.enabled:
            self.discord_notifier = DiscordNotifier(
                self.config.discord_webhook_url,
                max_retries=self.config.max_retries
            )
            self.message_formatter = MessageFormatter()
            self.rate_limiter = NotificationRateLimiter(self.config.max_notifications_per_hour)
        else:
            logger.info("Discord notifications disabled in configuration")
            self.discord_notifier = None
            self.message_formatter = None
            self.rate_limiter = None
        
        # Redis connection
        self.redis_client = None
        self._last_notification_times = {}  # Track last notification per symbol
        
        logger.info(f"Notification service initialized (enabled: {self.config.enabled})")
    
    async def start(self):
        """Start the notification service."""
        if not self.config.enabled:
            logger.info("Notification service disabled - not starting")
            return
        
        if self.running:
            logger.warning("Notification service already running")
            return
        
        logger.info("Starting Discord notification service...")
        
        try:
            # Initialize Redis connection
            self._initialize_redis()
            
            # Send startup notification
            await self._send_startup_notification()
            
            # Start monitoring tasks
            self.running = True
            
            # Monitor calculation events stream for signals
            task = asyncio.create_task(self._monitor_signal_events())
            self.tasks.append(task)
            
            # Optional: Monitor system status
            if self.config.include_system_status:
                status_task = asyncio.create_task(self._monitor_system_status())
                self.tasks.append(status_task)
            
            logger.info("Discord notification service started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Error starting notification service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the notification service."""
        if not self.running:
            return
        
        logger.info("Stopping Discord notification service...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        logger.info("Discord notification service stopped")
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            from src.core.constants import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
            
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection initialized for notifications")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def _send_startup_notification(self):
        """Send notification that the service has started."""
        try:
            if not self.discord_notifier:
                return
            
            status_data = {
                'service': 'Discord Notification Service',
                'status': 'running',
                'symbol_count': len(self.config.symbols_to_notify) if self.config.symbols_to_notify else 'All',
                'data_provider': 'Alpaca Markets',
                'message': f'Monitoring {", ".join(self.config.trading_profiles)} signals'
            }
            
            success = self.discord_notifier.send_system_status(status_data)
            if success:
                logger.info("Startup notification sent to Discord")
            else:
                logger.warning("Failed to send startup notification")
                
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
    
    async def _monitor_signal_events(self):
        """Monitor Redis streams for calculation events that contain signals."""
        logger.info("Starting signal event monitoring...")
        
        stream_name = "calculation_events_stream"
        consumer_group = "discord_notifications"
        consumer_name = "discord_notifier"
        
        # Create consumer group if it doesn't exist
        try:
            self.redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Error creating consumer group: {e}")
                return
        
        while self.running:
            try:
                # Read from stream
                messages = self.redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_name: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            await self._process_signal_event(msg_id, fields)
                            
                            # Acknowledge message
                            self.redis_client.xack(stream_name, consumer_group, msg_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing signal event {msg_id}: {e}")
                
            except redis.exceptions.ConnectionError:
                logger.warning("Redis connection lost, attempting to reconnect...")
                try:
                    self._initialize_redis()
                except Exception as e:
                    logger.error(f"Failed to reconnect to Redis: {e}")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error in signal monitoring: {e}")
                await asyncio.sleep(1)
    
    async def _process_signal_event(self, msg_id: str, fields: Dict[str, str]):
        """
        Process a signal event from Redis stream.
        
        Args:
            msg_id: Redis message ID
            fields: Event fields
        """
        try:
            # Parse event data
            event_data = self._parse_signal_event(fields)
            if not event_data:
                return
            
            # Check if this is a signal we should notify about
            if not self._should_notify_signal(event_data):
                logger.debug(f"Skipping notification for {event_data.get('symbol', 'unknown')}")
                return
            
            # Check rate limiting
            if not self.rate_limiter.can_send_notification():
                logger.warning("Rate limit exceeded, skipping notification")
                return
            
            # Send Discord notification
            success = self.discord_notifier.send_signal_notification(event_data)
            
            if success:
                self.rate_limiter.record_notification()
                symbol = event_data.get('symbol', 'unknown')
                self._last_notification_times[symbol] = time.time()
                logger.info(f"Sent Discord notification for {symbol} {event_data.get('signal', 'signal')}")
            else:
                logger.error("Failed to send Discord notification")
                
        except Exception as e:
            logger.error(f"Error processing signal event: {e}")
    
    def _parse_signal_event(self, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Parse signal event fields into notification data.
        
        Args:
            fields: Redis stream event fields
            
        Returns:
            Parsed signal data or None if not a signal event
        """
        try:
            # Check if this is a signal event
            if 'results' not in fields:
                return None
            
            # Parse results JSON
            results = json.loads(fields['results'])
            
            # Look for signal data in results
            signal_data = results.get('signal')
            if not signal_data:
                return None
            
            # Extract symbol
            symbol = fields.get('symbol', results.get('symbol', 'UNKNOWN'))
            
            # Build notification data
            notification_data = {
                'symbol': symbol,
                'signal': signal_data.get('action', 'UNKNOWN').upper(),
                'price': signal_data.get('price', 0.0),
                'price_change': signal_data.get('price_change', 0.0),
                'price_change_pct': signal_data.get('price_change_pct', 0.0),
                'strategy': signal_data.get('strategy', 'Unknown Strategy'),
                'confidence': signal_data.get('confidence', 0.0),
                'details': signal_data.get('details', {}),
                'timestamp': datetime.now()
            }
            
            # Add company name if available
            company_names = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corp.',
                'GOOGL': 'Alphabet Inc.',
                'TSLA': 'Tesla Inc.',
                'SPY': 'SPDR S&P 500 ETF',
                'QQQ': 'Invesco QQQ ETF',
                'IWM': 'iShares Russell 2000 ETF'
            }
            
            if symbol in company_names:
                notification_data['company_name'] = company_names[symbol]
            
            return notification_data
            
        except Exception as e:
            logger.error(f"Error parsing signal event: {e}")
            return None
    
    def _should_notify_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Check if signal should trigger notification.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            True if notification should be sent
        """
        try:
            symbol = signal_data.get('symbol', '')
            confidence = signal_data.get('confidence', 0.0)
            signal_type = signal_data.get('signal', '').upper()
            
            # Check symbol filter
            if not self.config.should_notify_symbol(symbol):
                return False
            
            # Check confidence threshold
            if not self.config.should_notify_confidence(confidence):
                logger.debug(f"Signal confidence {confidence} below threshold {self.config.min_confidence_threshold}")
                return False
            
            # Trust the core strategy engine - if it detected a signal change, notify
            # No artificial time-based filtering for same symbol
            
            # Only notify for BUY/SELL signals, not HOLD (unless explicitly configured)
            if signal_type == 'HOLD':
                return False  # Skip HOLD signals for now
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification criteria: {e}")
            return False
    
    async def _monitor_system_status(self):
        """Monitor system status and send periodic updates."""
        logger.info("Starting system status monitoring...")
        
        while self.running:
            try:
                # Wait 1 hour between status updates
                await asyncio.sleep(3600)
                
                if not self.running:
                    break
                
                # Send status update
                await self._send_status_update()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _send_status_update(self):
        """Send periodic status update."""
        try:
            rate_status = self.rate_limiter.get_status()
            
            status_data = {
                'service': 'Discord Notifications',
                'status': 'running',
                'message': f"Sent {rate_status['notifications_sent_last_hour']} notifications in last hour"
            }
            
            success = self.discord_notifier.send_system_status(status_data)
            if success:
                logger.info("Status update sent to Discord")
                
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get notification service statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'enabled': self.config.enabled,
            'running': self.running,
            'webhook_configured': bool(self.config.discord_webhook_url),
            'last_notifications': self._last_notification_times.copy()
        }
        
        if self.rate_limiter:
            stats['rate_limiter'] = self.rate_limiter.get_status()
        
        return stats