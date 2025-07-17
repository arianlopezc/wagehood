"""
Notification worker for processing and sending Discord notifications.
"""

import asyncio
import signal
from datetime import datetime
from ..utils.timezone_utils import utc_now
from typing import Optional, Dict, Any
import logging

from .models import NotificationMessage, NotificationDatabase
from .queue import NotificationQueueManager
from .deduplication import DeduplicationService
from .discord_client import DiscordNotificationSender
from .routing import ChannelConfigManager, MessageRouter

logger = logging.getLogger(__name__)


class NotificationWorker:
    """
    Single-worker notification processor.
    
    Processes notification queue, applies deduplication, and sends
    messages to Discord with reliable delivery guarantees.
    """
    
    def __init__(self, db_path: Optional[str] = None, worker_interval: float = 1.0):
        # Initialize components
        self.db = NotificationDatabase(db_path)
        self.queue_manager = NotificationQueueManager(db_path, worker_interval)
        self.dedup_service = DeduplicationService(self.db, dedup_window_minutes=10)
        
        # Channel and routing management
        self.config_manager = ChannelConfigManager()
        self.router = MessageRouter(self.config_manager)
        
        # Discord sender (will be initialized after config validation)
        self.discord_sender: Optional[DiscordNotificationSender] = None
        
        # Worker state
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.stats = {
            'messages_processed': 0,
            'messages_sent': 0,
            'messages_failed': 0,
            'messages_deduplicated': 0,
            'start_time': None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the notification worker.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Validate configuration
            validation = self.config_manager.validate_configuration()
            if validation['errors']:
                logger.error("Configuration validation failed:")
                for error in validation['errors']:
                    logger.error(f"  - {error}")
                return False
            
            if validation['warnings']:
                logger.warning("Configuration warnings:")
                for warning in validation['warnings']:
                    logger.warning(f"  - {warning}")
            
            # Initialize Discord sender
            self.discord_sender = DiscordNotificationSender(
                self.config_manager.get_all_configs()
            )
            
            # Set up queue message handler
            self.queue_manager.set_message_handler(self._handle_message)
            
            logger.info("Notification worker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize notification worker: {e}")
            return False
    
    async def start(self):
        """Start the notification worker (for backward compatibility)."""
        # This method is now deprecated in favor of the NotificationService wrapper
        # which properly handles initialization and background execution
        logger.warning("NotificationWorker.start() called directly - use NotificationService instead")
        
        if self._running:
            logger.warning("Notification worker already running")
            return
        
        if not await self.initialize():
            raise RuntimeError("Failed to initialize notification worker")
        
        self._running = True
        self.stats['start_time'] = utc_now()
        
        logger.info("Starting notification worker")
        
        # Set up signal handlers for graceful shutdown
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            # Start queue processing
            await self.queue_manager.start_processing()
            
            # Start periodic maintenance tasks
            maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Cancel maintenance task
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
            
        except Exception as e:
            logger.error(f"Error in notification worker: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def stop(self):
        """Stop the notification worker gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping notification worker")
        self._running = False
        self._shutdown_event.set()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.stop())
    
    async def _handle_message(self, message: NotificationMessage) -> bool:
        """
        Handle a single notification message.
        
        Args:
            message: Notification message to process
            
        Returns:
            True if message processed successfully, False otherwise
        """
        try:
            self.stats['messages_processed'] += 1
            
            # Apply deduplication
            if self.dedup_service.is_duplicate(message):
                logger.debug(f"Duplicate message blocked: {message.id}")
                self.stats['messages_deduplicated'] += 1
                return True  # Consider duplicates as "successfully handled"
            
            # Route message to appropriate channel
            channel_config = self.router.route_message(message)
            if not channel_config:
                logger.error(f"Failed to route message {message.id}")
                self.stats['messages_failed'] += 1
                return False
            
            # Send to Discord
            if self.discord_sender:
                success = await self.discord_sender.send_notification(message)
                
                if success:
                    # Mark as seen for deduplication
                    self.dedup_service.mark_message_seen(message)
                    self.stats['messages_sent'] += 1
                    return True
                else:
                    self.stats['messages_failed'] += 1
                    return False
            else:
                logger.error("Discord sender not initialized")
                self.stats['messages_failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    async def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while self._running:
            try:
                # Clean up old deduplication records every 15 minutes
                await asyncio.sleep(900)  # 15 minutes
                
                if self._running:
                    self.dedup_service.cleanup_old_records()
                    await self._log_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
    
    async def _log_stats(self):
        """Log worker performance statistics."""
        if not self.stats['start_time']:
            return
        
        uptime = utc_now() - self.stats['start_time']
        
        logger.info(f"Notification worker stats:")
        logger.info(f"  Uptime: {uptime}")
        logger.info(f"  Messages processed: {self.stats['messages_processed']}")
        logger.info(f"  Messages sent: {self.stats['messages_sent']}")
        logger.info(f"  Messages failed: {self.stats['messages_failed']}")
        logger.info(f"  Messages deduplicated: {self.stats['messages_deduplicated']}")
        
        # Queue stats
        queue_stats = await self.queue_manager.get_queue_stats()
        logger.info(f"  Queue pending: {queue_stats.get('pending', 0)}")
        logger.info(f"  Queue processing: {queue_stats.get('processing', 0)}")
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            # Stop queue processing
            await self.queue_manager.stop_processing()
            
            # Close Discord client
            if self.discord_sender:
                await self.discord_sender.close()
            
            logger.info("Notification worker cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def enqueue_signal(self, symbol: str, strategy: str, timeframe: str, content: str) -> bool:
        """
        Enqueue a signal notification.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name  
            timeframe: Signal timeframe
            content: Notification content
            
        Returns:
            True if enqueued successfully
        """
        return await self.queue_manager.enqueue_signal(symbol, strategy, timeframe, content)
    
    async def enqueue_service(self, content: str, priority: int = 2) -> bool:
        """
        Enqueue a service notification.
        
        Args:
            content: Notification content
            priority: Message priority (1=high, 2=normal, 3=low)
            
        Returns:
            True if enqueued successfully
        """
        return await self.queue_manager.enqueue_service(content, priority)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dictionary with health information
        """
        try:
            # Queue health
            queue_health = await self.queue_manager.health_check()
            
            # Discord connectivity - NO TESTS, just report configuration
            discord_health = {}
            if self.discord_sender:
                try:
                    # Just count configured channels, don't test them
                    channel_count = len(self.discord_sender.router.channel_configs)
                    enabled_count = sum(1 for c in self.discord_sender.router.channel_configs.values() if c.enabled)
                    discord_health = {
                        'channels_configured': channel_count,
                        'channels_enabled': enabled_count,
                        'channels_healthy': enabled_count  # Assume enabled channels are healthy
                    }
                except Exception as e:
                    discord_health = {'error': str(e)}
            
            # Deduplication stats
            dedup_stats = self.dedup_service.get_deduplication_stats()
            
            # Overall health determination
            healthy = (
                queue_health.get('healthy', False) and
                self._running and
                discord_health.get('channels_healthy', 0) > 0
            )
            
            return {
                'healthy': healthy,
                'running': self._running,
                'stats': self.stats,
                'queue_health': queue_health,
                'discord_health': discord_health,
                'deduplication_stats': dedup_stats,
                'configuration': {
                    'channels_configured': len(self.config_manager.configs),
                    'missing_channels': self.config_manager.get_missing_channels()
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'running': self._running
            }


class NotificationService:
    """
    High-level notification service interface.
    
    Provides simple interface for sending notifications while
    managing the underlying worker infrastructure.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.worker = NotificationWorker(db_path)
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the notification service."""
        logger.debug("NotificationService.start() called")
        if self._worker_task:
            logger.warning("Notification service already running")
            return
        
        # First initialize the worker
        logger.debug("Initializing notification worker...")
        if not await self.worker.initialize():
            raise RuntimeError("Failed to initialize notification worker")
        logger.debug("Worker initialized successfully")
        
        # Then start it as a background task
        logger.debug("Starting worker background task...")
        self._worker_task = asyncio.create_task(self._run_worker())
        
        # Wait a moment to ensure it starts successfully
        await asyncio.sleep(0.5)
        
        # Check if the task failed immediately
        if self._worker_task.done():
            # Worker failed to start
            logger.error("Worker task failed immediately")
            try:
                await self._worker_task
            except Exception as e:
                raise RuntimeError(f"Failed to start notification service: {e}")
        logger.debug("NotificationService.start() completed")
        logger.info(f"Notification service ready - worker running: {self.worker._running}")
                
    async def _run_worker(self):
        """Run the worker in the background."""
        logger.info("_run_worker called - starting background worker")
        try:
            # Set running flag before starting
            self.worker._running = True
            self.worker.stats['start_time'] = utc_now()
            
            logger.info("Starting notification worker")
            
            # Start queue processing
            logger.info("Starting queue processing...")
            try:
                await self.worker.queue_manager.start_processing()
                logger.info("Queue processing started")
            except Exception as e:
                logger.error(f"Failed to start queue processing: {e}", exc_info=True)
                raise
            
            # Start periodic maintenance tasks
            logger.info("Creating maintenance task...")
            maintenance_task = asyncio.create_task(self.worker._maintenance_loop())
            logger.info("Maintenance task created")
            
            # Wait for shutdown signal
            logger.info("Waiting for shutdown signal...")
            await self.worker._shutdown_event.wait()
            
            # Cancel maintenance task
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logger.error(f"Error in notification worker: {e}", exc_info=True)
            raise
        finally:
            await self.worker._cleanup()
    
    async def stop(self):
        """Stop the notification service."""
        if self.worker._running:
            await self.worker.stop()
        
        if self._worker_task:
            try:
                await self._worker_task
            except Exception as e:
                logger.error(f"Error stopping notification service: {e}")
            finally:
                self._worker_task = None
    
    async def send_signal(self, symbol: str, strategy: str, timeframe: str, content: str) -> bool:
        """Send a signal notification."""
        if not self.worker._running:
            logger.error("Notification service not running")
            return False
        
        return await self.worker.enqueue_signal(symbol, strategy, timeframe, content)
    
    async def send_service_notification(self, content: str, priority: int = 2) -> bool:
        """Send a service notification."""
        if not self.worker._running:
            logger.error("Notification service not running")
            return False
        
        return await self.worker.enqueue_service(content, priority)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return await self.worker.get_health_status()
    
    async def test_channels(self) -> Dict[str, bool]:
        """Test connectivity to all Discord channels."""
        if self.worker.discord_sender:
            return await self.worker.discord_sender.test_all_channels()
        return {}


# Global notification service instance
_notification_service: Optional[NotificationService] = None


async def get_notification_service() -> NotificationService:
    """
    Get global notification service instance.
    
    Returns:
        Singleton NotificationService instance
    """
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService()
    
    return _notification_service


async def send_signal_notification(symbol: str, strategy: str, timeframe: str, content: str) -> bool:
    """
    Convenience function to send signal notification.
    
    Args:
        symbol: Trading symbol
        strategy: Strategy name
        timeframe: Signal timeframe  
        content: Notification content
        
    Returns:
        True if notification sent successfully
    """
    service = await get_notification_service()
    return await service.send_signal(symbol, strategy, timeframe, content)


async def send_service_notification(content: str, priority: int = 2) -> bool:
    """
    Convenience function to send service notification.
    
    Args:
        content: Notification content
        priority: Message priority (1=high, 2=normal, 3=low)
        
    Returns:
        True if notification sent successfully
    """
    service = await get_notification_service()
    return await service.send_service_notification(content, priority)