"""
Message queue implementation for notification system.
"""

import asyncio
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
import logging

from .models import NotificationMessage, NotificationDatabase

logger = logging.getLogger(__name__)


class NotificationQueue:
    """Thread-safe notification message queue with persistent storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db = NotificationDatabase(db_path)
        self._shutdown = False
        
    async def enqueue(self, message: NotificationMessage) -> bool:
        """
        Add message to queue.
        Returns False if duplicate detected within deduplication window.
        """
        try:
            return self.db.add_message(message)
        except Exception as e:
            logger.error(f"Failed to enqueue message {message.id}: {e}")
            return False
    
    async def dequeue(self) -> Optional[NotificationMessage]:
        """Get next message from queue (highest priority first)."""
        try:
            return self.db.get_next_message()
        except Exception as e:
            logger.error(f"Failed to dequeue message: {e}")
            return None
    
    async def mark_sent(self, message_id: str):
        """Mark message as successfully sent."""
        try:
            self.db.mark_message_sent(message_id)
            logger.debug(f"Message marked as sent: {message_id}")
        except Exception as e:
            logger.error(f"Failed to mark message as sent {message_id}: {e}")
    
    async def mark_failed(self, message_id: str, retry_after: Optional[timedelta] = None):
        """Mark message as failed with optional retry scheduling."""
        try:
            self.db.mark_message_failed(message_id, retry_after)
            if retry_after:
                logger.debug(f"Message scheduled for retry {message_id}: {retry_after}")
            else:
                logger.warning(f"Message permanently failed: {message_id}")
        except Exception as e:
            logger.error(f"Failed to mark message as failed {message_id}: {e}")
    
    async def get_stats(self) -> dict:
        """Get queue statistics."""
        try:
            return self.db.get_queue_stats()
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def cleanup_old_records(self, older_than_hours: int = 1):
        """Clean up old deduplication records."""
        try:
            self.db.cleanup_old_records(older_than_hours)
            logger.debug(f"Cleaned up deduplication records older than {older_than_hours}h")
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")


class QueueProcessor:
    """Processes messages from notification queue with worker pattern."""
    
    def __init__(self, queue: NotificationQueue, 
                 message_handler: Callable[[NotificationMessage], Any],
                 worker_interval: float = 1.0):
        self.queue = queue
        self.message_handler = message_handler
        self.worker_interval = worker_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the queue processor."""
        if self._running:
            logger.warning("Queue processor already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._worker_loop())
        logger.info("Queue processor started")
    
    async def stop(self):
        """Stop the queue processor gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue processor stopped")
    
    async def _worker_loop(self):
        """Main worker processing loop."""
        logger.info("Queue processor worker loop started")
        logger.info(f"Processing interval: {self.worker_interval}s")
        
        while self._running:
            try:
                # Get next message
                logger.debug("Checking for next message...")
                message = await self.queue.dequeue()
                
                if message:
                    logger.info(f"Got message from queue: {message.id} - {message.content[:30]}...")
                    await self._process_message(message)
                else:
                    # No messages, wait before checking again
                    logger.debug("No messages in queue, sleeping...")
                    await asyncio.sleep(self.worker_interval)
                
                # Periodic cleanup every 100 iterations
                if hasattr(self, '_iteration_count'):
                    self._iteration_count += 1
                else:
                    self._iteration_count = 1
                
                if self._iteration_count % 100 == 0:
                    await self.queue.cleanup_old_records()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor worker loop: {e}")
                await asyncio.sleep(self.worker_interval)
        
        logger.info("Queue processor worker loop stopped")
    
    async def _process_message(self, message: NotificationMessage):
        """Process a single message."""
        try:
            logger.debug(f"Processing message: {message.id}")
            
            # Call the message handler
            if asyncio.iscoroutinefunction(self.message_handler):
                result = await self.message_handler(message)
            else:
                result = self.message_handler(message)
            
            # Handle result
            if result is True:
                await self.queue.mark_sent(message.id)
                logger.debug(f"Message sent successfully: {message.id}")
            else:
                # Failed - calculate retry delay with exponential backoff
                retry_delay = self._calculate_retry_delay(message.retry_count)
                await self.queue.mark_failed(message.id, retry_delay)
                logger.warning(f"Message failed, retry scheduled: {message.id}")
        
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            # Schedule retry on error
            retry_delay = self._calculate_retry_delay(message.retry_count)
            await self.queue.mark_failed(message.id, retry_delay)
    
    def _calculate_retry_delay(self, retry_count: int) -> timedelta:
        """Calculate exponential backoff delay."""
        # Exponential backoff: 1s, 2s, 4s
        delay_seconds = min(2 ** retry_count, 8)  # Cap at 8 seconds
        return timedelta(seconds=delay_seconds)


class NotificationQueueManager:
    """High-level manager for notification queue operations."""
    
    def __init__(self, db_path: Optional[str] = None, worker_interval: float = 1.0):
        self.queue = NotificationQueue(db_path)
        self.processor: Optional[QueueProcessor] = None
        self.worker_interval = worker_interval
        self._message_handler: Optional[Callable] = None
    
    def set_message_handler(self, handler: Callable[[NotificationMessage], Any]):
        """Set the message handler function."""
        self._message_handler = handler
    
    async def start_processing(self):
        """Start queue processing with registered handler."""
        logger.debug("NotificationQueueManager.start_processing() called")
        if not self._message_handler:
            raise ValueError("Message handler not set. Call set_message_handler() first.")
        
        if self.processor:
            logger.warning("Queue processor already running")
            return
        
        logger.debug(f"Creating QueueProcessor with interval {self.worker_interval}s")
        self.processor = QueueProcessor(
            self.queue, 
            self._message_handler, 
            self.worker_interval
        )
        logger.debug("Calling processor.start()")
        await self.processor.start()
        logger.debug("processor.start() completed")
    
    async def stop_processing(self):
        """Stop queue processing."""
        if self.processor:
            await self.processor.stop()
            self.processor = None
    
    async def enqueue_signal(self, symbol: str, strategy: str, timeframe: str, content: str) -> bool:
        """Enqueue a signal notification."""
        try:
            message = NotificationMessage.create_signal_notification(
                symbol=symbol,
                strategy=strategy,
                timeframe=timeframe,
                content=content
            )
            return await self.queue.enqueue(message)
        except ValueError as e:
            logger.error(f"Invalid signal notification parameters: {e}")
            return False
    
    async def enqueue_service(self, content: str, priority: int = 2) -> bool:
        """Enqueue a service notification."""
        message = NotificationMessage.create_service_notification(content, priority)
        return await self.queue.enqueue(message)
    
    async def get_queue_stats(self) -> dict:
        """Get current queue statistics."""
        return await self.queue.get_stats()
    
    async def health_check(self) -> dict:
        """Perform health check and return status."""
        try:
            stats = await self.get_queue_stats()
            
            # Check for concerning conditions
            warnings = []
            if stats.get('pending', 0) > 100:
                warnings.append("High pending message count")
            if stats.get('failed', 0) > 50:
                warnings.append("High failed message count")
            
            return {
                'healthy': len(warnings) == 0,
                'warnings': warnings,
                'stats': stats,
                'processor_running': self.processor is not None
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'processor_running': self.processor is not None
            }