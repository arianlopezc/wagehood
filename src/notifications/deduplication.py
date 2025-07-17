"""
Message deduplication service for notification system.
"""

import hashlib
from datetime import datetime, timedelta
from ..utils.timezone_utils import utc_now
from typing import Optional, Set, Dict
import logging

from .models import NotificationMessage, DeduplicationRecord, NotificationDatabase

logger = logging.getLogger(__name__)


class DeduplicationService:
    """
    Content-based deduplication service for notifications.
    
    Uses MD5 hashing of message content combined with time windows
    to prevent duplicate notifications from being sent.
    """
    
    def __init__(self, db: NotificationDatabase, dedup_window_minutes: int = 5):
        self.db = db
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self._in_memory_cache: Dict[str, datetime] = {}
        
    def is_duplicate(self, message: NotificationMessage) -> bool:
        """
        Check if message is a duplicate within the deduplication window.
        
        Args:
            message: The notification message to check
            
        Returns:
            True if message is a duplicate and should be rejected
        """
        content_hash = message.get_content_hash()
        channel_key = f"{content_hash}:{message.channel}"
        
        # Check in-memory cache first for performance
        if channel_key in self._in_memory_cache:
            last_seen = self._in_memory_cache[channel_key]
            if utc_now() - last_seen < self.dedup_window:
                logger.debug(f"Duplicate detected in memory cache: {content_hash}")
                return True
        
        # Check database for persistent deduplication
        cutoff_time = utc_now() - self.dedup_window
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                existing = conn.execute('''
                    SELECT last_seen FROM message_deduplication 
                    WHERE content_hash = ? AND channel = ? AND last_seen > ?
                ''', (content_hash, message.channel, cutoff_time)).fetchone()
                
                if existing:
                    logger.debug(f"Duplicate detected in database: {content_hash}")
                    # Update in-memory cache
                    self._in_memory_cache[channel_key] = datetime.fromisoformat(existing[0])
                    return True
                
        except Exception as e:
            logger.error(f"Error checking deduplication database: {e}")
            # Fall back to in-memory cache only
            pass
        
        return False
    
    def mark_message_seen(self, message: NotificationMessage):
        """
        Mark message as seen for future deduplication checks.
        
        Args:
            message: The notification message to mark as seen
        """
        content_hash = message.get_content_hash()
        channel_key = f"{content_hash}:{message.channel}"
        now = utc_now()
        
        # Update in-memory cache
        self._in_memory_cache[channel_key] = now
        
        # Update database
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO message_deduplication
                    (content_hash, channel, first_seen, last_seen)
                    VALUES (?, ?, COALESCE((SELECT first_seen FROM message_deduplication 
                            WHERE content_hash = ? AND channel = ?), ?), ?)
                ''', (content_hash, message.channel, content_hash, message.channel, now, now))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating deduplication database: {e}")
    
    def cleanup_old_records(self, older_than_hours: int = 1):
        """
        Clean up old deduplication records to prevent unbounded growth.
        
        Args:
            older_than_hours: Remove records older than this many hours
        """
        cutoff_time = utc_now() - timedelta(hours=older_than_hours)
        
        # Clean in-memory cache
        expired_keys = [
            key for key, last_seen in self._in_memory_cache.items()
            if last_seen < cutoff_time
        ]
        for key in expired_keys:
            del self._in_memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired entries from memory cache")
        
        # Clean database
        try:
            self.db.cleanup_old_records(older_than_hours)
        except Exception as e:
            logger.error(f"Error cleaning up deduplication database: {e}")
    
    def get_deduplication_stats(self) -> Dict[str, int]:
        """
        Get statistics about deduplication performance.
        
        Returns:
            Dictionary with cache size and database record count
        """
        stats = {
            'memory_cache_size': len(self._in_memory_cache),
            'database_records': 0
        }
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                count = conn.execute(
                    'SELECT COUNT(*) FROM message_deduplication'
                ).fetchone()[0]
                stats['database_records'] = count
                
        except Exception as e:
            logger.error(f"Error getting deduplication stats: {e}")
        
        return stats
    
    def force_allow_message(self, message: NotificationMessage):
        """
        Force allow a message by removing it from deduplication tracking.
        
        This is useful for emergency notifications that need to bypass
        normal deduplication rules.
        
        Args:
            message: The message to force allow
        """
        content_hash = message.get_content_hash()
        channel_key = f"{content_hash}:{message.channel}"
        
        # Remove from memory cache
        self._in_memory_cache.pop(channel_key, None)
        
        # Remove from database
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('''
                    DELETE FROM message_deduplication 
                    WHERE content_hash = ? AND channel = ?
                ''', (content_hash, message.channel))
                conn.commit()
                
            logger.info(f"Forced allow for message: {content_hash}")
            
        except Exception as e:
            logger.error(f"Error force allowing message: {e}")


class DuplicateDetectionError(Exception):
    """Raised when a duplicate message is detected."""
    pass


class SmartDeduplicationService(DeduplicationService):
    """
    Enhanced deduplication service with smart duplicate detection.
    
    Provides more sophisticated duplicate detection including:
    - Similar content detection (fuzzy matching)
    - Time-based clustering
    - Strategy-specific rules
    """
    
    def __init__(self, db: NotificationDatabase, dedup_window_minutes: int = 5,
                 similarity_threshold: float = 0.8):
        super().__init__(db, dedup_window_minutes)
        self.similarity_threshold = similarity_threshold
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two message contents.
        
        Uses simple word-based similarity for now, could be enhanced
        with more sophisticated NLP techniques.
        
        Args:
            content1: First message content
            content2: Second message content
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def is_similar_message(self, message: NotificationMessage) -> bool:
        """
        Check if message is similar to recent messages for the same channel.
        
        This is more computationally expensive than exact duplicate detection
        but can catch paraphrased or slightly modified duplicate content.
        
        Args:
            message: The message to check for similarity
            
        Returns:
            True if similar message found within time window
        """
        cutoff_time = utc_now() - self.dedup_window
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                # Get recent messages for same channel
                recent_messages = conn.execute('''
                    SELECT content FROM notification_queue
                    WHERE channel = ? AND created_at > ? AND status = 'sent'
                    ORDER BY created_at DESC LIMIT 10
                ''', (message.channel, cutoff_time)).fetchall()
                
                for (recent_content,) in recent_messages:
                    similarity = self._calculate_content_similarity(
                        message.content, recent_content
                    )
                    
                    if similarity >= self.similarity_threshold:
                        logger.debug(f"Similar message detected (similarity: {similarity:.2f})")
                        return True
                
        except Exception as e:
            logger.error(f"Error checking message similarity: {e}")
        
        return False
    
    def should_allow_message(self, message: NotificationMessage) -> bool:
        """
        Comprehensive check whether message should be allowed.
        
        Combines exact duplicate detection with similarity checking.
        
        Args:
            message: The message to evaluate
            
        Returns:
            True if message should be allowed (not a duplicate)
        """
        # Check exact duplicates first (faster)
        if self.is_duplicate(message):
            return False
        
        # Check for similar messages (slower)
        if self.is_similar_message(message):
            return False
        
        return True