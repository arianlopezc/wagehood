"""
Notification system data models and database schema.
"""

import sqlite3
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from ..utils.timezone_utils import utc_now
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Notification type enumeration."""
    SIGNAL = "signal"
    SERVICE = "service"
    SUMMARY = "summary"


class NotificationStatus(Enum):
    """Notification status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"


@dataclass
class NotificationMessage:
    """Immutable notification message data structure."""
    id: str
    type: NotificationType
    content: str
    channel: str
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    symbol: Optional[str] = None
    priority: int = 2  # 1=high, 2=normal, 3=low
    created_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    status: NotificationStatus = NotificationStatus.PENDING
    
    def __post_init__(self):
        if self.created_at is None:
            object.__setattr__(self, 'created_at', utc_now())
    
    @classmethod
    def create_signal_notification(cls, symbol: str, strategy: str, timeframe: str, 
                                 content: str) -> 'NotificationMessage':
        """Create a signal notification."""
        channel = CHANNEL_ROUTING.get((strategy, timeframe))
        if not channel:
            raise ValueError(f"No channel mapping for strategy={strategy}, timeframe={timeframe}")
        
        return cls(
            id=str(uuid.uuid4()),
            type=NotificationType.SIGNAL,
            content=content,
            channel=channel,
            strategy=strategy,
            timeframe=timeframe,
            symbol=symbol,
            priority=1  # High priority for signals
        )
    
    @classmethod
    def create_service_notification(cls, content: str, priority: int = 2) -> 'NotificationMessage':
        """Create a service notification."""
        return cls(
            id=str(uuid.uuid4()),
            type=NotificationType.SERVICE,
            content=content,
            channel='infra',
            priority=priority
        )
    
    @classmethod
    def create_summary_notification(cls, content: str, priority: int = 1) -> 'NotificationMessage':
        """Create an end-of-day summary notification."""
        return cls(
            id=str(uuid.uuid4()),
            type=NotificationType.SUMMARY,
            content=content,
            channel='eod-summary',
            priority=priority
        )
    
    def get_content_hash(self) -> str:
        """Generate content hash for deduplication."""
        hash_content = f"{self.channel}:{self.content}:{self.symbol}:{self.strategy}:{self.timeframe}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.next_retry_at:
            data['next_retry_at'] = self.next_retry_at.isoformat()
        return data


@dataclass
class DeduplicationRecord:
    """Deduplication tracking record."""
    content_hash: str
    channel: str
    first_seen: datetime
    last_seen: datetime


@dataclass
class ChannelConfig:
    """Channel configuration record."""
    channel_name: str
    webhook_url: str
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    enabled: bool = True


# Channel routing configuration
CHANNEL_ROUTING = {
    # MACD+RSI strategy - ONLY 1d signals to macd-rsi channel
    ('macd_rsi', '1d'): 'macd-rsi',
    
    # Support/Resistance breakout - ONLY 1d signals to support-resistance channel
    ('sr_breakout', '1d'): 'support-resistance',
    
    # RSI Trend Following - ONLY 1h signals to rsi-trend-following channel
    ('rsi_trend', '1h'): 'rsi-trend-following',
    
    # Bollinger Band Breakout - ONLY 1h signals to bollinger-band-breakout channel
    ('bollinger_breakout', '1h'): 'bollinger-band-breakout',
    
    # Crypto Bollinger Band Breakout - ONLY 1d signals to crypto-bollinger-1d channel
    ('bollinger_breakout_crypto', '1d'): 'crypto-bollinger-1d',
    
    # End-of-day summary routes to eod-summary channel
    ('eod_summary', None): 'eod-summary',
    
    # Service messages route to infra channel
    ('service', None): 'infra'
}


class NotificationDatabase:
    """SQLite database manager for notification system."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            wagehood_dir = Path.home() / '.wagehood'
            wagehood_dir.mkdir(exist_ok=True)
            db_path = str(wagehood_dir / 'notifications.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notification_queue (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    strategy TEXT,
                    timeframe TEXT,
                    symbol TEXT,
                    priority INTEGER DEFAULT 2,
                    created_at TIMESTAMP NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    next_retry_at TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS message_deduplication (
                    content_hash TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    PRIMARY KEY (content_hash, channel)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS channel_config (
                    channel_name TEXT PRIMARY KEY,
                    webhook_url TEXT NOT NULL,
                    strategy TEXT,
                    timeframe TEXT,
                    enabled BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Create indexes for performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_queue_status_priority 
                ON notification_queue(status, priority, created_at)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_queue_retry_time 
                ON notification_queue(next_retry_at) 
                WHERE next_retry_at IS NOT NULL
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_dedup_last_seen 
                ON message_deduplication(last_seen)
            ''')
            
            conn.commit()
    
    def add_message(self, message: NotificationMessage) -> bool:
        """Add message to queue. Returns False if duplicate detected."""
        content_hash = message.get_content_hash()
        
        with sqlite3.connect(self.db_path) as conn:
            # Check for duplicates within 5-minute window
            cutoff_time = utc_now() - timedelta(minutes=5)
            
            existing = conn.execute('''
                SELECT 1 FROM message_deduplication 
                WHERE content_hash = ? AND channel = ? AND last_seen > ?
            ''', (content_hash, message.channel, cutoff_time)).fetchone()
            
            if existing:
                logger.debug(f"Duplicate message detected: {content_hash}")
                return False
            
            # Add to queue
            conn.execute('''
                INSERT INTO notification_queue 
                (id, type, content, channel, strategy, timeframe, symbol, 
                 priority, created_at, retry_count, max_retries, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id, message.type.value, message.content, message.channel,
                message.strategy, message.timeframe, message.symbol,
                message.priority, message.created_at, message.retry_count,
                message.max_retries, message.status.value
            ))
            
            # Update deduplication record
            conn.execute('''
                INSERT OR REPLACE INTO message_deduplication
                (content_hash, channel, first_seen, last_seen)
                VALUES (?, ?, COALESCE((SELECT first_seen FROM message_deduplication 
                        WHERE content_hash = ? AND channel = ?), ?), ?)
            ''', (content_hash, message.channel, content_hash, message.channel,
                  message.created_at, message.created_at))
            
            conn.commit()
            logger.debug(f"Message queued: {message.id}")
            return True
    
    def get_next_message(self) -> Optional[NotificationMessage]:
        """Get next pending message by priority."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get highest priority pending message or ready retry
            now = utc_now()
            row = conn.execute('''
                SELECT * FROM notification_queue 
                WHERE (status = 'pending' OR (status = 'failed' AND next_retry_at <= ?))
                ORDER BY priority ASC, created_at ASC 
                LIMIT 1
            ''', (now,)).fetchone()
            
            if not row:
                return None
            
            # Convert to NotificationMessage
            message = NotificationMessage(
                id=row['id'],
                type=NotificationType(row['type']),
                content=row['content'],
                channel=row['channel'],
                strategy=row['strategy'],
                timeframe=row['timeframe'],
                symbol=row['symbol'],
                priority=row['priority'],
                created_at=datetime.fromisoformat(row['created_at']),
                retry_count=row['retry_count'],
                max_retries=row['max_retries'],
                next_retry_at=datetime.fromisoformat(row['next_retry_at']) if row['next_retry_at'] else None,
                status=NotificationStatus(row['status'])
            )
            
            # Mark as processing
            conn.execute('''
                UPDATE notification_queue 
                SET status = 'processing' 
                WHERE id = ?
            ''', (message.id,))
            conn.commit()
            
            return message
    
    def mark_message_sent(self, message_id: str):
        """Mark message as successfully sent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE notification_queue 
                SET status = 'sent' 
                WHERE id = ?
            ''', (message_id,))
            conn.commit()
    
    def mark_message_failed(self, message_id: str, retry_after: Optional[timedelta] = None):
        """Mark message as failed and schedule retry if applicable."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current retry count
            row = conn.execute('''
                SELECT retry_count, max_retries FROM notification_queue 
                WHERE id = ?
            ''', (message_id,)).fetchone()
            
            if not row:
                return
            
            retry_count, max_retries = row
            new_retry_count = retry_count + 1
            
            if new_retry_count <= max_retries and retry_after:
                # Schedule retry
                next_retry = utc_now() + retry_after
                conn.execute('''
                    UPDATE notification_queue 
                    SET retry_count = ?, next_retry_at = ?, status = 'failed'
                    WHERE id = ?
                ''', (new_retry_count, next_retry, message_id))
            else:
                # Mark as permanently failed
                conn.execute('''
                    UPDATE notification_queue 
                    SET retry_count = ?, status = 'failed'
                    WHERE id = ?
                ''', (new_retry_count, message_id))
            
            conn.commit()
    
    def cleanup_old_records(self, older_than_hours: int = 1):
        """Clean up old deduplication records."""
        cutoff = utc_now() - timedelta(hours=older_than_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM message_deduplication 
                WHERE last_seen < ?
            ''', (cutoff,))
            conn.commit()
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            for status in ['pending', 'processing', 'sent', 'failed']:
                count = conn.execute('''
                    SELECT COUNT(*) FROM notification_queue 
                    WHERE status = ?
                ''', (status,)).fetchone()[0]
                stats[status] = count
            
            # Count ready retries
            now = utc_now()
            retry_count = conn.execute('''
                SELECT COUNT(*) FROM notification_queue 
                WHERE status = 'failed' AND next_retry_at <= ?
            ''', (now,)).fetchone()[0]
            stats['ready_retries'] = retry_count
            
            return stats
    
    def save_channel_config(self, config: ChannelConfig):
        """Save channel configuration."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO channel_config
                (channel_name, webhook_url, strategy, timeframe, enabled)
                VALUES (?, ?, ?, ?, ?)
            ''', (config.channel_name, config.webhook_url, config.strategy,
                  config.timeframe, config.enabled))
            conn.commit()
    
    def get_channel_config(self, channel_name: str) -> Optional[ChannelConfig]:
        """Get channel configuration."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('''
                SELECT * FROM channel_config 
                WHERE channel_name = ?
            ''', (channel_name,)).fetchone()
            
            if not row:
                return None
            
            return ChannelConfig(
                channel_name=row['channel_name'],
                webhook_url=row['webhook_url'],
                strategy=row['strategy'],
                timeframe=row['timeframe'],
                enabled=bool(row['enabled'])
            )
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get overall processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count total messages by status
            total_processed = conn.execute('''
                SELECT COUNT(*) FROM notification_queue 
                WHERE status IN ('sent', 'failed')
            ''').fetchone()[0]
            stats['total_processed'] = total_processed
            
            # Count sent messages
            total_sent = conn.execute('''
                SELECT COUNT(*) FROM notification_queue 
                WHERE status = 'sent'
            ''').fetchone()[0]
            stats['total_sent'] = total_sent
            
            # Count failed messages
            total_failed = conn.execute('''
                SELECT COUNT(*) FROM notification_queue 
                WHERE status = 'failed' AND retry_count >= max_retries
            ''').fetchone()[0]
            stats['total_failed'] = total_failed
            
            return stats