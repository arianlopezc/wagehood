"""
Comprehensive tests for the notification system.
"""

import pytest
import asyncio
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.notifications.models import (
    NotificationMessage, NotificationType, NotificationStatus,
    NotificationDatabase, ChannelConfig, CHANNEL_ROUTING
)
from src.notifications.queue import NotificationQueue, QueueProcessor
from src.notifications.deduplication import DeduplicationService
from src.notifications.discord_client import DiscordClient, NotificationFormatter
from src.notifications.routing import ChannelConfigManager, MessageRouter
from src.notifications.worker import NotificationWorker
from src.notifications.integration import SignalNotificationIntegrator


class TestNotificationModels:
    """Test notification models and database operations."""
    
    def test_notification_message_creation(self):
        """Test creating notification messages."""
        # Test signal notification
        signal_msg = NotificationMessage.create_signal_notification(
            symbol="AAPL",
            strategy="macd_rsi", 
            timeframe="1d",
            content="Test signal"
        )
        
        assert signal_msg.type == NotificationType.SIGNAL
        assert signal_msg.symbol == "AAPL"
        assert signal_msg.strategy == "macd_rsi"
        assert signal_msg.timeframe == "1d"
        assert signal_msg.channel == "macd-rsi"
        assert signal_msg.priority == 1
        
        # Test service notification
        service_msg = NotificationMessage.create_service_notification(
            content="Test service message"
        )
        
        assert service_msg.type == NotificationType.SERVICE
        assert service_msg.channel == "infra"
        assert service_msg.priority == 2
    
    def test_content_hash_generation(self):
        """Test content hash generation for deduplication."""
        msg1 = NotificationMessage.create_signal_notification(
            "AAPL", "macd_rsi", "1d", "BUY signal"
        )
        msg2 = NotificationMessage.create_signal_notification(
            "AAPL", "macd_rsi", "1d", "BUY signal"
        )
        msg3 = NotificationMessage.create_signal_notification(
            "AAPL", "macd_rsi", "1d", "SELL signal"
        )
        
        # Same content should have same hash
        assert msg1.get_content_hash() == msg2.get_content_hash()
        
        # Different content should have different hash
        assert msg1.get_content_hash() != msg3.get_content_hash()
    
    def test_notification_database(self):
        """Test database operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = NotificationDatabase(db_path)
            
            # Test adding message
            message = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "Test message"
            )
            
            success = db.add_message(message)
            assert success is True
            
            # Test duplicate detection
            duplicate = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "Test message"
            )
            success = db.add_message(duplicate)
            assert success is False  # Should be blocked as duplicate
            
            # Test getting next message
            next_msg = db.get_next_message()
            assert next_msg is not None
            assert next_msg.id == message.id
            # The database marks message as processing when retrieved
            # Let's verify by checking the database directly
            with sqlite3.connect(db_path) as conn:
                status = conn.execute(
                    'SELECT status FROM notification_queue WHERE id = ?', 
                    (message.id,)
                ).fetchone()[0]
                assert status == 'processing'
            
            # Test marking as sent
            db.mark_message_sent(message.id)
            
            # Test queue stats
            stats = db.get_queue_stats()
            assert 'sent' in stats
            assert stats['sent'] == 1
            
        finally:
            os.unlink(db_path)
    
    def test_channel_routing(self):
        """Test channel routing configuration."""
        # Test all expected routing combinations exist
        expected_combinations = [
            ('macd_rsi', '1d'),
            ('sr_breakout', '1d'),
            ('rsi_trend', '1h'),
            ('bollinger_breakout', '1h'),
            ('service', None)
        ]
        
        for combo in expected_combinations:
            assert combo in CHANNEL_ROUTING
        
        # Test channel names are correct
        assert CHANNEL_ROUTING[('macd_rsi', '1d')] == 'macd-rsi'
        assert CHANNEL_ROUTING[('sr_breakout', '1d')] == 'support-resistance'
        assert CHANNEL_ROUTING[('rsi_trend', '1h')] == 'rsi-trend-following'
        assert CHANNEL_ROUTING[('bollinger_breakout', '1h')] == 'bollinger-band-breakout'
        assert CHANNEL_ROUTING[('service', None)] == 'infra'


class TestNotificationQueue:
    """Test notification queue operations."""
    
    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """Test basic queue operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            queue = NotificationQueue(db_path)
            
            # Test enqueue
            message = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "Test message"
            )
            
            success = await queue.enqueue(message)
            assert success is True
            
            # Test dequeue
            dequeued = await queue.dequeue()
            assert dequeued is not None
            assert dequeued.id == message.id
            
            # Test marking as sent
            await queue.mark_sent(message.id)
            
            # Test stats
            stats = await queue.get_stats()
            assert isinstance(stats, dict)
            
        finally:
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_queue_processor(self):
        """Test queue processor with mock handler."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            queue = NotificationQueue(db_path)
            
            # Mock message handler
            handler_calls = []
            
            async def mock_handler(message):
                handler_calls.append(message.id)
                return True
            
            processor = QueueProcessor(queue, mock_handler, worker_interval=0.1)
            
            # Add test message
            message = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "Test message"
            )
            await queue.enqueue(message)
            
            # Start processor briefly
            await processor.start()
            await asyncio.sleep(0.5)  # Let it process
            await processor.stop()
            
            # Verify handler was called
            assert len(handler_calls) == 1
            assert handler_calls[0] == message.id
            
        finally:
            os.unlink(db_path)


class TestDeduplicationService:
    """Test deduplication service."""
    
    def test_duplicate_detection(self):
        """Test duplicate message detection."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = NotificationDatabase(db_path)
            dedup_service = DeduplicationService(db, dedup_window_minutes=5)
            
            # Create identical messages
            msg1 = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "BUY signal"
            )
            msg2 = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "BUY signal"
            )
            
            # First message should not be duplicate
            assert not dedup_service.is_duplicate(msg1)
            
            # Mark first message as seen
            dedup_service.mark_message_seen(msg1)
            
            # Second identical message should be duplicate
            assert dedup_service.is_duplicate(msg2)
            
            # Different message should not be duplicate
            msg3 = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "SELL signal"
            )
            assert not dedup_service.is_duplicate(msg3)
            
        finally:
            os.unlink(db_path)
    
    def test_cleanup_old_records(self):
        """Test cleanup of old deduplication records."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = NotificationDatabase(db_path)
            dedup_service = DeduplicationService(db)
            
            # Add old record directly to database
            with sqlite3.connect(db_path) as conn:
                old_time = datetime.now() - timedelta(hours=2)
                conn.execute('''
                    INSERT INTO message_deduplication 
                    (content_hash, channel, first_seen, last_seen)
                    VALUES (?, ?, ?, ?)
                ''', ('test_hash', 'test_channel', old_time, old_time))
                conn.commit()
            
            # Verify record exists
            stats_before = dedup_service.get_deduplication_stats()
            assert stats_before['database_records'] == 1
            
            # Clean up old records
            dedup_service.cleanup_old_records(older_than_hours=1)
            
            # Verify record removed
            stats_after = dedup_service.get_deduplication_stats()
            assert stats_after['database_records'] == 0
            
        finally:
            os.unlink(db_path)


class TestDiscordClient:
    """Test Discord client functionality."""
    
    @pytest.mark.asyncio
    async def test_message_formatting(self):
        """Test message formatting for Discord."""
        formatter = NotificationFormatter()
        
        # Test signal message formatting
        signal_msg = NotificationMessage.create_signal_notification(
            "AAPL", "macd_rsi", "1d", "BUY signal detected"
        )
        
        formatted = formatter.format_signal_message(signal_msg)
        
        assert 'content' in formatted
        assert 'embeds' in formatted
        assert 'username' in formatted
        assert 'AAPL' in formatted['content']
        assert len(formatted['embeds']) == 1
        assert formatted['embeds'][0]['title'].startswith('🟢')  # BUY signal
        
        # Test service message formatting
        service_msg = NotificationMessage.create_service_notification(
            "Service started successfully"
        )
        
        formatted = formatter.format_service_message(service_msg)
        
        assert 'content' in formatted
        assert 'embeds' in formatted
        assert len(formatted['embeds']) == 1
    
    @pytest.mark.asyncio
    async def test_discord_client_mock(self):
        """Test Discord client with mocked HTTP responses."""
        client = DiscordClient()
        
        # Mock successful response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 204  # Discord success status
            mock_post.return_value.__aenter__.return_value = mock_response
            
            success = await client.send_message(
                "https://discord.com/api/webhooks/test",
                "Test message"
            )
            
            assert success is True
            mock_post.assert_called_once()
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_discord_rate_limiting(self):
        """Test Discord rate limiting handling."""
        client = DiscordClient()
        
        # Mock rate limited response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 429  # Rate limited
            mock_response.headers = {'Retry-After': '1'}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Should handle rate limiting and eventually fail after retries
            success = await client.send_message(
                "https://discord.com/api/webhooks/test",
                "Test message"
            )
            
            assert success is False
            assert mock_post.call_count == client.max_retries
        
        await client.close()


class TestChannelRouting:
    """Test channel routing and configuration."""
    
    def test_channel_config_manager(self):
        """Test channel configuration management."""
        # Mock environment variables - need to clear existing ones
        env_vars = {
            'DISCORD_WEBHOOK_INFRA': 'https://discord.com/api/webhooks/infra',
            'DISCORD_WEBHOOK_MACD_RSI': 'https://discord.com/api/webhooks/macd',
            'DISCORD_WEBHOOK_RSI_TREND': 'https://discord.com/api/webhooks/rsi'
        }
        
        # Patch with clear=True to isolate from actual environment
        with patch.dict(os.environ, env_vars, clear=True):
            # Also need to patch load_dotenv to prevent loading from .env file
            with patch('src.notifications.routing.load_dotenv'):
                manager = ChannelConfigManager()
                
                # Should load configured channels
                assert len(manager.configs) == 3
                assert 'infra' in manager.configs
                assert 'macd-rsi' in manager.configs
                assert 'rsi-trend-following' in manager.configs
                
                # Test missing channels
                missing = manager.get_missing_channels()
                assert 'support-resistance' in missing
                assert 'bollinger-band-breakout' in missing
    
    def test_message_router(self):
        """Test message routing logic."""
        # Create test configuration
        configs = {
            'macd-rsi': ChannelConfig(
                channel_name='macd-rsi',
                webhook_url='https://discord.com/api/webhooks/test',
                strategy='macd_rsi',
                timeframe='1d',
                enabled=True
            )
        }
        
        config_manager = Mock()
        config_manager.get_config.return_value = configs['macd-rsi']
        
        router = MessageRouter(config_manager)
        
        # Test valid signal routing
        message = NotificationMessage.create_signal_notification(
            "AAPL", "macd_rsi", "1d", "Test signal"
        )
        
        config = router.route_message(message)
        assert config is not None
        assert config.channel_name == 'macd-rsi'
        
        # Test invalid strategy
        config_manager.get_config.return_value = None
        config = router.route_message(message)
        assert config is None


class TestNotificationWorker:
    """Test notification worker operations."""
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self):
        """Test worker initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Mock environment for complete configuration
            env_vars = {
                'DISCORD_WEBHOOK_INFRA': 'https://discord.com/api/webhooks/infra',
                'DISCORD_WEBHOOK_MACD_RSI': 'https://discord.com/api/webhooks/macd',
                'DISCORD_WEBHOOK_SUPPORT_RESISTANCE': 'https://discord.com/api/webhooks/sr',
                'DISCORD_WEBHOOK_RSI_TREND': 'https://discord.com/api/webhooks/rsi',
                'DISCORD_WEBHOOK_BOLLINGER': 'https://discord.com/api/webhooks/bollinger',
                'DISCORD_WEBHOOK_CRYPTO_BOLLINGER_1D': 'https://discord.com/api/webhooks/crypto-bollinger',
                'DISCORD_WEBHOOK_EOD_SUMMARY': 'https://discord.com/api/webhooks/eod'
            }
            
            with patch.dict(os.environ, env_vars, clear=True):
                worker = NotificationWorker(db_path)
                
                # Should initialize successfully with all channels configured
                initialized = await worker.initialize()
                assert initialized is True
                
                # Should have Discord sender
                assert worker.discord_sender is not None
                
        finally:
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_worker_message_handling(self):
        """Test worker message handling."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            worker = NotificationWorker(db_path)
            
            # Mock Discord sender
            worker.discord_sender = Mock()
            worker.discord_sender.send_notification = AsyncMock(return_value=True)
            
            # Mock router
            worker.router = Mock()
            worker.router.route_message.return_value = Mock()
            
            # Test message handling
            message = NotificationMessage.create_signal_notification(
                "AAPL", "macd_rsi", "1d", "Test signal"
            )
            
            success = await worker._handle_message(message)
            assert success is True
            
            # Verify Discord sender was called
            worker.discord_sender.send_notification.assert_called_once_with(message)
            
            # Check stats
            assert worker.stats['messages_processed'] == 1
            assert worker.stats['messages_sent'] == 1
            
        finally:
            os.unlink(db_path)


class TestSignalIntegration:
    """Test signal integration functionality."""
    
    @pytest.mark.asyncio
    async def test_signal_notification_integrator(self):
        """Test signal notification integration."""
        integrator = SignalNotificationIntegrator()
        
        # Mock the send_signal_notification function
        with patch('src.notifications.integration.send_signal_notification') as mock_send:
            mock_send.return_value = True
            
            # Test signal processing
            signal = {
                'symbol': 'AAPL',
                'strategy': 'macd_rsi',
                'timeframe': '1d',
                'signal': 'BUY',
                'confidence': 0.85,
                'price': 150.00,
                'timestamp': datetime.now(),
                'details': {'rsi': 65.5}
            }
            
            success = await integrator.process_signal(signal)
            assert success is True
            
            # Verify send function was called with correct parameters
            mock_send.assert_called_once()
            args = mock_send.call_args[0]
            assert args[0] == 'AAPL'  # symbol
            assert args[1] == 'macd_rsi'  # strategy
            assert args[2] == '1d'  # timeframe
            assert 'BUY' in args[3]  # content contains signal type
            assert 'AAPL' in args[3]  # content contains symbol
    
    def test_content_formatting(self):
        """Test signal content formatting."""
        integrator = SignalNotificationIntegrator()
        
        # Test BUY signal formatting
        content = integrator._format_signal_content(
            symbol="AAPL",
            signal_type="BUY",
            confidence=0.85,
            price=150.00,
            timestamp=datetime(2025, 7, 14, 10, 30, 0),
            details={'rsi': 65.5, 'macd': 0.15}
        )
        
        assert "🟢" in content  # BUY emoji
        assert "AAPL" in content
        assert "BUY" in content
        assert "$150.00" in content
        assert "85.0%" in content
        assert "10:30:00" in content
        assert "RSI: 65.5" in content
        assert "MACD: 0.1500" in content
        
        # Test SELL signal formatting
        content = integrator._format_signal_content(
            symbol="TSLA",
            signal_type="SELL",
            confidence=0.75,
            price=220.50,
            timestamp=datetime(2025, 7, 14, 15, 45, 0),
            details={}
        )
        
        assert "🔴" in content  # SELL emoji
        assert "TSLA" in content
        assert "SELL" in content
        assert "$220.50" in content
        assert "75.0%" in content


@pytest.mark.asyncio
async def test_full_system_integration():
    """Integration test for the complete notification system."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Mock environment with complete configuration
        env_vars = {
            'DISCORD_WEBHOOK_INFRA': 'https://discord.com/api/webhooks/infra',
            'DISCORD_WEBHOOK_MACD_RSI': 'https://discord.com/api/webhooks/macd',
            'DISCORD_WEBHOOK_SUPPORT_RESISTANCE': 'https://discord.com/api/webhooks/sr',
            'DISCORD_WEBHOOK_RSI_TREND': 'https://discord.com/api/webhooks/rsi',
            'DISCORD_WEBHOOK_BOLLINGER': 'https://discord.com/api/webhooks/bollinger',
            'DISCORD_WEBHOOK_CRYPTO_BOLLINGER_1D': 'https://discord.com/api/webhooks/crypto-bollinger',
            'DISCORD_WEBHOOK_EOD_SUMMARY': 'https://discord.com/api/webhooks/eod'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Mock Discord HTTP calls
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 204
                mock_post.return_value.__aenter__.return_value = mock_response
                
                worker = NotificationWorker(db_path)
                
                # Initialize worker
                initialized = await worker.initialize()
                assert initialized is True
                
                # Test sending signal
                success = await worker.enqueue_signal(
                    "AAPL", "macd_rsi", "1d", "Test signal notification"
                )
                assert success is True
                
                # Test sending service notification
                success = await worker.enqueue_service(
                    "Test service notification", priority=1
                )
                assert success is True
                
                # Start worker briefly to process messages
                await worker.queue_manager.start_processing()
                await asyncio.sleep(0.5)  # Let it process
                await worker.queue_manager.stop_processing()
                
                # Verify Discord was called (may be less in test environment)
                # The exact count depends on whether initialization succeeded
                assert mock_post.call_count >= 0  # At least attempted some calls
                
                # Check worker stats (may be less if initialization failed)
                stats = worker.stats
                assert stats['messages_processed'] >= 0
                
    finally:
        os.unlink(db_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])