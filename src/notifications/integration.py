"""
Integration module for connecting notification service with signal detection.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .worker import NotificationService
from .client import send_signal_notification, send_service_notification

logger = logging.getLogger(__name__)


class SignalNotificationIntegrator:
    """
    Integrates signal detection with notification service.
    
    Formats signals from the streaming system and sends them to
    appropriate Discord channels based on strategy and timeframe.
    """
    
    def __init__(self, notification_service: Optional[NotificationService] = None):
        self.notification_service = notification_service
        self._enabled = True
        
    async def process_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Process a signal from the streaming system and send notification.
        
        Args:
            signal: Signal dictionary from streaming worker
            
        Returns:
            True if notification sent successfully
        """
        if not self._enabled:
            return True
        
        try:
            # Extract signal information
            symbol = signal.get('symbol', 'UNKNOWN')
            strategy = signal.get('strategy', 'unknown')
            timeframe = signal.get('timeframe', '1h')
            signal_type = signal.get('signal', 'SIGNAL')
            confidence = signal.get('confidence', 0.0)
            price = signal.get('price', 0.0)
            timestamp = signal.get('timestamp', datetime.now())
            
            # Format message content
            content = self._format_signal_content(
                symbol, signal_type, confidence, price, timestamp, signal.get('details', {})
            )
            
            # Send notification
            success = await send_signal_notification(symbol, strategy, timeframe, content)
            
            if success:
                logger.debug(f"Signal notification sent: {symbol} {strategy} {timeframe}")
            else:
                logger.warning(f"Failed to send signal notification: {symbol} {strategy} {timeframe}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing signal notification: {e}")
            return False
    
    def _format_signal_content(self, symbol: str, signal_type: str, confidence: float,
                             price: float, timestamp: datetime, details: Dict[str, Any]) -> str:
        """
        Format signal information into notification content.
        
        Args:
            symbol: Trading symbol
            signal_type: Type of signal (BUY/SELL)
            confidence: Signal confidence level
            price: Current price
            timestamp: Signal timestamp
            details: Additional signal details
            
        Returns:
            Formatted message content
        """
        # Base message
        action_emoji = "🟢" if signal_type.upper() == "BUY" else "🔴"
        
        content = f"{action_emoji} **{signal_type.upper()}** signal for **{symbol}**\n"
        content += f"Price: ${price:.2f}\n"
        content += f"Confidence: {confidence:.1%}\n"
        content += f"Time: {timestamp.strftime('%H:%M:%S')}"
        
        # Add relevant details
        if details:
            detail_lines = []
            
            # Common technical indicator values
            if 'rsi' in details:
                detail_lines.append(f"RSI: {details['rsi']:.1f}")
            if 'macd' in details:
                detail_lines.append(f"MACD: {details['macd']:.4f}")
            if 'bb_position' in details:
                detail_lines.append(f"BB Position: {details['bb_position']}")
            if 'volume_ratio' in details:
                detail_lines.append(f"Volume Ratio: {details['volume_ratio']:.2f}")
            
            if detail_lines:
                content += "\n" + " | ".join(detail_lines)
        
        return content
    
    def enable(self):
        """Enable signal notifications."""
        self._enabled = True
        logger.info("Signal notifications enabled")
    
    def disable(self):
        """Disable signal notifications."""
        self._enabled = False
        logger.info("Signal notifications disabled")
    
    def is_enabled(self) -> bool:
        """Check if signal notifications are enabled."""
        return self._enabled


class ServiceNotificationHelper:
    """
    Helper class for sending service notifications from various components.
    """
    
    @staticmethod
    async def notify_service_started(service_name: str, details: Optional[str] = None):
        """Send notification when service starts."""
        message = f"🚀 {service_name} started"
        if details:
            message += f"\n{details}"
        
        await send_service_notification(message, priority=2)
    
    @staticmethod
    async def notify_service_stopped(service_name: str, details: Optional[str] = None):
        """Send notification when service stops."""
        message = f"⏹️ {service_name} stopped"
        if details:
            message += f"\n{details}"
        
        await send_service_notification(message, priority=2)
    
    @staticmethod
    async def notify_error(component: str, error: str, priority: int = 1):
        """Send error notification."""
        message = f"🚨 Error in {component}\n{error}"
        await send_service_notification(message, priority=priority)
    
    @staticmethod
    async def notify_warning(component: str, warning: str):
        """Send warning notification."""
        message = f"⚠️ Warning in {component}\n{warning}"
        await send_service_notification(message, priority=2)
    
    @staticmethod
    async def notify_performance_issue(component: str, metrics: Dict[str, Any]):
        """Send performance issue notification."""
        message = f"📊 Performance issue in {component}\n"
        
        for metric, value in metrics.items():
            message += f"{metric}: {value}\n"
        
        await send_service_notification(message, priority=1)
    
    @staticmethod
    async def notify_system_stats(stats: Dict[str, Any]):
        """Send periodic system statistics."""
        message = "📈 System Statistics\n"
        
        for key, value in stats.items():
            if isinstance(value, dict):
                message += f"{key}:\n"
                for subkey, subvalue in value.items():
                    message += f"  {subkey}: {subvalue}\n"
            else:
                message += f"{key}: {value}\n"
        
        await send_service_notification(message, priority=3)


class StreamingManagerNotificationMixin:
    """
    Mixin class to add notification capabilities to StreamingManager.
    
    This can be used to extend the existing StreamingManager with
    notification functionality without modifying the original class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_integrator = SignalNotificationIntegrator()
        self.notification_helper = ServiceNotificationHelper()
        self._notification_service: Optional[NotificationService] = None
    
    async def _init_notifications(self):
        """Initialize notification service."""
        try:
            from .worker import get_notification_service
            self._notification_service = await get_notification_service()
            await self._notification_service.start()
            
            # Send startup notification
            await self.notification_helper.notify_service_started(
                "Streaming Manager",
                f"Processing {len(self.symbols)} symbols with {self.num_workers} workers"
            )
            
            logger.info("Notification service initialized for streaming manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {e}")
            # Continue without notifications rather than failing
    
    async def _shutdown_notifications(self):
        """Shutdown notification service."""
        if self._notification_service:
            try:
                await self.notification_helper.notify_service_stopped("Streaming Manager")
                await self._notification_service.stop()
                logger.info("Notification service shutdown completed")
            except Exception as e:
                logger.error(f"Error during notification service shutdown: {e}")
    
    async def _handle_signal_with_notifications(self, signal: Dict[str, Any]):
        """Handle signal and send notification."""
        try:
            # Process signal normally
            await self._original_handle_signal(signal)
            
            # Send notification
            await self.signal_integrator.process_signal(signal)
            
        except Exception as e:
            logger.error(f"Error handling signal with notifications: {e}")
            await self.notification_helper.notify_error(
                "Signal Processing",
                f"Failed to process signal for {signal.get('symbol', 'unknown')}: {str(e)}"
            )
    
    async def _notify_performance_metrics(self, metrics: Dict[str, Any]):
        """Send performance metrics notification."""
        # Only send if metrics indicate issues
        issues = []
        
        if metrics.get('memory_usage', 0) > 90:
            issues.append(f"High memory usage: {metrics['memory_usage']:.1f}%")
        
        if metrics.get('queue_depth', 0) > 1000:
            issues.append(f"High queue depth: {metrics['queue_depth']}")
        
        if metrics.get('error_rate', 0) > 0.1:
            issues.append(f"High error rate: {metrics['error_rate']:.1%}")
        
        if issues:
            await self.notification_helper.notify_performance_issue(
                "Streaming Manager",
                metrics
            )


def create_enhanced_streaming_manager(*args, **kwargs):
    """
    Factory function to create StreamingManager with notification capabilities.
    
    This creates a new class that inherits from both StreamingManager and
    the notification mixin, providing notification functionality.
    """
    try:
        from ..streaming.manager import StreamingManager
        
        class EnhancedStreamingManager(StreamingManagerNotificationMixin, StreamingManager):
            """StreamingManager with notification capabilities."""
            
            async def start(self):
                # Initialize notifications first
                await self._init_notifications()
                
                # Start the original streaming manager
                await super().start()
            
            async def shutdown(self):
                # Shutdown notifications after main shutdown
                await super().shutdown()
                await self._shutdown_notifications()
        
        return EnhancedStreamingManager(*args, **kwargs)
        
    except ImportError:
        logger.error("Could not import StreamingManager for enhancement")
        return None


async def integrate_notifications_with_existing_signals():
    """
    Function to integrate notifications with existing signal processing.
    
    This can be called from the existing streaming system to add
    notification capabilities without major refactoring.
    """
    integrator = SignalNotificationIntegrator()
    
    # Return the signal processing function that can be used as a callback
    return integrator.process_signal


def setup_notification_logging():
    """Set up logging for notification system."""
    # Configure notification-specific logging
    notification_logger = logging.getLogger('src.notifications')
    notification_logger.setLevel(logging.INFO)
    
    # Ensure notification logs are visible
    if not notification_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        notification_logger.addHandler(handler)
    
    logger.info("Notification logging configured")