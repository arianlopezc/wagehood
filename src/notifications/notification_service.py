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

from .constants import COMPANY_NAMES
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
            raise ImportError(
                "Redis package required for notifications. Install with: pip install redis"
            )

        self.config = config or NotificationConfig.from_environment()
        self.running = False
        self.tasks = []

        # Initialize components
        if self.config.enabled:
            self.discord_notifier = DiscordNotifier(
                self.config.discord_webhook_url, max_retries=self.config.max_retries
            )
            self.message_formatter = MessageFormatter()
            self.rate_limiter = NotificationRateLimiter(
                self.config.max_notifications_per_hour
            )
        else:
            logger.info("Discord notifications disabled in configuration")
            self.discord_notifier = None
            self.message_formatter = None
            self.rate_limiter = None

        # Redis connection
        self.redis_client = None
        self._last_notification_times = {}  # Track last notification per symbol

        logger.info(
            f"Notification service initialized (enabled: {self.config.enabled})"
        )

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

            # Monitor signal detection events stream for signals
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
            from src.core.constants import (
                REDIS_HOST,
                REDIS_PORT,
                REDIS_DB,
                REDIS_PASSWORD,
            )

            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
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

            timeframes_str = (
                ", ".join(self.config.timeframes_to_notify)
                if self.config.timeframes_to_notify
                else "All"
            )
            status_data = {
                "service": "Discord Notification Service",
                "status": "running",
                "symbol_count": (
                    len(self.config.symbols_to_notify)
                    if self.config.symbols_to_notify
                    else "All"
                ),
                "data_provider": "Alpaca Markets",
                "message": f'Monitoring {", ".join(self.config.signal_profiles)} signals for {timeframes_str} timeframes',
            }

            success = self.discord_notifier.send_system_status(status_data)
            if success:
                logger.info("Startup notification sent to Discord")
            else:
                logger.warning("Failed to send startup notification")

        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")

    async def _monitor_signal_events(self):
        """Monitor Redis streams for signal detection events that contain signals."""
        logger.info("Starting signal event monitoring...")

        stream_name = "signal_events_stream"
        consumer_group = "discord_notifications"
        consumer_name = "discord_notifier"

        # Create consumer group if it doesn't exist
        try:
            self.redis_client.xgroup_create(
                stream_name, consumer_group, id="0", mkstream=True
            )
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
                    {stream_name: ">"},
                    count=10,
                    block=1000,  # 1 second timeout
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
                logger.debug(
                    f"Skipping notification for {event_data.get('symbol', 'unknown')}"
                )
                return

            # Enhanced signal quality filtering
            signal_quality = event_data.get("signal_quality", {})
            if not self._meets_discord_quality_threshold(signal_quality):
                logger.debug(
                    f"Signal quality below Discord threshold for {event_data.get('symbol', 'unknown')}"
                )
                return

            # Check rate limiting
            if not self.rate_limiter.can_send_notification():
                logger.warning("Rate limit exceeded, skipping notification")
                return

            # Enhance notification with quality information
            enhanced_event_data = self._enhance_notification_with_quality_info(
                event_data
            )

            # Send Discord notification
            success = self.discord_notifier.send_signal_notification(
                enhanced_event_data
            )

            if success:
                self.rate_limiter.record_notification()
                symbol = event_data.get("symbol", "unknown")
                self._last_notification_times[symbol] = time.time()
                logger.info(
                    f"Sent Discord notification for {symbol} {event_data.get('signal', 'signal')}"
                )
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
            if "results" not in fields:
                return None

            # Parse results JSON
            results = json.loads(fields["results"])

            # Extract symbol
            symbol = fields.get("symbol", results.get("symbol", "UNKNOWN"))

            # Look for timeframe-specific signals (prioritize 1d timeframe for notifications)
            timeframe_results = results.get("timeframe_results", {})
            signal_data = None
            notification_timeframe = None

            # Check for 1d timeframe signals first (priority for Discord notifications)
            for timeframe in ["1d", "4h", "1h", "5m", "1m"]:
                if timeframe in timeframe_results:
                    tf_data = timeframe_results[timeframe]
                    if "signal" in tf_data and tf_data["signal"]:
                        # Check if this timeframe should trigger notifications
                        if self.config.should_notify_timeframe(timeframe):
                            signal_data = tf_data["signal"]
                            notification_timeframe = timeframe
                            break

            # If no timeframe-specific signal found, check composite signal
            if not signal_data:
                composite_signal = results.get("composite_signal")
                if composite_signal:
                    # Use composite signal but mark as 1d timeframe for consistency
                    notification_timeframe = "1d"

                    # Build notification data from composite signal
                    notification_data = {
                        "symbol": symbol,
                        "signal": composite_signal.get("direction", "UNKNOWN").upper(),
                        "price": 0.0,  # Price not directly available in composite signal
                        "price_change": 0.0,
                        "price_change_pct": 0.0,
                        "strategy": "Multi-Timeframe Composite",
                        "confidence": composite_signal.get("confidence", 0.0),
                        "timeframe": notification_timeframe,
                        "details": {
                            "timeframes_analyzed": composite_signal.get(
                                "timeframes_analyzed", 0
                            ),
                            "strength": composite_signal.get("strength", "unknown"),
                            "timeframe_alignment": composite_signal.get(
                                "timeframe_alignment", 0.0
                            ),
                            "strategy_alignment": composite_signal.get(
                                "strategy_alignment", 0.0
                            ),
                            "contributing_signals": composite_signal.get(
                                "contributing_signals", 0
                            ),
                        },
                        "timestamp": datetime.now(),
                        "timeframe": "composite",  # Mark as composite signal
                        "source_timeframes": self._extract_source_timeframes(results),
                        "signal_quality": results.get(
                            "signal_quality", {}
                        ),  # Add signal quality for filtering
                    }

                # Add company name if available
                if symbol in COMPANY_NAMES:
                    notification_data["company_name"] = COMPANY_NAMES[symbol]

                return notification_data

            # Process timeframe-specific signal data
            if signal_data and notification_timeframe:
                # Build notification data from timeframe-specific signal
                notification_data = {
                    "symbol": symbol,
                    "signal": signal_data.get("action", "UNKNOWN").upper(),
                    "price": signal_data.get("price", 0.0),
                    "price_change": signal_data.get("price_change", 0.0),
                    "price_change_pct": signal_data.get("price_change_pct", 0.0),
                    "strategy": signal_data.get("strategy", "Unknown Strategy"),
                    "confidence": signal_data.get("confidence", 0.0),
                    "timeframe": notification_timeframe,
                    "details": signal_data.get("details", {}),
                    "timestamp": datetime.now(),
                    "signal_quality": results.get(
                        "signal_quality", {}
                    ),  # Add signal quality for filtering
                }

                # Add company name if available
                if symbol in COMPANY_NAMES:
                    notification_data["company_name"] = COMPANY_NAMES[symbol]

                return notification_data

            # Look for legacy signal data (backward compatibility)
            legacy_signal_data = results.get("signal")
            if legacy_signal_data:
                # Build notification data from legacy signal
                notification_data = {
                    "symbol": symbol,
                    "signal": legacy_signal_data.get("action", "UNKNOWN").upper(),
                    "price": legacy_signal_data.get("price", 0.0),
                    "price_change": legacy_signal_data.get("price_change", 0.0),
                    "price_change_pct": legacy_signal_data.get("price_change_pct", 0.0),
                    "strategy": legacy_signal_data.get("strategy", "Unknown Strategy"),
                    "confidence": legacy_signal_data.get("confidence", 0.0),
                    "details": legacy_signal_data.get("details", {}),
                    "timestamp": datetime.now(),
                    "timeframe": "legacy",  # Mark as legacy signal
                    "source_timeframes": ["unknown"],
                    "signal_quality": results.get(
                        "signal_quality", {}
                    ),  # Add signal quality for filtering
                }

                # Add company name if available
                if symbol in COMPANY_NAMES:
                    notification_data["company_name"] = COMPANY_NAMES[symbol]

                return notification_data

            # No signal data found
            return None

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
            symbol = signal_data.get("symbol", "")
            confidence = signal_data.get("confidence", 0.0)
            signal_type = signal_data.get("signal", "").upper()
            timeframe = signal_data.get("timeframe", "unknown")

            # Check symbol filter
            if not self.config.should_notify_symbol(symbol):
                logger.debug(f"Symbol {symbol} not in notification list")
                return False

            # Check timeframe filter (CRITICAL: Only notify for allowed timeframes)
            if not self.config.should_notify_timeframe(timeframe):
                logger.debug(
                    f"Timeframe {timeframe} not in notification list - skipping Discord notification"
                )
                return False

            # Confidence threshold removed - send all signals regardless of confidence
            logger.debug(f"Signal confidence {confidence} - sending notification (no threshold)")
            # Note: Confidence threshold check removed to send all detected signals

            # Check source timeframes if available
            source_timeframes = signal_data.get("source_timeframes", [])
            if source_timeframes and not self._should_notify_timeframes(
                source_timeframes
            ):
                logger.debug(
                    f"Signal timeframes {source_timeframes} not in notification list {self.config.timeframes_to_notify}"
                )
                return False

            # Only notify for BUY/SELL signals, not HOLD (unless explicitly configured)
            if signal_type == "HOLD":
                logger.debug(f"Skipping HOLD signal for {symbol}")
                return False

            logger.debug(
                f"Signal {symbol} {signal_type} passed all filters - will notify"
            )
            return True

        except Exception as e:
            logger.error(f"Error checking notification criteria: {e}")
            return False

    def _extract_source_timeframes(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract source timeframes from calculation results.

        Args:
            results: Calculation results dictionary

        Returns:
            List of source timeframes
        """
        try:
            timeframes = []

            # Check timeframe_results for multi-timeframe signals
            timeframe_results = results.get("timeframe_results", {})
            if timeframe_results:
                timeframes.extend(timeframe_results.keys())

            # Check symbol_config for configured timeframes
            symbol_config = results.get("symbol_config", {})
            if symbol_config:
                config_timeframes = symbol_config.get("timeframes", [])
                timeframes.extend(config_timeframes)

            # Remove duplicates and return
            return list(set(timeframes)) if timeframes else ["unknown"]

        except Exception as e:
            logger.error(f"Error extracting source timeframes: {e}")
            return ["unknown"]

    def _should_notify_timeframes(self, source_timeframes: List[str]) -> bool:
        """
        Check if any of the source timeframes should trigger notifications.

        Args:
            source_timeframes: List of source timeframes

        Returns:
            True if any timeframe should trigger notifications
        """
        try:
            # If no source timeframes provided, default to allowing
            if not source_timeframes or source_timeframes == ["unknown"]:
                logger.debug("No source timeframes provided, allowing notification")
                return True

            # Check if any source timeframe is in the notification list
            for timeframe in source_timeframes:
                if self.config.should_notify_timeframe(timeframe):
                    logger.debug(f"Timeframe {timeframe} is in notification list")
                    return True

            # Also check for composite signals that include 1d timeframe
            if "composite" in source_timeframes and "1d" in source_timeframes:
                if self.config.should_notify_timeframe("1d"):
                    logger.debug("Composite signal includes 1d timeframe")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking timeframe notification criteria: {e}")
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
                "service": "Discord Notifications",
                "status": "running",
                "message": f"Sent {rate_status['notifications_sent_last_hour']} notifications in last hour",
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
            "enabled": self.config.enabled,
            "running": self.running,
            "webhook_configured": bool(self.config.discord_webhook_url),
            "last_notifications": self._last_notification_times.copy(),
        }

        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_status()

        return stats

    def _meets_discord_quality_threshold(self, signal_quality: Dict[str, Any]) -> bool:
        """
        Check if signal quality meets Discord notification threshold.

        Args:
            signal_quality: Signal quality metrics from signal detection

        Returns:
            True if signal should be sent to Discord
        """
        try:
            if not signal_quality:
                return False

            # Check if signal is explicitly marked as Discord-ready
            if signal_quality.get("discord_ready", False):
                return True

            # Check quality score threshold
            quality_score = signal_quality.get("quality_score", 0.0)
            if quality_score < 0.6:  # Minimum quality for Discord
                return False

            # Check noise level
            noise_level = signal_quality.get("noise_level", "high")
            if noise_level == "high":
                return False

            # Check validation recommendation
            recommendation = signal_quality.get("recommendation", "filter_out")
            if recommendation == "filter_out":
                return False

            # Check confidence threshold
            confidence = signal_quality.get("confidence_score", 0.0)
            if confidence < 0.6:
                return False

            # Check timeframe alignment
            timeframe_alignment = signal_quality.get("timeframe_alignment_score", 0.0)
            if timeframe_alignment < 0.5:
                return False

            # Check contributing signals
            contributing_signals = signal_quality.get("contributing_signals_count", 0)
            if contributing_signals < 2:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking signal quality for Discord: {e}")
            return False

    def _enhance_notification_with_quality_info(
        self, notification_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance notification data with signal quality information.

        Args:
            notification_data: Base notification data

        Returns:
            Enhanced notification data with quality metrics
        """
        try:
            signal_quality = notification_data.get("signal_quality", {})

            # Add quality metrics to details
            if "details" not in notification_data:
                notification_data["details"] = {}

            notification_data["details"].update(
                {
                    "quality_score": signal_quality.get("quality_score", 0.0),
                    "noise_level": signal_quality.get("noise_level", "unknown"),
                    "signal_persistence": signal_quality.get(
                        "signal_persistence_score", 0.0
                    ),
                    "validation_reasons": signal_quality.get("validation_reasons", []),
                }
            )

            # Add quality badge to strategy name
            quality_score = signal_quality.get("quality_score", 0.0)
            if quality_score >= 0.8:
                quality_badge = "🟢"  # High quality
            elif quality_score >= 0.6:
                quality_badge = "🟡"  # Medium quality
            else:
                quality_badge = "🔴"  # Low quality

            notification_data["strategy"] = (
                f"{quality_badge} {notification_data.get('strategy', 'Unknown')}"
            )

            return notification_data

        except Exception as e:
            logger.error(f"Error enhancing notification with quality info: {e}")
            return notification_data
