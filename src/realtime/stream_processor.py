"""
Stream Processor - Main Orchestration Service

This module provides the main orchestration service that coordinates
market data ingestion, calculation processing, and system monitoring
for the real-time trading analysis system.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json

from .config_manager import ConfigManager
from .data_ingestion import MarketDataIngestionService, create_ingestion_service
from .calculation_engine import SignalDetectionEngine, create_signal_detection_engine

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Main orchestration service for real-time market data processing.

    This service coordinates all components of the real-time system:
    - Configuration management
    - Market data ingestion
    - Indicator calculations
    - Signal generation
    - System monitoring
    """

    def __init__(self):
        """Initialize the stream processor."""
        self.config_manager = ConfigManager()
        self.ingestion_service = None
        self.calculation_engine = None

        self._running = False
        self._shutdown_event = asyncio.Event()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info("Stream processor initialized")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            # Handle SIGINT (Ctrl+C) and SIGTERM
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers setup completed")
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())

    async def start(self):
        """Start the stream processor and all components."""
        if self._running:
            logger.warning("Stream processor is already running")
            return

        try:
            logger.info("Starting stream processor...")
            self._running = True

            # Validate configuration
            validation_result = self.config_manager.validate_configuration()
            if not validation_result["is_valid"]:
                logger.error(
                    f"Configuration validation failed: {validation_result['errors']}"
                )
                raise Exception("Invalid configuration")

            if validation_result["warnings"]:
                logger.warning(
                    f"Configuration warnings: {validation_result['warnings']}"
                )

            # Initialize services
            logger.info("Initializing services...")
            self.ingestion_service = create_ingestion_service(self.config_manager)
            self.calculation_engine = create_signal_detection_engine(
                self.config_manager, self.ingestion_service
            )

            # Log configuration summary
            config_summary = self.config_manager.get_configuration_summary()
            logger.info(f"Configuration: {json.dumps(config_summary, indent=2)}")

            # Start services concurrently
            logger.info("Starting services...")
            tasks = [
                asyncio.create_task(self.ingestion_service.start()),
                asyncio.create_task(self.calculation_engine.start()),
                asyncio.create_task(self._monitor_system()),
            ]

            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error starting stream processor: {e}")
            raise
        finally:
            self._running = False
            logger.info("Stream processor stopped")

    async def shutdown(self):
        """Shutdown the stream processor gracefully."""
        if not self._running:
            return

        logger.info("Shutting down stream processor...")

        try:
            # Stop services
            if self.calculation_engine:
                await self.calculation_engine.stop()

            if self.ingestion_service:
                await self.ingestion_service.stop()

            # Signal shutdown
            self._shutdown_event.set()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            # Cleanup resources
            if self.calculation_engine:
                self.calculation_engine.cleanup()

            if self.ingestion_service:
                self.ingestion_service.cleanup()

            logger.info("Stream processor shutdown completed")

    async def _monitor_system(self):
        """Monitor system health and performance."""
        logger.info("Starting system monitoring...")

        while self._running:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes

                # Collect stats from all components
                stats = await self.get_system_stats()

                # Log system health
                self._log_system_health(stats)

                # Check for issues and send alerts if needed
                await self._check_system_health(stats)

            except asyncio.CancelledError:
                logger.info("System monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")

    def _log_system_health(self, stats: Dict[str, Any]):
        """Log system health information."""
        try:
            # Calculate uptime
            if self.ingestion_service and hasattr(self.ingestion_service, "_stats"):
                ingestion_stats = stats.get("ingestion", {})
                calculation_stats = stats.get("calculation", {})

                logger.info(
                    f"System Health - "
                    f"Ingestion Events: {ingestion_stats.get('events_published', 0)}, "
                    f"Calculations: {calculation_stats.get('calculations_performed', 0)}, "
                    f"Signals: {calculation_stats.get('signals_generated', 0)}, "
                    f"Errors: {ingestion_stats.get('errors', 0) + calculation_stats.get('errors', 0)}, "
                    f"Active Symbols: {len(calculation_stats.get('symbols_processed', []))}"
                )
        except Exception as e:
            logger.error(f"Error logging system health: {e}")

    async def _check_system_health(self, stats: Dict[str, Any]):
        """Check system health and send alerts for issues."""
        try:
            alerts = []

            # Check ingestion service health
            ingestion_stats = stats.get("ingestion", {})
            if ingestion_stats.get("errors", 0) > 10:
                alerts.append(
                    {
                        "type": "error",
                        "message": f"High error rate in ingestion service: {ingestion_stats['errors']} errors",
                        "component": "ingestion",
                    }
                )

            # Check calculation engine health
            calculation_stats = stats.get("calculation", {})
            if calculation_stats.get("errors", 0) > 5:
                alerts.append(
                    {
                        "type": "error",
                        "message": f"High error rate in calculation engine: {calculation_stats['errors']} errors",
                        "component": "calculation",
                    }
                )

            # Check if no data is being processed
            last_calculation = calculation_stats.get("last_calculation_time")
            if last_calculation is None:
                alerts.append(
                    {
                        "type": "warning",
                        "message": "No calculations have been performed yet",
                        "component": "calculation",
                    }
                )

            # Send alerts
            for alert in alerts:
                if self.ingestion_service:
                    await self.ingestion_service.publish_alert(
                        alert["type"],
                        "SYSTEM",
                        alert["message"],
                        {"component": alert["component"]},
                    )

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system statistics
        """
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "running": self._running,
                "configuration": self.config_manager.get_configuration_summary(),
            }

            if self.ingestion_service:
                stats["ingestion"] = self.ingestion_service.get_stats()

            if self.calculation_engine:
                stats["calculation"] = self.calculation_engine.get_stats()

            return stats

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

    def get_latest_results(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest calculation results for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest results or None
        """
        if self.calculation_engine:
            return self.calculation_engine.get_latest_results(symbol)
        return None

    def add_symbol(self, symbol: str, **kwargs) -> bool:
        """
        Add a symbol to the watchlist.

        Args:
            symbol: Trading symbol to add
            **kwargs: Additional configuration parameters

        Returns:
            True if added successfully
        """
        return self.config_manager.add_symbol(symbol, **kwargs)

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the watchlist.

        Args:
            symbol: Trading symbol to remove

        Returns:
            True if removed successfully
        """
        return self.config_manager.remove_symbol(symbol)

    def update_configuration(
        self, config_type: str, config_data: Dict[str, Any]
    ) -> bool:
        """
        Update system configuration.

        Args:
            config_type: Type of configuration (watchlist, indicators, strategies, system)
            config_data: Configuration data

        Returns:
            True if updated successfully
        """
        try:
            if config_type == "system":
                from .config_manager import SystemConfig

                system_config = SystemConfig(**config_data)
                return self.config_manager.update_system_config(system_config)

            elif config_type == "indicators":
                from .config_manager import IndicatorConfig

                indicators = [IndicatorConfig(**indicator) for indicator in config_data]
                return self.config_manager.update_indicator_configs(indicators)

            elif config_type == "strategies":
                from .config_manager import StrategyConfig

                strategies = [StrategyConfig(**strategy) for strategy in config_data]
                return self.config_manager.update_strategy_configs(strategies)

            else:
                logger.error(f"Unknown configuration type: {config_type}")
                return False

        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False


async def main():
    """Main entry point for the stream processor."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("realtime_processor.log"),
        ],
    )

    # Create and start stream processor
    processor = StreamProcessor()

    try:
        await processor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await processor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
