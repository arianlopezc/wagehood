#!/usr/bin/env python3
"""
Alpaca Real-time Trading System Launcher

This script launches the real-time trading system with Alpaca Markets integration,
replacing the mock data provider with live market data from Alpaca.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging before importing other modules
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logger = logging.getLogger(__name__)

# Import project modules
from src.realtime.config_manager import ConfigManager, AssetConfig
from src.realtime.alpaca_ingestion import AlpacaIngestionService
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.incremental_indicators import IncrementalIndicatorCalculator
from src.data.providers.alpaca_provider import AlpacaProvider
from src.trading.alpaca_client import AlpacaTradingClient


class AlpacaRealtimeSystem:
    """
    Main orchestrator for the Alpaca-powered real-time trading system.
    
    This class coordinates all components of the real-time system:
    - Alpaca data ingestion via WebSocket
    - Incremental indicator calculations
    - Strategy signal generation
    - Portfolio management and trading
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time system.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or {}
        self.running = False
        
        # Components
        self.config_manager: Optional[ConfigManager] = None
        self.alpaca_ingestion: Optional[AlpacaIngestionService] = None
        self.calculation_engine: Optional[CalculationEngine] = None
        self.alpaca_provider: Optional[AlpacaProvider] = None
        self.trading_client: Optional[AlpacaTradingClient] = None
        
        # Configuration
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'db': int(os.getenv('REDIS_DB', 0))
        }
        
        self.alpaca_config = {
            'paper': os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true',
            'feed': os.getenv('ALPACA_DATA_FEED', 'iex'),
            'max_retries': int(os.getenv('ALPACA_MAX_RETRIES', 3)),
            'retry_delay': float(os.getenv('ALPACA_RETRY_DELAY', 1.0))
        }
        
        # Watchlist symbols from environment
        symbols_env = os.getenv('WATCHLIST_SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,SPY')
        self.watchlist_symbols = [s.strip() for s in symbols_env.split(',') if s.strip()]
        
        logger.info(f"Initialized AlpacaRealtimeSystem")
        logger.info(f"  Paper Trading: {self.alpaca_config['paper']}")
        logger.info(f"  Data Feed: {self.alpaca_config['feed']}")
        logger.info(f"  Watchlist: {self.watchlist_symbols}")
    
    async def start(self) -> None:
        """Start all system components."""
        logger.info("🚀 Starting Alpaca Real-time Trading System...")
        
        try:
            # Initialize configuration manager
            await self._initialize_config_manager()
            
            # Initialize Alpaca data provider
            await self._initialize_alpaca_provider()
            
            # Initialize trading client
            await self._initialize_trading_client()
            
            # Initialize data ingestion service
            await self._initialize_data_ingestion()
            
            # Initialize calculation engine
            await self._initialize_calculation_engine()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.running = True
            logger.info("✅ All components started successfully!")
            logger.info("📊 Real-time market data processing is now active")
            
            # Keep the system running
            await self._run_main_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop all system components gracefully."""
        logger.info("🛑 Stopping Alpaca Real-time Trading System...")
        
        self.running = False
        
        try:
            # Stop data ingestion
            if self.alpaca_ingestion:
                await self.alpaca_ingestion.stop()
            
            # Stop calculation engine
            if self.calculation_engine:
                await self.calculation_engine.stop()
            
            # Disconnect trading client
            if self.trading_client:
                await self.trading_client.disconnect()
            
            # Disconnect data provider
            if self.alpaca_provider:
                await self.alpaca_provider.disconnect()
            
            logger.info("✅ System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _initialize_config_manager(self) -> None:
        """Initialize the configuration manager."""
        logger.info("Initializing configuration manager...")
        
        try:
            self.config_manager = ConfigManager()
            
            # Setup default watchlist
            await self._setup_default_watchlist()
            
            logger.info("✅ Configuration manager initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize config manager: {e}")
    
    async def _setup_default_watchlist(self) -> None:
        """Setup default watchlist from environment variables."""
        logger.info("Setting up default watchlist...")
        
        try:
            # Clear existing watchlist
            current_watchlist = self.config_manager.get_watchlist()
            for asset in current_watchlist:
                self.config_manager.remove_asset_from_watchlist(asset.symbol)
            
            # Add symbols from environment
            for symbol in self.watchlist_symbols:
                # Determine asset type
                asset_type = "crypto" if "/" in symbol else "stock"
                
                asset_config = AssetConfig(
                    symbol=symbol,
                    enabled=True,
                    data_provider="alpaca",
                    timeframes=["1min", "5min", "1hour", "1day"],
                    priority=1,
                    asset_type=asset_type,
                    last_updated=datetime.now()
                )
                
                self.config_manager.add_asset_to_watchlist(asset_config)
            
            logger.info(f"✅ Added {len(self.watchlist_symbols)} symbols to watchlist")
            
        except Exception as e:
            logger.error(f"Error setting up watchlist: {e}")
    
    async def _initialize_alpaca_provider(self) -> None:
        """Initialize the Alpaca data provider."""
        logger.info("Initializing Alpaca data provider...")
        
        try:
            self.alpaca_provider = AlpacaProvider(self.alpaca_config)
            
            connected = await self.alpaca_provider.connect()
            if not connected:
                raise ConnectionError("Failed to connect to Alpaca Markets")
            
            logger.info("✅ Alpaca data provider connected")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Alpaca provider: {e}")
    
    async def _initialize_trading_client(self) -> None:
        """Initialize the Alpaca trading client."""
        logger.info("Initializing Alpaca trading client...")
        
        try:
            self.trading_client = AlpacaTradingClient(self.alpaca_config)
            
            connected = await self.trading_client.connect()
            if not connected:
                raise ConnectionError("Failed to connect to Alpaca Trading")
            
            # Get account info
            account = await self.trading_client.get_account()
            logger.info(f"✅ Trading client connected - Account: {account['id']}")
            logger.info(f"   Status: {account['status']}")
            logger.info(f"   Buying Power: ${account['buying_power']:,.2f}")
            logger.info(f"   Paper Trading: {self.alpaca_config['paper']}")
            
        except Exception as e:
            logger.warning(f"⚠️  Trading client initialization failed: {e}")
            logger.info("   Continuing without trading functionality...")
            self.trading_client = None
    
    async def _initialize_data_ingestion(self) -> None:
        """Initialize the Alpaca data ingestion service."""
        logger.info("Initializing Alpaca data ingestion...")
        
        try:
            self.alpaca_ingestion = AlpacaIngestionService(
                config_manager=self.config_manager,
                redis_config=self.redis_config
            )
            
            await self.alpaca_ingestion.start()
            
            logger.info("✅ Data ingestion service started")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize data ingestion: {e}")
    
    async def _initialize_calculation_engine(self) -> None:
        """Initialize the calculation engine."""
        logger.info("Initializing calculation engine...")
        
        try:
            # Create incremental calculator
            indicator_calculator = IncrementalIndicatorCalculator()
            
            # Initialize calculation engine
            self.calculation_engine = CalculationEngine(
                config_manager=self.config_manager,
                indicator_calculator=indicator_calculator,
                redis_config=self.redis_config
            )
            
            await self.calculation_engine.start()
            
            logger.info("✅ Calculation engine started")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize calculation engine: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_main_loop(self) -> None:
        """Main system loop."""
        logger.info("🔄 Entering main system loop...")
        
        # Status reporting interval
        status_interval = 30  # seconds
        last_status_time = datetime.now()
        
        try:
            while self.running:
                # Check system health
                await self._check_system_health()
                
                # Report status periodically
                now = datetime.now()
                if (now - last_status_time).total_seconds() >= status_interval:
                    await self._report_system_status()
                    last_status_time = now
                
                # Sleep for a short time
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
    
    async def _check_system_health(self) -> None:
        """Check the health of all system components."""
        try:
            # Check data ingestion health
            if self.alpaca_ingestion and not self.alpaca_ingestion.is_healthy():
                logger.warning("⚠️  Data ingestion service is unhealthy, attempting recovery...")
                try:
                    await self.alpaca_ingestion.reconnect()
                except Exception as e:
                    logger.error(f"Failed to recover data ingestion: {e}")
            
            # Check calculation engine health
            if self.calculation_engine and not self.calculation_engine.is_running():
                logger.warning("⚠️  Calculation engine stopped, attempting restart...")
                try:
                    await self.calculation_engine.start()
                except Exception as e:
                    logger.error(f"Failed to restart calculation engine: {e}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    async def _report_system_status(self) -> None:
        """Report current system status."""
        try:
            logger.info("📊 System Status Report:")
            
            # Data ingestion metrics
            if self.alpaca_ingestion:
                metrics = self.alpaca_ingestion.get_metrics()
                logger.info(f"   Data Ingestion:")
                logger.info(f"     Events Processed: {metrics['events_processed']}")
                logger.info(f"     Events/Second: {metrics['events_per_second']}")
                logger.info(f"     Subscribed Symbols: {len(metrics['subscribed_symbols'])}")
                logger.info(f"     Connected: {metrics['connected']}")
            
            # Trading client status
            if self.trading_client:
                account = await self.trading_client.get_account()
                positions = await self.trading_client.get_positions()
                logger.info(f"   Trading:")
                logger.info(f"     Portfolio Value: ${account['portfolio_value']:,.2f}")
                logger.info(f"     Buying Power: ${account['buying_power']:,.2f}")
                logger.info(f"     Open Positions: {len(positions)}")
            
            # Memory and performance info
            logger.info(f"   System: Running normally at {datetime.now()}")
            
        except Exception as e:
            logger.error(f"Error reporting status: {e}")


async def main():
    """Main entry point."""
    logger.info("🎯 Alpaca Real-time Trading System")
    logger.info("=" * 60)
    
    # Check environment
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.error("❌ Alpaca API credentials not found!")
        logger.info("Please set the following environment variables:")
        logger.info("  ALPACA_API_KEY=your_api_key")
        logger.info("  ALPACA_SECRET_KEY=your_secret_key")
        logger.info("\nOr copy .env.example to .env and configure your credentials")
        return 1
    
    # Create and start the system
    system = AlpacaRealtimeSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    finally:
        await system.stop()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)