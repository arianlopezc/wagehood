#!/usr/bin/env python3
"""
Wagehood Production Real-time Service Launcher
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from src.realtime.data_ingestion import create_ingestion_service
from src.realtime.config_manager import ConfigManager
from src.realtime.calculation_engine import CalculationEngine
from src.jobs.job_processor import JobProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wagehood_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionService:
    """Production real-time trading service."""
    
    def __init__(self):
        self.config = None
        self.ingestion_service = None
        self.calc_engine = None
        self.job_processor = None
        self.running = False
        
    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("🚀 Initializing Wagehood Production Service")
            
            # Initialize configuration
            self.config = ConfigManager()
            logger.info("✅ Configuration manager initialized")
            
            # Create ingestion service
            self.ingestion_service = create_ingestion_service(self.config)
            logger.info("✅ Data ingestion service created")
            
            # Create calculation engine
            self.calc_engine = CalculationEngine(self.config, self.ingestion_service)
            logger.info("✅ Calculation engine initialized")
            
            # Create job processor
            self.job_processor = JobProcessor()
            logger.info("✅ Job processor initialized")
            
            # Log configuration
            enabled_symbols = self.config.get_enabled_symbols()
            logger.info(f"🎯 Configured symbols: {enabled_symbols}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def start(self):
        """Start the production service."""
        if not await self.initialize():
            return False
            
        try:
            self.running = True
            logger.info("🚀 Starting real-time processing...")
            
            # Start job processor
            job_task = asyncio.create_task(self.job_processor.start())
            logger.info("✅ Job processor started")
            
            # Start the main ingestion service
            await self.ingestion_service.start()
            
        except KeyboardInterrupt:
            logger.info("⏹️  Shutdown requested by user")
            await self.stop()
        except Exception as e:
            logger.error(f"❌ Service error: {e}", exc_info=True)
            await self.stop()
            return False
        
        return True
    
    async def stop(self):
        """Stop the production service cleanly."""
        if self.running:
            logger.info("🛑 Stopping production service...")
            self.running = False
            
            if self.job_processor:
                await self.job_processor.stop()
                logger.info("✅ Job processor stopped")
                
            if self.ingestion_service:
                await self.ingestion_service.stop()
                logger.info("✅ Ingestion service stopped")

async def main():
    """Main entry point for production service."""
    service = ProductionService()
    
    # Set up signal handlers for clean shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if hasattr(asyncio, 'current_task'):
            task = asyncio.current_task()
            if task:
                task.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    logger.info(f"🎯 Wagehood Production Service - {datetime.now()}")
    success = await service.start()
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
