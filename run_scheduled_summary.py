#!/usr/bin/env python3
"""
Scheduled Summary Script for Daily EOD Notifications

This script is designed to be called by cron jobs and will:
1. Run the summary generation with Discord notifications enabled
2. Handle timezone conversion for 5pm ET scheduling
3. Log results for monitoring
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure logging for scheduled runs
log_dir = Path.home() / '.wagehood' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'scheduled_summary.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_summary import main as run_summary


async def main():
    """
    Main entry point for scheduled summary generation.
    
    This function runs the summary with Discord notifications enabled
    and handles all necessary error logging for scheduled execution.
    """
    try:
        logger.info("Starting scheduled summary generation")
        
        # Run summary with scheduled flag (enables Discord notifications)
        result = await run_summary(symbols=None, scheduled=True)
        
        logger.info(f"Summary completed successfully:")
        logger.info(f"  - Symbols processed: {result.symbols_processed}")
        logger.info(f"  - Symbols with signals: {result.symbols_with_signals}")
        logger.info(f"  - Total signals: {result.total_signals}")
        logger.info(f"  - Execution time: {result.execution_duration_seconds:.2f}s")
        
        if result.errors:
            logger.warning(f"Summary completed with {len(result.errors)} errors:")
            for error in result.errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
        
        logger.info("Scheduled summary generation completed successfully")
        
    except Exception as e:
        logger.error(f"Scheduled summary generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())