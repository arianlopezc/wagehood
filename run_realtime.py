#!/usr/bin/env python3
"""
Real-time Market Data Processing Startup Script

This script starts the real-time market data processing system including:
- Market data ingestion from configured providers
- Incremental technical indicator calculations
- Trading strategy signal generation
- Redis Streams event processing
- System monitoring and alerting

Usage:
    python run_realtime.py [--config-file CONFIG] [--symbols SYMBOLS] [--log-level LEVEL]

Examples:
    # Start with default configuration
    python run_realtime.py

    # Start with specific symbols
    python run_realtime.py --symbols "SPY,QQQ,IWM,AAPL"

    # Start with custom log level
    python run_realtime.py --log-level DEBUG

    # Start with custom configuration file
    python run_realtime.py --config-file custom_config.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.realtime.stream_processor import StreamProcessor
from src.realtime.config_manager import ConfigManager


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(f"realtime_processor_{timestamp}.log"))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce noise from some third-party libraries
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config_file(config_file: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {config_file}: {e}")
        return {}


def setup_environment(args):
    """Setup environment variables from arguments."""
    if args.symbols:
        os.environ['WATCHLIST_SYMBOLS'] = args.symbols
    
    if args.data_provider:
        os.environ['DATA_PROVIDER'] = args.data_provider
    
    if args.update_interval:
        os.environ['DATA_UPDATE_INTERVAL'] = str(args.update_interval)
    
    if args.workers:
        os.environ['CALCULATION_WORKERS'] = str(args.workers)


def validate_environment():
    """Validate required environment variables and dependencies."""
    errors = []
    
    # Check Redis connection
    try:
        import redis
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        
        r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=5)
        r.ping()
        logging.info(f"Redis connection validated: {redis_host}:{redis_port}")
    except Exception as e:
        errors.append(f"Redis connection failed: {e}")
    
    # Check required packages
    try:
        import numpy
        import pandas
        logging.info(f"NumPy {numpy.__version__}, Pandas {pandas.__version__} available")
    except ImportError as e:
        errors.append(f"Required packages missing: {e}")
    
    if errors:
        logging.error("Environment validation failed:")
        for error in errors:
            logging.error(f"  - {error}")
        return False
    
    return True


async def run_system(args):
    """Run the real-time processing system."""
    logging.info("Starting real-time market data processing system...")
    
    # Load custom configuration if provided
    if args.config_file:
        config_data = load_config_file(args.config_file)
        if config_data:
            logging.info(f"Loaded configuration from {args.config_file}")
    
    # Setup environment
    setup_environment(args)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create and configure system
    processor = StreamProcessor()
    
    # Apply custom configuration if provided
    if args.config_file and 'config_data' in locals():
        # Apply configuration updates
        for config_type, config_values in config_data.items():
            if config_type in ['indicators', 'strategies', 'system']:
                success = processor.update_configuration(config_type, config_values)
                if success:
                    logging.info(f"Applied {config_type} configuration")
                else:
                    logging.warning(f"Failed to apply {config_type} configuration")
    
    # Display startup information
    config_summary = processor.config_manager.get_configuration_summary()
    logging.info("=== System Configuration ===")
    logging.info(f"Enabled Symbols: {config_summary['watchlist']['symbols']}")
    logging.info(f"Enabled Indicators: {config_summary['indicators']['indicator_names']}")
    logging.info(f"Enabled Strategies: {config_summary['strategies']['strategy_names']}")
    logging.info("============================")
    
    try:
        # Start the system
        await processor.start()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)
    finally:
        await processor.shutdown()
        logging.info("System shutdown completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time Market Data Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration options
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to monitor (e.g., "SPY,QQQ,IWM")'
    )
    
    parser.add_argument(
        '--data-provider',
        type=str,
        default='mock',
        choices=['mock', 'yahoo', 'alphavantage'],
        help='Data provider to use (default: mock)'
    )
    
    parser.add_argument(
        '--update-interval',
        type=int,
        default=1,
        help='Data update interval in seconds (default: 1)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of calculation workers (default: 4)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: auto-generated with timestamp)'
    )
    
    # System options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and environment then exit'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Handle special modes
    if args.validate_only:
        logging.info("Validation mode: checking environment and configuration...")
        setup_environment(args)
        if validate_environment():
            logging.info("✓ Environment validation passed")
            sys.exit(0)
        else:
            logging.error("✗ Environment validation failed")
            sys.exit(1)
    
    if args.show_config:
        logging.info("Configuration display mode...")
        setup_environment(args)
        config_manager = ConfigManager()
        config_summary = config_manager.get_configuration_summary()
        print(json.dumps(config_summary, indent=2, default=str))
        sys.exit(0)
    
    # Run the system
    try:
        asyncio.run(run_system(args))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()