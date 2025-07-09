#!/usr/bin/env python3
"""
Docker Health Check Script for Wagehood Trading System

This script validates that all critical components are working correctly:
1. Redis connectivity
2. Alpaca API credentials and connectivity
3. Core system imports and functionality
4. Market data retrieval capability

Exit codes:
- 0: All checks passed
- 1: Critical component failure
- 2: Configuration error
- 3: Network/connectivity issue
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# Suppress logging to keep health check output clean
logging.getLogger().setLevel(logging.CRITICAL)

def log_error(message: str):
    """Log error message to stderr"""
    print(f"HEALTH CHECK ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log info message to stdout"""
    print(f"HEALTH CHECK: {message}")

def check_redis_connectivity():
    """Check Redis server connectivity"""
    try:
        from src.storage.cache import cache_manager
        
        # Test basic cache operations
        test_key = f"health_check_{datetime.now().timestamp()}"
        cache_manager.set('health', test_key, 'ok')
        result = cache_manager.get('health', test_key)
        
        if result != 'ok':
            log_error("Redis cache operations failed")
            return False
        
        # Clean up test key
        cache_manager.delete('health', test_key)
        log_info("Redis connectivity verified")
        return True
        
    except Exception as e:
        log_error(f"Redis connectivity check failed: {e}")
        return False

def check_core_imports():
    """Check that core system components can be imported"""
    try:
        from src.strategies import create_strategy
        from src.data.mock_generator import MockDataGenerator
        from src.backtest.engine import BacktestEngine
        from src.core.models import MarketData, TimeFrame
        
        log_info("Core system imports verified")
        return True
        
    except Exception as e:
        log_error(f"Core system imports failed: {e}")
        return False

def check_alpaca_credentials():
    """Check Alpaca credentials are present and valid"""
    try:
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            log_error("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required")
            return False
        
        # Validate credential format (basic check)
        if len(api_key) < 20 or len(secret_key) < 40:
            log_error("Alpaca credentials appear to be invalid (too short)")
            return False
        
        log_info("Alpaca credentials present and format validated")
        return True
        
    except Exception as e:
        log_error(f"Alpaca credential validation failed: {e}")
        return False

async def check_alpaca_connectivity():
    """Check actual connectivity to Alpaca Markets API"""
    try:
        from src.realtime.data_ingestion import MinimalAlpacaProvider
        
        config = {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'paper': True,
            'feed': 'iex'
        }
        
        provider = MinimalAlpacaProvider(config)
        await provider.connect()
        await provider.disconnect()
        
        log_info("Alpaca Markets connectivity verified")
        return True
        
    except Exception as e:
        log_error(f"Alpaca connectivity check failed: {e}")
        return False

async def check_market_data_retrieval():
    """Test basic market data retrieval capability"""
    try:
        from src.data.providers.alpaca_provider import AlpacaProvider
        from src.core.models import TimeFrame
        
        config = {
            'api_key': os.getenv('ALPACA_API_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'paper': True,
            'feed': 'iex'
        }
        
        provider = AlpacaProvider(config)
        await provider.connect()
        
        # Test basic data retrieval
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        data = await provider.get_historical_data(
            symbol='AAPL',
            timeframe=TimeFrame.DAILY,
            start_date=start_date,
            end_date=end_date,
            limit=1
        )
        
        await provider.disconnect()
        
        if not data:
            log_error("No market data retrieved")
            return False
        
        log_info("Market data retrieval verified")
        return True
        
    except Exception as e:
        log_error(f"Market data retrieval check failed: {e}")
        return False

def check_system_configuration():
    """Check critical system configuration"""
    try:
        # Check required environment variables
        required_vars = [
            'WAGEHOOD_ENV',
            'REDIS_HOST',
            'REDIS_PORT',
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            log_error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        log_info("System configuration verified")
        return True
        
    except Exception as e:
        log_error(f"System configuration check failed: {e}")
        return False

async def main():
    """Main health check function"""
    log_info("Starting comprehensive health check...")
    
    checks = [
        ("System Configuration", check_system_configuration),
        ("Core Imports", check_core_imports),
        ("Redis Connectivity", check_redis_connectivity),
        ("Alpaca Credentials", check_alpaca_credentials),
        ("Alpaca Connectivity", check_alpaca_connectivity),
        ("Market Data Retrieval", check_market_data_retrieval),
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            log_info(f"Running {check_name} check...")
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if result:
                passed += 1
                log_info(f"✅ {check_name} check passed")
            else:
                failed += 1
                log_error(f"❌ {check_name} check failed")
        
        except Exception as e:
            failed += 1
            log_error(f"❌ {check_name} check failed with exception: {e}")
    
    log_info(f"Health check completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        log_info("🎉 All health checks passed - system is ready for production!")
        return 0
    else:
        log_error(f"❌ {failed} health checks failed - system not ready for production")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_error("Health check interrupted")
        sys.exit(1)
    except Exception as e:
        log_error(f"Health check failed with unexpected error: {e}")
        sys.exit(1)