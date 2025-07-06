# Wagehood Configuration Guide

Guide for configuring the multi-strategy multi-timeframe trading system.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Environment Configuration](#environment-configuration)
3. [Trading Profile Configuration](#trading-profile-configuration)
4. [Strategy Parameters](#strategy-parameters)
5. [Redis Configuration](#redis-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring Configuration](#monitoring-configuration)
8. [Alpaca Integration](#alpaca-integration)

## Quick Setup

### Minimal Configuration

For a basic setup, you only need:

```bash
# 1. Start Redis
redis-server

# 2. Set basic environment variables
export WATCHLIST_SYMBOLS="SPY,QQQ,AAPL"
export DATA_PROVIDER="mock"

# 3. Start the system
python run_realtime.py
```

### Production Setup

```bash
# Environment variables for production
export REDIS_HOST="localhost"
export REDIS_PORT=6379
export WATCHLIST_SYMBOLS="SPY,QQQ,IWM,AAPL,MSFT,GOOGL,TSLA,NVDA"
export DATA_PROVIDER="alpaca"
export ALPACA_PAPER_TRADING="true"
export CALCULATION_WORKERS=4
export LOG_LEVEL="INFO"
```

## Environment Configuration

### Core System Variables

**Redis Configuration:**
```bash
# Redis server settings
export REDIS_HOST="localhost"           # Redis host address
export REDIS_PORT=6379                  # Redis port
export REDIS_DB=0                       # Redis database number
export REDIS_PASSWORD=""                # Redis password (if required)

# Redis performance settings
export REDIS_STREAMS_MAXLEN=10000       # Maximum stream length
export REDIS_MAX_MEMORY="2gb"           # Redis memory limit
export REDIS_CONNECTION_POOL_SIZE=10    # Connection pool size
```

**Data Provider Configuration:**
```bash
# Data source settings
export DATA_PROVIDER="mock"             # Data provider: mock, alpaca
export DATA_UPDATE_INTERVAL=1           # Update frequency (seconds)
export WATCHLIST_SYMBOLS="SPY,QQQ,AAPL" # Comma-separated symbol list
export DEFAULT_TIMEFRAME="1d"           # Default timeframe
```

**System Performance:**
```bash
# Worker configuration
export CALCULATION_WORKERS=4            # Number of worker processes
export MAX_CONCURRENT_CALCULATIONS=100  # Concurrent calculation limit
export BATCH_CALCULATION_SIZE=10        # Batch processing size

# Memory management
export MAX_MEMORY_MB=1024               # Memory limit per worker
export CACHE_TTL_SECONDS=3600           # Cache time-to-live
export MAX_CONCURRENT_BACKTESTS=5       # Backtest concurrency
```

### Logging Configuration

```bash
# Logging settings
export LOG_LEVEL="INFO"                 # DEBUG, INFO, WARNING, ERROR
export LOG_FILE="wagehood.log"          # Log file path
export LOG_MAX_SIZE="100MB"             # Maximum log file size
export LOG_BACKUP_COUNT=5               # Number of backup log files

# Structured logging
export LOG_FORMAT="json"                # json, text
export LOG_INCLUDE_TIMESTAMP=true       # Include timestamps
export LOG_INCLUDE_LEVEL=true           # Include log levels
```

### Monitoring Configuration

```bash
# Monitoring settings
export ENABLE_MONITORING=true           # Enable system monitoring
export ENABLE_ALERTS=true               # Enable alert system
export METRICS_COLLECTION_INTERVAL=60   # Metrics collection frequency
export PERFORMANCE_TRACKING=true        # Track performance metrics

# Alert thresholds
export ALERT_HIGH_CPU_THRESHOLD=80      # CPU usage alert threshold
export ALERT_HIGH_MEMORY_THRESHOLD=85   # Memory usage alert threshold
export ALERT_REDIS_CONNECTION_TIMEOUT=5 # Redis connection timeout
```

## Trading Profile Configuration

### Profile Templates

**Day Trading Setup:**
```bash
# Day trading environment
export DEFAULT_TRADING_PROFILE="day"
export DAY_TRADING_TIMEFRAMES="1m,5m,15m"
export DAY_TRADING_STRATEGIES="rsi_trend,bollinger_breakout"
export DAY_TRADING_MIN_CONFIDENCE=0.7
export DAY_TRADING_MAX_POSITIONS=10
```

**Swing Trading Setup:**
```bash
# Swing trading environment
export DEFAULT_TRADING_PROFILE="swing"
export SWING_TRADING_TIMEFRAMES="30m,1h,4h"
export SWING_TRADING_STRATEGIES="macd_rsi,rsi_trend,bollinger_breakout"
export SWING_TRADING_MIN_CONFIDENCE=0.6
export SWING_TRADING_MAX_POSITIONS=5
```

**Position Trading Setup:**
```bash
# Position trading environment
export DEFAULT_TRADING_PROFILE="position"
export POSITION_TRADING_TIMEFRAMES="1d,1w,1M"
export POSITION_TRADING_STRATEGIES="ma_crossover,sr_breakout"
export POSITION_TRADING_MIN_CONFIDENCE=0.5
export POSITION_TRADING_MAX_POSITIONS=3
```

### Custom Profile Configuration

Create custom trading profiles by defining timeframes and strategies:

```python
# Custom profile in Python configuration
CUSTOM_TRADING_PROFILES = {
    'scalping': {
        'name': 'Scalping',
        'timeframes': ['1m', '5m'],
        'strategies': ['rsi_trend', 'bollinger_breakout'],
        'min_confidence': 0.8,
        'max_positions': 15,
        'risk_level': 'Very High'
    },
    'conservative': {
        'name': 'Conservative Long-term',
        'timeframes': ['1d', '1w'],
        'strategies': ['ma_crossover'],
        'min_confidence': 0.4,
        'max_positions': 2,
        'risk_level': 'Very Low'
    }
}
```

## Strategy Parameters

### Global Strategy Settings

```bash
# Global strategy configuration
export MIN_SIGNAL_CONFIDENCE=0.6        # Minimum confidence for signals
export ENABLE_VOLUME_CONFIRMATION=true  # Require volume confirmation
export VOLUME_CONFIRMATION_THRESHOLD=1.2 # Volume threshold multiplier
export ENABLE_DIVERGENCE_DETECTION=true # Enable divergence analysis
export SIGNAL_COOLDOWN_PERIOD=300       # Cooldown between signals (seconds)
```

### Individual Strategy Parameters

**Moving Average Crossover:**
```bash
# MA Crossover strategy parameters
export MA_CROSSOVER_SHORT_PERIOD=50
export MA_CROSSOVER_LONG_PERIOD=200
export MA_CROSSOVER_MIN_CONFIDENCE=0.6
export MA_CROSSOVER_VOLUME_CONFIRMATION=true
export MA_CROSSOVER_VOLUME_THRESHOLD=1.2
```

**MACD + RSI Combined:**
```bash
# MACD RSI strategy parameters
export MACD_RSI_FAST=12
export MACD_RSI_SLOW=26
export MACD_RSI_SIGNAL=9
export MACD_RSI_PERIOD=14
export MACD_RSI_OVERSOLD=30
export MACD_RSI_OVERBOUGHT=70
export MACD_RSI_MIN_CONFIDENCE=0.6
export MACD_RSI_VOLUME_CONFIRMATION=true
export MACD_RSI_DIVERGENCE_DETECTION=true
```

**RSI Trend Following:**
```bash
# RSI Trend strategy parameters
export RSI_TREND_PERIOD=14
export RSI_TREND_MAIN_PERIOD=21
export RSI_TREND_UPTREND_THRESHOLD=50
export RSI_TREND_DOWNTREND_THRESHOLD=50
export RSI_TREND_PULLBACK_LOW=40
export RSI_TREND_PULLBACK_HIGH=60
export RSI_TREND_MIN_CONFIDENCE=0.6
```

**Bollinger Band Breakout:**
```bash
# Bollinger Bands strategy parameters
export BB_BREAKOUT_PERIOD=20
export BB_BREAKOUT_STD=2.0
export BB_BREAKOUT_CONSOLIDATION_PERIODS=10
export BB_BREAKOUT_VOLUME_CONFIRMATION=true
export BB_BREAKOUT_VOLUME_THRESHOLD=1.5
export BB_BREAKOUT_SQUEEZE_THRESHOLD=0.1
```

**Support/Resistance Breakout:**
```bash
# S/R Breakout strategy parameters
export SR_BREAKOUT_LOOKBACK_PERIODS=50
export SR_BREAKOUT_MIN_TOUCHES=2
export SR_BREAKOUT_TOUCH_TOLERANCE=0.02
export SR_BREAKOUT_VOLUME_CONFIRMATION=true
export SR_BREAKOUT_VOLUME_THRESHOLD=1.5
export SR_BREAKOUT_MIN_CONFIDENCE=0.7
```

## Redis Configuration

### Basic Redis Setup

**Redis Server Configuration (redis.conf):**
```conf
# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Network
bind 127.0.0.1
port 6379
timeout 0
keepalive 300

# Security (optional)
# requirepass your_password_here
```

### Redis Streams Configuration

```bash
# Stream configuration for real-time data
export REDIS_STREAM_MARKET_DATA="wagehood:stream:market_data"
export REDIS_STREAM_SIGNALS="wagehood:stream:signals"
export REDIS_STREAM_PERFORMANCE="wagehood:stream:performance"

# Stream retention
export REDIS_STREAMS_MAXLEN=10000       # Maximum messages per stream
export REDIS_STREAMS_BLOCK_MS=1000      # Blocking read timeout
export REDIS_STREAMS_COUNT=10           # Messages per read
```

### Redis Key Patterns

The system uses organized key patterns for data storage:

```bash
# Watchlist keys
wagehood:watchlist:enhanced              # Enhanced watchlist

# Signal keys
wagehood:signals:{symbol}:{strategy}:{timeframe}

# Performance keys
wagehood:performance:{strategy}:{timeframe}

# Configuration keys
wagehood:config:strategies               # Strategy configuration
wagehood:config:profiles                # Trading profiles
wagehood:config:system                  # System settings

# Portfolio keys
wagehood:portfolio:positions            # Current positions
wagehood:portfolio:stats                # Portfolio statistics
```

## Performance Tuning

### Redis Performance Optimization

```bash
# Redis performance settings
export REDIS_TCP_KEEPALIVE=60           # TCP keepalive
export REDIS_CONNECTION_POOL_SIZE=20    # Connection pool size
export REDIS_CONNECTION_TIMEOUT=5       # Connection timeout
export REDIS_SOCKET_TIMEOUT=5           # Socket timeout

# Memory optimization
export REDIS_HASH_MAX_ZIPLIST_ENTRIES=512
export REDIS_HASH_MAX_ZIPLIST_VALUE=64
export REDIS_LIST_MAX_ZIPLIST_SIZE=-2
export REDIS_SET_MAX_INTSET_ENTRIES=512
```

### System Performance Tuning

```bash
# CPU optimization
export CALCULATION_WORKERS=4            # Match CPU cores
export ENABLE_MULTIPROCESSING=true      # Use multiprocessing
export CPU_AFFINITY_ENABLED=false       # CPU affinity (optional)

# Memory optimization
export MAX_MEMORY_PER_WORKER=256        # Memory limit per worker (MB)
export ENABLE_MEMORY_MONITORING=true    # Monitor memory usage
export MEMORY_WARNING_THRESHOLD=80      # Memory warning threshold (%)

# I/O optimization
export ASYNC_IO_ENABLED=true            # Enable async I/O
export MAX_CONCURRENT_REQUESTS=50       # Concurrent API requests
export REQUEST_TIMEOUT=30               # Request timeout (seconds)
```

### Database Optimization

```bash
# Data processing optimization
export BATCH_SIZE=100                   # Batch processing size
export PREFETCH_ENABLED=true            # Enable data prefetching
export CACHE_WARMING_ENABLED=true       # Warm cache on startup

# Calculation optimization
export VECTORIZED_CALCULATIONS=true     # Use vectorized operations
export PARALLEL_INDICATOR_CALC=true     # Parallel indicator calculations
export INCREMENTAL_UPDATES=true         # Incremental data updates
```

## Monitoring Configuration

### Performance Monitoring

```bash
# Performance metrics
export ENABLE_PERFORMANCE_MONITORING=true
export PERFORMANCE_METRICS_INTERVAL=60  # Collection interval (seconds)
export PERFORMANCE_METRICS_RETENTION=7  # Retention period (days)

# System metrics
export MONITOR_CPU_USAGE=true
export MONITOR_MEMORY_USAGE=true
export MONITOR_DISK_USAGE=true
export MONITOR_NETWORK_USAGE=true
export MONITOR_REDIS_PERFORMANCE=true
```

### Alerting Configuration

```bash
# Alert thresholds
export ALERT_CPU_THRESHOLD=80           # CPU usage alert (%)
export ALERT_MEMORY_THRESHOLD=85        # Memory usage alert (%)
export ALERT_DISK_THRESHOLD=90          # Disk usage alert (%)
export ALERT_REDIS_MEMORY_THRESHOLD=90  # Redis memory alert (%)

# Alert delivery
export ALERT_EMAIL_ENABLED=false        # Email alerts
export ALERT_SLACK_ENABLED=false        # Slack alerts
export ALERT_LOG_ENABLED=true           # Log alerts
export ALERT_CONSOLE_ENABLED=true       # Console alerts
```

### Logging Configuration

```bash
# Application logging
export APP_LOG_LEVEL="INFO"
export APP_LOG_FILE="logs/wagehood.log"
export APP_LOG_MAX_SIZE="100MB"
export APP_LOG_BACKUP_COUNT=5

# Performance logging
export PERF_LOG_ENABLED=true
export PERF_LOG_LEVEL="INFO"
export PERF_LOG_FILE="logs/performance.log"

# Error logging
export ERROR_LOG_ENABLED=true
export ERROR_LOG_LEVEL="ERROR"
export ERROR_LOG_FILE="logs/errors.log"
```

## Alpaca Integration

### API Configuration

```bash
# Alpaca API credentials
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
# export ALPACA_BASE_URL="https://api.alpaca.markets"      # Live trading

# Data feed configuration
export ALPACA_DATA_FEED="iex"           # iex (free) or sip ($99/month)
export ALPACA_PAPER_TRADING=true        # Enable paper trading
```

### Trading Configuration

```bash
# Trading parameters
export ALPACA_ENABLE_TRADING=false      # Enable actual trading
export ALPACA_DEFAULT_ORDER_TYPE="market" # market, limit, stop
export ALPACA_DEFAULT_TIME_IN_FORCE="day" # day, gtc, ioc, fok

# Risk management
export ALPACA_MAX_POSITION_SIZE=1000     # Maximum position size ($)
export ALPACA_MAX_DAILY_TRADES=10        # Daily trade limit
export ALPACA_BUYING_POWER_LIMIT=0.8     # Use 80% of buying power

# Order management
export ALPACA_ORDER_TIMEOUT=300         # Order timeout (seconds)
export ALPACA_ENABLE_STOP_LOSS=true     # Enable stop loss orders
export ALPACA_DEFAULT_STOP_LOSS=0.05    # Default stop loss (5%)
```

### Data Feed Configuration

```bash
# Real-time data
export ALPACA_ENABLE_REALTIME=true      # Enable real-time data
export ALPACA_REALTIME_SYMBOLS="SPY,QQQ,AAPL" # Real-time symbols
export ALPACA_DATA_BUFFER_SIZE=1000     # Data buffer size

# Historical data
export ALPACA_HISTORICAL_LIMIT=1000     # Historical data limit
export ALPACA_RATE_LIMIT_REQUESTS=200   # Rate limit (requests/minute)
export ALPACA_RATE_LIMIT_ENABLED=true   # Enable rate limiting
```

## Configuration File Examples

### development.env
```bash
# Development environment
NODE_ENV=development
LOG_LEVEL=DEBUG

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Data
DATA_PROVIDER=mock
WATCHLIST_SYMBOLS=SPY,QQQ,AAPL

# System
CALCULATION_WORKERS=2
MAX_CONCURRENT_CALCULATIONS=50
```

### production.env
```bash
# Production environment
NODE_ENV=production
LOG_LEVEL=INFO

# Redis
REDIS_HOST=redis.production.com
REDIS_PORT=6379
REDIS_PASSWORD=secure_password

# Data
DATA_PROVIDER=alpaca
WATCHLIST_SYMBOLS=SPY,QQQ,IWM,AAPL,MSFT,GOOGL,TSLA,NVDA

# System
CALCULATION_WORKERS=8
MAX_CONCURRENT_CALCULATIONS=200

# Monitoring
ENABLE_MONITORING=true
ENABLE_ALERTS=true

# Alpaca
ALPACA_PAPER_TRADING=false
ALPACA_DATA_FEED=sip
```

## Configuration Validation

### Validation Script

Create a configuration validation script:

```bash
# validate_config.py
python -c "
import os
import redis
from src.strategies import STRATEGY_REGISTRY

# Validate Redis connection
try:
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'))
    r.ping()
    print('✓ Redis connection successful')
except:
    print('✗ Redis connection failed')

# Validate strategies
strategies = os.getenv('DEFAULT_STRATEGIES', '').split(',')
for strategy in strategies:
    if strategy.strip() in STRATEGY_REGISTRY:
        print(f'✓ Strategy {strategy} found')
    else:
        print(f'✗ Strategy {strategy} not found')

print('Configuration validation complete')
"
```

### Health Check Endpoint

```bash
# Check system health
curl -X GET http://localhost:8080/health

# Expected response:
{
    "status": "healthy",
    "redis": "connected",
    "strategies": "loaded",
    "workers": "running"
}
```

## Troubleshooting Configuration

### Common Issues

**Redis Connection Issues:**
```bash
# Check Redis status
redis-cli ping

# Check Redis configuration
redis-cli config get "*"

# Test Redis performance
redis-cli --latency -h localhost -p 6379
```

**Environment Variable Issues:**
```bash
# List all Wagehood environment variables
env | grep -E "(REDIS|ALPACA|WAGEHOOD|CALCULATION)"

# Validate environment variables
python -c "import os; print({k:v for k,v in os.environ.items() if 'REDIS' in k})"
```

**Performance Issues:**
```bash
# Monitor system resources
top -p $(pgrep -f run_realtime.py)

# Monitor Redis performance
redis-cli --stat

# Check logs for errors
tail -f logs/wagehood.log | grep ERROR
```

---

This configuration guide provides setup instructions for all aspects of the Wagehood multi-strategy multi-timeframe trading system.