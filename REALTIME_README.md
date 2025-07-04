# Real-Time Market Data Processing System

A high-performance, Redis Streams-based real-time market data processing system for technical analysis and trading strategy evaluation.

## 🚀 Quick Start

### Prerequisites

1. **Redis Server** (required for streaming and caching)
   ```bash
   # Install Redis
   brew install redis         # macOS
   sudo apt install redis     # Ubuntu/Debian
   
   # Start Redis
   redis-server
   ```

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```bash
# Start with default configuration (SPY, QQQ, IWM)
python run_realtime.py

# Start with custom symbols
python run_realtime.py --symbols "AAPL,TSLA,NVDA,MSFT"

# Start with debug logging
python run_realtime.py --log-level DEBUG

# Validate environment before starting
python run_realtime.py --validate-only
```

## 🏗️ Architecture Overview

```
Market Data Feeds → Redis Streams → Calculation Engine → Redis Cache → APIs
                                        ↓
                                   Trading Signals → Alert System
```

### Core Components

1. **ConfigManager**: Manages watchlists, indicator settings, and system configuration
2. **MarketDataIngestionService**: Fetches and publishes market data to Redis Streams  
3. **CalculationEngine**: Processes streams and performs incremental calculations
4. **IncrementalIndicatorCalculator**: High-performance incremental algorithms
5. **StreamProcessor**: Main orchestration service

### Key Features

- **Sub-second Updates**: Real-time data processing every 1 second
- **Incremental Calculations**: O(1) updates for most indicators
- **Redis Streams**: Event-driven architecture with guaranteed delivery
- **Circuit Breakers**: Fault tolerance for external data feeds
- **Horizontal Scaling**: Add workers for more symbols
- **Comprehensive Monitoring**: Performance metrics and alerting

## 📊 Supported Indicators & Strategies

### Technical Indicators

- **Simple Moving Average (SMA)** - Configurable periods
- **Exponential Moving Average (EMA)** - Configurable periods  
- **Relative Strength Index (RSI)** - Wilder's smoothing method
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Standard deviation bands

### Trading Strategies

- **MACD + RSI Strategy** - Combined momentum signals
- **Moving Average Crossover** - Golden/Death cross detection
- **RSI Trend Following** - Trend-aware RSI signals
- **Bollinger Band Breakouts** - Volatility-based signals
- **Support/Resistance Breakouts** - Level-based trading

## ⚙️ Configuration

### Environment Variables

```bash
# Market Data Configuration
WATCHLIST_SYMBOLS="SPY,QQQ,IWM,AAPL,TSLA"    # Symbols to monitor
DATA_UPDATE_INTERVAL=1                        # Update frequency (seconds)
DATA_PROVIDER="mock"                          # Data provider (mock, yahoo, etc.)

# System Performance  
CALCULATION_WORKERS=4                         # Number of worker processes
MAX_CONCURRENT_CALCULATIONS=100               # Concurrent calculation limit
BATCH_CALCULATION_SIZE=10                     # Batch processing size

# Redis Configuration
REDIS_HOST="localhost"                        # Redis server host
REDIS_PORT=6379                              # Redis server port
REDIS_STREAMS_MAXLEN=10000                   # Stream message retention
REDIS_MAX_MEMORY="2gb"                       # Redis memory limit

# Monitoring
ENABLE_MONITORING=true                       # Enable system monitoring
ENABLE_ALERTS=true                          # Enable alert system
```

### Configuration File (JSON)

```json
{
  "indicators": [
    {
      "name": "sma_50",
      "enabled": true,
      "parameters": {"period": 50},
      "update_frequency_seconds": 1,
      "ttl_seconds": 300
    },
    {
      "name": "rsi_14", 
      "enabled": true,
      "parameters": {"period": 14},
      "update_frequency_seconds": 1,
      "ttl_seconds": 300
    }
  ],
  "strategies": [
    {
      "name": "macd_rsi_strategy",
      "enabled": true,
      "parameters": {
        "rsi_overbought": 70,
        "rsi_oversold": 30
      },
      "required_indicators": ["rsi_14", "macd"]
    }
  ],
  "system": {
    "max_concurrent_calculations": 100,
    "data_update_interval_seconds": 1,
    "calculation_workers": 4
  }
}
```

## 📈 Performance Characteristics

### Target Metrics

- **Data Ingestion**: 1-second updates per asset
- **Calculation Latency**: <100ms per indicator update  
- **API Response Time**: <10ms for cached data
- **System Throughput**: 1000+ assets simultaneously
- **Memory Usage**: Optimized with rolling windows

### Scalability

- **Horizontal**: Add Redis consumer groups for parallel processing
- **Vertical**: Utilize Redis Cluster for data distribution
- **Load Balancing**: Multiple processor instances with shared Redis
- **Failover**: Redis Sentinel for high availability

## 🔧 Advanced Usage

### Adding Custom Indicators

```python
from src.realtime.incremental_indicators import IncrementalIndicatorCalculator

calculator = IncrementalIndicatorCalculator()

# Add custom indicator logic
def calculate_custom_indicator(symbol: str, price: float) -> float:
    # Your custom calculation here
    return result

# Register with calculation engine
# See calculation_engine.py for integration examples
```

### Custom Data Provider

```python
from src.data.providers.base import BaseDataProvider

class CustomProvider(BaseDataProvider):
    def get_latest_data(self, symbol: str) -> OHLCV:
        # Implement your data fetching logic
        return ohlcv_data

# Register provider
ingestion_service.add_provider("custom", CustomProvider())
```

### Redis Streams Integration

```python
import redis

# Connect to Redis Streams
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Read market data stream
messages = r.xread({"market_data_stream": "$"}, block=1000)

# Read calculation events
calc_messages = r.xread({"calculation_events_stream": "$"}, block=1000)
```

## 📊 Monitoring & Observability

### System Health Endpoints

```python
# Get system statistics
stats = processor.get_system_stats()

# Get latest results for a symbol
results = processor.get_latest_results("SPY")

# Check configuration
config = processor.config_manager.get_configuration_summary()
```

### Metrics Available

- **Ingestion Metrics**: Events published, errors, provider performance
- **Calculation Metrics**: Indicators calculated, signals generated, latency
- **System Metrics**: Memory usage, Redis performance, worker status
- **Business Metrics**: Signal accuracy, strategy performance

### Log Analysis

```bash
# Monitor real-time logs
tail -f realtime_processor_*.log

# Filter for errors
grep ERROR realtime_processor_*.log

# Monitor specific symbol
grep "SPY" realtime_processor_*.log
```

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Check configuration
   echo $REDIS_HOST $REDIS_PORT
   ```

2. **No Data Processing**
   ```bash
   # Validate configuration
   python run_realtime.py --validate-only
   
   # Check enabled symbols
   python run_realtime.py --show-config
   ```

3. **High Memory Usage**
   ```bash
   # Monitor Redis memory
   redis-cli info memory
   
   # Adjust stream retention
   export REDIS_STREAMS_MAXLEN=5000
   ```

### Debug Mode

```bash
# Run with debug logging
python run_realtime.py --log-level DEBUG

# Check specific component logs
grep "CalculationEngine" realtime_processor_*.log
grep "MarketDataIngestion" realtime_processor_*.log
```

## 🔐 Security Considerations

- **Redis Security**: Use AUTH and TLS for production deployments
- **Input Validation**: All market data is validated before processing
- **Rate Limiting**: Circuit breakers prevent API abuse
- **Monitoring**: Comprehensive logging for audit trails

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY run_realtime.py .

CMD ["python", "run_realtime.py"]
```

### Kubernetes Integration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realtime-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: realtime-processor
  template:
    metadata:
      labels:
        app: realtime-processor
    spec:
      containers:
      - name: processor
        image: wagehood/realtime-processor:latest
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: WATCHLIST_SYMBOLS
          value: "SPY,QQQ,IWM"
```

### Health Checks

```bash
# System health check
curl -X GET "http://localhost:8000/api/v1/realtime/health"

# Performance metrics
curl -X GET "http://localhost:8000/api/v1/realtime/stats"
```

## 📚 API Integration

The real-time system integrates with the existing FastAPI service:

```python
# Get real-time indicator values
GET /api/v1/realtime/indicators/{symbol}

# Get trading signals
GET /api/v1/realtime/signals/{symbol}

# Update watchlist
POST /api/v1/realtime/watchlist

# System health
GET /api/v1/realtime/health
```

## 🤝 Contributing

1. Follow the existing code patterns and architecture
2. Add comprehensive tests for new features  
3. Update documentation for any API changes
4. Ensure performance meets target metrics
5. Add monitoring for new components

## 📄 License

This project follows the same license as the main Wagehood project.