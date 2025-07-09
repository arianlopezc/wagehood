# Comprehensive Real-Time Data Integration Tests

This directory contains comprehensive integration tests for the Wagehood real-time trading system, focusing on thorough validation of the entire Alpaca API integration pipeline with real market data.

## Test Categories

### 1. Alpaca API Integration Tests (`test_alpaca_api_integration.py`)
- **Authentication and Connection Validation**: Tests API key validation, connection establishment, and provider information
- **Historical Data Retrieval**: Validates data fetching across different symbols and timeframes
- **Data Quality and Consistency**: Ensures OHLCV data integrity, chronological ordering, and reasonable value ranges
- **Timestamp Synchronization**: Validates timezone handling and timestamp accuracy
- **Error Handling**: Tests invalid symbols, date ranges, and connection failures
- **Rate Limiting**: Validates API rate limiting and throttling behavior
- **Concurrent Requests**: Tests simultaneous data requests for multiple symbols
- **Market Hours Handling**: Validates market hours vs after-hours data
- **Data Completeness**: Checks for data gaps and missing periods
- **Performance Benchmarks**: Tests connection times, request latency, and throughput
- **Memory Usage**: Monitors memory efficiency during data operations
- **Cryptocurrency Support**: Validates crypto data handling and 24/7 market coverage

### 2. Real-Time Streaming Tests (`test_realtime_streaming.py`)
- **Streaming Connection Establishment**: Tests WebSocket connection setup and teardown
- **Data Flow Validation**: Ensures real-time data flows correctly through the pipeline
- **Data Quality in Streaming**: Validates OHLC consistency and timestamp accuracy in real-time
- **Error Recovery and Reconnection**: Tests automatic reconnection after connection failures
- **Latency Measurement**: Measures and validates streaming latency under various conditions
- **Throughput Under Load**: Tests high-frequency data streaming capabilities
- **Circuit Breaker Functionality**: Validates circuit breaker patterns for resilience
- **Memory Management**: Ensures no memory leaks during extended streaming
- **Data Persistence**: Tests Redis stream storage and retrieval
- **Multi-Provider Streaming**: Validates concurrent streaming from multiple data sources

### 3. Data Processing Pipeline Tests (`test_data_processing_pipeline.py`)
- **End-to-End Pipeline Flow**: Tests complete data flow from ingestion to signal generation
- **Data Transformation and Validation**: Ensures proper data format conversion and validation
- **Redis Storage Performance**: Tests Redis storage and retrieval efficiency
- **Real-Time Indicator Calculations**: Validates live technical indicator computation
- **Signal Generation Pipeline**: Tests trading signal creation from real-time data
- **Error Handling and Recovery**: Tests pipeline resilience to various failure modes
- **Performance Under Load**: Validates pipeline performance with high data volumes
- **Data Consistency**: Ensures data integrity across all pipeline stages

### 4. Performance Benchmarks (`test_performance_benchmarks.py`)
- **Baseline Latency Benchmarks**: Establishes performance baselines for core operations
- **Throughput Capacity Limits**: Tests maximum sustainable operation rates
- **Memory Usage Efficiency**: Monitors memory consumption and leak detection
- **Sustained Load Stability**: Tests system stability under continuous load
- **Concurrent Symbol Processing**: Validates performance with multiple simultaneous data streams
- **Error Recovery Performance**: Tests performance during and after error conditions
- **Resource Cleanup Efficiency**: Ensures proper resource management and cleanup
- **Performance Regression Detection**: Compares against established performance baselines

### 5. System Resilience Tests (`test_system_resilience.py`)
- **Network Failure Recovery**: Tests recovery from network connectivity issues
- **Data Corruption Handling**: Validates detection and handling of corrupted data
- **Cascading Failure Resilience**: Tests system behavior during cascading failures
- **Circuit Breaker Behavior**: Comprehensive circuit breaker functionality testing
- **Resource Exhaustion Handling**: Tests behavior under memory and connection limits
- **Gradual Degradation Detection**: Validates detection of performance degradation
- **System Recovery After Outage**: Tests complete system recovery capabilities

## Prerequisites

### Required Environment Variables
```bash
# Alpaca Markets API Configuration
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_PAPER_TRADING=true
ALPACA_DATA_FEED=iex

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Required Services
- **Redis Server**: Must be running locally or accessible remotely
- **Alpaca Markets API Access**: Valid API credentials for paper trading
- **Network Connectivity**: Required for real-time data streaming tests

### Python Dependencies
```bash
pip install alpaca-py redis pytest pytest-asyncio psutil
```

## Running the Tests

### Run All Integration Tests
```bash
# Run all comprehensive integration tests
pytest tests/comprehensive/integration/ -v

# Run with detailed output
pytest tests/comprehensive/integration/ -v -s --tb=short
```

### Run Specific Test Categories
```bash
# Alpaca API integration only
pytest tests/comprehensive/integration/test_alpaca_api_integration.py -v

# Real-time streaming tests only
pytest tests/comprehensive/integration/test_realtime_streaming.py -v

# Performance benchmarks only
pytest tests/comprehensive/integration/test_performance_benchmarks.py -v

# System resilience tests only
pytest tests/comprehensive/integration/test_system_resilience.py -v
```

### Run Tests by Markers
```bash
# Run only performance tests
pytest tests/comprehensive/integration/ -m performance -v

# Run only tests requiring Alpaca API
pytest tests/comprehensive/integration/ -m alpaca -v

# Run only network-dependent tests
pytest tests/comprehensive/integration/ -m network -v

# Skip slow tests
pytest tests/comprehensive/integration/ -m "not slow" -v
```

### Run with Coverage
```bash
# Run with coverage reporting
pytest tests/comprehensive/integration/ --cov=src --cov-report=html -v
```

## Test Configuration

### Performance Thresholds
The tests include configurable performance thresholds:

```python
PERFORMANCE_BASELINES = {
    'max_avg_latency_ms': 5.0,
    'min_throughput_ops_per_sec': 1000,
    'max_error_rate': 0.01,
    'max_memory_per_op_kb': 0.5,
    'max_p95_latency_ms': 20.0
}
```

### Test Environment Configuration
```python
TEST_CONFIG = {
    'test_duration_short': 5.0,    # seconds
    'test_duration_medium': 15.0,  # seconds  
    'test_duration_long': 30.0,    # seconds
    'max_memory_usage_mb': 100,    # MB
    'test_symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
    'crypto_symbols': ['BTC/USD', 'ETH/USD', 'LTC/USD']
}
```

## Expected Results

### Normal Test Run Output
```
✓ Alpaca API connection validation passed
✓ Historical data retrieval test passed: 30 symbols processed
✓ Data quality validation passed for 1000 data points
✓ Real-time streaming test passed: 500 messages processed
✓ Performance benchmark passed: 1250 ops/s achieved
✓ System resilience test passed: 95% recovery rate
```

### Performance Metrics
The tests will report detailed performance metrics:
- **Latency**: Average, P95, P99, and maximum response times
- **Throughput**: Operations per second under various loads
- **Memory Usage**: Peak memory consumption and growth patterns
- **Error Rates**: Percentage of failed operations
- **Recovery Times**: Time to recover from various failure scenarios

## Troubleshooting

### Common Issues

1. **Missing Alpaca Credentials**
   ```
   SKIP: Alpaca credentials not available
   ```
   Solution: Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables

2. **Redis Connection Failed**
   ```
   SKIP: Redis not available for integration tests
   ```
   Solution: Ensure Redis server is running on localhost:6379

3. **Network Connectivity Issues**
   ```
   ConnectionError: Failed to connect to Alpaca API
   ```
   Solution: Check internet connectivity and Alpaca API status

4. **Performance Test Failures**
   ```
   AssertionError: Throughput too low: 800 ops/s
   ```
   Solution: Check system resources and reduce concurrent test load

### Test Debugging
```bash
# Run with debug logging
pytest tests/comprehensive/integration/ -v -s --log-cli-level=DEBUG

# Run single test with full output
pytest tests/comprehensive/integration/test_alpaca_api_integration.py::TestAlpacaAPIIntegration::test_alpaca_connection_validation -v -s

# Run with profiling
pytest tests/comprehensive/integration/ --profile
```

## Continuous Integration

### GitHub Actions Configuration
```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:6-alpine
        ports:
          - 6379:6379
    env:
      ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
      ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest tests/comprehensive/integration/ -v
```

## Contributing

When adding new integration tests:

1. **Follow Naming Conventions**: Use descriptive test names starting with `test_`
2. **Add Appropriate Markers**: Mark tests with `@pytest.mark.integration`, `@pytest.mark.slow`, etc.
3. **Include Performance Assertions**: Validate both functionality and performance
4. **Handle External Dependencies**: Gracefully handle missing credentials or services
5. **Document Test Purpose**: Include comprehensive docstrings explaining test objectives
6. **Clean Up Resources**: Ensure proper cleanup of test artifacts

### Example Test Structure
```python
@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.alpaca
async def test_new_integration_feature(self, alpaca_provider):
    """
    Test description explaining what this test validates.
    
    This test ensures that [specific functionality] works correctly
    under [specific conditions] and meets [performance requirements].
    """
    # Setup
    # Test execution
    # Assertions
    # Cleanup
    
    logger.info("✓ Test passed with metrics: ...")
```

This comprehensive test suite ensures the robustness, performance, and reliability of the real-time data integration system under various conditions and failure scenarios.