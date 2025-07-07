# Comprehensive Test Framework - Wagehood Trading System

This directory contains the comprehensive test framework for the Wagehood trading system, designed to validate all aspects of the platform including mathematical accuracy, real-time data integration, worker processes, and end-to-end system functionality.

## Quick Start

### Run All Tests
```bash
# Run the complete test suite
python tests/comprehensive/run_comprehensive_tests.py

# Run with performance monitoring
python tests/comprehensive/run_comprehensive_tests.py --performance

# Run in quick mode (faster subset)
python tests/comprehensive/run_comprehensive_tests.py --quick
```

### Run Specific Test Suites
```bash
# Mathematical validation only
python tests/comprehensive/run_comprehensive_tests.py --suite mathematical

# Integration tests
python tests/comprehensive/run_comprehensive_tests.py --suite integration --environment integration

# End-to-end tests
python tests/comprehensive/run_comprehensive_tests.py --suite e2e --environment e2e

# Performance tests
python tests/comprehensive/run_comprehensive_tests.py --suite performance --environment performance
```

### CI/CD Integration
```bash
# Optimized for CI/CD pipelines
python tests/comprehensive/run_comprehensive_tests.py --ci --quick --timeout 900
```

## Test Framework Architecture

### Directory Structure
```
tests/comprehensive/
├── mathematical/           # Strategy and indicator math validation
├── integration/           # API and data source integration tests
├── workers/              # Worker process validation tests
├── e2e/                  # End-to-end system tests
├── performance/          # Performance and stress tests
├── data/                # Test data management
├── logs/                # Centralized test logging
├── reports/             # Generated test reports
├── fixtures/            # Shared test fixtures
└── utils/               # Test framework utilities
```

### Test Categories

#### 1. Mathematical Validation (`mathematical/`)
- **Purpose**: Ensure mathematical correctness of trading strategies and indicators
- **Coverage**: RSI, MACD, Bollinger Bands, Moving Averages, Strategy Logic
- **Validation**: Cross-validation with TA-Lib, boundary condition testing
- **Execution**: Fully parallel, fast execution

#### 2. Integration Tests (`integration/`)
- **Purpose**: Validate integration with external APIs and services
- **Coverage**: Alpaca API, Redis cache, data providers, external services
- **Validation**: Connection handling, error recovery, rate limiting
- **Execution**: Limited parallel (API rate limits)

#### 3. Worker Process Tests (`workers/`)
- **Purpose**: Validate background worker processes
- **Coverage**: Calculation engine, data ingestion, signal processing
- **Validation**: Process isolation, inter-process communication, resource usage
- **Execution**: Parallel with resource monitoring

#### 4. End-to-End Tests (`e2e/`)
- **Purpose**: Validate complete trading workflows
- **Coverage**: Data ingestion → Signal generation → Trade execution → Portfolio update
- **Validation**: System integration, real-world scenarios, performance under load
- **Execution**: Sequential (shared state dependencies)

#### 5. Performance Tests (`performance/`)
- **Purpose**: Validate system performance and identify bottlenecks
- **Coverage**: Benchmarking, stress testing, memory leak detection
- **Validation**: Performance thresholds, resource utilization, scalability
- **Execution**: Isolated execution with monitoring

## Test Configuration

### Environment Types
- **`unit`**: Isolated tests with no external dependencies
- **`integration`**: Tests with mocked external services
- **`e2e`**: Tests with live external services (sandbox)
- **`performance`**: Performance testing with monitoring enabled

### Test Markers
Use pytest markers to run specific test types:
```bash
# Run only mathematical tests
pytest -m mathematical

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m "performance and not stress"

# Skip slow tests
pytest -m "not slow"
```

## Test Data Management

### Generated Test Data
- Programmatically created scenarios for consistent testing
- Configurable market conditions (trending, sideways, volatile)
- Realistic price movements with proper OHLC relationships

### Historical Data
- Real market data for backtesting validation
- Multiple timeframes and symbols
- Data quality validation

### Mock Data
- Controlled test scenarios for edge cases
- Predictable outcomes for validation
- Extreme market conditions

## Reporting and Analysis

### Generated Reports
- **HTML Report**: Comprehensive visual report with charts and metrics
- **JSON Report**: Programmatic access to test results
- **Summary Report**: Concise text summary for CI/CD
- **Performance Report**: Detailed performance analysis with visualizations

### Performance Monitoring
- Real-time system resource monitoring during test execution
- Memory usage tracking and leak detection
- CPU utilization and optimization opportunities
- Network and disk I/O analysis

### Validation Results
- Mathematical accuracy validation against external libraries
- Data integrity checks and quality metrics
- Trading signal validation and backtesting results
- Portfolio calculation verification

## Advanced Usage

### Custom Test Configurations
```python
# Custom test runner configuration
from tests.comprehensive.utils.test_runner import TestRunner, TestSuite

runner = TestRunner()
custom_suite = TestSuite(
    name="custom_mathematical",
    path="tests/comprehensive/mathematical",
    dependencies=[],
    parallel=True,
    timeout=300,
    environment="unit",
    markers=["mathematical", "custom"]
)

results = runner.run_test_suite(custom_suite)
```

### Performance Profiling
```python
# Custom performance profiling
from tests.comprehensive.utils.performance_monitor import PerformanceMonitor, PerformanceProfile

monitor = PerformanceMonitor()
monitor.start()

with PerformanceProfile(monitor, "Custom Test"):
    # Your test code here
    pass

monitor.stop()
performance_report = monitor.generate_report()
```

### Data Validation
```python
# Custom data validation
from tests.comprehensive.utils.data_validator import DataValidator

validator = DataValidator()
suite = validator.create_suite("Custom Validation")

result = validator.validate_numerical_equality(
    expected=expected_values,
    actual=actual_values,
    name="Custom Calculation",
    suite_name=suite.name
)
```

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Comprehensive Tests
  run: |
    python tests/comprehensive/run_comprehensive_tests.py \
      --ci \
      --quick \
      --timeout 900 \
      --report-dir ${{ github.workspace }}/test-reports

- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: test-reports/
```

### Test Quality Gates
- **Code Coverage**: Minimum 95% coverage for mathematical tests
- **Performance**: No regression >10% in performance benchmarks
- **Mathematical Accuracy**: 100% validation pass rate required
- **Integration**: All external API integrations must pass

## Troubleshooting

### Common Issues

#### Redis Connection Errors
```bash
# Start Redis for integration tests
redis-server --port 6379 --daemonize yes

# Or use Docker
docker run -d -p 6379:6379 redis:alpine
```

#### Alpaca API Rate Limits
- Tests automatically handle rate limiting with exponential backoff
- Use sandbox environment for integration tests
- Configure API keys in environment variables

#### Memory Issues
- Large datasets may require increased memory limits
- Use `--quick` mode for faster execution with smaller datasets
- Monitor memory usage with `--performance` flag

#### Test Timeouts
- Increase timeout with `--timeout SECONDS`
- Use `--parallel` for faster execution where possible
- Check system resources and dependencies

### Debug Mode
```bash
# Run with detailed logging
python tests/comprehensive/run_comprehensive_tests.py --log-level DEBUG

# Run specific failing test
pytest tests/comprehensive/mathematical/test_indicators.py::TestRSICalculator::test_rsi_calculation_accuracy -v -s
```

## Contributing

### Adding New Tests
1. Choose appropriate test category (`mathematical`, `integration`, `workers`, `e2e`, `performance`)
2. Follow existing test structure and naming conventions
3. Use framework utilities for validation and reporting
4. Add appropriate pytest markers
5. Update documentation

### Test Development Guidelines
- Tests should be deterministic and repeatable
- Use appropriate tolerance levels for numerical comparisons
- Include both positive and negative test cases
- Validate edge cases and boundary conditions
- Document expected behavior and assumptions

### Performance Considerations
- Mathematical tests should complete in <5 seconds
- Integration tests should handle API rate limits gracefully
- Performance tests should establish realistic benchmarks
- Memory usage should be monitored and optimized

## Support

For questions or issues with the comprehensive test framework:
1. Check the troubleshooting section above
2. Review test logs in `tests/comprehensive/logs/`
3. Examine generated reports in `tests/comprehensive/reports/`
4. Consult the detailed architecture document in `.local/comprehensive-test-framework-architecture.md`