# Comprehensive Mathematical Validation Tests

This directory contains a comprehensive suite of mathematical validation tests for all technical indicators and trading strategies in the Wagehood trading system. These tests ensure mathematical precision, correctness, and reliability of all calculations.

## Overview

The test suite provides mathematical validation for:

### Technical Indicators
- **Moving Averages**: SMA, EMA, WMA, VWMA, MA Crossovers, MA Envelopes
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillator, Williams %R, CCI, Momentum, ROC
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels, Donchian Channels, Historical Volatility

### Trading Strategies
- **Moving Average Crossover Strategy**: Golden Cross/Death Cross detection and validation
- **MACD+RSI Combined Strategy**: Signal combinations, divergence detection, confidence calculations
- **Additional strategies**: Framework for testing all 5 core strategies

### Validation Categories
1. **Mathematical Precision**: Tests with known input/output vectors
2. **Edge Cases**: Boundary conditions, error handling, extreme scenarios
3. **Incremental Validation**: Real-time vs batch calculation consistency
4. **Performance Testing**: Scalability and efficiency validation

## Directory Structure

```
tests/comprehensive/mathematical/
├── README.md                           # This file
├── __init__.py                         # Package initialization
├── test_runner.py                      # Comprehensive test runner
├── test_edge_cases.py                  # Edge case and boundary testing
├── test_incremental_validation.py     # Incremental vs batch validation
├── fixtures/
│   ├── __init__.py
│   └── test_data.py                    # Test data generators and validators
├── indicators/
│   ├── __init__.py
│   ├── test_moving_averages.py         # Moving average tests
│   ├── test_momentum.py                # Momentum indicator tests
│   └── test_volatility.py              # Volatility indicator tests
└── strategies/
    ├── __init__.py
    ├── test_moving_average_crossover.py # MA crossover strategy tests
    └── test_macd_rsi.py                 # MACD+RSI strategy tests
```

## Running Tests

### Complete Test Suite
```bash
# Run all mathematical validation tests
python tests/comprehensive/mathematical/test_runner.py

# Run with verbose output
python tests/comprehensive/mathematical/test_runner.py --verbose
```

### Specific Categories
```bash
# Test only technical indicators
python tests/comprehensive/mathematical/test_runner.py --category indicators

# Test only trading strategies
python tests/comprehensive/mathematical/test_runner.py --category strategies

# Test only edge cases
python tests/comprehensive/mathematical/test_runner.py --category edge_cases

# Test only incremental validation
python tests/comprehensive/mathematical/test_runner.py --category incremental
```

### Quick Validation
```bash
# Run quick validation subset (core functions only)
python tests/comprehensive/mathematical/test_runner.py --quick
```

### Individual Test Files
```bash
# Run specific test files
pytest tests/comprehensive/mathematical/indicators/test_moving_averages.py -v
pytest tests/comprehensive/mathematical/strategies/test_moving_average_crossover.py -v
pytest tests/comprehensive/mathematical/test_edge_cases.py -v
```

## Test Categories

### 1. Technical Indicator Tests

#### Moving Averages (`indicators/test_moving_averages.py`)
- **SMA Tests**: Basic calculation, linearity properties, edge cases
- **EMA Tests**: Mathematical properties, convergence, responsiveness
- **WMA Tests**: Weight distribution, performance characteristics
- **VWMA Tests**: Volume weighting, zero volume handling
- **Crossover Tests**: Signal generation, mathematical accuracy
- **Integration Tests**: Consistency between different MA types

#### Momentum Indicators (`indicators/test_momentum.py`)
- **RSI Tests**: Known test vectors, mathematical properties, extreme values
- **MACD Tests**: Component relationships, crossover detection, parameter validation
- **Stochastic Tests**: Mathematical bounds, smoothing effects, zero range handling
- **Williams %R Tests**: Range validation, mathematical precision
- **CCI Tests**: Typical price calculation, mean deviation handling
- **Momentum/ROC Tests**: Mathematical relationships, boundary conditions

#### Volatility Indicators (`indicators/test_volatility.py`)
- **Bollinger Bands Tests**: Standard deviation scaling, squeeze/expansion detection
- **ATR Tests**: True Range components, Wilder's smoothing, volatility response
- **Keltner Channels Tests**: ATR relationships, multiplier scaling
- **Donchian Channels Tests**: High/low detection, breakout identification
- **Volatility Tests**: Annualization, numerical stability

### 2. Trading Strategy Tests

#### Moving Average Crossover (`strategies/test_moving_average_crossover.py`)
- **Golden Cross Detection**: Mathematical accuracy, timing validation
- **Death Cross Detection**: Precision testing, confidence calculation
- **Volume Confirmation**: Impact analysis, threshold validation
- **Edge Cases**: Insufficient data, extreme volatility, constant prices
- **Signal Validation**: Confidence thresholds, metadata requirements

#### MACD+RSI Strategy (`strategies/test_macd_rsi.py`)
- **Signal Combinations**: Primary and secondary signal logic
- **Divergence Detection**: Bullish/bearish pattern recognition
- **Confidence Calculation**: Multi-factor analysis, weight distribution
- **Mathematical Precision**: Crossover timing, RSI boundary conditions
- **Volume Integration**: Impact on signal generation

### 3. Edge Case Testing (`test_edge_cases.py`)

#### Data Quality Issues
- **Zero Division Protection**: Safe handling of zero values
- **Negative Prices**: Graceful degradation or appropriate errors
- **Infinite Values**: Input validation and error handling
- **NaN Values**: Proper propagation or filtering
- **Empty Arrays**: Appropriate error responses

#### Extreme Scenarios
- **Constant Data**: Mathematical correctness for flat markets
- **Extreme Volatility**: Stability under market stress
- **Minimal Data**: Boundary condition handling
- **Large Datasets**: Memory efficiency and performance
- **Numerical Precision**: Floating-point accuracy

#### Strategy Edge Cases
- **Insufficient Data**: Graceful degradation
- **Missing Indicators**: Error handling and recovery
- **Invalid Parameters**: Input validation
- **Concurrent Signals**: Timing conflict resolution

### 4. Incremental Validation (`test_incremental_validation.py`)

#### Real-Time Consistency
- **Streaming Calculations**: Point-by-point vs batch consistency
- **State Management**: Stateless calculation verification
- **Partial Updates**: Incremental data handling
- **Memory Efficiency**: Long-running calculation stability

#### Mathematical Accuracy
- **Floating-Point Precision**: Accumulation error detection
- **Numerical Stability**: Large number handling
- **Calculation Order**: Independence from processing sequence
- **Performance Scaling**: Efficiency with data size

## Test Data and Fixtures

### Test Data Generators (`fixtures/test_data.py`)
- **Linear Trends**: Predictable patterns for validation
- **Sinusoidal Data**: Oscillating patterns for indicator testing
- **Volatile Data**: Realistic market simulation
- **OHLCV Generation**: Complete market data creation
- **Edge Case Data**: Boundary condition datasets

### Precision Test Vectors
- **Known Input/Output**: Mathematically verified results
- **Reference Implementations**: Cross-validation data
- **Boundary Values**: Edge case verification
- **Error Scenarios**: Invalid input handling

### Validation Helpers
- **Array Comparison**: NaN-aware precision testing
- **Error Calculation**: Percentage error analysis
- **Timestamp Generation**: Realistic time series
- **Market Data Creation**: Complete data structure building

## Mathematical Validation Standards

### Precision Requirements
- **Decimal Precision**: 6+ decimal places for most indicators
- **Floating Point**: 1e-10 tolerance for exact calculations
- **Percentage Error**: < 0.01% for critical calculations
- **Range Validation**: Proper bounds checking (RSI: 0-100, etc.)

### Performance Standards
- **Processing Time**: < 1 second for 5000 data points
- **Memory Usage**: Linear scaling with data size
- **Concurrent Access**: Thread-safe calculations
- **Real-Time Capability**: Streaming calculation support

### Correctness Criteria
- **Mathematical Properties**: Linearity, convergence, monotonicity
- **Statistical Validity**: Proper variance, mean calculations
- **Signal Accuracy**: Correct timing and confidence
- **Error Handling**: Graceful failure modes

## Integration with CI/CD

### Automated Testing
```bash
# Add to CI/CD pipeline
python tests/comprehensive/mathematical/test_runner.py --quick
```

### Quality Gates
- All mathematical tests must pass before deployment
- Performance benchmarks must be met
- Code coverage requirements for new indicators/strategies

### Regression Testing
- Automatic execution on code changes
- Performance regression detection
- Mathematical accuracy verification

## Contributing

### Adding New Indicators
1. Create test file in `indicators/` directory
2. Include mathematical validation tests
3. Add edge case coverage
4. Implement incremental validation
5. Update test runner configuration

### Adding New Strategies
1. Create test file in `strategies/` directory
2. Test signal generation logic
3. Validate confidence calculations
4. Include parameter optimization tests
5. Add performance benchmarks

### Test Development Guidelines
- Use known mathematical results for validation
- Include comprehensive edge case coverage
- Test both batch and incremental calculations
- Validate all parameters and error conditions
- Document expected behaviors and tolerances

## Troubleshooting

### Common Issues
- **Floating Point Precision**: Use appropriate tolerances
- **NaN Propagation**: Check for proper NaN handling
- **Performance Issues**: Profile with large datasets
- **Memory Leaks**: Validate with repeated calculations

### Debugging Tips
- Run individual test files for isolation
- Use verbose mode for detailed output
- Check test data generation for edge cases
- Validate mathematical formulas against references

## References

### Mathematical Sources
- Technical Analysis formulas and standards
- Statistical calculation references
- Numerical analysis best practices
- Financial mathematics literature

### Implementation Standards
- IEEE 754 floating-point arithmetic
- Numerical stability guidelines
- Real-time system requirements
- Performance optimization techniques

---

This comprehensive test suite ensures the mathematical integrity and reliability of the Wagehood trading system. All calculations are validated against known standards and tested under extreme conditions to guarantee accuracy in live trading environments.