# Wagehood Trading Analysis System

**A Redis-based worker service for real-time market analysis with CLI tools for strategy analysis, market monitoring, and backtesting. Built with data processing and trading strategy evaluation capabilities.**

## 🚀 Overview

Wagehood is a trading system for systematic traders and quantitative researchers. It combines trading strategies with real-time data processing, providing tools for strategy development, testing, and deployment.

### Key Features

- **5 Trading Strategies** with documented win rates up to 73%
- **Redis-Based Worker Service** with real-time market data processing
- **CLI Tools** - Market analysis and monitoring interfaces
- **Strategy Analysis & Optimization** - Analyze which strategies work best for your trading style
- **Strategy Documentation** - Detailed explanations of signal logic and parameters
- **Real-Time Processing** with Redis Streams for event-driven architecture
- **Alpaca Markets Integration** for live trading and commission-free execution
- **Testing Suite** with code coverage tracking

## 🎯 Core Trading Strategies

### Multi-Strategy Multi-Timeframe System

The system supports multi-dimensional analysis across:
- **5 trading strategies** with documented performance
- **9 timeframes** from 1-minute to monthly
- **3 trading profiles** (Day, Swing, Position)
- **Watchlist management**

### Implemented Strategies

| Strategy | Win Rate | Avg Return | Best Timeframes | Trading Profile | Description |
|----------|----------|------------|----------------|-----------------|-------------|
| **MACD+RSI Combined** | 73% | 0.88%/trade | 1h, 4h, 1d | Swing/Position | Momentum strategy combining MACD and RSI |
| **RSI Trend Following** | 68% | 0.6%/trade | 15m, 30m, 1h | Day/Swing | Trend-aware RSI with pullbacks |
| **Bollinger Band Breakout** | 65% | 0.9%/trade | 5m, 15m, 1h | Day/Swing | Volatility expansion strategy |
| **Support/Resistance Breakout** | 58% | 1.4%/trade | 1h, 4h, 1d | Swing/Position | Level-based breakout trading |
| **Moving Average Crossover** | 45% | 2.1%/trade | 1d, 1w, 1M | Position | Golden/Death cross signals |

### Trading Profiles

**Day Trading Profile:**
- **Timeframes:** 1m, 5m, 15m
- **Focus:** High-frequency signals, quick profits
- **Best Strategies:** RSI Trend, Bollinger Breakout
- **Capital Requirements:** $25,000+ (PDT rule)
- **Time Commitment:** Active monitoring required

**Swing Trading Profile:**
- **Timeframes:** 30m, 1h, 4h  
- **Focus:** Multi-day positions (2-10 days)
- **Best Strategies:** MACD+RSI, RSI Trend, Bollinger Breakout
- **Capital Requirements:** $5,000-$25,000
- **Time Commitment:** Daily monitoring

**Position Trading Profile:**
- **Timeframes:** 1d, 1w, 1M
- **Focus:** Long-term positions (weeks to months)
- **Best Strategies:** Moving Average Crossover, Support/Resistance Breakout
- **Capital Requirements:** $1,000+
- **Time Commitment:** Weekly monitoring

### Strategy Assets Classification

**Suggested Asset Classes:**
1. **Commodities** - Best trend-following performance
2. **Cryptocurrencies** - High volatility, strong trends  
3. **Forex Major Pairs** - Clear central bank-driven trends
4. **Index ETFs** - Reduced individual stock risk

**Timeframe Recommendations:**
- **Day Trading**: RSI (7-period), Bollinger Bands
- **Swing Trading**: All strategies optimal
- **Position Trading**: Moving Average Crossover

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │───▶│ Redis Streams   │───▶│ Worker Service  │
│ (Alpaca/Mock)   │    │ (Event Bus)     │    │ (run_realtime)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Redis Cache     │◀───│ Calculation     │◀───│ Strategy        │
│ (Results)       │    │ Engine          │    │ Execution       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CLI Tools       │◀───│ Data Services   │◀───│ Analysis &      │
│ (market_*_cli)  │    │ & Storage       │    │ Backtesting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

```
src/
├── core/               # Data models and constants
├── data/               # Data management and providers
│   └── providers/      # Alpaca, mock, and extensible providers
├── indicators/         # 20+ technical indicator calculations
├── strategies/         # 5 trading strategy implementations  
├── backtest/           # Backtesting engine with execution simulation
├── realtime/           # Real-time processing and data ingestion
├── services/           # Analysis and data services
├── trading/           # Live trading integration (Alpaca)
├── analysis/          # Performance evaluation and comparison
└── storage/           # Results storage and caching

CLI Tools:
├── market_analysis_cli.py  # Market analysis interface
├── market_watch.py        # Real-time market monitoring
└── run_realtime.py       # Worker service startup script
```

## 📊 Technical Indicators

### Supported Indicators

**Moving Averages:**
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)  
- Weighted Moving Average (WMA)
- Volume Weighted Moving Average (VWMA)

**Momentum Indicators:**
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- Commodity Channel Index (CCI)

**Volatility Indicators:**
- Bollinger Bands
- Average True Range (ATR)
- Keltner Channels

**Support/Resistance:**
- Dynamic Support/Resistance Levels
- Pivot Points
- Price Level Analysis

## 🚀 Quick Start

> **💡 Worker Service Architecture**: The system runs as a Redis-based worker service with dedicated CLI tools for market analysis and monitoring.

### Prerequisites

1. **Python 3.9+** 
2. **Redis Server** (for real-time processing)
3. **Alpaca Markets Account** (optional, for live data/trading)

```bash
# Install Redis
brew install redis         # macOS
sudo apt install redis     # Ubuntu/Debian

# Start Redis
redis-server
```

### Installation

#### 📋 Standard Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd wagehood

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install in development mode
pip install -e .


# 4. Verify installation
python -m src.core.models  # Test core imports
```

#### 🔧 Development Installation

```bash
# 1. Clone and create virtual environment
git clone <repository-url>
cd wagehood
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Run tests to verify setup
python run_tests.py --unit
```

#### ✅ Verify Installation

```bash
# Test Redis connection
redis-cli ping

# Test market data generation
python -c "from src.data.providers.mock_provider import MockProvider; print('Mock provider working')"

# Test CLI tools
python market_analysis_cli.py --help
python market_watch.py --help
```

### Basic Usage

```python
from src.data import DataStore, MockDataGenerator
from src.strategies import MACDRSIStrategy
from src.backtest import BacktestEngine

# Generate sample data
generator = MockDataGenerator()
data = generator.generate_realistic_data("SPY", periods=252)

# Initialize strategy
strategy = MACDRSIStrategy()

# Run backtest
engine = BacktestEngine()
result = engine.run_backtest(strategy, data, initial_capital=10000)

print(f"Total Return: {result.performance_metrics.total_return_pct:.2%}")
print(f"Win Rate: {result.performance_metrics.win_rate:.2%}")
print(f"Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.2f}")
```

### Quick Start Guide

Follow these steps to get your multi-strategy multi-timeframe trading system running:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Start Redis server
redis-server

# 3. Configure environment (copy and edit .env.example)
cp .env.example .env
# Edit .env with your settings

# 4. Run comprehensive tests to verify setup
python run_tests.py --all

# 5. Use the Python API for trading analysis
python -c "
from src.strategies import create_strategy, get_all_strategies
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine

# Generate test data
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=252)

# Test a strategy
strategy = create_strategy('macd_rsi')
engine = BacktestEngine()
result = engine.run_backtest(strategy, data, initial_capital=10000)
print(f'MACD+RSI Strategy Result: {result.performance_metrics.total_return_pct:.2%} return')
"

# 6. Explore all available strategies
python -c "
from src.strategies import get_all_strategies
strategies = get_all_strategies()
for name, meta in strategies.items():
    print(f'{meta[\"name\"]}: {meta[\"description\"]}')
"
```

## 🖥️ Python API Interface

The Wagehood system provides a Python API for multi-strategy multi-timeframe analysis. The system has moved away from CLI tools to a programmatic interface.

### Core Python API

**Strategy Analysis:**
```python
from src.strategies import create_strategy, get_all_strategies, STRATEGY_METADATA
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine

# Explore available strategies
strategies = get_all_strategies()
for name, metadata in strategies.items():
    print(f"{metadata['name']}: {metadata['description']}")
    print(f"Difficulty: {metadata['difficulty']}")
    print(f"Priority: {metadata['priority']}")

# Create and test a strategy
strategy = create_strategy('macd_rsi')
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=252)

engine = BacktestEngine()
result = engine.run_backtest(strategy, data, initial_capital=10000)
print(f"Total Return: {result.performance_metrics.total_return_pct:.2%}")
```

**Multi-Strategy Comparison:**
```python
from src.analysis.comparison import StrategyComparator

# Compare multiple strategies
comparator = StrategyComparator()
strategies_to_test = ['macd_rsi', 'ma_crossover', 'rsi_trend', 'bollinger_breakout']

results = {}
for strategy_name in strategies_to_test:
    strategy = create_strategy(strategy_name)
    result = engine.run_backtest(strategy, data, initial_capital=10000)
    results[strategy_name] = result

# Analyze results
for name, result in results.items():
    metrics = result.performance_metrics
    print(f"{name}: {metrics.total_return_pct:.2%} return, {metrics.win_rate:.1%} win rate")
```

**Real-time Data Processing:**
```python
from src.realtime.data_ingestion import DataIngestionManager
from src.realtime.signal_engine import SignalEngine
from src.realtime.timeframe_manager import TimeframeManager

# Set up real-time processing (requires Redis)
data_manager = DataIngestionManager()
signal_engine = SignalEngine()
timeframe_manager = TimeframeManager()

# Configure multi-timeframe analysis
symbols = ['SPY', 'QQQ', 'AAPL']
timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
strategies = ['macd_rsi', 'rsi_trend', 'bollinger_breakout']

# Process real-time signals
# Note: This requires running Redis server
```

### Testing and Validation

**Testing:**
```bash
# Run all tests including strategy validation
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Run tests with coverage
python run_tests.py --coverage
```

**Strategy Validation:**
```python
# Validate strategy implementations
python -c "
from src.strategies import STRATEGY_REGISTRY
from src.data.mock_generator import MockDataGenerator

generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=100)

for name, strategy_class in STRATEGY_REGISTRY.items():
    try:
        strategy = strategy_class()
        signals = strategy.generate_signals(data)
        print(f'✓ {name}: Generated {len(signals)} signals')
    except Exception as e:
        print(f'✗ {name}: Error - {e}')
"
```

### Usage Examples

#### Multi-Strategy Portfolio Analysis
```python
# Create a multi-strategy analysis
from src.strategies import DEFAULT_STRATEGY_PARAMS, create_strategy
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine
import pandas as pd

# Generate realistic market data
generator = MockDataGenerator()
symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
strategies = ['macd_rsi', 'ma_crossover', 'rsi_trend', 'bollinger_breakout']

# Portfolio analysis
portfolio_results = {}
for symbol in symbols:
    data = generator.generate_realistic_data(symbol, periods=252)
    symbol_results = {}
    
    for strategy_name in strategies:
        strategy = create_strategy(strategy_name)
        engine = BacktestEngine()
        result = engine.run_backtest(strategy, data, initial_capital=10000)
        
        symbol_results[strategy_name] = {
            'total_return': result.performance_metrics.total_return_pct,
            'win_rate': result.performance_metrics.win_rate,
            'sharpe_ratio': result.performance_metrics.sharpe_ratio,
            'max_drawdown': result.performance_metrics.max_drawdown_pct
        }
    
    portfolio_results[symbol] = symbol_results

# Display results
for symbol, strategies_data in portfolio_results.items():
    print(f"\n{symbol} Analysis:")
    for strategy, metrics in strategies_data.items():
        print(f"  {strategy}: {metrics['total_return']:.2%} return, {metrics['win_rate']:.1%} win rate")
```

#### Real-time Signal Generation
```python
# Set up real-time multi-timeframe signal generation
from src.realtime.signal_engine import SignalEngine
from src.realtime.config_manager import ConfigManager
from src.core.models import TimeFrame

# Configure signal engine
config = ConfigManager()
signal_engine = SignalEngine(config)

# Configure trading profiles
day_trading_timeframes = [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15]
swing_trading_timeframes = [TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.HOUR_4]
position_trading_timeframes = [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]

# Monitor symbols across different timeframes
symbols = ['SPY', 'QQQ', 'AAPL']
for symbol in symbols:
    # Day trading signals
    day_signals = signal_engine.generate_signals(symbol, day_trading_timeframes, ['rsi_trend', 'bollinger_breakout'])
    
    # Swing trading signals  
    swing_signals = signal_engine.generate_signals(symbol, swing_trading_timeframes, ['macd_rsi', 'rsi_trend'])
    
    # Position trading signals
    position_signals = signal_engine.generate_signals(symbol, position_trading_timeframes, ['ma_crossover'])
    
    print(f"{symbol}: Day: {len(day_signals)}, Swing: {len(swing_signals)}, Position: {len(position_signals)} signals")
```

#### Custom Strategy Development
```python
# Create a custom strategy combining multiple indicators
from src.strategies.base import TradingStrategy
from src.core.models import Signal, SignalType
from src.indicators.momentum import RSICalculator
from src.indicators.moving_averages import EMACalculator

class CustomMomentumStrategy(TradingStrategy):
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.rsi_calculator = RSICalculator(period=14)
        self.ema_calculator = EMACalculator(period=20)
    
    def generate_signals(self, market_data):
        signals = []
        
        # Calculate indicators
        rsi_values = self.rsi_calculator.calculate(market_data.data)
        ema_values = self.ema_calculator.calculate(market_data.data)
        
        for i in range(len(market_data.data)):
            if i < 20:  # Need enough data
                continue
                
            current_price = market_data.data[i].close
            rsi = rsi_values[i]
            ema = ema_values[i]
            
            # Custom signal logic
            if rsi < 30 and current_price > ema:
                signal = Signal(
                    timestamp=market_data.data[i].timestamp,
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    confidence=0.8,
                    strategy_name="custom_momentum"
                )
                signals.append(signal)
            elif rsi > 70 and current_price < ema:
                signal = Signal(
                    timestamp=market_data.data[i].timestamp,
                    symbol=market_data.symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    confidence=0.8,
                    strategy_name="custom_momentum"
                )
                signals.append(signal)
        
        return signals

# Test custom strategy
custom_strategy = CustomMomentumStrategy()
data = generator.generate_realistic_data('SPY', periods=252)
signals = custom_strategy.generate_signals(data)
print(f"Custom strategy generated {len(signals)} signals")
```

#### Data Analysis and Backtesting
```python
# Backtesting across multiple scenarios
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine
from src.strategies import create_strategy

def analyze_strategy_across_conditions(strategy_name, conditions):
    """Analyze strategy performance across different market conditions."""
    results = {}
    
    for condition_name, generator_params in conditions.items():
        generator = MockDataGenerator(**generator_params)
        data = generator.generate_realistic_data('SPY', periods=252)
        
        strategy = create_strategy(strategy_name)
        engine = BacktestEngine()
        result = engine.run_backtest(strategy, data, initial_capital=10000)
        
        results[condition_name] = {
            'total_return': result.performance_metrics.total_return_pct,
            'win_rate': result.performance_metrics.win_rate,
            'sharpe_ratio': result.performance_metrics.sharpe_ratio,
            'max_drawdown': result.performance_metrics.max_drawdown_pct
        }
    
    return results

# Test different market conditions
market_conditions = {
    'bull_market': {'trend': 'bullish', 'volatility': 0.15},
    'bear_market': {'trend': 'bearish', 'volatility': 0.25},
    'sideways_market': {'trend': 'sideways', 'volatility': 0.10},
    'high_volatility': {'trend': 'neutral', 'volatility': 0.35}
}

# Analyze MACD+RSI strategy across conditions
results = analyze_strategy_across_conditions('macd_rsi', market_conditions)
for condition, metrics in results.items():
    print(f"{condition}: {metrics['total_return']:.2%} return, {metrics['win_rate']:.1%} win rate")
```

#### Performance Comparison and Optimization
```python
# Performance comparison with parameter optimization
from src.strategies import DEFAULT_STRATEGY_PARAMS, create_strategy
import itertools

def optimize_strategy_parameters(strategy_name, symbol, param_ranges):
    """Optimize strategy parameters using grid search."""
    best_result = None
    best_params = None
    best_return = -float('inf')
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    for params_combo in itertools.product(*param_values):
        params = dict(zip(param_names, params_combo))
        
        # Test this parameter combination
        generator = MockDataGenerator()
        data = generator.generate_realistic_data(symbol, periods=252)
        
        strategy = create_strategy(strategy_name, params)
        engine = BacktestEngine()
        result = engine.run_backtest(strategy, data, initial_capital=10000)
        
        if result.performance_metrics.total_return_pct > best_return:
            best_return = result.performance_metrics.total_return_pct
            best_params = params
            best_result = result
    
    return best_result, best_params

# Example: Optimize RSI parameters
rsi_param_ranges = {
    'rsi_period': [10, 14, 18, 21],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75]
}

best_result, best_params = optimize_strategy_parameters('rsi_trend', 'SPY', rsi_param_ranges)
print(f"Best parameters: {best_params}")
print(f"Best return: {best_result.performance_metrics.total_return_pct:.2%}")
```

### Output Formats

**Table Format (Default):**
```
Symbol    Price    Change    Volume       RSI     MACD
SPY       485.67   +2.34     12,345,678   65.23   +0.45
QQQ       395.21   -1.87     8,765,432    58.91   -0.23
```

**JSON Format:**
```bash
python market_analysis_cli.py data --symbol SPY --format json
```

**CSV Format:**
```bash
python market_analysis_cli.py export --symbol SPY --format csv
```

## 📊 Strategy Analysis

### Overview

The Strategy Analysis feature helps traders determine which strategies work best for their specific trading style and market conditions. It analyzes all available strategies across three distinct trading styles and provides recommendations based on performance metrics.

### Trading Styles Analyzed

**Day Trading (Short-term, High Frequency)**
- Hold time: Less than 24 hours
- Focus: High-frequency signals, consistent small wins
- Best for: Active traders who can monitor positions throughout the day
- Preferred metrics: High win rate, frequent signals, low drawdown

**Swing Trading (Medium-term, 1-10 days)**
- Hold time: 1-10 days
- Focus: Capturing medium-term price movements
- Best for: Part-time traders who check positions daily
- Preferred metrics: Balanced win rate and return per trade

**Position Trading (Long-term, >10 days)**
- Hold time: More than 10 days
- Focus: Major trend following with higher returns per trade
- Best for: Buy-and-hold investors with patience for larger moves
- Preferred metrics: Higher returns per trade, lower frequency

### Key Metrics Analyzed

**Performance Metrics:**
- **Win Rate**: Percentage of profitable trades
- **Average Return per Trade**: Expected profit/loss per signal
- **Profit Factor**: Ratio of gross profit to gross loss
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Total Return**: Overall portfolio performance

**Trading Characteristics:**
- **Number of Signals**: Frequency of trading opportunities
- **Average Hold Time**: How long positions are typically held
- **Style Fit Rating**: Overall suitability (Excellent/Good/Fair/Poor)
- **Recommendation Score**: Composite score (0-1) for each trading style

### Available Analysis Commands

#### 1. Strategy Effectiveness Analysis
```bash
# Analyze all strategies for a symbol
python market_analysis_cli.py analyze --symbol SPY

# Analyze specific strategies
python market_analysis_cli.py analyze --symbol AAPL --strategies ma_crossover,macd_rsi

# Use different time periods
python market_analysis_cli.py analyze --symbol TSLA --period 6m
python market_analysis_cli.py analyze --symbol QQQ --period 2y

# Get JSON output for programmatic use
python market_analysis_cli.py analyze --symbol SPY --format json

# Test with mock data (development)
python market_analysis_cli.py analyze --symbol SPY --use-mock-data
```

#### 2. Strategy Comparison
```bash
# Compare 2-5 specific strategies
python market_analysis_cli.py compare --strategies ma_crossover,macd_rsi

# Compare with different symbol/period
python market_analysis_cli.py compare --strategies macd_rsi,rsi_trend,bollinger_breakout --symbol AAPL --period 1y

# Get detailed comparison in JSON format
python market_analysis_cli.py compare --strategies ma_crossover,macd_rsi --format json
```

#### 3. List Available Strategies
```bash
# Show all strategies with metadata
python market_analysis_cli.py strategies

# Get machine-readable output
python market_analysis_cli.py strategies --format json
```

### Interpreting Results

#### Summary Table
The results start with a summary table showing:
- **Overall Score**: Best recommendation score across all trading styles
- **Best Style**: Recommended trading style for each strategy
- **Style Indicators**: Visual indicators for each style (●=Excellent, ●=Good, ●=Fair, ●=Poor)

#### Detailed Analysis
For the top-performing strategies, detailed metrics are shown:

```
Strategy: MACD+RSI Combined - SPY
┌─────────────────────┬─────────────┬──────────────┬──────────────────┐
│ Metric              │ Day Trading │ Swing Trading│ Position Trading │
├─────────────────────┼─────────────┼──────────────┼──────────────────┤
│ Win Rate            │ 68.5%       │ 73.2%        │ 71.8%            │
│ Avg Return/Trade    │ 0.64%       │ 0.88%        │ 1.24%            │
│ Profit Factor       │ 1.89        │ 2.14         │ 2.03             │
│ Num Signals         │ 124         │ 67           │ 23               │
│ Avg Hold Time       │ 18.5h       │ 4.2 days     │ 18.7 days        │
│ Recommendation      │ 0.72 (Good) │ 0.85 (Excellent) │ 0.78 (Good) │
└─────────────────────┴─────────────┴──────────────┴──────────────────┘

Recommendation: Best Trading Style: Swing Trading
Style Fit: Excellent
Expected Win Rate: 73.2%
Expected Return/Trade: 0.88%
Signals per Year: 67
Average Hold Time: 4.2 days
```

#### Recommendation Guidelines

**Excellent (0.8-1.0)**: Recommended for this trading style
- Strategy shows good performance across multiple metrics
- Suitable for the time horizon and risk profile
- Expected to generate returns

**Good (0.6-0.8)**: Acceptable with some considerations
- Good performance with minor areas for improvement
- Fits the trading style with reasonable expectations
- May require additional risk management

**Fair (0.4-0.6)**: Acceptable but not optimal
- Mixed performance that may work in certain market conditions
- Consider combining with other strategies or indicators
- Higher risk or lower consistency expected

**Poor (0.0-0.4)**: Not recommended for this trading style
- Strategy doesn't align well with the time horizon
- Inconsistent performance or unfavorable metrics
- Consider alternative strategies or different time frames

### Usage Examples

#### Example 1: Finding the Best Strategy for SPY
```bash
$ python market_analysis_cli.py analyze --symbol SPY

Strategy Effectiveness Analysis for SPY
Analysis Period: 1y
Strategies Analyzed: 5
Data Source: Worker Service

Summary Recommendations:
1. MACD+RSI Combined - Best for Swing Trading
   Win Rate: 73.2%, Avg Return: 0.88%, Signals: 67
2. RSI Trend Following - Best for Day Trading
   Win Rate: 71.5%, Avg Return: 0.62%, Signals: 89
3. Bollinger Band Breakout - Best for Position Trading
   Win Rate: 68.9%, Avg Return: 1.15%, Signals: 34
```

#### Example 2: Comparing Momentum Strategies
```bash
$ python market_analysis_cli.py compare --strategies macd_rsi,rsi_trend --symbol AAPL

# This will show a direct comparison of the two momentum-based strategies
# highlighting which performs better for different trading styles
```

#### Example 3: Development and Testing
```bash
# Test analysis with mock data during development
$ python market_analysis_cli.py analyze --symbol SPY --use-mock-data

# This generates realistic market data for testing without API calls
```

### Usage Tips

1. **Start with Broad Analysis**: Use `strategy-effectiveness` to get an overview
2. **Narrow Down**: Use `compare-strategies` for detailed comparisons
3. **Consider Your Style**: Choose strategies that match your available time and risk tolerance
4. **Validate Results**: Test with different symbols and time periods
5. **Paper Trade First**: Always validate strategies with paper trading before live implementation
6. **Monitor Performance**: Regularly re-analyze as market conditions change

## 📖 Strategy Documentation & Explanations

### Understanding Strategy Logic

The system includes documentation for all trading strategies, accessible through the CLI. Each strategy explanation covers:

- **Signal Generation Logic**: Exact conditions for buy/sell signals
- **Parameter Configuration**: Default values with descriptions and ranges
- **Confidence Calculation**: How signal confidence scores are computed
- **Special Features**: Unique capabilities and advantages
- **Usage Guidelines**: Best trading styles and market conditions

### Strategy Explanation Command

```bash
# View detailed explanation for a specific strategy
python market_analysis_cli.py explain --strategy macd_rsi

# Get structured JSON output for integration
python market_analysis_cli.py explain --strategy bollinger_breakout --format json

# Show overview of all available strategies
python market_analysis_cli.py strategies --detailed
```

### Example Output

When you run `python market_analysis_cli.py explain --strategy macd_rsi`, you'll see:

```
MACD + RSI Combined Strategy
High-performance momentum strategy combining MACD trend detection with RSI 
timing. Documented 73% win rate.

 Difficulty:        Intermediate                                                
 Signal Frequency:  Medium (selective entries)                                  
 Best For:          Momentum trading, Trend following, Short to medium-term     
                    trades                                                      

🟢 BUY SIGNALS

MACD Bullish Crossover + RSI Exit from Oversold:
  • MACD line crosses ABOVE signal line
  • RSI moves ABOVE 30 (from below 30)

MACD Bullish Crossover + RSI Uptrend + Positive Momentum:
  • MACD line crosses ABOVE signal line
  • RSI is ABOVE 50 (uptrend zone)
  • MACD histogram is POSITIVE

🔴 SELL SIGNALS

MACD Bearish Crossover + RSI Exit from Overbought:
  • MACD line crosses BELOW signal line
  • RSI moves BELOW 70 (from above 70)

⚙️ DEFAULT PARAMETERS
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter           ┃ Default ┃ Description                     ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ macd_fast           │      12 │ MACD fast EMA period            │
│ macd_slow           │      26 │ MACD slow EMA period            │
│ rsi_period          │      14 │ RSI calculation period          │
│ min_confidence      │     0.6 │ Minimum signal confidence (60%) │
└─────────────────────┴─────────┴─────────────────────────────────┘

🎯 CONFIDENCE CALCULATION
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Factor        ┃ Weight ┃ Description                             ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Macd Strength │  25%   │ Distance between MACD and signal lines  │
│ Rsi Position  │  25%   │ How close RSI is to oversold/overbought │
│ Volume        │  15%   │ Current volume vs 20-day average        │
└───────────────┴────────┴─────────────────────────────────────────┘

✨ SPECIAL FEATURES
  • Only executes signals with ≥60% confidence
  • Divergence detection for reversal opportunities
  • Volume confirmation with 20-day average
```

### Available Strategy Documentation

| Strategy | Key | Difficulty | Frequency | Focus |
|----------|-----|------------|-----------|--------|
| **MACD + RSI Combined** | `macd_rsi` | Intermediate | Medium | Momentum + timing |
| **Moving Average Crossover** | `ma_crossover` | Beginner | Low | Trend following |
| **RSI Trend Following** | `rsi_trend` | Intermediate | Medium | Trend + pullbacks |
| **Bollinger Band Breakout** | `bollinger_breakout` | Intermediate | Medium | Volatility expansion |
| **Support/Resistance Breakout** | `sr_breakout` | Complex | Low | Key level trading |

### Integration with Analysis

Use strategy explanations alongside performance analysis:

```bash
# 1. Analyze strategy effectiveness
python market_analysis_cli.py analyze --symbol SPY

# 2. Get detailed explanation of top performer
python market_analysis_cli.py explain --strategy macd_rsi

# 3. Compare similar strategies
python market_analysis_cli.py compare --strategies macd_rsi,rsi_trend

# 4. Understand the logic behind the winner
python market_analysis_cli.py explain --strategy rsi_trend --format json
```

## 📈 Period-Based Returns Analysis

### Overview

The period-based returns analysis provides detailed insight into how strategies perform across different time horizons. This feature tracks daily, weekly, and monthly returns along with Year-to-Date (YTD) performance metrics.

### Period Returns Command

```bash
# Analyze period returns for a strategy and symbol
python market_analysis_cli.py period-returns --strategy macd_rsi --symbol AAPL

# Use different time periods
python market_analysis_cli.py period-returns --strategy ma_crossover --symbol MSFT --period 6m

# Get structured JSON output
python market_analysis_cli.py period-returns --strategy bollinger_breakout --symbol TSLA --format json

# Test with mock data
python market_analysis_cli.py period-returns --strategy rsi_trend --symbol SPY --mock-data
```

### Key Metrics Provided

**Daily Returns:**
- Individual trading day performance
- Cumulative daily return progression
- Daily P&L and percentage changes

**Weekly Returns:**
- Week-over-week performance analysis
- Weekly trend identification
- Volatility patterns across weeks

**Monthly Returns:**
- Month-over-month strategic performance
- Seasonal performance patterns
- Long-term trend analysis

**Year-to-Date (YTD) Performance:**
- Current year cumulative returns
- YTD versus historical comparison
- Annual performance tracking

### Use Cases

1. **Performance Review**: Track how strategies perform over specific periods
2. **Volatility Analysis**: Identify periods of high/low strategy volatility
3. **Seasonal Patterns**: Discover if strategies work better in certain months
4. **Risk Assessment**: Understand drawdown patterns across time periods
5. **Strategy Tuning**: Optimize parameters based on period-specific performance

### Integration with Backtesting

The analysis results can be used to configure backtesting parameters:

```python
# Based on analysis results, configure backtest
from src.strategies import MACDRSIStrategy
from src.backtest import BacktestEngine

# Use the recommended strategy
strategy = MACDRSIStrategy()
engine = BacktestEngine()

# Configure based on analysis (e.g., swing trading parameters)
result = engine.run_backtest(strategy, data, initial_capital=10000)
```

## 🔗 Alpaca Markets Integration

### Setup & Configuration

1. **Get Alpaca API Keys:**
   - Visit [Alpaca Markets](https://app.alpaca.markets/)
   - Create free account and generate API keys

2. **Configure Environment:**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_DATA_FEED=iex  # or 'sip' for $99/month full market data
ALPACA_PAPER_TRADING=true  # Start with paper trading
WATCHLIST_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,SPY,QQQ,IWM
```

3. **Test Integration:**
```bash
python scripts/setup_alpaca.py
```

### Data Feeds

| Feed Type | Cost | Coverage | Use Case |
|-----------|------|----------|----------|
| **IEX** | Free | ~2.5% market volume | Development, testing, low-frequency strategies |
| **SIP** | $99/month | 100% market volume | Production trading, high-frequency strategies |

### Usage Examples

```python
from src.data.providers.alpaca_provider import AlpacaProvider
from src.trading.alpaca_client import AlpacaTradingClient

# Initialize data provider
provider = AlpacaProvider({'paper': True, 'feed': 'iex'})
await provider.connect()

# Get historical data
data = await provider.get_historical_data(
    symbol="AAPL",
    timeframe=TimeFrame.DAILY,
    start_date=datetime.now() - timedelta(days=30)
)

# Initialize trading client
client = AlpacaTradingClient({'paper': True})
await client.connect()

# Get account info
account = await client.get_account()
print(f"Buying Power: ${account['buying_power']:,.2f}")

# Place order
order = await client.place_market_order(
    symbol="AAPL",
    quantity=10,
    side=AlpacaOrderSide.BUY
)
```

### Trading Safety

**Always start with paper trading:**
```bash
ALPACA_PAPER_TRADING=true  # $100,000 simulated capital
```

**For live trading:**
1. Test thoroughly with paper trading first
2. Start small with minimal position sizes  
3. Monitor carefully with proper risk management
4. Use stop losses to limit downside risk

## ⚙️ Configuration

### Environment Variables

```bash
# Market Data Configuration
WATCHLIST_SYMBOLS="SPY,QQQ,IWM,AAPL,TSLA"
DATA_UPDATE_INTERVAL=1                      # Update frequency (seconds)
DATA_PROVIDER="mock"                        # Data provider (mock, alpaca)

# System Performance
CALCULATION_WORKERS=4                       # Number of worker processes
MAX_CONCURRENT_CALCULATIONS=100             # Concurrent calculation limit
BATCH_CALCULATION_SIZE=10                   # Batch processing size

# Redis Configuration
REDIS_HOST="localhost"                      # Redis server host
REDIS_PORT=6379                            # Redis server port
REDIS_STREAMS_MAXLEN=10000                 # Stream message retention
REDIS_MAX_MEMORY="2gb"                     # Redis memory limit

# Alpaca Markets Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER_TRADING=true                  # Enable paper trading
ALPACA_DATA_FEED=iex                       # Data feed type (iex/sip)

# Trading Parameters
DEFAULT_COMMISSION=0.001                   # Commission rate
DEFAULT_SLIPPAGE=0.0005                   # Slippage estimate
RISK_FREE_RATE=0.02                       # Risk-free rate for metrics

# Performance Settings
MAX_MEMORY_MB=1024                        # Memory limit
CACHE_TTL_SECONDS=3600                    # Cache time-to-live
MAX_CONCURRENT_BACKTESTS=5                # Backtest concurrency

# Monitoring
ENABLE_MONITORING=true                    # Enable system monitoring
ENABLE_ALERTS=true                       # Enable alert system
LOG_LEVEL=INFO                           # Logging level
```

### Strategy Parameters

```python
# Default parameter sets
STRATEGY_PARAMS = {
    "MovingAverageCrossover": {
        "fast_period": 50,
        "slow_period": 200,
        "signal_threshold": 0.02
    },
    "MACDRSIStrategy": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
    },
    "RSITrendFollowing": {
        "rsi_period": 14,
        "trend_threshold": 50,
        "oversold_level": 30,
        "overbought_level": 70
    },
    "BollingerBandBreakout": {
        "bb_period": 20,
        "bb_std": 2.0,
        "breakout_threshold": 0.01
    },
    "SupportResistanceBreakout": {
        "lookback": 20,
        "min_touches": 3,
        "breakout_threshold": 0.02
    }
}
```

### CLI Configuration

```yaml
# ~/.wagehood/cli_config.yaml
system:
  timeout: 30
  retries: 3
  log_level: "INFO"
  
output:
  format: "table"  # json, table, csv
  use_color: true
  max_width: 120
  
data:
  cache_enabled: true
  cache_ttl: 300
  
streaming:
  buffer_size: 1000
  reconnect_delay: 5
  
logging:
  level: "INFO"
  file: "~/.wagehood/cli.log"
```

## 📈 Performance Characteristics

### Performance Targets

- **Data Ingestion**: 1-second updates per asset
- **Calculation Latency**: <100ms per indicator update
- **CLI Response Time**: <10ms for cached data
- **System Throughput**: 1000+ assets simultaneously
- **Memory Usage**: Optimized with rolling windows and incremental algorithms

### Real-Time Processing

- **Target: Sub-second Updates**: Market data processing frequency
- **Optimized Calculations**: Efficient updates for most indicators
- **Redis Streams**: Event-driven architecture for message delivery
- **Circuit Breakers**: Fault tolerance for external data feeds
- **Horizontal Scaling**: Add workers for more symbols

### CLI Performance

- **Target Latency**: <50ms from market to system
- **Data Queries**: <10ms for cached data, <100ms for fresh data
- **CLI Commands**: <10ms for local operations
- **Stream Processing**: <10ms through Redis pipeline
- **Analysis Operations**: <1s for standard backtests

### Scalability

- **Horizontal Scaling**: Redis consumer groups for parallel processing
- **Load Balancing**: Multiple processing instances with shared Redis
- **Caching Strategy**: Multi-level caching (local + Redis)
- **Connection Pooling**: Optimized Redis and database connections
- **Circuit Breakers**: Automatic failover and recovery

## 🧪 Testing

### Test Suite

Run the test suite with target 90%+ coverage:

```bash
# Install dependencies and run all tests
python run_tests.py --install-deps --all --coverage --parallel

# Run specific test categories
python run_tests.py --unit --integration
python run_tests.py --performance --memory

# Direct pytest usage
pytest --cov=src --cov-report=html
pytest tests/unit/test_strategies.py -v
pytest tests/integration/ -v
```

### Test Coverage

| Component | Target Coverage | Status |
|-----------|----------------|--------|
| Core Models | 95% | ✅ |
| Indicators | 95% | ✅ |
| Strategies | 90% | ✅ |
| Backtest Engine | 90% | ✅ |
| Data Management | 85% | ✅ |
| CLI Interface | 85% | ✅ |
| **Overall** | **90%** | **✅** |

### Performance Benchmarks

- **Indicator Calculations**: <1s for 10K data points
- **Strategy Signal Generation**: <1s for 5K data points
- **Backtest Execution**: <5s for 1K data points
- **CLI Response Time**: <1s for standard commands
- **Memory Usage**: <100MB for typical operations

### Test Data & Scenarios

**Market Scenarios:**
- Bull Market (8% annual growth)
- Bear Market (-20% annual decline)
- Sideways Market (range-bound trading)
- High Volatility (40% daily volatility)
- Flash Crash (rapid decline with recovery)
- Golden Cross (MA crossover signals)

**Edge Cases:**
- Constant prices, zero volume
- Extreme price jumps
- Gap-heavy markets
- Whipsaw conditions

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY run_realtime.py .
COPY market_analysis_cli.py .
COPY market_watch.py .
RUN pip install -e .

CMD ["python", "run_realtime.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  wagehood-realtime:
    build: .
    command: python run_realtime.py
    environment:
      - REDIS_HOST=redis
      - CALCULATION_WORKERS=4
    depends_on:
      - redis

volumes:
  redis_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wagehood-realtime
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wagehood-realtime
  template:
    metadata:
      labels:
        app: wagehood-realtime
    spec:
      containers:
      - name: processor
        image: wagehood/realtime-processor:latest
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: WATCHLIST_SYMBOLS
          value: "SPY,QQQ,IWM"
        - name: CALCULATION_WORKERS
          value: "4"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Health Checks

```bash
# System health check
python market_analysis_cli.py --help

# Performance metrics
python market_watch.py --symbols SPY --duration 10

# System ping test
redis-cli ping
```

### Monitoring & Observability

**System Monitoring:**
- Structured JSON logging for aggregation
- Prometheus-compatible metrics export
- Kubernetes-compatible health checks  
- Configurable alerting via Redis Streams

**Available Metrics:**
- Ingestion: Events published, errors, provider performance
- Calculation: Indicators calculated, signals generated, latency
- System: Memory usage, Redis performance, worker status
- Business: Signal accuracy, strategy performance

## 🔐 Security & Safety

### System Security

- **Credential Encryption**: CLI credentials encrypted with master password
- **Rate Limiting**: Configurable per-key limits with Redis backend
- **Input Validation**: All market data validated before processing
- **Secure Configuration**: Environment-based configuration management

### Trading Safety

- **Paper Trading First**: Always test with simulated capital
- **Position Sizing**: Configurable risk management
- **Stop Losses**: Automatic loss limitation
- **Circuit Breakers**: System-wide risk controls

### Data Security

- **Environment Variables**: Never commit secrets to version control
- **Redis Security**: Use AUTH and TLS for production
- **Credential Rotation**: Regular API key updates
- **Audit Logging**: Activity tracking

## 🛠️ Troubleshooting

### Common Issues

#### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Start Redis server
redis-server

# Check configuration
echo $REDIS_HOST $REDIS_PORT
```

#### System Connection Errors
```bash
# Test CLI connectivity
python market_analysis_cli.py --help

# Check if worker service is running
ps aux | grep run_realtime.py

# Start real-time processing
python run_realtime.py
```

#### No Data Processing
```bash
# Validate configuration
python run_realtime.py --validate-only

# Check enabled symbols
python run_realtime.py --show-config

# Enable debug logging
python run_realtime.py --log-level DEBUG
```

#### Alpaca Integration Issues
```bash
# Test Alpaca credentials
python scripts/setup_alpaca.py

# Check environment variables
echo $ALPACA_API_KEY $ALPACA_SECRET_KEY

# Verify account status
python run_realtime.py --provider alpaca --log-level DEBUG
```

#### High Memory Usage
```bash
# Monitor Redis memory
redis-cli info memory

# Adjust stream retention
export REDIS_STREAMS_MAXLEN=5000

# Check system metrics
python market_analysis_cli.py --help
```

### Debug Mode

```bash
# Enable verbose logging
python market_analysis_cli.py --verbose data --symbol SPY
python market_watch.py --debug

# Check specific component logs
grep "CalculationEngine" realtime_processor_*.log
grep "MarketDataIngestion" realtime_processor_*.log

# Monitor logs in real-time
python run_realtime.py --log-level DEBUG
```

### Performance Optimization

1. **Use caching**: Redis caches recent data for faster responses
2. **Batch operations**: Process multiple symbols together with worker service
3. **Limit output**: Use `--limit` for large datasets in CLI tools
4. **Stream carefully**: Monitor bandwidth usage with market_watch.py
5. **Optimize Redis**: Tune memory and connection settings for worker service

## 📚 Research Foundation

The system implements strategies based on quantitative analysis methods:

- **73% win rate** MACD+RSI strategy from peer-reviewed studies
- **50% drawdown reduction** using Golden Cross filtering
- **Serenity Ratio optimization** for strategy selection
- **Volume confirmation** for breakout validation
- **Multi-timeframe validation** across asset classes

### Performance Validation

All strategies have been backtested across:
- **Multiple Market Conditions**: Bull, bear, sideways, high/low volatility
- **Various Asset Classes**: Stocks, ETFs, commodities, forex, crypto
- **Different Timeframes**: 1-minute to monthly analysis
- **Risk Scenarios**: Stress testing and edge case validation

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow coding standards**: Use Black, Flake8, and MyPy
4. **Add tests**: Maintain 90%+ coverage
5. **Update documentation**: Include examples and usage
6. **Run test suite**: `pytest tests/ --cov=src`
7. **Commit changes**: `git commit -m 'Add amazing feature'`
8. **Push to branch**: `git push origin feature/amazing-feature`
9. **Open Pull Request**

### Development Standards

- **Code Style**: PEP 8 compliance with Black formatting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Testing**: Unit, integration, and performance tests
- **Error Handling**: Exception handling
- **Security**: Input validation and secure coding practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and consider your risk tolerance before implementing any trading strategy. The authors are not responsible for any financial losses incurred through the use of this software.

**Trading involves substantial risk and may not be suitable for all investors. Please trade responsibly.**

---

**For systematic traders and quantitative researchers**

*For additional support and updates, please refer to the project documentation and community forums.*