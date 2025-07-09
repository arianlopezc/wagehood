# Wagehood Signal Detection System

**A Redis-based worker service for real-time market signal detection with Python API for strategy analysis and market monitoring. Built with data processing and signal detection capabilities.**

> **⚠️ IMPORTANT DISCLAIMER**: This is a signal detection system only. It does not execute trades or manage portfolios. All signals are for analysis and educational purposes only.

## 🚀 Overview

Wagehood is a signal detection system for systematic traders and quantitative researchers. It combines signal generation strategies with real-time data processing, providing tools for strategy development, testing, and signal analysis.

### Key Features

- **5 Signal Detection Strategies** with comprehensive signal analysis capabilities
- **Redis-Based Worker Service** with real-time market data processing
- **Python API** - Market analysis and signal monitoring interfaces
- **Strategy Analysis & Optimization** - Analyze which strategies generate the highest quality signals
- **Strategy Documentation** - Detailed explanations of signal generation logic and parameters
- **Real-Time Processing** with Redis Streams for event-driven architecture
- **Alpaca Markets Integration** for historical data and market analysis
- **Testing Suite** with code coverage tracking

## 🎯 Core Trading Strategies

### Multi-Strategy Multi-Timeframe System

The system supports multi-dimensional signal analysis across:
- **5 signal detection strategies** with comprehensive documentation
- **9 timeframes** from 1-minute to monthly
- **3 analysis profiles** (Short-term, Medium-term, Long-term)
- **Watchlist management**

### Implemented Strategies

| Strategy | Signal Logic | Best Timeframes | Analysis Profile | Description |
|----------|-------------|----------------|------------------|-------------|
| **MACD+RSI Combined** | Momentum convergence | 1h, 4h, 1d | Medium/Long-term | Momentum strategy combining MACD and RSI signals |
| **RSI Trend Following** | Trend-aware oscillation | 15m, 30m, 1h | Short/Medium-term | Trend-aware RSI with pullback signals |
| **Bollinger Band Breakout** | Volatility expansion | 5m, 15m, 1h | Short/Medium-term | Volatility expansion signal detection |
| **Support/Resistance Breakout** | Level breakouts | 1h, 4h, 1d | Medium/Long-term | Level-based breakout signal detection |
| **Moving Average Crossover** | MA crossover signals | 1d, 1w, 1M | Long-term | Golden/Death cross signal detection |

### Analysis Profiles

**Short-Term Analysis:**
- **Timeframes:** 1m, 5m, 15m
- **Focus:** High-frequency signals, rapid signal generation
- **Best Strategies:** RSI Trend, Bollinger Breakout
- **Signal Frequency:** Multiple signals per hour
- **Use Case:** Active market monitoring and quick signal identification

**Medium-Term Analysis:**
- **Timeframes:** 30m, 1h, 4h  
- **Focus:** Multi-hour signal patterns (hours to days)
- **Best Strategies:** MACD+RSI, RSI Trend, Bollinger Breakout
- **Signal Frequency:** Daily signal generation
- **Use Case:** Swing analysis and trend identification

**Long-Term Analysis:**
- **Timeframes:** 1d, 1w, 1M
- **Focus:** Long-term signal patterns (days to weeks)
- **Best Strategies:** Moving Average Crossover, Support/Resistance Breakout
- **Signal Frequency:** Weekly signal generation
- **Use Case:** Position analysis and long-term trend detection

### Signal Detection Asset Classes

**Recommended Asset Classes for Signal Analysis:**
1. **Commodities** - Clear trend-following signals
2. **Cryptocurrencies** - High volatility, strong signal patterns  
3. **Forex Major Pairs** - Clear central bank-driven signal trends
4. **Index ETFs** - Consistent signal patterns with reduced noise

**Timeframe Recommendations:**
- **Short-Term Analysis**: RSI (7-period), Bollinger Bands
- **Medium-Term Analysis**: All strategies provide quality signals
- **Long-Term Analysis**: Moving Average Crossover signals

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │───▶│ Redis Streams   │───▶│ Worker Service  │
│ (Alpaca/Mock)   │    │ (Event Bus)     │    │ (run_realtime)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Redis Cache     │◀───│ Calculation     │◀───│ Signal          │
│ (Results)       │    │ Engine          │    │ Generation      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Discord         │◀───│ Data Services   │◀───│ Signal Analysis │
│ Notifications   │    │ & Storage       │    │ & Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

```
src/
├── core/               # Data models and constants
├── data/               # Data management and providers
│   └── providers/      # Alpaca, mock, and extensible providers
├── indicators/         # 20+ technical indicator calculations
├── strategies/         # 5 signal detection strategy implementations  
├── backtest/           # Signal analysis engine with historical validation
├── realtime/           # Real-time processing and data ingestion
├── services/           # Analysis and data services
├── analysis/          # Signal evaluation and comparison
└── storage/           # Results storage and caching

Python API:
└── src/                   # Main API package structure
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

> **💡 Worker Service Architecture**: The system runs as a Redis-based worker service with Python API for market signal detection and analysis.

### Prerequisites

1. **Python 3.9+** 
2. **Redis Server** (for real-time processing)
3. **Alpaca Markets Account** (optional, for live data/historical analysis)

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
python -c "from src.data.mock_generator import MockDataGenerator; print('Mock provider working')"

# Test Python API
python -c "from src.core.models import OHLCV; print('Core models working')"
python -c "from src.strategies import create_strategy; print('Strategies working')"
```

### Basic Usage

```python
from src.data.mock_generator import MockDataGenerator
from src.strategies import create_strategy
from src.backtest.engine import SignalAnalysisEngine
from src.core.models import MarketData, TimeFrame

# Generate sample data
generator = MockDataGenerator()
ohlcv_data = generator.generate_realistic_data("SPY", periods=252)

# Create MarketData object for analysis
market_data = MarketData(
    symbol="SPY",
    timeframe=TimeFrame.DAILY,
    data=ohlcv_data,
    indicators={},
    last_updated=ohlcv_data[-1].timestamp
)

# Initialize strategy
strategy = create_strategy('macd_rsi')

# Run signal analysis
engine = SignalAnalysisEngine()
result = engine.run_signal_analysis(strategy, market_data)

print(f"Total Signals: {result.total_signals}")
print(f"Buy Signals: {result.buy_signals}")
print(f"Sell Signals: {result.sell_signals}")
print(f"Average Confidence: {result.avg_confidence:.2f}")
```

### Quick Start Guide

Follow these steps to get your multi-strategy multi-timeframe signal detection system running:

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

# 5. Use the Python API for signal analysis
python -c "
from src.strategies import create_strategy
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine
from src.core.models import MarketData, TimeFrame

# Generate test data
generator = MockDataGenerator()
ohlcv_data = generator.generate_realistic_data('SPY', periods=252)

# Create MarketData object
market_data = MarketData(
    symbol='SPY',
    timeframe=TimeFrame.DAILY,
    data=ohlcv_data,
    indicators={},
    last_updated=ohlcv_data[-1].timestamp
)

# Test a strategy
strategy = create_strategy('macd_rsi')
engine = BacktestEngine()
result = engine.run_backtest(strategy, market_data, initial_capital=10000)
print(f'MACD+RSI Strategy Result: {result.total_signals} signals, {result.avg_confidence:.2f} avg confidence')
"

# 6. Explore all available strategies
python -c "
from src.strategies import STRATEGY_REGISTRY
for name, strategy_class in STRATEGY_REGISTRY.items():
    print(f'{name}: {strategy_class.__doc__ or \"Trading strategy\"}')
"
```

## 🖥️ Python API Interface

The Wagehood system provides a Python API for multi-strategy multi-timeframe signal analysis and a Job Submission CLI for signal analysis against the running production instance.

## 📋 Job Submission CLI

Submit signal analysis jobs to the running production instance and get detailed results with all signals and analysis metrics.

```bash
# Submit a backtest job
python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \
                    --start 2024-01-01 --end 2024-12-31

# Output includes:
# - Real-time progress monitoring
# - Complete performance metrics  
# - All trading signals generated
# - Signal quality analysis
# - Signal frequency and confidence analysis
```

**Key Features:**
- **Single Command**: Submit, monitor, and view results in one command
- **Real-time Progress**: Live progress updates with visual progress bar
- **Comprehensive Results**: All signals, trades, and performance metrics
- **Production Integration**: Uses running Docker instance for analysis

See [Job Submission CLI Documentation](docs/JOB_SUBMISSION_CLI.md) for complete usage guide.

### Core Python API

**Strategy Analysis:**
```python
from src.strategies import create_strategy, STRATEGY_REGISTRY
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine

# Explore available strategies
for name, strategy_class in STRATEGY_REGISTRY.items():
    print(f"{name}: {strategy_class.__doc__ or 'Trading strategy'}")

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
from src.realtime.data_ingestion import create_ingestion_service
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.config_manager import ConfigManager

# Set up real-time processing (requires Redis)
config = ConfigManager()
ingestion_service = create_ingestion_service(config)
calc_engine = CalculationEngine()

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

### Data Analysis Output

The Python API returns structured data objects that can be easily processed:

```python
# Example: Getting strategy performance data
result = engine.run_backtest(strategy, data, initial_capital=10000)
performance = result.performance_metrics

print(f"Total Return: {performance.total_return_pct:.2%}")
print(f"Win Rate: {performance.win_rate:.2%}")
print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
print(f"Max Drawdown: {performance.max_drawdown_pct:.2%}")
```

## 📊 Strategy Analysis

The system provides comprehensive strategy analysis through the Python API, helping traders determine which strategies work best for their specific trading style and market conditions.

### Trading Style Analysis

The Python API can analyze strategies across different trading styles:

```python
from src.analysis.strategy_analyzer import StrategyAnalyzer
from src.data.mock_generator import MockDataGenerator

# Initialize analyzer
analyzer = StrategyAnalyzer()
generator = MockDataGenerator()

# Analyze strategy across different timeframes/styles
strategy_results = {}
for style in ['day_trading', 'swing_trading', 'position_trading']:
    data = generator.generate_realistic_data('SPY', periods=252)
    strategy = create_strategy('macd_rsi')
    engine = BacktestEngine()
    result = engine.run_backtest(strategy, data, initial_capital=10000)
    strategy_results[style] = result.performance_metrics

# Compare results
for style, metrics in strategy_results.items():
    print(f"{style}: {metrics.total_return_pct:.2%} return, {metrics.win_rate:.1%} win rate")
```

### Performance Evaluation

The system provides detailed performance metrics for strategy evaluation:

```python
# Comprehensive strategy evaluation
def evaluate_strategy(strategy_name, symbol, periods=252):
    strategy = create_strategy(strategy_name)
    generator = MockDataGenerator()
    data = generator.generate_realistic_data(symbol, periods=periods)
    
    engine = BacktestEngine()
    result = engine.run_backtest(strategy, data, initial_capital=10000)
    
    metrics = result.performance_metrics
    return {
        'win_rate': metrics.win_rate,
        'total_return': metrics.total_return_pct,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown_pct,
        'profit_factor': metrics.profit_factor,
        'total_trades': metrics.total_trades
    }

# Evaluate multiple strategies
strategies = ['macd_rsi', 'ma_crossover', 'rsi_trend', 'bollinger_breakout']
for strategy_name in strategies:
    metrics = evaluate_strategy(strategy_name, 'SPY')
    print(f"{strategy_name}: {metrics['total_return']:.2%} return, {metrics['win_rate']:.1%} win rate")
```

## 📖 Strategy Documentation & Explanations

### Understanding Strategy Logic

The system provides comprehensive strategy documentation through the Python API. Each strategy includes:

- **Signal Generation Logic**: Exact conditions for buy/sell signals
- **Parameter Configuration**: Default values with descriptions and ranges
- **Confidence Calculation**: How signal confidence scores are computed
- **Special Features**: Unique capabilities and advantages
- **Usage Guidelines**: Best trading styles and market conditions

### Accessing Strategy Information

```python
from src.strategies import STRATEGY_REGISTRY, create_strategy

# View all available strategies
for name, strategy_class in STRATEGY_REGISTRY.items():
    print(f"{name}: {strategy_class.__doc__ or 'Trading strategy'}")

# Get strategy with default parameters
strategy = create_strategy('macd_rsi')
print(f"Strategy: {strategy.__class__.__name__}")

# Access strategy documentation
strategy = create_strategy('macd_rsi')
if hasattr(strategy, 'get_strategy_info'):
    info = strategy.get_strategy_info()
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Parameters: {info.get('parameters', {})}")
```

### Available Strategy Information

| Strategy | Key | Focus | Implementation |
|----------|-----|-------|----------------|
| **MACD + RSI Combined** | `macd_rsi` | Momentum + timing | High-performance momentum strategy |
| **Moving Average Crossover** | `ma_crossover` | Trend following | Classic trend-following approach |
| **RSI Trend Following** | `rsi_trend` | Trend + pullbacks | Trend-aware RSI with pullbacks |
| **Bollinger Band Breakout** | `bollinger_breakout` | Volatility expansion | Volatility expansion strategy |
| **Support/Resistance Breakout** | `sr_breakout` | Key level trading | Level-based breakout trading |

### Strategy Parameter Inspection

```python
# Inspect strategy parameters and signals
strategy = create_strategy('macd_rsi')
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=100)

# Generate signals to understand strategy behavior
signals = strategy.generate_signals(data)
print(f"Generated {len(signals)} signals")

# Analyze signal distribution
signal_types = {}
for signal in signals:
    signal_types[signal.signal_type.value] = signal_types.get(signal.signal_type.value, 0) + 1

for signal_type, count in signal_types.items():
    print(f"{signal_type}: {count} signals")
```

## 📈 Period-Based Returns Analysis

### Overview

The system provides detailed period-based returns analysis through the Python API, offering insight into how strategies perform across different time horizons. This includes daily, weekly, and monthly returns along with Year-to-Date (YTD) performance metrics.

### Period Returns Analysis

```python
from src.backtest.engine import BacktestEngine
from src.strategies import create_strategy
from src.data.mock_generator import MockDataGenerator
from datetime import datetime, timedelta

# Analyze strategy returns across different periods
def analyze_period_returns(strategy_name, symbol, periods=252):
    strategy = create_strategy(strategy_name)
    generator = MockDataGenerator()
    data = generator.generate_realistic_data(symbol, periods=periods)
    
    engine = BacktestEngine()
    result = engine.run_backtest(strategy, data, initial_capital=10000)
    
    # Access period-based returns from performance metrics
    performance = result.performance_metrics
    
    return {
        'daily_returns': performance.daily_returns,
        'weekly_returns': performance.weekly_returns,
        'monthly_returns': performance.monthly_returns,
        'ytd_return': performance.ytd_return,
        'avg_daily_return': performance.avg_daily_return,
        'avg_weekly_return': performance.avg_weekly_return,
        'avg_monthly_return': performance.avg_monthly_return
    }

# Example usage
returns_analysis = analyze_period_returns('macd_rsi', 'AAPL')
print(f"Average Daily Return: {returns_analysis['avg_daily_return']:.4f}")
print(f"Average Weekly Return: {returns_analysis['avg_weekly_return']:.4f}")
print(f"Average Monthly Return: {returns_analysis['avg_monthly_return']:.4f}")
print(f"YTD Return: {returns_analysis['ytd_return']:.2%}")
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

### Advanced Period Analysis

```python
# Advanced period-based analysis
def detailed_period_analysis(strategy_name, symbol):
    strategy = create_strategy(strategy_name)
    generator = MockDataGenerator()
    data = generator.generate_realistic_data(symbol, periods=365)  # One year of data
    
    engine = BacktestEngine()
    result = engine.run_backtest(strategy, data, initial_capital=10000)
    
    performance = result.performance_metrics
    
    # Analyze period returns
    if performance.daily_returns:
        daily_volatility = np.std([r.return_pct for r in performance.daily_returns])
        best_day = max(performance.daily_returns, key=lambda x: x.return_pct)
        worst_day = min(performance.daily_returns, key=lambda x: x.return_pct)
        
        print(f"Daily Volatility: {daily_volatility:.4f}")
        print(f"Best Day: {best_day.return_pct:.2%} on {best_day.period_start.date()}")
        print(f"Worst Day: {worst_day.return_pct:.2%} on {worst_day.period_start.date()}")
    
    # Seasonal analysis
    if performance.monthly_returns:
        monthly_performance = {}
        for month_return in performance.monthly_returns:
            month = month_return.period_start.month
            if month not in monthly_performance:
                monthly_performance[month] = []
            monthly_performance[month].append(month_return.return_pct)
        
        print("\nSeasonal Performance:")
        for month, returns in monthly_performance.items():
            avg_return = sum(returns) / len(returns)
            print(f"Month {month}: {avg_return:.2%} average return")

# Example usage
detailed_period_analysis('macd_rsi', 'SPY')
```

### Integration with Strategy Development

Use period analysis results to optimize strategy parameters:

```python
# Use period analysis for strategy optimization
def optimize_strategy_by_periods(strategy_name, symbol):
    base_strategy = create_strategy(strategy_name)
    generator = MockDataGenerator()
    
    # Test different parameter sets
    parameter_sets = [
        {'rsi_period': 10, 'rsi_overbought': 75, 'rsi_oversold': 25},
        {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30},
        {'rsi_period': 21, 'rsi_overbought': 65, 'rsi_oversold': 35}
    ]
    
    best_params = None
    best_monthly_return = -float('inf')
    
    for params in parameter_sets:
        strategy = create_strategy(strategy_name, params)
        data = generator.generate_realistic_data(symbol, periods=252)
        
        engine = BacktestEngine()
        result = engine.run_backtest(strategy, data, initial_capital=10000)
        
        avg_monthly_return = result.performance_metrics.avg_monthly_return
        if avg_monthly_return > best_monthly_return:
            best_monthly_return = avg_monthly_return
            best_params = params
    
    return best_params, best_monthly_return

# Example optimization
best_params, best_return = optimize_strategy_by_periods('rsi_trend', 'SPY')
print(f"Best parameters: {best_params}")
print(f"Best monthly return: {best_return:.4f}")
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
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_DATA_FEED=iex  # or 'sip' for $99/month full market data
ALPACA_PAPER_TRADING=true  # For data access
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

# Initialize data provider
provider = AlpacaProvider({'paper': True, 'feed': 'iex'})
await provider.connect()

# Get historical data
data = await provider.get_historical_data(
    symbol="AAPL",
    timeframe=TimeFrame.DAILY,
    start_date=datetime.now() - timedelta(days=30)
)

# Get real-time market data
market_data = await provider.get_market_data(
    symbol="AAPL",
    timeframe=TimeFrame.MINUTE_1
)

# Get account info (for data access validation)
account = await provider.get_account()
print(f"Account Status: {account['status']}")
print(f"Data Feed: {account['data_feed']}")
```

### Data Access Safety

**Always use paper trading for development:**
```bash
ALPACA_PAPER_TRADING=true  # For safe development and testing
```

**For production signal detection:**
1. Test thoroughly with paper trading first
2. Use appropriate data feed (IEX for development, SIP for production)
3. Monitor data quality and API rate limits
4. Implement proper error handling for data feed interruptions

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

## 🎯 Discord Multi-Channel Integration

The system features advanced multi-channel Discord integration with strategy-specific routing, trading profile support, and sophisticated rate limiting.

### Key Features

- **Strategy-Specific Channels**: Each trading strategy can route notifications to dedicated Discord channels
- **Trading Profile Integration**: Day trading vs swing trading timeframe filtering
- **Individual Rate Limiting**: Customizable notification limits per strategy
- **Alert Symbol Configuration**: Separate watchlist for notifications
- **Strategy-Timeframe Mapping**: Automated routing based on timeframe and strategy combinations
- **Rich Embed Formatting**: Color-coded messages with comprehensive signal information

### Trading Profile Configuration

**Day Trading Profile:**
- **Timeframes:** 1h notifications for rapid signal delivery
- **Strategies:** RSI Trend Following, Bollinger Band Breakout
- **Rate Limits:** Higher frequency (12-15 notifications/hour)
- **Use Case:** Active intraday monitoring and quick entry/exit signals

**Swing Trading Profile:**
- **Timeframes:** 1d notifications for position trades
- **Strategies:** MACD+RSI Combined, Support/Resistance Breakout
- **Rate Limits:** Lower frequency (6-8 notifications/hour)
- **Use Case:** Multi-day position signals with longer holding periods

### Discord Configuration

```bash
# Core Discord Settings (.env)
DISCORD_NOTIFICATIONS_ENABLED=true
DISCORD_MULTI_CHANNEL_ENABLED=true
DISCORD_NOTIFY_TIMEFRAMES=1h,1d

# Alert Symbol List (separate from main watchlist)
DISCORD_ALERT_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,SPY

# Strategy-Timeframe Mapping
DISCORD_STRATEGY_TIMEFRAME_MAPPING=macd_rsi:1d,sr_breakout:1d,rsi_trend:1h,bollinger_breakout:1h

# Strategy-Specific Webhook URLs
DISCORD_WEBHOOK_MACD_RSI=https://discord.com/api/webhooks/your_macd_rsi_webhook
DISCORD_WEBHOOK_RSI_TREND=https://discord.com/api/webhooks/your_rsi_trend_webhook
DISCORD_WEBHOOK_BOLLINGER_BREAKOUT=https://discord.com/api/webhooks/your_bollinger_webhook
DISCORD_WEBHOOK_SR_BREAKOUT=https://discord.com/api/webhooks/your_sr_breakout_webhook

# Individual Rate Limits (notifications per hour)
DISCORD_RATE_LIMIT_MACD_RSI=8      # Swing trading
DISCORD_RATE_LIMIT_SR_BREAKOUT=6   # Swing trading
DISCORD_RATE_LIMIT_RSI_TREND=12    # Day trading
DISCORD_RATE_LIMIT_BOLLINGER_BREAKOUT=15  # Day trading

# Strategy Enable/Disable
DISCORD_ENABLED_MACD_RSI=true
DISCORD_ENABLED_RSI_TREND=true
DISCORD_ENABLED_BOLLINGER_BREAKOUT=true
DISCORD_ENABLED_SR_BREAKOUT=true
```

### Multi-Channel Benefits

1. **Organized Signal Flow**: Each strategy sends notifications to dedicated channels
2. **Profile-Based Filtering**: Day traders get 1h signals, swing traders get 1d signals
3. **Rate Limit Management**: Prevents notification spam with strategy-specific limits
4. **Selective Symbol Monitoring**: Configure separate symbol lists for notifications
5. **Backward Compatibility**: Fallback to single-channel mode when multi-channel is disabled

## 📈 Performance Characteristics

### Current System Capacity

**Recommended Production Limits:**
- **Symbols**: Up to 20 symbols safely for real-time processing
- **Timeframes**: 6 concurrent timeframes per symbol
- **Strategies**: All 5 strategies simultaneously
- **Memory Usage**: ~512MB for typical production workload
- **CPU Usage**: 1-2 cores for optimal performance

### Signal Processing Performance

- **Signal Generation**: <100ms per signal calculation
- **Analysis Latency**: <50ms per indicator update
- **API Response Time**: <10ms for cached signal data
- **Notification Delivery**: <5 seconds from signal to Discord
- **System Throughput**: 20 symbols with 6 timeframes each

### Scaling Guidelines

**For 20+ Symbols:**
- Increase CPU allocation to 2+ cores
- Allocate 1GB+ memory
- Consider separate Redis instance
- Monitor disk I/O for logging

**For High-Frequency Trading:**
- Reduce timeframes to 1m, 5m only
- Increase Redis memory allocation
- Use SSD storage for faster data access
- Consider horizontal scaling with multiple instances

**Resource Requirements:**
- **Minimum**: 1 CPU core, 512MB RAM, Redis server
- **Recommended**: 2 CPU cores, 1GB RAM, dedicated Redis instance
- **High Scale**: 4+ CPU cores, 2GB+ RAM, Redis cluster

### Real-Time Signal Processing

- **Target: Sub-second Updates**: Signal detection processing frequency
- **Optimized Calculations**: Efficient updates for signal generation
- **Redis Streams**: Event-driven architecture for signal delivery
- **Circuit Breakers**: Fault tolerance for external data feeds
- **Horizontal Scaling**: Add workers for more symbols

### API Performance

- **Target Latency**: <50ms from market data to signal
- **Signal Queries**: <10ms for cached signals, <100ms for fresh analysis
- **Local Operations**: <10ms for cached operations
- **Stream Processing**: <10ms through Redis pipeline
- **Analysis Operations**: <1s for standard signal analysis

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
| Discord Integration | 95% | ✅ |
| **Overall** | **90%** | **✅** |

### Performance Benchmarks

- **Indicator Calculations**: <1s for 10K data points
- **Strategy Signal Generation**: <1s for 5K data points
- **Backtest Execution**: <5s for 1K data points
- **API Response Time**: <1s for standard operations
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

### Production Docker Configuration

The system includes a production-ready Docker setup with security best practices, health checks, and Alpaca credential validation.

**Multi-Stage Dockerfile Features:**
- Security-optimized build with non-root user
- Integrated Redis server for standalone operation
- Comprehensive health checks with Alpaca connectivity validation
- Proper signal handling for graceful shutdown
- Production logging and monitoring

```dockerfile
# Multi-stage production build
FROM python:3.9-slim as builder
RUN groupadd -r builduser && useradd -r -g builduser builduser

# Install dependencies with security updates
RUN apt-get update && apt-get install -y build-essential gcc g++ \
    && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Production stage
FROM python:3.9-slim
RUN apt-get update && apt-get install -y redis-server curl \
    && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Create secure app user
RUN groupadd -r wagehood && useradd -r -g wagehood wagehood

# Configure Redis for container
RUN echo "maxmemory 256mb" >> /etc/redis/redis.conf \
    && echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf

# Health check with Alpaca validation
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python docker-healthcheck.py || exit 1

USER wagehood
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["production"]
```

### Production Docker Compose

Complete production deployment with environment variable validation, resource limits, and comprehensive monitoring.

```yaml
services:
  wagehood:
    build: .
    image: wagehood:latest
    container_name: wagehood-trading
    restart: unless-stopped
    
    # MANDATORY: Alpaca credentials validation
    environment:
      - ALPACA_API_KEY=${ALPACA_API_KEY:?ALPACA_API_KEY environment variable is required}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY environment variable is required}
      - ALPACA_PAPER_TRADING=${ALPACA_PAPER_TRADING:-true}
      - ALPACA_DATA_FEED=${ALPACA_DATA_FEED:-iex}
      
      # Discord Multi-Channel Configuration
      - DISCORD_NOTIFICATIONS_ENABLED=${DISCORD_NOTIFICATIONS_ENABLED:-false}
      - DISCORD_MULTI_CHANNEL_ENABLED=${DISCORD_MULTI_CHANNEL_ENABLED:-false}
      - DISCORD_NOTIFY_TIMEFRAMES=${DISCORD_NOTIFY_TIMEFRAMES:-1d}
      - DISCORD_ALERT_SYMBOLS=${DISCORD_ALERT_SYMBOLS:-}
      - DISCORD_STRATEGY_TIMEFRAME_MAPPING=${DISCORD_STRATEGY_TIMEFRAME_MAPPING:-}
      
      # Strategy-Specific Webhooks
      - DISCORD_WEBHOOK_MACD_RSI=${DISCORD_WEBHOOK_MACD_RSI:-}
      - DISCORD_WEBHOOK_RSI_TREND=${DISCORD_WEBHOOK_RSI_TREND:-}
      - DISCORD_WEBHOOK_BOLLINGER_BREAKOUT=${DISCORD_WEBHOOK_BOLLINGER_BREAKOUT:-}
      - DISCORD_WEBHOOK_SR_BREAKOUT=${DISCORD_WEBHOOK_SR_BREAKOUT:-}
      
      # Individual Rate Limits
      - DISCORD_RATE_LIMIT_MACD_RSI=${DISCORD_RATE_LIMIT_MACD_RSI:-8}
      - DISCORD_RATE_LIMIT_RSI_TREND=${DISCORD_RATE_LIMIT_RSI_TREND:-12}
      - DISCORD_RATE_LIMIT_BOLLINGER_BREAKOUT=${DISCORD_RATE_LIMIT_BOLLINGER_BREAKOUT:-15}
      - DISCORD_RATE_LIMIT_SR_BREAKOUT=${DISCORD_RATE_LIMIT_SR_BREAKOUT:-6}
    
    # Resource limits for production
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    
    # Comprehensive health check
    healthcheck:
      test: ["CMD", "python", "docker-healthcheck.py"]
      interval: 60s
      timeout: 30s
      retries: 3
      start_period: 120s
    
    # Data persistence
    volumes:
      - wagehood-data:/app/data
      - wagehood-logs:/app/logs
    
    ports:
      - "6379:6379"  # Redis port

volumes:
  wagehood-data:
  wagehood-logs:
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

### Production Health Checks

Comprehensive health monitoring with Alpaca connectivity validation and system status verification.

```bash
# Docker health check (runs automatically)
docker exec wagehood-trading python docker-healthcheck.py

# Manual system verification
docker exec wagehood-trading python -c "
from src.strategies import create_strategy
from src.data.mock_generator import MockDataGenerator
from src.realtime.data_ingestion import MinimalAlpacaProvider
import asyncio
import os

# Test core system
strategy = create_strategy('macd_rsi')
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=10)
signals = strategy.generate_signals(data)
print(f'✅ Strategy system: Generated {len(signals)} signals')

# Test Alpaca connectivity
async def test_alpaca():
    config = {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'paper': True,
        'feed': 'iex'
    }
    provider = MinimalAlpacaProvider(config)
    await provider.connect()
    print('✅ Alpaca connectivity validated')

asyncio.run(test_alpaca())
"

# Check container resource usage
docker stats wagehood-trading --no-stream

# View real-time logs
docker logs -f wagehood-trading

# Redis connectivity test
docker exec wagehood-trading redis-cli ping
```

### Environment Variable Reference

**Required Variables:**
```bash
# MANDATORY: Alpaca Markets credentials
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
```

**Optional Configuration:**
```bash
# Alpaca Settings
ALPACA_PAPER_TRADING=true                    # Use paper trading
ALPACA_DATA_FEED=iex                         # Free IEX data feed

# System Performance
CALCULATION_WORKERS=4                        # Worker processes
MAX_CONCURRENT_CALCULATIONS=100              # Calculation limit
REDIS_STREAMS_MAXLEN=10000                   # Stream retention

# Discord Multi-Channel (All optional)
DISCORD_NOTIFICATIONS_ENABLED=false         # Enable notifications
DISCORD_MULTI_CHANNEL_ENABLED=false         # Multi-channel mode
DISCORD_ALERT_SYMBOLS=AAPL,MSFT,GOOGL       # Notification symbols
DISCORD_WEBHOOK_MACD_RSI=https://...         # Strategy webhooks
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

- **Credential Encryption**: API credentials encrypted with environment variables
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

#### Python API Connection Errors
```bash
# Test Python API connectivity
python -c "from src.core.models import OHLCV; print('Core API working')"

# Test strategy imports
python -c "from src.strategies import create_strategy; print('Strategies working')"

# Test data providers
python -c "from src.data.mock_generator import MockDataGenerator; print('Data providers working')"
```

#### No Data Processing
```bash
# Test data generation
python -c "
from src.data.mock_generator import MockDataGenerator
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=10)
print(f'Generated {len(data)} data points')
"

# Test strategy execution
python -c "
from src.strategies import create_strategy
from src.data.mock_generator import MockDataGenerator
strategy = create_strategy('macd_rsi')
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=100)
signals = strategy.generate_signals(data)
print(f'Generated {len(signals)} signals')
"
```

#### Alpaca Integration Issues
```bash
# Test Alpaca credentials
python -c "
import os
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
print(f'API Key: {\"Set\" if api_key else \"Not set\"}')
print(f'Secret Key: {\"Set\" if secret_key else \"Not set\"}')
"

# Test Alpaca data provider
python -c "
from src.data.providers.alpaca_provider import AlpacaProvider
import os
if os.getenv('ALPACA_API_KEY'):
    provider = AlpacaProvider()
    print('Alpaca provider initialized')
else:
    print('Alpaca credentials not found')
"
```

#### High Memory Usage
```bash
# Monitor Redis memory
redis-cli info memory

# Adjust stream retention
export REDIS_STREAMS_MAXLEN=5000

# Test system memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Current process memory: {memory_mb:.1f} MB')
"
```

#### Discord Notifications Not Working
```bash
# Test Discord webhook connectivity
curl -X POST "https://discord.com/api/webhooks/your_webhook_id/your_webhook_token" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test message from Wagehood"}'

# Check Discord configuration
python -c "
import os
print('Discord Config Status:')
print(f'  Notifications enabled: {os.getenv(\"DISCORD_NOTIFICATIONS_ENABLED\", \"false\")}')
print(f'  Multi-channel enabled: {os.getenv(\"DISCORD_MULTI_CHANNEL_ENABLED\", \"false\")}')
print(f'  Alert symbols: {os.getenv(\"DISCORD_ALERT_SYMBOLS\", \"not set\")}')
print(f'  Timeframes: {os.getenv(\"DISCORD_NOTIFY_TIMEFRAMES\", \"not set\")}')
"

# Test multi-channel notification service
docker exec wagehood-trading python -c "
from src.notifications.multi_channel_service import MultiChannelNotificationService
from src.notifications.multi_channel_config import MultiChannelNotificationConfig

config = MultiChannelNotificationConfig.from_environment()
print(f'Multi-channel config: enabled={config.enabled}, channels={len(config.strategy_channels)}')
"
```

#### Production Container Issues
```bash
# Check container status and logs
docker ps -a | grep wagehood
docker logs wagehood-trading --tail 50

# Test Alpaca credentials in container
docker exec wagehood-trading python -c "
import os
print('Alpaca Credentials Check:')
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
print(f'  API Key: {\"✅ SET\" if api_key else \"❌ MISSING\"}')
print(f'  Secret Key: {\"✅ SET\" if secret_key else \"❌ MISSING\"}')
if api_key and secret_key:
    print(f'  API Key length: {len(api_key)}')
    print(f'  Secret Key length: {len(secret_key)}')
"

# Check container resource usage
docker exec wagehood-trading python -c "
import psutil
import os
print('Container Resource Usage:')
print(f'  CPU usage: {psutil.cpu_percent()}%')
print(f'  Memory usage: {psutil.virtual_memory().percent}%')
print(f'  Disk usage: {psutil.disk_usage(\"/\").percent}%')
"
```

#### Performance Issues
```bash
# Check Redis performance
docker exec wagehood-trading redis-cli --latency-history

# Monitor stream processing
docker exec wagehood-trading redis-cli XINFO STREAM market_data_stream

# Check calculation engine performance
docker exec wagehood-trading python -c "
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.config_manager import ConfigManager
from src.storage.cache import cache_manager

config = ConfigManager()
print('System Performance Check:')
print(f'  Enabled symbols: {len(config.get_enabled_symbols())}')
print(f'  Redis connection: {\"✅ OK\" if cache_manager.redis_client.ping() else \"❌ FAILED\"}')
"
```

### Debug Mode

```bash
# Enable verbose logging for Python API
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.strategies import create_strategy
from src.data.mock_generator import MockDataGenerator
strategy = create_strategy('macd_rsi')
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=50)
signals = strategy.generate_signals(data)
print(f'Debug: Generated {len(signals)} signals')
"

# Test with detailed error handling
python -c "
try:
    from src.backtest.engine import BacktestEngine
    print('BacktestEngine imported successfully')
except Exception as e:
    print(f'Error importing BacktestEngine: {e}')
"
```

### Performance Optimization

1. **Use caching**: Redis caches recent data for faster responses
2. **Batch operations**: Process multiple symbols together with worker service
3. **Limit data size**: Use reasonable periods (e.g., 252 days) for backtesting
4. **Optimize calculations**: Use NumPy arrays when available for indicator calculations
5. **Optimize Redis**: Tune memory and connection settings for worker service

## 🆕 Recent Feature Updates

### Strategy-Timeframe Mapping

Automatic routing of trading signals based on strategy and timeframe combinations:

```bash
# Configure strategy-timeframe mappings
DISCORD_STRATEGY_TIMEFRAME_MAPPING=macd_rsi:1d,sr_breakout:1d,rsi_trend:1h,bollinger_breakout:1h
```

**Mapping Logic:**
- **MACD+RSI**: 1d timeframe → Swing trading channel
- **Support/Resistance**: 1d timeframe → Swing trading channel  
- **RSI Trend**: 1h timeframe → Day trading channel
- **Bollinger Breakout**: 1h timeframe → Day trading channel

### Separate Alert Symbol Lists

Independent configuration of symbols for notifications vs main processing:

```bash
# Main watchlist for processing (all symbols)
WATCHLIST_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,SPY,QQQ,IWM,AMZN,NVDA,META

# Focused list for Discord notifications (high-priority symbols only)
DISCORD_ALERT_SYMBOLS=AAPL,MSFT,GOOGL,TSLA,SPY
```

**Benefits:**
- Process comprehensive symbol list for analysis
- Limit notifications to key trading symbols
- Reduce notification noise and focus on priority symbols
- Separate day trading symbols from swing trading symbols

### Multi-Channel Webhook Configuration

Strategy-specific webhook routing with individual rate limiting:

```bash
# Swing Trading Strategies (1d timeframe)
DISCORD_WEBHOOK_MACD_RSI=https://discord.com/api/webhooks/swing-macd-rsi/token
DISCORD_RATE_LIMIT_MACD_RSI=8                 # 8 notifications/hour

DISCORD_WEBHOOK_SR_BREAKOUT=https://discord.com/api/webhooks/swing-sr/token
DISCORD_RATE_LIMIT_SR_BREAKOUT=6              # 6 notifications/hour

# Day Trading Strategies (1h timeframe)
DISCORD_WEBHOOK_RSI_TREND=https://discord.com/api/webhooks/day-rsi/token
DISCORD_RATE_LIMIT_RSI_TREND=12               # 12 notifications/hour

DISCORD_WEBHOOK_BOLLINGER_BREAKOUT=https://discord.com/api/webhooks/day-bollinger/token
DISCORD_RATE_LIMIT_BOLLINGER_BREAKOUT=15      # 15 notifications/hour
```

### Enhanced Production Deployment

**Docker Container Features:**
- Integrated Redis server for standalone operation
- Mandatory Alpaca credential validation
- Comprehensive health checks with connectivity testing
- Security-hardened container with non-root user
- Graceful shutdown handling and resource management

**Key Improvements:**
- Credential validation prevents container startup without valid API keys
- Health checks verify Alpaca connectivity and system functionality
- Resource limits prevent runaway resource consumption
- Persistent volumes for data and log storage
- Production-ready logging and monitoring

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

This software is for educational and research purposes only. **This is a signal detection system that does not execute trades or manage portfolios.** All signals are for analysis purposes only. The authors are not responsible for any financial decisions made based on the signals generated by this software.

**Signal detection involves analysis of historical patterns that may not predict future market behavior. Always conduct thorough testing and consider your risk tolerance before making any trading decisions based on these signals.**

---

**For systematic traders and quantitative researchers**

*For additional support and updates, please refer to the project documentation and community forums.*