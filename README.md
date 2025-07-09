# Wagehood Signal Detection System

**A distributed signal detection system for real-time market analysis with multi-strategy signal generation and comprehensive analysis capabilities.**

> **⚠️ IMPORTANT DISCLAIMER**: This is a signal detection system only. It does not execute trades or manage portfolios. All signals are for analysis and educational purposes only.

## 🚀 Overview

Wagehood is a high-performance signal detection system designed for systematic traders and quantitative researchers. It provides real-time multi-strategy signal analysis across multiple timeframes with advanced filtering and notification capabilities.

### Key Features

- **Multi-Strategy Signal Engine** - 5 sophisticated signal detection strategies
- **Distributed Architecture** - Redis-based worker service with horizontal scaling
- **Real-Time Processing** - Sub-second signal detection with event-driven architecture
- **Multi-Timeframe Analysis** - Simultaneous analysis across 9 timeframes
- **Advanced Filtering** - Strategy-specific timeframe mapping and confidence scoring
- **Multi-Channel Notifications** - Discord integration with strategy-specific routing
- **Comprehensive Analysis** - Signal quality assessment and performance metrics
- **Production-Ready** - Docker deployment with health checks and monitoring

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

## 🎯 Signal Generation Conditions

Each strategy generates buy/sell signals based on specific technical conditions:

### MACD+RSI Combined Strategy
**Buy Signal Conditions:**
- MACD line crosses above signal line (bullish crossover)
- RSI exits oversold territory (above 30) OR RSI is above 50 with positive momentum
- Volume exceeds 1.3x average volume (confirmation)
- MACD histogram shows positive momentum

**Sell Signal Conditions:**
- MACD line crosses below signal line (bearish crossover)
- RSI exits overbought territory (below 70) OR RSI is below 50 with negative momentum
- Volume exceeds 1.3x average volume (confirmation)
- MACD histogram shows negative momentum

**Confidence Factors:**
- MACD signal strength (20%), RSI position (20%), histogram momentum (15%)
- Volume confirmation (15%), price momentum (15%), trend alignment (10%)
- Signal timing quality (5%)

### RSI Trend Following Strategy
**Buy Signal Conditions:**
- Primary RSI (14-period) in pullback zone (40-50) during uptrend
- RSI turns upward from pullback zone
- Trend confirmed by 70% of periods with RSI above 50
- Volume above average for confirmation

**Sell Signal Conditions:**
- Primary RSI (14-period) in rally zone (50-60) during downtrend
- RSI turns downward from rally zone
- Trend confirmed by 70% of periods with RSI below 50
- Volume above average for confirmation

**Confidence Factors:**
- RSI position relative to trend zones, trend strength validation
- RSI momentum direction, price momentum confirmation

### Bollinger Band Breakout Strategy
**Buy Signal Conditions:**
- Price closes above upper Bollinger Band (20-period, 2 std dev)
- High of the period breaks above upper band
- Volume significantly above 20-period average
- Band width expansion after consolidation (squeeze)

**Sell Signal Conditions:**
- Price closes below lower Bollinger Band (20-period, 2 std dev)
- Low of the period breaks below lower band
- Volume significantly above 20-period average
- Band width expansion after consolidation (squeeze)

**Confidence Factors:**
- Breakout strength (30%), band width expansion (20%)
- Volume confirmation (30%), price momentum (20%)

### Support/Resistance Breakout Strategy
**Buy Signal Conditions:**
- Price closes above identified resistance level
- High of the period breaks above resistance level
- Resistance level validated by minimum 2 touches
- Volume exceeds 20-period average
- Level strength confirmed by touch count and hold ratio

**Sell Signal Conditions:**
- Price closes below identified support level
- Low of the period breaks below support level
- Support level validated by minimum 2 touches
- Volume exceeds 20-period average
- Level strength confirmed by touch count and hold ratio

**Confidence Factors:**
- Level strength (30%), breakout strength (40%)
- Volume confirmation (20%), touch count validation (10%)

### Moving Average Crossover Strategy
**Buy Signal Conditions:**
- Short EMA (50-period) crosses above long EMA (200-period) - Golden Cross
- Price closes above both EMAs
- Volume exceeds 1.2x average volume
- EMA separation indicates strong momentum

**Sell Signal Conditions:**
- Short EMA (50-period) crosses below long EMA (200-period) - Death Cross
- Price closes below both EMAs
- Volume exceeds 1.2x average volume
- EMA separation indicates strong momentum

**Confidence Factors:**
- EMA separation strength (40%), volume confirmation (30%)
- Trend strength and momentum (30%)

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

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │───▶│ Redis Streams   │───▶│ Signal Engine   │
│ (Alpaca/Mock)   │    │ (Event Bus)     │    │ (Multi-Strategy)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Notifications   │◀───│ Analysis Engine │◀───│ Multi-Timeframe │
│ (Discord/API)   │    │ (Filtering)     │    │ Processor       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

| Component | Responsibility | Key Features |
|-----------|---------------|--------------|
| **Data Layer** | Market data ingestion and validation | Alpaca integration, mock data, WebSocket streams |
| **Signal Engine** | Multi-strategy signal detection | 5 strategies, confidence scoring, real-time processing |
| **Analysis Engine** | Signal filtering and evaluation | Quality assessment, timeframe alignment, performance metrics |
| **Notification System** | Multi-channel alert routing | Discord webhooks, strategy-specific channels, rate limiting |
| **Storage Layer** | Results caching and persistence | Redis streams, indicator caching, historical analysis |

### Distributed Processing

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Ingestion  │    │ Signal Workers  │    │ Analysis Workers│
│ (Primary)       │    │ (Horizontal     │    │ (Background)    │
│                 │    │  Scaling)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Redis Message   │
                    │ Bus & Cache     │
                    └─────────────────┘
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

**System Initialization:**
1. **Dependencies**: Install Python packages and Redis server
2. **Configuration**: Set up environment variables for data providers
3. **Testing**: Run comprehensive test suite to verify all components
4. **API Access**: Use Python API for signal analysis and multi-strategy comparison

**Core Workflow:**
- **Data Pipeline**: Market data → Redis Streams → Signal Engine
- **Multi-Strategy Processing**: 5 strategies across 9 timeframes simultaneously
- **Signal Generation**: Real-time signal detection with confidence scoring
- **Quality Assessment**: Advanced filtering and signal validation

**Available Strategies:**
- MACD+RSI Combined Signal Detection
- RSI Trend Following with Momentum
- Bollinger Band Breakout Detection
- Support/Resistance Level Analysis
- Moving Average Crossover Signals

**Setup Steps:**

1. **Install dependencies** and set up Python environment
2. **Start Redis server** for real-time data processing
3. **Configure environment** variables for data providers
4. **Run comprehensive tests** to verify all components
5. **Use Python API** for signal analysis and strategy testing
6. **Explore strategies** through the strategy registry system

## 🖥️ Python API Interface

The Wagehood system provides a Python API for multi-strategy multi-timeframe signal analysis and a Job Submission CLI for signal analysis against the running production instance.

## 📋 Job Submission CLI

**Architecture**: Submit signal analysis jobs to the running production instance and get detailed results with all signals and analysis metrics.

**Key Features:**
- **Single Command Interface**: Submit, monitor, and view results in one command
- **Real-time Progress**: Live progress updates with visual progress bar
- **Comprehensive Results**: All signals, performance metrics, and quality analysis
- **Production Integration**: Uses running Docker instance for analysis
- **Multi-Strategy Support**: Run analysis across all 5 available strategies
- **Multi-Timeframe Analysis**: Simultaneous analysis across 9 timeframes

See [Job Submission CLI Documentation](docs/JOB_SUBMISSION_CLI.md) for complete usage guide.

### Core Python API

**Strategy Analysis Architecture:**
- **Strategy Registry**: Centralized registry of all 5 signal detection strategies
- **Data Generation**: Mock data generator for testing and validation
- **Signal Analysis Engine**: Core engine for multi-strategy signal processing
- **Performance Evaluation**: Comprehensive metrics and quality assessment

**Key Components:**
- **Multi-Strategy Processing**: Simultaneous analysis across all strategies
- **Timeframe Management**: Automatic handling of 9 different timeframes
- **Signal Quality Scoring**: Advanced confidence and reliability metrics
- **Real-time Processing**: Redis-based event-driven architecture

**Multi-Strategy Comparison:**
- **Strategy Comparator**: Advanced comparison engine for multi-strategy analysis
- **Performance Metrics**: Comprehensive evaluation across all signal quality dimensions
- **Ranking System**: Automated ranking based on composite scoring algorithms
- **Quality Assessment**: Signal frequency, confidence, and reliability analysis

**Real-time Data Processing:**
- **Data Ingestion Service**: Multi-source data ingestion with Redis Streams
- **Signal Detection Engine**: Real-time multi-strategy signal processing
- **Configuration Management**: Dynamic configuration for symbols, timeframes, and strategies
- **Event-Driven Architecture**: Scalable processing with horizontal worker scaling

### Testing and Validation

**Testing Architecture:**
- **Comprehensive Test Suite**: Unit, integration, and performance testing
- **Strategy Validation**: Automated validation of all signal detection strategies
- **Coverage Analysis**: Code coverage reporting and quality metrics
- **Performance Benchmarking**: Execution time and resource usage testing

**Test Categories:**
- **Unit Tests**: Individual component testing and validation
- **Integration Tests**: End-to-end system testing with real data flows
- **Performance Tests**: Throughput, latency, and resource usage validation
- **Strategy Tests**: Signal generation quality and accuracy validation

**Validation Features:**
- **Strategy Registry Validation**: Automated testing of all 5 strategies
- **Signal Quality Assessment**: Comprehensive signal validation and scoring
- **Data Integrity Checks**: Market data validation and consistency testing
- **Performance Metrics**: Execution time, memory usage, and throughput analysis

### Usage Examples

#### Multi-Strategy Portfolio Analysis

**Architecture Pattern**: Multi-dimensional analysis across symbols and strategies

**Key Components:**
- **Data Generation**: Realistic market data simulation for multiple symbols
- **Strategy Matrix**: Systematic analysis across all 5 available strategies
- **Performance Aggregation**: Comprehensive metrics collection and comparison
- **Results Analysis**: Automated ranking and performance evaluation

**Analysis Dimensions:**
- **Symbol Analysis**: Multi-symbol portfolio evaluation
- **Strategy Comparison**: Performance comparison across all strategies
- **Risk Assessment**: Drawdown, volatility, and risk-adjusted returns
- **Quality Metrics**: Signal frequency, confidence, and reliability scoring

#### Real-time Signal Generation

**Architecture Pattern**: Multi-timeframe real-time signal processing

**Key Components:**
- **Signal Engine**: Core real-time signal processing engine
- **Configuration Manager**: Dynamic configuration for symbols and strategies
- **Timeframe Profiles**: Pre-configured timeframe sets for different trading styles
- **Multi-Symbol Processing**: Simultaneous processing across multiple symbols

**Trading Profiles:**
- **Day Trading**: 1m, 5m, 15m timeframes with high-frequency strategies
- **Swing Trading**: 30m, 1h, 4h timeframes with medium-term strategies
- **Position Trading**: 1d, 1w, 1M timeframes with long-term strategies

**Signal Processing:**
- **Real-time Data Ingestion**: Continuous market data processing
- **Multi-Strategy Execution**: Simultaneous signal generation across strategies
- **Quality Filtering**: Advanced signal filtering and confidence scoring
- **Performance Optimization**: Efficient processing with horizontal scaling

#### Custom Strategy Development

**Architecture Pattern**: Extensible strategy framework for custom signal logic

**Key Components:**
- **Base Strategy Class**: Abstract base class with standardized interface
- **Signal Models**: Structured signal representation with confidence scoring
- **Indicator Integration**: Seamless integration with technical indicator library
- **Performance Validation**: Automated testing and validation framework

**Development Process:**
- **Strategy Inheritance**: Extend TradingStrategy base class
- **Indicator Integration**: Use built-in RSI, EMA, MACD, and other calculators
- **Signal Generation**: Implement custom signal logic with confidence scoring
- **Testing Framework**: Automated validation with realistic market data

**Custom Strategy Features:**
- **Multi-Indicator Combinations**: Combine multiple technical indicators
- **Confidence Scoring**: Dynamic confidence calculation based on signal strength
- **Signal Filtering**: Advanced filtering based on market conditions
- **Performance Metrics**: Comprehensive performance evaluation and comparison

#### Data Analysis and Backtesting

**Architecture Pattern**: Comprehensive signal analysis across multiple market conditions

**Key Components:**
- **Multi-Condition Analysis**: Systematic testing across different market scenarios
- **Performance Aggregation**: Comprehensive metrics collection and comparison
- **Market Condition Simulation**: Realistic market data generation for various conditions
- **Results Evaluation**: Automated analysis and performance ranking

**Market Condition Testing:**
- **Bull Market**: Upward trending market with moderate volatility
- **Bear Market**: Downward trending market with higher volatility
- **Sideways Market**: Range-bound market with low volatility
- **High Volatility**: Neutral trend with elevated volatility

**Analysis Features:**
- **Strategy Robustness**: Performance validation across market conditions
- **Risk Assessment**: Comprehensive risk metrics for different scenarios
- **Performance Comparison**: Comparative analysis across strategies and conditions
- **Quality Scoring**: Signal quality evaluation under varying market conditions

#### Performance Comparison and Optimization

**Architecture Pattern**: Systematic parameter optimization and performance comparison

**Key Components:**
- **Grid Search Optimization**: Systematic parameter space exploration
- **Performance Comparison**: Multi-strategy performance evaluation
- **Parameter Tuning**: Automated optimization across parameter ranges
- **Results Validation**: Statistical validation of optimization results

**Optimization Features:**
- **Multi-Parameter Optimization**: Simultaneous optimization across multiple parameters
- **Cross-Validation**: Performance validation across different data periods
- **Overfitting Prevention**: Robust validation to prevent parameter overfitting
- **Performance Metrics**: Comprehensive evaluation beyond simple returns

**Parameter Ranges:**
- **RSI Parameters**: Period, oversold/overbought thresholds
- **MACD Parameters**: Fast/slow periods, signal smoothing
- **Moving Average Parameters**: Period selection and type optimization
- **Bollinger Band Parameters**: Period and standard deviation multipliers

### Data Analysis Output

**Architecture Pattern**: Structured data objects for comprehensive signal analysis

**Key Components:**
- **Performance Metrics**: Comprehensive signal performance evaluation
- **Signal Quality Assessment**: Confidence scoring and reliability metrics
- **Risk Assessment**: Drawdown analysis and risk-adjusted returns
- **Comparative Analysis**: Multi-strategy performance comparison

**Data Structure Features:**
- **Signal Quality Metrics**: Confidence scoring, frequency analysis, reliability assessment
- **Performance Evaluation**: Return analysis, win rate calculation, risk metrics
- **Temporal Analysis**: Signal timing, frequency distribution, consistency metrics
- **Risk Metrics**: Drawdown analysis, volatility assessment, risk-adjusted performance

## 📊 Strategy Analysis

The system provides comprehensive strategy analysis through the Python API, helping traders determine which strategies work best for their specific trading style and market conditions.

### Trading Style Analysis

**Architecture Pattern**: Multi-dimensional strategy analysis across trading styles

**Key Components:**
- **Strategy Analyzer**: Comprehensive analysis engine for trading style evaluation
- **Style Classification**: Systematic categorization of trading approaches
- **Performance Comparison**: Multi-style performance evaluation and ranking
- **Adaptive Configuration**: Dynamic parameter adjustment for different styles

**Trading Style Categories:**
- **Day Trading**: High-frequency signals with short holding periods
- **Swing Trading**: Medium-term signals with multi-day holding periods
- **Position Trading**: Long-term signals with extended holding periods

**Analysis Features:**
- **Style-Specific Metrics**: Performance evaluation tailored to each trading style
- **Comparative Analysis**: Cross-style performance comparison and ranking
- **Risk Assessment**: Style-specific risk metrics and drawdown analysis
- **Optimization Recommendations**: Strategy parameter suggestions for each style

### Performance Evaluation

**Architecture Pattern**: Comprehensive performance metrics for strategy evaluation

**Key Components:**
- **Performance Metrics Engine**: Comprehensive evaluation of signal quality and performance
- **Multi-Strategy Evaluation**: Systematic comparison across all available strategies
- **Risk Assessment**: Advanced risk metrics and drawdown analysis
- **Quality Scoring**: Signal confidence and reliability assessment

**Performance Metrics:**
- **Signal Quality**: Win rate, signal frequency, confidence scoring
- **Return Analysis**: Total return, risk-adjusted returns, consistency metrics
- **Risk Metrics**: Maximum drawdown, volatility, Sharpe ratio
- **Trade Analysis**: Signal frequency, profit factor, consistency evaluation

**Evaluation Features:**
- **Multi-Symbol Analysis**: Performance evaluation across different asset classes
- **Time-Series Analysis**: Performance consistency over different time periods
- **Comparative Ranking**: Automated ranking based on composite scoring
- **Risk-Adjusted Performance**: Sharpe ratio, Sortino ratio, and other risk metrics

## 📖 Strategy Documentation & Explanations

### Understanding Strategy Logic

The system provides comprehensive strategy documentation through the Python API. Each strategy includes:

- **Signal Generation Logic**: Exact conditions for buy/sell signals
- **Parameter Configuration**: Default values with descriptions and ranges
- **Confidence Calculation**: How signal confidence scores are computed
- **Special Features**: Unique capabilities and advantages
- **Usage Guidelines**: Best trading styles and market conditions

### Accessing Strategy Information

**Architecture Pattern**: Comprehensive strategy documentation and metadata access

**Key Components:**
- **Strategy Registry**: Centralized repository of all available strategies
- **Metadata Access**: Comprehensive strategy documentation and parameter information
- **Dynamic Configuration**: Parameter inspection and validation
- **Usage Guidelines**: Best practices and implementation recommendations

**Information Categories:**
- **Strategy Description**: Detailed explanation of signal generation logic
- **Parameter Documentation**: Default values, ranges, and optimization guidelines
- **Performance Characteristics**: Expected performance metrics and risk profiles
- **Implementation Details**: Technical implementation and computational requirements

### Available Strategy Information

| Strategy | Key | Focus | Implementation |
|----------|-----|-------|----------------|
| **MACD + RSI Combined** | `macd_rsi` | Momentum + timing | High-performance momentum strategy |
| **Moving Average Crossover** | `ma_crossover` | Trend following | Classic trend-following approach |
| **RSI Trend Following** | `rsi_trend` | Trend + pullbacks | Trend-aware RSI with pullbacks |
| **Bollinger Band Breakout** | `bollinger_breakout` | Volatility expansion | Volatility expansion strategy |
| **Support/Resistance Breakout** | `sr_breakout` | Key level trading | Level-based breakout trading |

### Strategy Parameter Inspection

**Architecture Pattern**: Dynamic strategy parameter inspection and signal analysis

**Key Components:**
- **Parameter Inspection**: Dynamic analysis of strategy parameters and defaults
- **Signal Generation Analysis**: Comprehensive signal behavior evaluation
- **Distribution Analysis**: Statistical analysis of signal patterns and types
- **Performance Validation**: Signal quality assessment and validation

**Inspection Features:**
- **Parameter Analysis**: Default values, ranges, and optimization potential
- **Signal Distribution**: Statistical analysis of signal frequency and types
- **Confidence Assessment**: Signal confidence scoring and reliability metrics
- **Temporal Analysis**: Signal timing distribution and frequency patterns

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