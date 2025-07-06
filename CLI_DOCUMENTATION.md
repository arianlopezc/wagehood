# Wagehood Python API Documentation

Reference for the Wagehood Python API with multi-strategy multi-timeframe capabilities.

**Note:** The system has transitioned from CLI-based tools to a Python API for integration and flexibility.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Commands](#core-commands)
3. [Watchlist Management](#watchlist-management)
4. [Signal Analysis](#signal-analysis)
5. [Multi-Dimensional Analysis](#multi-dimensional-analysis)
6. [Performance Analysis](#performance-analysis)
7. [Trading Profiles](#trading-profiles)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Python 3.9+
- Redis server (for real-time features)
- Project dependencies installed

### Basic Workflow
```python
# 1. Import required modules
from src.strategies import create_strategy, get_all_strategies
from src.data.mock_generator import MockDataGenerator
from src.backtest.engine import BacktestEngine

# 2. Generate market data
generator = MockDataGenerator()
data = generator.generate_realistic_data('SPY', periods=252)

# 3. Create and test a strategy
strategy = create_strategy('macd_rsi')
engine = BacktestEngine()
result = engine.run_backtest(strategy, data, initial_capital=10000)

# 4. View results
print(f"Total Return: {result.performance_metrics.total_return_pct:.2%}")
print(f"Win Rate: {result.performance_metrics.win_rate:.1%}")
print(f"Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.2f}")

# 5. Explore all strategies
strategies = get_all_strategies()
for name, metadata in strategies.items():
    print(f"{metadata['name']}: {metadata['description']}")
```

## Core Commands

### System Status
```bash
# Show system overview with active components
python market_analysis_cli.py status
```

**Output:**
- Active watchlist symbols
- Enabled strategies
- Portfolio overview
- System health

### Real-Time Monitoring
```bash
# Watch live updates (refreshes every 5 seconds)
python market_analysis_cli.py watch
```

**Features:**
- Latest trading signals
- Portfolio statistics
- Auto-refresh display
- Press Ctrl+C to stop

### Position Monitoring
```bash
# View current open positions
python market_analysis_cli.py positions
```

**Information:**
- Symbol and quantity
- Entry and current prices
- Unrealized P&L
- Percentage gains/losses

## Watchlist Management

The watchlist system supports multi-strategy multi-timeframe configuration with three trading profiles.

### Add Symbols

**Basic Addition:**
```bash
# Add with default swing trading profile
python market_analysis_cli.py watchlist add SPY

# Add with specific trading profile
python market_analysis_cli.py watchlist add AAPL --trading-profile day

# Add with custom strategies
python market_analysis_cli.py watchlist add TSLA --trading-profile swing --strategies macd_rsi,rsi_trend
```

**Custom Addition:**
```bash
# Add with custom timeframes
python market_analysis_cli.py watchlist add QQQ --trading-profile swing --timeframes 1h,4h,1d

# Add with priority and notes
python market_analysis_cli.py watchlist add MSFT --trading-profile position --priority 1 --notes "Blue chip tech stock"

# Add with all parameters
python market_analysis_cli.py watchlist add NVDA \
  --trading-profile day \
  --strategies rsi_trend,bollinger_breakout \
  --timeframes 5m,15m,30m \
  --priority 2 \
  --notes "High volatility semiconductor"
```

### List Watchlist

**Basic Listing:**
```bash
# List all symbols in table format
python market_analysis_cli.py watchlist list

# Detailed view with full information
python market_analysis_cli.py watchlist list --detailed
```

**Filtered Listing:**
```bash
# Filter by trading profile
python market_analysis_cli.py watchlist list --trading-profile day
python market_analysis_cli.py watchlist list --trading-profile swing
python market_analysis_cli.py watchlist list --trading-profile position

# Filter by strategy
python market_analysis_cli.py watchlist list --strategy macd_rsi
python market_analysis_cli.py watchlist list --strategy ma_crossover
```

### Configure Symbols

**Trading Profile Changes:**
```bash
# Change trading profile (updates timeframes automatically)
python market_analysis_cli.py watchlist configure AAPL --trading-profile position
```

**Strategy Management:**
```bash
# Add strategy to existing symbol
python market_analysis_cli.py watchlist configure SPY --add-strategy bollinger_breakout

# Remove strategy from symbol
python market_analysis_cli.py watchlist configure SPY --remove-strategy sr_breakout
```

**Timeframe Management:**
```bash
# Add custom timeframe
python market_analysis_cli.py watchlist configure TSLA --add-timeframe 1h

# Remove timeframe
python market_analysis_cli.py watchlist configure TSLA --remove-timeframe 5m
```

**Other Settings:**
```bash
# Update priority
python market_analysis_cli.py watchlist configure AAPL --priority 1

# Update notes
python market_analysis_cli.py watchlist configure AAPL --notes "Updated analysis focus"
```

### Remove Symbols
```bash
# Remove symbol from watchlist
python market_analysis_cli.py watchlist remove AAPL
```

## Signal Analysis

### Multi-Dimensional Signals

**Basic Signal Views:**
```bash
# Get all signals for a symbol
python market_analysis_cli.py signals multi --symbol SPY

# Simple signal view for quick overview
python market_analysis_cli.py signals-simple --symbol SPY --limit 10
```

**Filtering Options:**
```bash
# Filter by timeframe
python market_analysis_cli.py signals multi --timeframe 1h --limit 20

# Filter by strategy
python market_analysis_cli.py signals multi --strategy macd_rsi --limit 15

# Filter by signal type
python market_analysis_cli.py signals multi --signal-type BUY --limit 25

# Combined filters
python market_analysis_cli.py signals multi \
  --symbol AAPL \
  --timeframe 4h \
  --strategy rsi_trend \
  --signal-type BUY \
  --limit 10
```

**Signal Output:**
- Timestamp and symbol
- Timeframe and strategy
- Signal type (BUY/SELL/HOLD)
- Confidence percentage
- Current price
- Signal distribution summary

## Multi-Dimensional Analysis

### Analysis Commands

**Basic Analysis:**
```bash
# Analyze symbol using watchlist configuration
python market_analysis_cli.py analyze multi SPY

# Quick analysis with default settings
python market_analysis_cli.py analyze-simple AAPL
```

**Custom Analysis:**
```bash
# Specify strategies and timeframes
python market_analysis_cli.py analyze multi TSLA \
  --strategies macd_rsi,rsi_trend,bollinger_breakout \
  --timeframes 15m,1h,4h
```

**Analysis Output:**
- Signal matrix across strategies and timeframes
- Color-coded signal types with confidence
- Strategy metadata and descriptions
- Best conditions for each strategy

### Signal Matrix Interpretation

The analysis displays a matrix showing:
- **Rows:** Trading strategies
- **Columns:** Timeframes
- **Cells:** Latest signal with confidence

**Color Coding:**
- 🟢 **Green (BUY):** Bullish signal
- 🔴 **Red (SELL):** Bearish signal  
- 🟡 **Yellow (HOLD):** Neutral signal
- **Gray:** No data available

**Confidence Levels:**
- **80-100%:** Very high confidence
- **60-79%:** High confidence
- **40-59%:** Moderate confidence
- **20-39%:** Low confidence
- **0-19%:** Very low confidence

## Performance Analysis

### Performance Matrix

**Matrix View:**
```bash
# View performance across all strategies and trading profiles
python market_analysis_cli.py performance --matrix
```

**Matrix Output:**
- Strategies vs Trading Profiles
- Win rate and profit factor for each combination
- Color-coded performance levels
- Performance legend

**Performance Color Coding:**
- 🟢 **Green:** Good (>60% win rate, >1.5 profit factor)
- 🟡 **Yellow:** Acceptable (>50% win rate, >1.2 profit factor)
- 🔴 **Red:** Needs improvement

### Individual Strategy Performance

**Detailed Performance:**
```bash
# Specific strategy performance
python market_analysis_cli.py performance --strategy macd_rsi

# Simple performance overview
python market_analysis_cli.py performance-simple --strategy ma_crossover
```

**Performance Metrics:**
- Total signals generated
- Win rate percentage
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Average win/loss amounts

## Trading Profiles

### Day Trading Profile

**Characteristics:**
- **Timeframes:** 1m, 5m, 15m
- **Capital Requirement:** $25,000+ (PDT rule)
- **Time Commitment:** Active monitoring required
- **Risk Level:** High
- **Best Strategies:** RSI Trend, Bollinger Breakout

**Setup Example:**
```bash
python market_analysis_cli.py watchlist add AAPL --trading-profile day --strategies rsi_trend,bollinger_breakout
python market_analysis_cli.py watchlist add TSLA --trading-profile day --strategies rsi_trend,bollinger_breakout
```

### Swing Trading Profile

**Characteristics:**
- **Timeframes:** 30m, 1h, 4h
- **Capital Requirement:** $5,000-$25,000
- **Time Commitment:** Daily monitoring
- **Risk Level:** Medium
- **Best Strategies:** MACD+RSI, RSI Trend, Bollinger Breakout

**Setup Example:**
```bash
python market_analysis_cli.py watchlist add SPY --trading-profile swing --strategies macd_rsi,rsi_trend
python market_analysis_cli.py watchlist add QQQ --trading-profile swing --strategies macd_rsi,bollinger_breakout
```

### Position Trading Profile

**Characteristics:**
- **Timeframes:** 1d, 1w, 1M
- **Capital Requirement:** $1,000+
- **Time Commitment:** Weekly monitoring
- **Risk Level:** Low
- **Best Strategies:** Moving Average Crossover, Support/Resistance Breakout

**Setup Example:**
```bash
python market_analysis_cli.py watchlist add VTI --trading-profile position --strategies ma_crossover,sr_breakout
python market_analysis_cli.py watchlist add SCHD --trading-profile position --strategies ma_crossover
```

## Best Practices

### Watchlist Management

1. **Start Simple:** Begin with one trading profile and a few symbols
2. **Match Profile to Lifestyle:** Choose profile based on available time
3. **Use Priority System:** Set priority 1 for most important symbols
4. **Regular Review:** Use `watchlist list --detailed` to review configurations
5. **Profile Consistency:** Keep similar symbols in the same trading profile

### Strategy Selection

1. **Beginner Strategies:** Start with Moving Average Crossover
2. **Intermediate Strategies:** Progress to MACD+RSI and RSI Trend
3. **Complex Strategies:** Use Support/Resistance Breakout when experienced
4. **Profile Matching:** Match strategies to trading profile timeframes
5. **Performance Monitoring:** Regularly check performance matrix

### Signal Analysis

1. **Multi-Timeframe Confirmation:** Look for agreement across timeframes
2. **High Confidence Signals:** Focus on signals >60% confidence
3. **Strategy Consensus:** Multiple strategies signaling same direction
4. **Volume Confirmation:** Verify with volume-based strategies
5. **Market Context:** Consider overall market conditions

### Risk Management

1. **Position Sizing:** Never risk more than 2% per trade
2. **Diversification:** Use multiple symbols and strategies
3. **Stop Losses:** Always set stop losses based on strategy signals
4. **Paper Trading:** Test new configurations with paper trading first
5. **Regular Review:** Weekly performance and configuration review

## Troubleshooting

### Common Issues

**No Signals Showing:**
```bash
# Check system status
python market_analysis_cli.py status

# Verify Redis is running
redis-cli ping

# Check if worker service is running
ps aux | grep run_realtime.py
```

**Empty Watchlist:**
```bash
# Check if watchlist exists
python market_analysis_cli.py watchlist list

# Add symbols if empty
python market_analysis_cli.py watchlist add SPY --trading-profile swing
```

**Performance Data Missing:**
```bash
# Verify strategies are running
python market_analysis_cli.py status

# Check if enough time has passed for data collection
# (Allow at least 1 hour for initial data)
```

### Error Messages

**"Strategy not found":**
- Check strategy spelling
- Use `python market_analysis_cli.py --help` for available strategies
- Valid strategies: `macd_rsi`, `ma_crossover`, `rsi_trend`, `bollinger_breakout`, `sr_breakout`

**"Invalid timeframe":**
- Check timeframe format
- Valid timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1M`

**"Redis connection failed":**
- Ensure Redis server is running: `redis-server`
- Check Redis configuration in environment variables

### Performance Issues

**Slow CLI Response:**
- Redis may be overloaded - restart Redis server
- Reduce number of symbols in watchlist
- Use `--limit` parameter to reduce output size

**High Memory Usage:**
- Monitor Redis memory: `redis-cli info memory`
- Adjust `REDIS_STREAMS_MAXLEN` environment variable
- Restart worker service periodically

### Getting Help

**Command Help:**
```bash
# General help
python market_analysis_cli.py --help

# Command-specific help
python market_analysis_cli.py watchlist --help
python market_analysis_cli.py signals --help
python market_analysis_cli.py analyze --help
python market_analysis_cli.py performance --help
```

**System Information:**
```bash
# Check system status
python market_analysis_cli.py status

# Monitor real-time updates
python market_analysis_cli.py watch
```

---

For additional support, please refer to the main README.md and project documentation.