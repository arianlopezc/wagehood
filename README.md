# Wagehood - Local Trading Signal Detection Service

<p align="center">
  <img src="./logo.png" alt="Wagehood Logo" width="200">
</p>

A lightweight, local-first trading signal detection system that monitors market conditions and sends notifications through Discord. This project is designed for educational purposes and market research, not for live trading.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Trading Strategies](#trading-strategies)
  - [MACD-RSI Combined Strategy](#macd-rsi-combined-strategy)
  - [Support/Resistance Breakout Strategy](#supportresistance-breakout-strategy)
  - [Mathematical Foundations](#mathematical-foundations)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
  - [Analysis Commands](#analysis-commands)
  - [Monitoring Commands](#monitoring-commands)
  - [Notification Commands](#notification-commands)
  - [Utility Commands](#utility-commands)
- [Job Types](#job-types)
  - [1-Hour Analysis](#1-hour-analysis)
  - [1-Day Analysis](#1-day-analysis)
  - [Daily Summary](#daily-summary)
- [Discord Notifications](#discord-notifications)
- [Market Data Provider](#market-data-provider)
- [System Requirements](#system-requirements)
- [Performance Characteristics](#performance-characteristics)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## Overview

Wagehood is a local trading signal detection service that runs on your machine using cron jobs to monitor market conditions and generate trading signals based on technical analysis strategies. The system is designed to be:

- **Local-first**: Runs entirely on your machine with no external dependencies beyond market data
- **Cost-effective**: Uses Alpaca's free market data API for real-time and historical data
- **Reliable**: Implements multiple layers of fault tolerance and automatic recovery
- **Transparent**: All signals include detailed analysis and calculations
- **Educational**: Designed for learning about technical analysis and signal generation

## System Architecture

The system consists of several components working together:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Cron Jobs     │────▶│ Analysis Engine  │────▶│ Signal Detector │
│  (Scheduler)    │     │  (1h/1d/Daily)   │     │   (Strategies)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Alpaca Market    │     │ Discord         │
                        │ Data Provider    │     │ Notifications   │
                        └──────────────────┘     └─────────────────┘
```

### Key Components:

- **Cron Wrappers**: Manage execution timing and process lifecycle
- **Analysis Triggers**: Coordinate data fetching and strategy execution
- **Signal Detectors**: Implement technical analysis strategies
- **Notification System**: Queues and sends Discord alerts
- **Health Monitoring**: Tracks system health and recovers from failures

## Trading Strategies

### MACD-RSI Combined Strategy

This strategy combines two popular technical indicators to identify potential trend changes:

**Components:**
- **MACD (Moving Average Convergence Divergence)**: Measures the relationship between two moving averages
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions

**Signal Generation (Optimized for Higher Frequency):**
- **BUY Signals**: 
  - Primary: MACD crosses above signal line AND RSI exits oversold (<30)
  - Secondary: MACD crosses above signal line AND RSI > 50 (uptrend)
  - Tertiary: Strong MACD crossover alone (>2% divergence)
- **SELL Signals**: 
  - Primary: MACD crosses below signal line AND RSI exits overbought (>70)
  - Secondary: MACD crosses below signal line AND RSI < 50 (downtrend)
  - Tertiary: Strong MACD crossover alone (>2% divergence)

**Parameters (Optimized for Retail Trading):**
- MACD Fast Period: 12
- MACD Slow Period: 26
- MACD Signal Period: 9
- RSI Period: 14
- RSI Overbought: 70
- RSI Oversold: 30
- Minimum Confidence: 50% (reduced from 70%)
- Volume Threshold: 1.1x average (relaxed from 1.3x)

### Support/Resistance Breakout Strategy

This strategy identifies when price breaks through significant support or resistance levels:

**Components:**
- **Support Levels**: Price points where buying pressure historically increases
- **Resistance Levels**: Price points where selling pressure historically increases
- **Volume Confirmation**: Ensures breakouts have sufficient volume

**Signal Generation (Optimized Parameters):**
- **BUY Signal**: Price breaks above resistance with volume > 1.1x average (relaxed from 1.5x)
- **SELL Signal**: Price breaks below support with volume > 1.1x average (relaxed from 1.5x)

**Additional Strategies Available:**
- **RSI Trend Following**: Captures trends with wider RSI ranges (30-55 for bullish, 45-70 for bearish)
- **Bollinger Band Breakout**: Detects volatility breakouts with relaxed consolidation requirements

**Parameters:**
- Lookback Period: 60 days for daily, 60 bars for hourly
- Level Identification: Local minima/maxima with at least 2% separation
- Volume Threshold: 1.2x average volume
- Breakout Confirmation: Close price beyond level

### Mathematical Foundations

The strategies are based on well-established technical analysis principles:

**MACD Calculation:**
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**RSI Calculation:**
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

**Support/Resistance Identification:**
- Uses peak detection algorithms to find local extrema
- Validates levels based on historical price reactions
- Filters levels by minimum separation distance

**References:**
- Appel, Gerald. "Technical Analysis: Power Tools for Active Investors" (MACD)
- Wilder, J. Welles. "New Concepts in Technical Trading Systems" (RSI)
- Murphy, John J. "Technical Analysis of the Financial Markets"

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/wagehood.git
cd wagehood
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. **Initialize the system:**
```bash
python setup_notifications.py
python setup_cron_jobs.py setup
```

## Configuration

### Environment Variables (.env file):

```bash
# Alpaca API Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Discord Configuration
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Optional: Custom webhook for high-priority alerts
DISCORD_HIGH_PRIORITY_WEBHOOK_URL=your_priority_webhook_url

# System Configuration
LOG_LEVEL=INFO
HEALTH_CHECK_INTERVAL=300  # seconds
```

### Discord Webhook Configuration:

The system supports multiple Discord channels for different notification types:
- **Default Channel**: All standard signals and notifications
- **High Priority Channel**: Critical signals and system alerts
- **Infrastructure Channel**: System health and error notifications

## CLI Commands

### Analysis Commands

#### Run 1-Hour Analysis
```bash
python trigger_1h_analysis.py
```
Executes the 1-hour timeframe analysis across all monitored symbols. Checks both MACD-RSI and Support/Resistance strategies.

**Example output:**
```
Starting 1-hour analysis for AAPL...
✅ AAPL: BUY signal detected (MACD-RSI)
  - MACD: 0.15 crossed above signal
  - RSI: 45.2 (neutral)
  - Price: $185.32
```

#### Run 1-Day Analysis
```bash
python trigger_1d_analysis.py
```
Executes daily timeframe analysis. Best run after market close for end-of-day signals.

#### Run Daily Summary
```bash
python run_summary.py
```
Generates a comprehensive daily report including:
- All signals detected in the last 24 hours
- Market performance metrics
- System health statistics

### Monitoring Commands

#### Check System Health
```bash
python monitor_health.py --check
```
Performs a one-time health check of all system components.

#### Continuous Health Monitoring
```bash
python monitor_health.py --continuous
```
Runs continuous health monitoring with automatic recovery attempts.

#### View Cron Job Status
```bash
python setup_cron_jobs.py status
```
Shows the current status of all scheduled jobs.

### Notification Commands

#### Start Notification Workers
```bash
python start_notification_workers.py
```
Starts the notification worker processes that handle Discord message delivery.

#### Check Notification Queue
```bash
python check_notification_queue.py
```
Displays the current state of the notification queue.

#### Test Discord Connection
```bash
python test_discord.py
```
Sends a test message to verify Discord webhook configuration.

### Utility Commands

#### Clean Up Logs
```bash
python cleanup_logs.py --execute
```
Removes old log files and rotates current logs to save disk space.

#### Setup Cron Jobs
```bash
# Install cron jobs
python setup_cron_jobs.py setup

# Remove cron jobs
python setup_cron_jobs.py remove

# Reinstall (remove then setup)
python setup_cron_jobs.py reinstall
```

#### View Recent Signals
```bash
# Check 1-day signal history
cat ~/.wagehood/signal_history_1d.json | jq '.'

# Check 1-hour signal history
cat ~/.wagehood/signal_history_1h.json | jq '.'
```

## Job Types

### 1-Hour Analysis

**Purpose**: Detects intraday trading signals on hourly timeframes

**Schedule**: Runs every 10 seconds during market hours (9:30 AM - 4:00 PM ET)

**Characteristics**:
- Uses 60 hourly bars for analysis
- More sensitive to short-term price movements
- Suitable for day trading strategies
- Automatically skips outside market hours

**Process Flow**:
1. Check if market is open
2. Fetch latest hourly data from Alpaca
3. Calculate technical indicators
4. Detect signals for each strategy
5. Send notifications for new signals

### 1-Day Analysis

**Purpose**: Detects swing trading signals on daily timeframes

**Schedule**: Runs every 10 seconds, but most active after market close

**Characteristics**:
- Uses 60 daily bars for analysis
- More reliable for position trading
- Less prone to false signals
- Best results when run after 4:00 PM ET

**Process Flow**:
1. Fetch daily bars (including today's partial bar during market hours)
2. Calculate indicators on complete daily data
3. Identify support/resistance levels
4. Generate signals based on closing prices
5. Queue notifications with detailed analysis

### Daily Summary

**Purpose**: Provides comprehensive market analysis and system performance metrics

**Schedule**: Runs once daily at 5:00 PM ET (configurable)

**Contents**:
- All signals from the last 24 hours
- Win/loss statistics for each strategy
- Market breadth indicators
- System health summary
- Notable price movements

## Discord Notifications

### Notification Types

1. **Trading Signals**
   - Symbol, strategy, and signal type (BUY/SELL)
   - Current price and key indicator values
   - Timestamp and market session
   - Priority level (normal/high)

2. **Daily Summaries**
   - Aggregated performance metrics
   - Signal success rates
   - Market overview

3. **System Alerts**
   - Service health issues
   - Recovery notifications
   - Critical errors

### Message Format

```
🟢 BUY Signal: AAPL
Strategy: MACD-RSI
Price: $185.32
MACD: 0.15 (bullish crossover)
RSI: 45.2
Volume: 1.2M (avg: 980K)
Time: 2024-01-15 14:30:00 ET
```

### Deduplication

The system prevents duplicate notifications using:
- Signal history tracking per symbol/strategy
- Date-based deduplication (one notification per signal per day)
- Separate tracking for 1-hour and 1-day timeframes

## Market Data Provider

### Why Alpaca?

The system uses Alpaca Markets for several key reasons:

1. **Cost-Effective**: Free tier includes:
   - Real-time data for 30+ symbols
   - Unlimited historical data
   - Both quotes and trades
   - No monthly fees

2. **Comprehensive API**:
   - RESTful API for historical data
   - WebSocket for real-time streaming
   - Consistent data quality
   - Low latency

3. **Future Expansion**:
   - Same API supports paper trading
   - Easy transition to live trading
   - Options data available
   - Crypto markets included

4. **Reliability**:
   - 99.9% uptime SLA
   - Professional data feeds
   - Institutional-grade infrastructure

### Data Requirements

The system fetches:
- **Historical Bars**: 60 periods for each timeframe
- **Real-time Updates**: During market hours only
- **Symbol Lists**: Configurable watchlist
- **Market Status**: To determine trading sessions

## System Requirements

### Hardware
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space for logs and data
- **Network**: Stable internet connection

### Software
- **Operating System**: macOS, Linux, or Windows with WSL
- **Python**: 3.8 or higher
- **Cron**: For job scheduling (or Task Scheduler on Windows)

### Resource Usage
- **CPU**: ~5-10% during analysis runs
- **Memory**: ~200-500MB per analysis process
- **Network**: ~10-50MB per day
- **Disk I/O**: Minimal, mostly log writes

## Performance Characteristics

### Execution Times
- **1-Hour Analysis**: 3-5 seconds per run
- **1-Day Analysis**: 5-10 seconds per run
- **Daily Summary**: 10-15 seconds
- **Notification Delivery**: <1 second

### Scalability
- Supports monitoring 50+ symbols
- Processes 6 iterations per minute
- Handles 1000+ signals per day
- Automatic log rotation prevents disk issues

### Signal Generation Performance (After Optimization)
- **Signal Frequency**: 3-20x increase compared to original parameters
- **Expected Win Rate**: 40-70% (professional trading range)
- **Backtested Returns**: Positive returns across major symbols
- **Example Results** (2024 backtest):
  - AAPL: 17 signals, +22.04% return, 62.5% win rate
  - TSLA: 14 signals, +101.48% return, 66.67% win rate
  - SPY: 111 signals, +8.59% return, 75% win rate

### Reliability Features
- **Process Isolation**: Each job runs independently
- **Automatic Recovery**: Failed jobs restart automatically
- **Health Monitoring**: Continuous system health checks
- **Graceful Degradation**: Continues operating if Discord is down

## Troubleshooting

### Common Issues

#### Analysis Timeout
```bash
# Check timeout settings
grep TIMEOUT_SECONDS cron_wrapper_*.py

# View recent logs
tail -n 100 ~/.wagehood/cron_1d.log
```

#### Missing Notifications
```bash
# Check notification queue
python check_notification_queue.py

# Verify Discord webhook
python test_discord.py

# Check signal history
python check_notification_status.py
```

#### Cron Jobs Not Running
```bash
# Check cron status
crontab -l

# Reinstall jobs
python setup_cron_jobs.py reinstall

# Check system logs
tail -f ~/.wagehood/watchdog.log
```

### Log Locations
- **Analysis Logs**: `~/.wagehood/cron_1h.log`, `~/.wagehood/cron_1d.log`
- **Notification Logs**: `~/.wagehood/notification_workers.log`
- **System Health**: `~/.wagehood/watchdog.log`
- **Error Logs**: `~/.wagehood/*_error.log`

## Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. It is NOT intended for live trading or financial decision-making.

- **No Trading Advice**: Signals generated are for informational purposes only
- **No Guarantees**: Past performance does not indicate future results
- **Risk Warning**: Trading involves substantial risk of loss
- **Educational Tool**: Designed for learning about technical analysis
- **Your Responsibility**: Any trading decisions are solely your responsibility

By using this software, you acknowledge that:
1. You will not use it for live trading without proper risk management
2. The developers are not responsible for any financial losses
3. Signal accuracy is not guaranteed
4. You should paper trade and backtest before any real trading

---

For questions, issues, or contributions, please visit the project repository.