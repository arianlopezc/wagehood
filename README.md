# Wagehood

A trading strategy analysis system that processes market data and generates trading signals using multiple technical analysis strategies.

## Features

- **Multiple Trading Strategies**: RSI Trend, Bollinger Breakout, MACD+RSI, Support/Resistance Breakout
- **Real-time Data**: Alpaca Markets integration for live and historical market data
- **Job Queue System**: Job processing with multiple workers
- **CLI Interface**: Easy-to-use command line interface for all operations
- **Comprehensive Testing**: Full integration test suite with real market data

## Quick Start
### 1. Install the CLI
```bash
python3 install_cli.py
```

### 2. Configure your credentials
```bash
wagehood configure
```

### 3. Start workers
```bash
wagehood workers start
```

### 4. Submit a job
```bash
wagehood submit -s AAPL -st rsi_trend -t 1d -sd 2024-01-01 -ed 2024-01-31
```

## Available Strategies

- `rsi_trend` - RSI-based trend following
- `bollinger_breakout` - Bollinger Bands breakout detection
- `macd_rsi` - Combined MACD and RSI signals
- `sr_breakout` - Support/Resistance level breakouts

## CLI Commands

### Job Management
```bash
wagehood submit -s <symbol> -st <strategy> -t <timeframe> -sd <start> -ed <end>
wagehood jobs                 # List all jobs
wagehood jobs -s completed    # Filter by status
```

### Worker Control
```bash
wagehood workers start
wagehood workers stop
wagehood workers status
```

### Symbol Management
```bash
wagehood symbols list
wagehood symbols add NVDA
wagehood symbols remove TSLA
```

### Testing
```bash
wagehood test    # Run full test suite
```

## Requirements

- Python 3.8+
- Alpaca Markets API account
- TA-Lib library

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the CLI installer: `python3 install_cli.py`

## Configuration

The CLI stores configuration in `~/.wagehood/config.json`. You can reconfigure at any time:
```bash
wagehood configure
```

## License

MIT