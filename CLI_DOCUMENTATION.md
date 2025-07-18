# Wagehood CLI Documentation

Comprehensive reference guide for all Wagehood command-line tools and utilities.

## Table of Contents

- [Core Analysis Commands](#core-analysis-commands)
- [Monitoring and Health Commands](#monitoring-and-health-commands)
- [Notification Management](#notification-management)
- [System Setup and Configuration](#system-setup-and-configuration)
- [Utility Commands](#utility-commands)
- [Troubleshooting Commands](#troubleshooting-commands)
- [Advanced Usage](#advanced-usage)

## Core Analysis Commands

### trigger_1h_analysis.py

**Purpose**: Executes technical analysis on 1-hour timeframe data

**Usage**:
```bash
python trigger_1h_analysis.py
```

**Behavior**:
- Checks if market is open before running
- Fetches last 60 hourly bars for each symbol
- Runs MACD-RSI and Support/Resistance strategies
- Sends Discord notifications for new signals
- Updates signal history to prevent duplicates

**Example Output**:
```
2024-01-15 10:30:15 - INFO - Starting 1-hour analysis...
2024-01-15 10:30:15 - INFO - Market is open - proceeding with analysis
2024-01-15 10:30:16 - INFO - Analyzing AAPL...
2024-01-15 10:30:17 - INFO - ✅ AAPL: BUY signal detected (macd_rsi)
2024-01-15 10:30:17 - INFO - 📤 Sending notification for AAPL macd_rsi BUY
```

**Exit Codes**:
- 0: Success
- 1: Analysis failed
- 2: Market closed (not an error)

### trigger_1d_analysis.py

**Purpose**: Executes technical analysis on daily timeframe data

**Usage**:
```bash
python trigger_1d_analysis.py
```

**Behavior**:
- Runs regardless of market status
- Fetches last 60 daily bars
- Best results when run after market close
- Analyzes closing prices for signal generation

**Configuration**:
No command-line arguments. Configured via:
- Symbol list in `src/utils/symbol_config.py`
- Strategy parameters in respective strategy files

### run_summary.py

**Purpose**: Generates daily summary report with all signals and metrics

**Usage**:
```bash
python run_summary.py
```

**Output Includes**:
- All signals from last 24 hours
- Performance metrics by strategy
- System health summary
- Market overview

**Schedule**: Typically run at 5:00 PM ET via cron

## Monitoring and Health Commands

### monitor_health.py

**Purpose**: Check and monitor system health

**Usage**:
```bash
# One-time health check
python monitor_health.py --check

# Continuous monitoring
python monitor_health.py --continuous

# Check with recovery attempts
python monitor_health.py --check --recover
```

**Options**:
- `--check`: Perform single health check and exit
- `--continuous`: Run continuous monitoring loop
- `--recover`: Attempt to fix issues found
- `--interval SECONDS`: Check interval (default: 300)

**Health Checks**:
- Cron job status
- Log file activity
- Notification worker status
- Disk space usage
- Process health

### watchdog.py

**Purpose**: Lightweight monitoring script run by cron

**Usage**:
```bash
python watchdog.py
```

**Features**:
- Runs every 5 minutes via cron
- Checks log file activity
- Verifies cron job installation
- Attempts automatic recovery
- Rotates logs if needed

## Notification Management

### start_notification_workers.py

**Purpose**: Start Discord notification worker processes

**Usage**:
```bash
# Start with default workers
python start_notification_workers.py

# Start with custom worker count
python start_notification_workers.py --workers 4

# Start in debug mode
python start_notification_workers.py --debug
```

**Options**:
- `--workers N`: Number of worker processes (default: 2)
- `--debug`: Enable debug logging
- `--daemon`: Run as background daemon

### check_notification_queue.py

**Purpose**: Inspect notification queue status

**Usage**:
```bash
python check_notification_queue.py
```

**Output Shows**:
- Queue size
- Pending notifications
- Failed notifications
- Worker status
- Last processed time

### test_discord.py

**Purpose**: Test Discord webhook configuration

**Usage**:
```bash
python test_discord.py
```

**Tests**:
- Default webhook
- High-priority webhook (if configured)
- Infrastructure webhook (if configured)
- Message formatting
- Delivery confirmation

## System Setup and Configuration

### setup_cron_jobs.py

**Purpose**: Manage cron job installation

**Usage**:
```bash
# Install cron jobs
python setup_cron_jobs.py setup

# Remove all cron jobs
python setup_cron_jobs.py remove

# Show current status
python setup_cron_jobs.py status

# Reinstall (remove + setup)
python setup_cron_jobs.py reinstall
```

**Installed Jobs**:
- 1-hour analysis wrapper (every minute)
- 1-day analysis wrapper (every minute)
- Watchdog (every 5 minutes)
- Daily summary (5:00 PM ET)

### setup_notifications.py

**Purpose**: Initialize notification system

**Usage**:
```bash
python setup_notifications.py
```

**Actions**:
- Creates notification database
- Sets up worker configuration
- Initializes queue tables
- Configures Discord channels

### install_cli.py

**Purpose**: Install wagehood CLI command (legacy)

**Usage**:
```bash
python install_cli.py
```

**Note**: This installs the old job-based CLI system. The current system uses direct script execution.

## Utility Commands

### cleanup_logs.py

**Purpose**: Clean up and rotate log files

**Usage**:
```bash
# Dry run (show what would be done)
python cleanup_logs.py

# Execute cleanup
python cleanup_logs.py --execute

# Setup rotation config
python cleanup_logs.py --setup-rotation
```

**Actions**:
- Removes old log files
- Compresses rotated logs
- Frees disk space
- Reports space saved

### check_notification_status.py

**Purpose**: Debug notification delivery issues

**Usage**:
```bash
python check_notification_status.py
```

**Checks**:
- Signal history files
- Deduplication status
- Recent notifications
- Failed deliveries

### cleanup_old_signals.py

**Purpose**: Archive old signal history

**Usage**:
```bash
python cleanup_old_signals.py --days 30
```

**Options**:
- `--days N`: Keep signals from last N days
- `--dry-run`: Show what would be deleted
- `--archive`: Archive before deletion

## Troubleshooting Commands

### Common Diagnostic Commands

**Check if analysis is running**:
```bash
# View recent 1-hour analysis logs
tail -f ~/.wagehood/cron_1h.log

# View recent 1-day analysis logs
tail -f ~/.wagehood/cron_1d.log

# Check for errors
grep ERROR ~/.wagehood/*.log | tail -20
```

**Verify cron job execution**:
```bash
# List active cron jobs
crontab -l | grep wagehood

# Check cron daemon logs (macOS)
log show --predicate 'process == "cron"' --last 1h

# Check cron logs (Linux)
grep CRON /var/log/syslog | tail -20
```

**Monitor real-time activity**:
```bash
# Watch all logs simultaneously
tail -f ~/.wagehood/*.log

# Monitor specific process
ps aux | grep trigger_1h_analysis

# Check system resources
top -pid $(pgrep -f trigger_1h_analysis)
```

## Advanced Usage

### Running Analysis for Specific Symbols

Currently, symbols are configured in `src/utils/symbol_config.py`. To analyze specific symbols:

1. Temporarily modify the symbol list:
```python
# In src/utils/symbol_config.py
SYMBOLS = ['AAPL', 'GOOGL']  # Your symbols
```

2. Run the analysis:
```bash
python trigger_1d_analysis.py
```

3. Restore original symbol list

### Custom Timeframes

To run analysis on custom timeframes:

```python
# Create custom trigger script
from src.data_providers.alpaca_provider import AlpacaProvider
from src.strategies.macd_rsi_strategy import MacdRsiStrategy

provider = AlpacaProvider()
bars = provider.get_bars_timeframe('AAPL', '15Min', limit=60)
strategy = MacdRsiStrategy()
signal = strategy.check_signal(bars)
```

### Batch Processing

For bulk analysis across many symbols:

```bash
# Create symbol list file
echo "AAPL
GOOGL
MSFT
AMZN" > symbols.txt

# Run analysis for each
while read symbol; do
    echo "Analyzing $symbol..."
    # Modify symbol config and run
done < symbols.txt
```

### Performance Monitoring

**Track execution times**:
```bash
# Time analysis execution
time python trigger_1d_analysis.py

# Profile memory usage
/usr/bin/time -l python trigger_1d_analysis.py  # macOS
/usr/bin/time -v python trigger_1d_analysis.py  # Linux
```

**Monitor resource usage**:
```bash
# Watch CPU and memory
python monitor_resources.py --command "python trigger_1d_analysis.py"
```

### Integration with External Tools

**Export signals to CSV**:
```bash
# Convert signal history to CSV
python -c "
import json
import csv
with open('~/.wagehood/signal_history_1d.json') as f:
    data = json.load(f)
with open('signals.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Symbol', 'Strategy', 'Signal', 'Date'])
    for key, value in data.items():
        symbol, strategy = key.split('|')
        writer.writerow([symbol, strategy, value['signal_type'], value['date']])
"
```

**Send to external webhook**:
```bash
# Forward signals to custom endpoint
python -c "
import requests
import json
with open('~/.wagehood/signal_history_1d.json') as f:
    signals = json.load(f)
requests.post('https://your-api.com/signals', json=signals)
"
```

## Environment Variables

All commands respect these environment variables:

```bash
# Alpaca API (required)
export ALPACA_API_KEY="your-key"
export ALPACA_SECRET_KEY="your-secret"
export ALPACA_BASE_URL="https://api.alpaca.markets"

# Discord (required for notifications)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Optional
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export HEALTH_CHECK_INTERVAL="300"  # seconds
export NOTIFICATION_WORKERS="2"  # number of workers
```

## Error Codes and Meanings

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 0 | Success | Normal operation |
| 1 | General failure | Check logs for details |
| 2 | Market closed | Expected during off-hours |
| 3 | Configuration error | Missing API keys or webhooks |
| 4 | Network error | API connection issues |
| 5 | Data error | Invalid or missing market data |
| 127 | Command not found | Python path issues |

## Best Practices

1. **Always check logs** when something seems wrong:
   ```bash
   tail -n 100 ~/.wagehood/*.log | grep -E "ERROR|WARNING"
   ```

2. **Run health checks** regularly:
   ```bash
   python monitor_health.py --check
   ```

3. **Monitor disk space** to prevent log overflow:
   ```bash
   df -h ~/.wagehood
   python cleanup_logs.py --execute
   ```

4. **Test changes** before deploying:
   ```bash
   # Backup current state
   cp ~/.wagehood/signal_history_*.json ~/.wagehood/backup/
   
   # Test changes
   python trigger_1d_analysis.py
   
   # Verify results
   python check_notification_status.py
   ```

5. **Use process isolation** for testing:
   ```bash
   # Run in isolated environment
   export DISCORD_WEBHOOK_URL="https://discord.com/test-webhook"
   python test_discord.py
   ```

---

For more information, see the main [README.md](README.md) or check the source code documentation.