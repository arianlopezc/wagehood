# Wagehood CLI Documentation

A comprehensive command-line interface for the Wagehood real-time trading system. The CLI provides full access to market data, configuration management, system monitoring, and administrative functions.

## Installation & Setup

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Make CLI executable
chmod +x wagehood_cli.py

# Optional: Add to PATH
ln -s $(pwd)/wagehood_cli.py /usr/local/bin/wagehood
```

### Configuration

Create a configuration file (optional):

```bash
mkdir -p ~/.wagehood
```

**~/.wagehood/cli_config.yaml:**
```yaml
api:
  url: "http://localhost:8000"
  timeout: 30
  retries: 3
  
output:
  format: "table"  # json, table, csv
  use_color: true
  max_width: 120
  
streaming:
  buffer_size: 1000
  reconnect_delay: 5
  
logging:
  level: "INFO"
  file: "~/.wagehood/cli.log"
```

## Quick Start

```bash
# Check system health
./wagehood_cli.py monitor health

# Get latest data for SPY
./wagehood_cli.py data latest SPY

# Stream real-time data for 30 seconds
./wagehood_cli.py data stream SPY QQQ --duration 30

# Add symbols to watchlist
./wagehood_cli.py config watchlist add AAPL TSLA NVDA

# Check performance stats
./wagehood_cli.py monitor stats
```

## Command Reference

### Global Options

```bash
./wagehood_cli.py [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

Global Options:
  --api-url TEXT          API base URL [default: http://localhost:8000]
  -c, --config PATH       Configuration file [default: ~/.wagehood/cli_config.yaml]
  -f, --output-format     Output format: json|table|csv [default: table]
  -v, --verbose           Enable verbose logging
  -q, --quiet             Suppress non-error output
  --no-color              Disable colored output
  --version               Show version and exit
  -h, --help              Show help message
```

## Data Commands

### Latest Data

Get the most recent market data and indicators for symbols.

```bash
./wagehood_cli.py data latest [OPTIONS] [SYMBOLS]...

Options:
  --indicators         Include technical indicators
  --signals            Include trading signals
  --raw                Output raw API response
  
Examples:
  # Latest data for SPY
  ./wagehood_cli.py data latest SPY
  
  # Multiple symbols with indicators
  ./wagehood_cli.py data latest SPY QQQ AAPL --indicators
  
  # With trading signals in JSON format
  ./wagehood_cli.py data latest SPY --signals -f json
```

### Real-time Streaming

Stream live market data via WebSocket connection.

```bash
./wagehood_cli.py data stream [OPTIONS] [SYMBOLS]...

Options:
  -d, --duration INTEGER  Stream duration in seconds [default: 60]
  --indicators           Include indicator updates
  --signals              Include signal updates  
  --reconnect            Auto-reconnect on disconnect
  
Examples:
  # Stream SPY for 60 seconds
  ./wagehood_cli.py data stream SPY
  
  # Stream multiple symbols with indicators
  ./wagehood_cli.py data stream SPY QQQ --indicators --duration 300
  
  # Stream with auto-reconnect
  ./wagehood_cli.py data stream AAPL --reconnect --duration 3600
```

### Historical Data

Query historical indicator data with date filtering.

```bash
./wagehood_cli.py data historical [OPTIONS] SYMBOL

Options:
  --start DATE           Start date (YYYY-MM-DD)
  --end DATE             End date (YYYY-MM-DD)
  --indicators TEXT      Comma-separated indicator list
  --limit INTEGER        Maximum records to return
  
Examples:
  # Last 30 days of SPY data
  ./wagehood_cli.py data historical SPY --start 2024-01-01
  
  # Specific indicators only
  ./wagehood_cli.py data historical AAPL --indicators "sma_50,rsi_14" --limit 100
```

### Data Export

Create and download bulk data exports.

```bash
# Create export job
./wagehood_cli.py data export create [OPTIONS] [SYMBOLS]...

Options:
  --start DATE           Start date
  --end DATE             End date
  --format TEXT          Export format: csv|json|parquet
  --indicators           Include indicators
  --signals              Include signals
  
# Check export status
./wagehood_cli.py data export status EXPORT_ID

# Download export results
./wagehood_cli.py data export download EXPORT_ID [OUTPUT_FILE]

Examples:
  # Export SPY data to CSV
  ./wagehood_cli.py data export create SPY --format csv --start 2024-01-01
  
  # Download export
  ./wagehood_cli.py data export download exp_123456 spy_data.csv
```

## Configuration Commands

### CLI Configuration

Manage CLI settings and preferences.

```bash
# Show current configuration
./wagehood_cli.py config cli show

# Set configuration values
./wagehood_cli.py config cli set KEY VALUE

# Reset to defaults
./wagehood_cli.py config cli reset

# Export configuration
./wagehood_cli.py config cli export config_backup.yaml

Examples:
  # Set API URL
  ./wagehood_cli.py config cli set api.url "http://api.example.com:8000"
  
  # Set output format
  ./wagehood_cli.py config cli set output.format json
  
  # Reset all settings
  ./wagehood_cli.py config cli reset --confirm
```

### Watchlist Management

Manage the list of symbols being monitored.

```bash
# Show current watchlist
./wagehood_cli.py config watchlist show

# Add symbols
./wagehood_cli.py config watchlist add [SYMBOLS]...

# Remove symbols  
./wagehood_cli.py config watchlist remove [SYMBOLS]...

# Clear all symbols
./wagehood_cli.py config watchlist clear

Examples:
  # Add symbols to watchlist
  ./wagehood_cli.py config watchlist add AAPL TSLA NVDA MSFT
  
  # Remove a symbol
  ./wagehood_cli.py config watchlist remove TSLA
  
  # View current watchlist
  ./wagehood_cli.py config watchlist show
```

### Indicator Configuration

Manage technical indicator settings.

```bash
# Show indicator configurations
./wagehood_cli.py config indicators show

# Update from file
./wagehood_cli.py config indicators update FILE

# Reset to defaults
./wagehood_cli.py config indicators reset

Examples:
  # Show all indicators
  ./wagehood_cli.py config indicators show
  
  # Update from JSON file
  ./wagehood_cli.py config indicators update new_indicators.json
```

### Strategy Configuration

Manage trading strategy settings.

```bash
# Show strategy configurations
./wagehood_cli.py config strategies show

# Update strategies
./wagehood_cli.py config strategies update FILE

# Enable/disable strategies
./wagehood_cli.py config strategies enable STRATEGY_NAME
./wagehood_cli.py config strategies disable STRATEGY_NAME

Examples:
  # Show all strategies
  ./wagehood_cli.py config strategies show
  
  # Disable a strategy
  ./wagehood_cli.py config strategies disable bollinger_breakout_strategy
```

### System Configuration

Manage system-wide settings.

```bash
# Show system configuration
./wagehood_cli.py config system show

# Update system settings
./wagehood_cli.py config system update FILE

# Set individual values
./wagehood_cli.py config system set KEY VALUE

Examples:
  # Show system config
  ./wagehood_cli.py config system show
  
  # Set calculation workers
  ./wagehood_cli.py config system set calculation_workers 8
```

## Monitoring Commands

### Health Checks

Check system health and component status.

```bash
# Basic health check
./wagehood_cli.py monitor health

# Detailed health information
./wagehood_cli.py monitor health --detailed

# Health check with JSON output
./wagehood_cli.py monitor health -f json

Examples:
  # Quick health check
  ./wagehood_cli.py monitor health
  
  # Detailed component status
  ./wagehood_cli.py monitor health --detailed
```

### Performance Statistics

View system performance metrics and statistics.

```bash
# Show performance stats
./wagehood_cli.py monitor stats

# Real-time stats monitoring
./wagehood_cli.py monitor stats --live

# Component-specific stats
./wagehood_cli.py monitor stats --component ingestion

Examples:
  # View current stats
  ./wagehood_cli.py monitor stats
  
  # Live monitoring (updates every 5 seconds)
  ./wagehood_cli.py monitor stats --live
```

### Alerts Management

View and manage system alerts.

```bash
# Show recent alerts
./wagehood_cli.py monitor alerts

# Filter alerts by type
./wagehood_cli.py monitor alerts --type error

# Filter alerts by time
./wagehood_cli.py monitor alerts --since "1 hour ago"

Examples:
  # Show all alerts
  ./wagehood_cli.py monitor alerts
  
  # Show only errors from last hour
  ./wagehood_cli.py monitor alerts --type error --since "1 hour ago"
```

### Live Monitoring

Real-time system monitoring dashboard.

```bash
# Start live monitoring
./wagehood_cli.py monitor live

# Monitor specific components
./wagehood_cli.py monitor live --components ingestion,calculation

# Set refresh interval
./wagehood_cli.py monitor live --interval 10

Examples:
  # Start live dashboard
  ./wagehood_cli.py monitor live
  
  # Monitor calculation engine only
  ./wagehood_cli.py monitor live --components calculation
```

### Connectivity Test

Test API connectivity and response times.

```bash
# Basic ping test
./wagehood_cli.py monitor ping

# Extended connectivity test
./wagehood_cli.py monitor ping --count 10 --interval 1

Examples:
  # Single ping
  ./wagehood_cli.py monitor ping
  
  # 10 pings with 1 second interval
  ./wagehood_cli.py monitor ping --count 10 --interval 1
```

## Administrative Commands

### Service Management

Start, stop, and manage system services.

```bash
# Start API server
./wagehood_cli.py admin service start-api [OPTIONS]

# Start real-time processor  
./wagehood_cli.py admin service start-realtime [OPTIONS]

# Stop services
./wagehood_cli.py admin service stop [SERVICE_NAME]

# Restart services
./wagehood_cli.py admin service restart [SERVICE_NAME]

# Service status
./wagehood_cli.py admin service status

Options:
  --background           Run in background
  --config FILE          Service configuration file
  --log-level LEVEL      Logging level
  
Examples:
  # Start API server in background
  ./wagehood_cli.py admin service start-api --background
  
  # Start real-time processor with debug logging
  ./wagehood_cli.py admin service start-realtime --log-level DEBUG
  
  # Check service status
  ./wagehood_cli.py admin service status
```

### Cache Management

Manage system caches and data stores.

```bash
# Clear specific cache
./wagehood_cli.py admin cache clear [CACHE_TYPE]

# Clear all caches
./wagehood_cli.py admin cache clear-all

# Cache statistics
./wagehood_cli.py admin cache stats

Cache Types:
  - indicators: Technical indicator cache
  - market_data: Market data cache  
  - strategies: Strategy results cache
  - all: All cache types

Examples:
  # Clear indicator cache
  ./wagehood_cli.py admin cache clear indicators
  
  # Clear all caches
  ./wagehood_cli.py admin cache clear-all --confirm
  
  # Show cache statistics
  ./wagehood_cli.py admin cache stats
```

### Log Management

View and analyze system logs.

```bash
# Show recent logs
./wagehood_cli.py admin logs show

# Filter logs by level
./wagehood_cli.py admin logs show --level ERROR

# Filter logs by component
./wagehood_cli.py admin logs show --component calculation

# Follow logs in real-time
./wagehood_cli.py admin logs follow

Examples:
  # Show last 100 log entries
  ./wagehood_cli.py admin logs show --limit 100
  
  # Show only errors
  ./wagehood_cli.py admin logs show --level ERROR
  
  # Follow logs in real-time
  ./wagehood_cli.py admin logs follow
```

### Backup & Restore

Create and restore system backups.

```bash
# Create backup
./wagehood_cli.py admin backup create [OPTIONS]

# List backups
./wagehood_cli.py admin backup list

# Restore from backup
./wagehood_cli.py admin backup restore BACKUP_ID

Options:
  --include TEXT         Components to include (config,data,logs)
  --compress             Compress backup files
  --description TEXT     Backup description
  
Examples:
  # Create full backup
  ./wagehood_cli.py admin backup create --description "Pre-upgrade backup"
  
  # Backup only configuration
  ./wagehood_cli.py admin backup create --include config
  
  # List available backups
  ./wagehood_cli.py admin backup list
  
  # Restore backup
  ./wagehood_cli.py admin backup restore backup_20240101_120000
```

### Maintenance Tasks

Run system maintenance and optimization tasks.

```bash
# Run cleanup tasks
./wagehood_cli.py admin maintenance cleanup

# Optimize database
./wagehood_cli.py admin maintenance optimize

# Reindex data
./wagehood_cli.py admin maintenance reindex

# Full maintenance cycle
./wagehood_cli.py admin maintenance full

Examples:
  # Clean up old files
  ./wagehood_cli.py admin maintenance cleanup
  
  # Run full maintenance
  ./wagehood_cli.py admin maintenance full --confirm
```

## Output Formats

### Table Format (Default)

Human-readable tables with colors and formatting:

```
Symbol    Price    Change    Volume       RSI     MACD
SPY       485.67   +2.34     12,345,678   65.23   +0.45
QQQ       395.21   -1.87     8,765,432    58.91   -0.23
```

### JSON Format

Machine-readable JSON output:

```bash
./wagehood_cli.py data latest SPY -f json
```

```json
{
  "symbol": "SPY",
  "timestamp": "2024-01-01T12:00:00Z",
  "price": 485.67,
  "volume": 12345678,
  "indicators": {
    "rsi_14": 65.23,
    "macd": 0.45
  }
}
```

### CSV Format

Comma-separated values for spreadsheet import:

```bash
./wagehood_cli.py data latest SPY QQQ -f csv
```

```csv
symbol,timestamp,price,volume,rsi_14,macd
SPY,2024-01-01T12:00:00Z,485.67,12345678,65.23,0.45
QQQ,2024-01-01T12:00:00Z,395.21,8765432,58.91,-0.23
```

## Environment Variables

```bash
# API Configuration
export WAGEHOOD_API_URL="http://localhost:8000"
export WAGEHOOD_API_TOKEN="your_api_token"

# CLI Configuration  
export WAGEHOOD_CONFIG_FILE="~/.wagehood/cli_config.yaml"
export WAGEHOOD_OUTPUT_FORMAT="table"
export WAGEHOOD_LOG_LEVEL="INFO"

# Development
export WAGEHOOD_DEBUG=1
```

## Shell Completion

Enable shell completion for better usability:

### Bash

```bash
# Add to ~/.bashrc
eval "$(_WAGEHOOD_COMPLETE=source_bash wagehood_cli.py)"

# Or generate completion script
./wagehood_cli.py completion --shell bash >> ~/.bashrc
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_WAGEHOOD_COMPLETE=source_zsh wagehood_cli.py)"

# Or generate completion script
./wagehood_cli.py completion --shell zsh >> ~/.zshrc
```

### Fish

```bash
# Generate completion script
./wagehood_cli.py completion --shell fish > ~/.config/fish/completions/wagehood.fish
```

## Error Handling

The CLI provides comprehensive error handling with helpful messages:

```bash
# Connection errors
Error: Unable to connect to API at http://localhost:8000
Suggestion: Check if the API server is running

# Configuration errors  
Error: Invalid symbol 'INVALID' in watchlist
Suggestion: Use valid ticker symbols (e.g., SPY, QQQ, AAPL)

# Authentication errors
Error: API request failed with status 401
Suggestion: Check your API token configuration
```

## Tips & Best Practices

### Performance Optimization

1. **Use caching**: CLI caches recent data for faster responses
2. **Batch operations**: Process multiple symbols together
3. **Limit output**: Use `--limit` for large datasets
4. **Stream carefully**: Monitor bandwidth usage with streaming

### Configuration Management

1. **Use config files**: Store preferences in ~/.wagehood/cli_config.yaml
2. **Environment variables**: Set WAGEHOOD_* variables for automation
3. **Backup configs**: Regular backups before major changes
4. **Validate settings**: Use validation commands before applying

### Monitoring Best Practices

1. **Regular health checks**: Monitor system health periodically
2. **Set up alerts**: Configure alerts for critical issues
3. **Log analysis**: Use log filtering for troubleshooting
4. **Performance tracking**: Monitor trends in system metrics

### Automation Scripts

Example shell script for automated monitoring:

```bash
#!/bin/bash
# health_check.sh

# Check system health
if ! ./wagehood_cli.py monitor health --quiet; then
    echo "System health check failed!"
    ./wagehood_cli.py monitor alerts --type error --since "5 minutes ago"
    exit 1
fi

# Check performance metrics
./wagehood_cli.py monitor stats -f json > /tmp/metrics.json

echo "Health check completed successfully"
```

## Troubleshooting

### Common Issues

1. **Connection refused**
   ```bash
   # Check if API is running
   ./wagehood_cli.py monitor ping
   
   # Start API server
   ./wagehood_cli.py admin service start-api
   ```

2. **Permission denied**
   ```bash
   # Make CLI executable
   chmod +x wagehood_cli.py
   
   # Check file permissions
   ls -la wagehood_cli.py
   ```

3. **Module not found**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

4. **Configuration errors**
   ```bash
   # Reset configuration
   ./wagehood_cli.py config cli reset
   
   # Validate configuration
   ./wagehood_cli.py config cli validate
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
./wagehood_cli.py -v data latest SPY
./wagehood_cli.py --verbose monitor health
```

### Log Files

Check CLI log files for detailed error information:

```bash
tail -f ~/.wagehood/cli.log
grep ERROR ~/.wagehood/cli.log
```

## Support

For additional help and support:

1. **Built-in help**: Use `--help` with any command
2. **GitHub Issues**: Report bugs and feature requests
3. **Documentation**: Refer to API documentation
4. **Community**: Join the Wagehood community forums

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-01  
**Compatibility**: Python 3.11+, Wagehood API v1.0+