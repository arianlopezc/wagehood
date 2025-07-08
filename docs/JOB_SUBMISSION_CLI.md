# Job Submission CLI Documentation

The Wagehood Job Submission CLI allows you to submit backtest jobs to the running production instance, monitor their progress in real-time, and view detailed results including all signals and trades.

## Overview

The CLI provides a single command that handles the entire backtest workflow:
1. **Submit** a backtest job to the production instance
2. **Monitor** job progress with real-time updates
3. **Display** comprehensive results with all signals and trades

## Prerequisites

1. **Running Production Instance**: The Wagehood Docker container must be running
2. **Redis Access**: CLI connects to Redis on port 6380 (Docker instance)
3. **Python Dependencies**: `redis` package required

## Installation

```bash
# Ensure Redis package is installed
pip install redis

# Make CLI executable
chmod +x submit_job.py
```

## Basic Usage

```bash
python submit_job.py --symbol SYMBOL --timeframe TIMEFRAME --strategy STRATEGY \
                    --start START_DATE --end END_DATE
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--symbol` | Trading symbol to analyze | `AAPL`, `SPY`, `MSFT` |
| `--timeframe` | Analysis timeframe | `1h`, `1d`, `5m` |
| `--strategy` | Trading strategy to test | `macd_rsi`, `rsi_trend` |
| `--start` | Start date (YYYY-MM-DD) | `2024-01-01` |
| `--end` | End date (YYYY-MM-DD) | `2024-12-31` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--redis-host` | `localhost` | Redis server host |
| `--redis-port` | `6380` | Redis server port |

## Available Strategies

| Strategy | Key | Best For | Win Rate | Description |
|----------|-----|----------|----------|-------------|
| **MACD + RSI Combined** | `macd_rsi` | Swing Trading | ~73% | Momentum strategy combining MACD and RSI |
| **RSI Trend Following** | `rsi_trend` | Day Trading | ~68% | Trend-aware RSI with pullbacks |
| **Bollinger Band Breakout** | `bollinger_breakout` | Volatility Trading | ~65% | Volatility expansion strategy |
| **Support/Resistance Breakout** | `sr_breakout` | Breakout Trading | ~58% | Level-based breakout trading |
| **Moving Average Crossover** | `ma_crossover` | Position Trading | ~45% | Golden/Death cross signals |

## Available Timeframes

### Day Trading (High Frequency)
- `1m` - 1 minute bars
- `5m` - 5 minute bars  
- `15m` - 15 minute bars
- `30m` - 30 minute bars

### Swing Trading (Medium Term)
- `1h` - 1 hour bars
- `4h` - 4 hour bars

### Position Trading (Long Term)
- `1d` - Daily bars
- `1w` - Weekly bars
- `1M` - Monthly bars

## Examples

### Example 1: AAPL Swing Trading
```bash
python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \
                    --start 2024-01-01 --end 2024-12-31
```

### Example 2: SPY Day Trading
```bash
python submit_job.py --symbol SPY --timeframe 5m --strategy rsi_trend \
                    --start 2024-06-01 --end 2024-06-30
```

### Example 3: QQQ Position Trading
```bash
python submit_job.py --symbol QQQ --timeframe 1d --strategy ma_crossover \
                    --start 2023-01-01 --end 2024-12-31
```

### Example 4: Custom Redis Connection
```bash
python submit_job.py --symbol TSLA --timeframe 1h --strategy bollinger_breakout \
                    --start 2024-01-01 --end 2024-06-30 \
                    --redis-host 192.168.1.100 --redis-port 6379
```

## Output Format

The CLI provides detailed output in several sections:

### 1. Job Submission
```
📊 Submitting backtest job...
Symbol: AAPL
Timeframe: 1h
Strategy: macd_rsi
Period: 2024-01-01 to 2024-12-31

✅ Job submitted with ID: job_20250108_130000_xyz123
```

### 2. Progress Monitoring
```
⏳ Monitoring job: job_20250108_130000_xyz123
🚀 Status: Running [████████░░░░░░░░] 50% - Processing 2024-06...
✅ Job completed successfully!
```

### 3. Performance Summary
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 BACKTEST RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbol: AAPL
Timeframe: 1h
Strategy: macd_rsi
Period: 2024-01-01 to 2024-12-31
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Return:     +15.42%
Win Rate:          68.2%
Sharpe Ratio:       1.85
Max Drawdown:      -8.30%
Profit Factor:      2.14
Total Trades:        127
Winning Trades:       87
Losing Trades:        40
```

### 4. Signals Summary
```
🎯 SIGNALS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Signals:       254
Buy Signals:         127
Sell Signals:        127
```

### 5. All Signals Details
```
📋 ALL SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date         Type Price      Confidence Strategy
────────────────────────────────────────────────────────────────────────────
2024-01-15   BUY  $185.23    0.85       macd_rsi
2024-01-22   SELL $192.45    0.78       macd_rsi
2024-02-05   BUY  $188.12    0.82       macd_rsi
2024-02-18   SELL $195.33    0.89       macd_rsi
...
```

### 6. All Trades Details
```
💰 ALL TRADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry Date   Exit Date    Side Entry $   Exit $    P&L       P&L %
────────────────────────────────────────────────────────────────────────────
2024-01-15   2024-01-22   BUY  $185.23   $192.45   +722.00   +3.9%
2024-02-05   2024-02-18   BUY  $188.12   $195.33   +721.00   +3.8%
2024-03-01   2024-03-08   BUY  $190.45   $187.20   -325.00   -1.7%
...
```

## Technical Details

### Architecture
- **Client**: `submit_job.py` (CLI script)
- **Queue**: Redis Stream (`jobs_stream`)
- **Processor**: `JobProcessor` in production service
- **Storage**: Redis (job status and results)

### Data Flow
1. CLI submits job to `jobs_stream`
2. Production service picks up job
3. Fetches historical data from Alpaca
4. Runs backtest using existing engine
5. Stores results in Redis
6. CLI polls status and displays results

### Redis Keys
- `jobs_stream` - Job queue
- `job:status:{job_id}` - Job status and progress
- `job:result:{job_id}` - Job results (TTL: 24 hours)

## Error Handling

### Common Errors and Solutions

#### Redis Connection Failed
```
❌ Failed to connect to Redis at localhost:6380
```
**Solution**: Ensure Wagehood Docker container is running
```bash
docker ps | grep wagehood
docker-compose up -d
```

#### Invalid Date Format
```
❌ Invalid date format: time data '2024/01/01' does not match format
```
**Solution**: Use YYYY-MM-DD format
```bash
--start 2024-01-01 --end 2024-12-31
```

#### Job Failed
```
❌ Job failed: No historical data available for INVALID_SYMBOL
```
**Solution**: Use valid trading symbols (AAPL, SPY, MSFT, etc.)

#### No Results Found
```
❌ No results found for job job_xyz
```
**Solution**: Job may have expired (24-hour TTL) or failed

## Performance Considerations

### Timeframe vs. Data Amount
- **1m timeframes**: Large data volumes, slower processing
- **1h timeframes**: Balanced data size and detail
- **1d timeframes**: Fast processing, less detail

### Date Range Recommendations
- **Day trading (1m-30m)**: 1-7 days
- **Swing trading (1h-4h)**: 1-6 months  
- **Position trading (1d-1M)**: 1-3 years

### Concurrent Jobs
- Multiple jobs can run simultaneously
- Each job is processed independently
- Results are stored separately

## Integration with Production System

### Connection to Running Instance
The CLI connects to the same Redis instance used by the production service:
- **Host**: localhost (Docker container)
- **Port**: 6380 (mapped from container's 6379)
- **Database**: 0 (default)

### Data Sources
Jobs use the same data sources as the production system:
- **Live Market Data**: Alpaca Markets API
- **Historical Data**: Alpaca Markets historical API
- **Strategies**: Same strategy implementations
- **Indicators**: Same technical indicator calculations

### Resource Usage
- Jobs run in the same environment as real-time processing
- Share CPU and memory resources with production workload
- Use configurable worker pool (default: 8 workers)

## Troubleshooting

### Check Production Service Status
```bash
docker logs wagehood-trading --tail 20
```

### Check Redis Connectivity
```bash
redis-cli -h localhost -p 6380 ping
```

### View Job Queue
```bash
redis-cli -h localhost -p 6380 XLEN jobs_stream
```

### View Job Status
```bash
redis-cli -h localhost -p 6380 HGETALL job:status:JOB_ID
```

### View Job Results
```bash
redis-cli -h localhost -p 6380 HGETALL job:result:JOB_ID
```

## Best Practices

### Strategy Selection
1. **Day Trading**: Use `rsi_trend` or `bollinger_breakout` with short timeframes
2. **Swing Trading**: Use `macd_rsi` with 1h-4h timeframes
3. **Position Trading**: Use `ma_crossover` or `sr_breakout` with daily timeframes

### Date Range Selection
1. **Testing Period**: Use recent 1-2 years for relevance
2. **Validation Period**: Test on different market conditions
3. **Development**: Start with shorter periods for faster iteration

### Performance Analysis
1. **Win Rate**: Look for >60% for reliable strategies
2. **Sharpe Ratio**: Target >1.0 for risk-adjusted returns
3. **Max Drawdown**: Keep <20% for manageable risk
4. **Trade Count**: Ensure sufficient sample size (>30 trades)

## Advanced Usage

### Batch Testing
Test multiple configurations:
```bash
# Test different timeframes
for tf in 1h 4h 1d; do
    python submit_job.py --symbol AAPL --timeframe $tf --strategy macd_rsi \
                        --start 2024-01-01 --end 2024-12-31
done

# Test different strategies
for strategy in macd_rsi rsi_trend bollinger_breakout; do
    python submit_job.py --symbol SPY --timeframe 1h --strategy $strategy \
                        --start 2024-01-01 --end 2024-12-31
done
```

### Custom Scripting
Integrate into larger analysis workflows:
```python
import subprocess
import json

def run_backtest(symbol, timeframe, strategy, start, end):
    cmd = [
        'python', 'submit_job.py',
        '--symbol', symbol,
        '--timeframe', timeframe, 
        '--strategy', strategy,
        '--start', start,
        '--end', end
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

# Run multiple tests
symbols = ['AAPL', 'MSFT', 'GOOGL']
for symbol in symbols:
    success = run_backtest(symbol, '1h', 'macd_rsi', '2024-01-01', '2024-12-31')
    print(f'{symbol}: {"✅" if success else "❌"}')
```

## Support

For issues or questions:
1. Check Docker container logs: `docker logs wagehood-trading`
2. Verify Redis connectivity: `redis-cli -h localhost -p 6380 ping`
3. Review job status in Redis: `redis-cli -h localhost -p 6380 XLEN jobs_stream`
4. Check system resources: `docker stats wagehood-trading`

The Job Submission CLI provides a powerful interface for backtesting strategies against historical data using the same production-grade infrastructure that powers real-time trading analysis.