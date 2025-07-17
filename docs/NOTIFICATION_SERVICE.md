# Discord Notification Service

The Wagehood notification service sends real-time trading signals and system alerts to Discord channels using webhooks.

## Features

- **Zero Duplicates**: Content-based deduplication prevents sending the same signal multiple times
- **Guaranteed Delivery**: Persistent SQLite queue with retry logic ensures messages are delivered
- **Channel Routing**: Signals are automatically routed to appropriate channels based on strategy and timeframe
- **Auto-Start**: Service can be configured to start automatically on system boot
- **Local Architecture**: Runs as a single worker alongside other Wagehood services

## Channel Configuration

The service uses 5 Discord channels for different types of notifications:

| Channel | Purpose | Webhook Environment Variable |
|---------|---------|------------------------------|
| `infra` | Service health and system alerts | `DISCORD_WEBHOOK_INFRA` |
| `macd-rsi` | MACD-RSI signals (1d timeframe only) | `DISCORD_WEBHOOK_MACD_RSI` |
| `support-resistance` | Support/Resistance signals (1d timeframe only) | `DISCORD_WEBHOOK_SUPPORT_RESISTANCE` |
| `rsi-trend-following` | RSI Trend signals (1h timeframe only) | `DISCORD_WEBHOOK_RSI_TREND` |
| `bollinger-band-breakout` | Bollinger Band signals (1h timeframe only) | `DISCORD_WEBHOOK_BOLLINGER` |

## Setup

### 1. Create Discord Webhooks

1. In your Discord server, go to Server Settings → Integrations → Webhooks
2. Create a webhook for each channel listed above
3. Copy the webhook URL for each channel

### 2. Configure Webhooks

Webhooks can be configured in two ways:

#### Option A: During Installation
```bash
python3 install_cli.py
```
The installer will prompt for webhook URLs during setup.

#### Option B: Manual Configuration
Add webhook URLs to your `.env` file:
```env
DISCORD_WEBHOOK_INFRA=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_MACD_RSI=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_SUPPORT_RESISTANCE=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_RSI_TREND=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_BOLLINGER=https://discord.com/api/webhooks/...
```

### 3. Start the Service

#### Manual Start
```bash
wagehood notifications start
```

#### Auto-Start on Boot
The installer can configure the notification service to start automatically:
```bash
python3 install_cli.py
# Choose option 4 (All workers) when prompted
```

## Usage

### CLI Commands

```bash
# Show configuration
wagehood notifications config

# Check service status
wagehood notifications status

# Test Discord connectivity
wagehood notifications test

# Start service
wagehood notifications start

# Stop service
wagehood notifications stop

# Restart service
wagehood notifications restart
```

### Signal Notifications

When the streaming service detects a trading signal, it automatically sends a notification to the appropriate Discord channel:

- **BUY signals**: Green indicator (🟢) with price, confidence, and technical details
- **SELL signals**: Red indicator (🔴) with price, confidence, and technical details

Example notification:
```
🟢 BUY signal for AAPL
Price: $150.25
Confidence: 85.0%
Time: 14:30:45
RSI: 65.5 | MACD: 0.1500
```

### Service Notifications

System health and operational alerts are sent to the `infra` channel:

- Service start/stop events
- Error notifications
- Performance warnings
- System statistics

## Architecture

### Components

1. **Message Queue**: SQLite-based persistent queue ensures reliable delivery
2. **Deduplication Service**: MD5 content hashing prevents duplicate messages within 5-minute windows
3. **Discord Client**: Webhook-based integration with rate limiting and retry logic
4. **Channel Router**: Routes messages to correct channels based on strategy and timeframe
5. **Notification Worker**: Single-worker process that manages the queue and sends messages

### Data Flow

```
Signal Detection → Message Queue → Deduplication → Channel Routing → Discord API
```

### Reliability Features

- **Persistent Queue**: Messages survive service restarts
- **Exponential Backoff**: Retry delays of 1s, 2s, 4s for failed messages
- **Circuit Breaker**: Prevents cascading failures
- **Health Monitoring**: Tracks queue depth, delivery rates, and errors

## Troubleshooting

### Service Won't Start

1. Check webhook configuration:
   ```bash
   wagehood notifications config
   ```

2. Verify webhook URLs are valid:
   - Must start with `https://discord.com/api/webhooks/`
   - Test each webhook in Discord

3. Check logs:
   ```bash
   tail -f ~/.wagehood/notification_workers.log
   ```

### Notifications Not Received

1. Verify service is running:
   ```bash
   wagehood notifications status
   ```

2. Test Discord connectivity:
   ```bash
   wagehood notifications test
   ```

3. Check for duplicate messages (5-minute deduplication window)

### High Memory Usage

The service automatically cleans up old deduplication records every hour. If memory usage is high:

1. Check queue depth for stuck messages
2. Restart the service to clear in-memory caches
3. Review error logs for failing webhooks

## Performance

- **Message Processing**: ~1000 messages/minute capacity
- **Deduplication Window**: 5 minutes (configurable)
- **Retry Attempts**: 3 attempts with exponential backoff
- **Queue Cleanup**: Automatic hourly cleanup of processed messages
- **Memory Usage**: ~50-100MB typical, depends on queue depth

## Security Notes

- Webhook URLs should be kept secret
- Never commit webhook URLs to version control
- Use environment variables or `.env` file for configuration
- Webhooks are write-only (cannot read Discord messages)