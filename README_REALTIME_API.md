# Real-time Market Data API

This document describes the comprehensive FastAPI endpoints for the real-time market data processing system.

## Overview

The Real-time Market Data API provides four main categories of endpoints:

1. **Real-time Data Endpoints** - Access live market data, indicators, and signals
2. **Configuration Endpoints** - Manage watchlists, indicators, and system settings
3. **Monitoring Endpoints** - System health, performance metrics, and alerts
4. **Data Query Endpoints** - Historical data retrieval and bulk exports

## Base URL

All endpoints are prefixed with `/realtime` when the API is running on the default configuration:

```
http://localhost:8000/realtime
```

## Authentication

Currently using mock authentication. In production, replace with proper JWT/API key authentication.

## 1. Real-time Data Endpoints

### Get Latest Data
```http
GET /realtime/data/latest/{symbol}
```
Get the most recent real-time data for a specific symbol.

**Parameters:**
- `symbol` (path): Trading symbol (e.g., 'AAPL', 'SPY')

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "data": [
    {
      "symbol": "AAPL",
      "timestamp": "2024-01-01T12:00:00Z",
      "price": 150.25,
      "volume": 1000000,
      "bid": 150.20,
      "ask": 150.30,
      "spread": 0.10
    }
  ],
  "count": 1
}
```

### Get Latest Indicators
```http
GET /realtime/indicators/{symbol}?indicators=rsi_14,macd
```
Get the latest technical indicator values for a symbol.

**Parameters:**
- `symbol` (path): Trading symbol
- `indicators` (query): Optional list of specific indicators to retrieve

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "symbol": "AAPL",
  "indicators": [
    {
      "name": "rsi_14",
      "value": 65.5,
      "timestamp": "2024-01-01T12:00:00Z",
      "metadata": null
    },
    {
      "name": "macd",
      "values": {
        "macd": 2.5,
        "signal": 2.1,
        "histogram": 0.4
      },
      "timestamp": "2024-01-01T12:00:00Z",
      "metadata": null
    }
  ]
}
```

### Get Trading Signals
```http
GET /realtime/signals/{symbol}?strategy=macd_rsi_strategy
```
Get the latest trading signals for a symbol.

**Parameters:**
- `symbol` (path): Trading symbol
- `strategy` (query): Optional specific strategy to filter signals

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "symbol": "AAPL",
  "signals": [
    {
      "symbol": "AAPL",
      "strategy": "macd_rsi_strategy",
      "signal_type": "BUY",
      "strength": 0.85,
      "price": 150.25,
      "timestamp": "2024-01-01T12:00:00Z",
      "indicators": {
        "rsi": 65.5,
        "macd": 2.5
      },
      "metadata": {
        "confidence": "high"
      }
    }
  ]
}
```

### WebSocket Streaming
```websocket
ws://localhost:8000/realtime/ws/{connection_id}
```
Real-time data streaming via WebSocket.

**Message Format:**
```json
// Subscribe to symbols
{
  "action": "subscribe",
  "symbols": ["AAPL", "SPY"]
}

// Unsubscribe from symbols
{
  "action": "unsubscribe",
  "symbols": ["AAPL"]
}

// Ping/Pong
{
  "action": "ping"
}
```

## 2. Configuration Endpoints

### Get Watchlist
```http
GET /realtime/config/watchlist
```
Get the current watchlist configuration.

### Update Watchlist
```http
PUT /realtime/config/watchlist
```
Update the complete watchlist configuration.

### Add Symbol to Watchlist
```http
POST /realtime/config/watchlist/add
```
Add a single symbol to the watchlist.

**Request:**
```json
{
  "symbol": "AAPL",
  "data_provider": "mock",
  "timeframes": ["1m", "5m", "1h"],
  "priority": 1
}
```

### Remove Symbol from Watchlist
```http
DELETE /realtime/config/watchlist/{symbol}
```
Remove a symbol from the watchlist.

### Get Indicator Configurations
```http
GET /realtime/config/indicators
```
Get all indicator configurations.

### Update Indicator Configurations
```http
PUT /realtime/config/indicators
```
Update indicator configurations.

### Get Strategy Configurations
```http
GET /realtime/config/strategies
```
Get all strategy configurations.

### Update Strategy Configurations
```http
PUT /realtime/config/strategies
```
Update strategy configurations.

### Get System Configuration
```http
GET /realtime/config/system
```
Get system configuration.

### Update System Configuration
```http
PUT /realtime/config/system
```
Update system configuration.

### Get Configuration Summary
```http
GET /realtime/config/summary
```
Get a summary of all configurations.

### Validate Configuration
```http
GET /realtime/config/validate
```
Validate the current configuration for issues.

## 3. Monitoring Endpoints

### Get System Health
```http
GET /realtime/monitor/health
```
Get comprehensive system health information.

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z",
  "status": "healthy",
  "uptime_seconds": 3600,
  "components": {
    "ingestion": "healthy",
    "calculation": "healthy"
  },
  "statistics": {
    "running": true,
    "configuration": {...}
  },
  "alerts": []
}
```

### Get System Statistics
```http
GET /realtime/monitor/stats
```
Get detailed system statistics and performance metrics.

### Get System Alerts
```http
GET /realtime/monitor/alerts?limit=50&alert_type=error
```
Get system alerts and notifications.

**Parameters:**
- `limit` (query): Maximum number of alerts to return
- `offset` (query): Number of alerts to skip
- `alert_type` (query): Filter by alert type (error, warning, info)
- `component` (query): Filter by component name
- `acknowledged` (query): Filter by acknowledgment status

## 4. Data Query Endpoints

### Get Historical Data
```http
GET /realtime/data/historical/{symbol}?indicator=rsi_14&start_date=2024-01-01&limit=1000
```
Get historical indicator data for a symbol.

**Parameters:**
- `symbol` (path): Trading symbol
- `indicator` (query): Specific indicator to retrieve
- `start_date` (query): Start date for data query
- `end_date` (query): End date for data query
- `limit` (query): Maximum number of records to return

### Create Bulk Export
```http
POST /realtime/data/export
```
Create a bulk data export job.

**Request:**
```json
{
  "symbols": ["AAPL", "SPY", "QQQ"],
  "indicators": ["rsi_14", "macd"],
  "strategies": ["macd_rsi_strategy"],
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "format": "json"
}
```

### Get Export Status
```http
GET /realtime/data/export/{export_id}
```
Get the status of a bulk export job.

### Download Export
```http
GET /realtime/data/export/{export_id}/download
```
Download the results of a completed export job.

## CLI Client Usage

A command-line client is provided for easy interaction with the API:

```bash
# Get latest data
python src/api/cli_client.py data latest AAPL

# Get indicators
python src/api/cli_client.py data indicators AAPL --indicators rsi_14 macd

# Get trading signals
python src/api/cli_client.py data signals AAPL --strategy macd_rsi_strategy

# Get watchlist
python src/api/cli_client.py config watchlist

# Add symbol to watchlist
python src/api/cli_client.py config add GOOGL --priority 2

# Remove symbol from watchlist
python src/api/cli_client.py config remove GOOGL

# Get system health
python src/api/cli_client.py monitor health

# Get system statistics
python src/api/cli_client.py monitor stats

# Get system alerts
python src/api/cli_client.py monitor alerts

# Stream real-time data
python src/api/cli_client.py stream AAPL SPY --duration 60

# Create export job
python src/api/cli_client.py export create AAPL SPY --format json

# Check export status
python src/api/cli_client.py export status <export_id>
```

## WebSocket Client Example

```python
import asyncio
import websockets
import json

async def websocket_example():
    uri = "ws://localhost:8000/realtime/ws/my_connection"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to symbols
        await websocket.send(json.dumps({
            "action": "subscribe",
            "symbols": ["AAPL", "SPY"]
        }))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

# Run the example
asyncio.run(websocket_example())
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `422` - Validation Error (invalid request data)
- `500` - Internal Server Error
- `503` - Service Unavailable (service not initialized)

Error responses follow this format:
```json
{
  "message": "Error description",
  "detail": "Additional error details",
  "timestamp": "2024-01-01T12:00:00Z",
  "path": "/realtime/data/latest/INVALID"
}
```

## Rate Limiting

Consider implementing rate limiting for production use:
- WebSocket connections: 10 per client IP
- API requests: 1000 per minute per IP
- Export jobs: 5 concurrent per user

## Security Considerations

1. **Authentication**: Replace mock authentication with proper JWT/API key system
2. **Input Validation**: All inputs are validated using Pydantic models
3. **Error Handling**: Comprehensive error handling prevents information leakage
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **CORS**: Configure CORS appropriately for production
6. **WebSocket Security**: Add proper authentication for WebSocket connections

## Performance Optimization

1. **Caching**: Redis-based caching for frequently accessed data
2. **Connection Pooling**: Efficient database and Redis connection management
3. **Async Processing**: All operations are asynchronous for better performance
4. **Background Tasks**: Long-running operations processed in background
5. **Streaming**: Efficient data streaming for large datasets

## Monitoring and Observability

1. **Health Checks**: Comprehensive health monitoring
2. **Metrics**: Performance metrics and statistics
3. **Alerts**: Automated alert system for issues
4. **Logging**: Structured logging for debugging and monitoring
5. **Tracing**: Request tracing for performance analysis