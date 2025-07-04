# Trading System API

A comprehensive FastAPI-based REST API for the advanced trading system with backtesting and analysis capabilities.

## Features

- **Data Management**: Upload, retrieve, and manage market data
- **Strategy Backtesting**: Run backtests with various trading strategies
- **Technical Analysis**: Calculate technical indicators and perform analysis
- **Parameter Optimization**: Optimize strategy parameters using grid search
- **Strategy Comparison**: Compare multiple strategies side-by-side
- **Results Management**: Store, retrieve, and analyze backtest results
- **Async Operations**: Support for long-running background tasks
- **Comprehensive Documentation**: Auto-generated OpenAPI/Swagger docs

## Installation

1. Install dependencies:
```bash
pip install -r requirements-api.txt
```

2. Run the API server:
```bash
python run_api.py
```

Or with custom settings:
```bash
python run_api.py --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

Once running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health check with service status

### Data Management
- `POST /data/upload` - Upload market data
- `GET /data/{symbol}/{timeframe}` - Get historical data
- `GET /data/symbols` - List available symbols
- `GET /data/{symbol}/{timeframe}/info` - Get data information
- `DELETE /data/{symbol}` - Clear symbol data

### Analysis
- `POST /analysis/backtest` - Run backtest
- `POST /analysis/backtest/async` - Run backtest asynchronously
- `GET /analysis/backtest/status/{job_id}` - Get async backtest status
- `POST /analysis/indicators` - Calculate technical indicators
- `POST /analysis/optimize` - Optimize strategy parameters
- `POST /analysis/optimize/async` - Run optimization asynchronously
- `GET /analysis/optimize/status/{job_id}` - Get optimization status
- `POST /analysis/compare` - Compare strategies
- `GET /analysis/strategies` - Get available strategies

### Results
- `GET /results/{backtest_id}` - Get backtest results
- `GET /results/` - List backtest results
- `GET /results/rankings` - Get strategy rankings
- `GET /results/best-strategy/{symbol}/{timeframe}` - Get best strategy
- `GET /results/export/{backtest_id}` - Export results
- `GET /results/summary` - Get results summary
- `DELETE /results/{backtest_id}` - Delete backtest results

## Usage Examples

### 1. Upload Market Data

```python
import requests

# Sample OHLCV data
data = [
    {
        "timestamp": "2023-01-01T00:00:00",
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000
    },
    # ... more data points
]

response = requests.post("http://localhost:8000/data/upload", json={
    "symbol": "AAPL",
    "timeframe": "daily",
    "data": data
})
```

### 2. Run a Backtest

```python
response = requests.post("http://localhost:8000/analysis/backtest", json={
    "symbol": "AAPL",
    "timeframe": "daily",
    "strategy": "sma_crossover",
    "parameters": {
        "parameters": {
            "fast_period": 10,
            "slow_period": 20
        }
    },
    "initial_capital": 10000.0,
    "commission": 0.001
})

backtest_result = response.json()
print(f"Backtest ID: {backtest_result['backtest_id']}")
print(f"Total Return: {backtest_result['metrics']['total_return']:.2%}")
```

### 3. Optimize Strategy Parameters

```python
response = requests.post("http://localhost:8000/analysis/optimize", json={
    "symbol": "AAPL",
    "timeframe": "daily",
    "strategy": "sma_crossover",
    "parameter_ranges": {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [20, 30, 40, 50]
    },
    "optimization_metric": "sharpe_ratio",
    "initial_capital": 10000.0,
    "commission": 0.001
})

optimization_result = response.json()
print(f"Best parameters: {optimization_result['best_parameters']}")
```

### 4. Calculate Technical Indicators

```python
response = requests.post("http://localhost:8000/analysis/indicators", json={
    "symbol": "AAPL",
    "timeframe": "daily",
    "indicators": ["SMA", "RSI", "MACD"],
    "parameters": {
        "SMA": {"period": 20},
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9}
    }
})

indicators = response.json()
print(f"Calculated indicators: {list(indicators['indicators'].keys())}")
```

### 5. Compare Strategies

```python
response = requests.post("http://localhost:8000/analysis/compare", json={
    "symbol": "AAPL",
    "timeframe": "daily",
    "strategies": [
        {
            "strategy": "sma_crossover",
            "parameters": {"fast_period": 10, "slow_period": 20}
        },
        {
            "strategy": "rsi_oversold",
            "parameters": {"rsi_period": 14, "oversold_level": 30}
        }
    ],
    "initial_capital": 10000.0,
    "commission": 0.001
})

comparison = response.json()
print(f"Best strategy: {comparison['best_strategy']['strategy']}")
```

## Supported Timeframes

- `1min`, `5min`, `15min`, `30min` - Intraday
- `1hour`, `4hour` - Hourly
- `daily`, `weekly`, `monthly` - Daily and above

## Supported Strategies

- **SMA Crossover**: Simple Moving Average crossover
- **EMA Crossover**: Exponential Moving Average crossover
- **RSI Oversold**: RSI-based oversold/overbought
- **MACD Signal**: MACD signal line crossover
- **Bollinger Bands**: Bollinger Bands mean reversion

## Technical Indicators

- **Moving Averages**: SMA, EMA, WMA
- **Momentum**: RSI, MACD, Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume-based indicators
- **Custom**: Support for custom indicators

## Error Handling

The API uses standard HTTP status codes and returns structured error responses:

```json
{
  "message": "Error description",
  "detail": "Additional error details",
  "timestamp": "2023-01-01T12:00:00",
  "path": "/api/endpoint"
}
```

## Testing

Run the test suite:
```bash
python test_api.py
```

## Architecture

The API follows a clean architecture pattern:

```
src/api/
├── __init__.py          # Module exports
├── app.py               # FastAPI application
├── dependencies.py      # Dependency injection
├── schemas.py           # Pydantic models
├── routes/
│   ├── data.py         # Data endpoints
│   ├── analysis.py     # Analysis endpoints
│   └── results.py      # Results endpoints
└── services/
    ├── data_service.py     # Data management
    ├── backtest_service.py # Backtesting
    └── analysis_service.py # Analysis operations
```

## Performance Considerations

- **Async Operations**: Long-running operations support async execution
- **Caching**: Results are cached for improved performance
- **Pagination**: Large result sets are paginated
- **Background Tasks**: CPU-intensive tasks run in the background

## Security

- **Input Validation**: All inputs are validated using Pydantic models
- **Error Handling**: Comprehensive error handling prevents information leakage
- **CORS**: Configurable CORS settings for web applications
- **Rate Limiting**: Can be configured for production use

## Production Deployment

For production deployment, consider:

1. **Environment Variables**: Use environment variables for configuration
2. **Database**: Configure persistent storage for data and results
3. **Load Balancing**: Use a reverse proxy like Nginx
4. **Monitoring**: Add monitoring and logging
5. **Security**: Implement authentication and authorization

Example production command:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.app:app
```

## License

This project is part of the advanced trading system and follows the same license terms.