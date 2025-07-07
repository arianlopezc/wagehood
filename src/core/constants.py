"""System-wide constants"""

# Trading Constants
DEFAULT_COMMISSION = 0.0  # Commission-free trading (unless specified)
DEFAULT_SLIPPAGE = 0.0005  # 0.05% slippage
MIN_DATA_POINTS = 200  # Minimum data points for analysis
MAX_POSITION_SIZE = 0.1  # Maximum 10% of portfolio per position
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# Indicator Defaults
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD_DEV = 2.0

MA_FAST = 50
MA_SLOW = 200

# Performance Thresholds
MIN_WIN_RATE = 0.4  # Minimum 40% win rate
MIN_PROFIT_FACTOR = 1.2  # Minimum 1.2 profit factor
MAX_DRAWDOWN_PCT = 0.2  # Maximum 20% drawdown

# System Limits
MAX_MEMORY_USAGE_MB = 1024  # 1GB memory limit
MAX_CONCURRENT_BACKTESTS = 5

# Cache Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL
LOCAL_CACHE_SIZE = 10000  # Maximum items in local cache (for optimization only)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
USE_LOCAL_ONLY = False  # Set to True ONLY for testing/development
REDIS_MAX_MEMORY = "2gb"  # Maximum Redis memory usage
REDIS_EVICTION_POLICY = "allkeys-lru"  # Memory eviction policy

# Redis is the PRIMARY DATASTORE - not a cache
# Local cache is used only for small object optimization and testing
# If Redis fails in production, the system should fail fast

# Mock Data Parameters
DEFAULT_VOLATILITY = 0.02  # 2% daily volatility
DEFAULT_TREND_STRENGTH = 0.0005  # 0.05% daily trend
DEFAULT_VOLUME_RATIO = 1.0  # Volume multiplier

# Supported Assets
SUPPORTED_SYMBOLS = [
    # Major Indices
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    # Major Cryptocurrencies
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    # Major Forex Pairs
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    # Major Commodities
    "GLD",
    "USO",
    "SLV",
]

# API Configuration removed - system uses Redis-based worker model
