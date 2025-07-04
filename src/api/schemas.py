"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class TimeFrame(str, Enum):
    """Supported timeframes."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StrategyType(str, Enum):
    """Supported strategy types."""
    SMA_CROSSOVER = "sma_crossover"
    EMA_CROSSOVER = "ema_crossover"
    RSI_OVERSOLD = "rsi_oversold"
    MACD_SIGNAL = "macd_signal"
    BOLLINGER_BANDS = "bollinger_bands"
    CUSTOM = "custom"


class OptimizationMetric(str, Enum):
    """Optimization metrics."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"


# Base models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    message: str
    detail: Optional[Union[str, List[Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None


# Data models
class DataUploadRequest(BaseModel):
    """Data upload request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    data: List[Dict[str, Any]] = Field(..., description="Market data records")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class DataUploadResponse(BaseResponse):
    """Data upload response model."""
    symbol: str
    timeframe: str
    records_count: int
    message: str


class MarketDataPoint(BaseModel):
    """Market data point model."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class MarketDataResponse(BaseResponse):
    """Market data response model."""
    symbol: str
    timeframe: str
    data: List[MarketDataPoint]
    total_records: int


class SymbolsResponse(BaseResponse):
    """Available symbols response model."""
    symbols: List[str]


# Analysis models
class StrategyParameters(BaseModel):
    """Strategy parameters model."""
    parameters: Dict[str, Any] = Field(..., description="Strategy-specific parameters")


class BacktestRequest(BaseModel):
    """Backtest request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    strategy: StrategyType = Field(..., description="Strategy type")
    parameters: StrategyParameters = Field(..., description="Strategy parameters")
    start_date: Optional[datetime] = Field(None, description="Backtest start date")
    end_date: Optional[datetime] = Field(None, description="Backtest end date")
    initial_capital: float = Field(10000.0, description="Initial capital")
    commission: float = Field(0.001, description="Commission rate")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class BacktestMetrics(BaseModel):
    """Backtest metrics model."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_trades: int
    avg_trade: float
    best_trade: float
    worst_trade: float


class Trade(BaseModel):
    """Trade model."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    return_pct: Optional[float] = None


class BacktestResponse(BaseResponse):
    """Backtest response model."""
    backtest_id: str
    symbol: str
    timeframe: str
    strategy: str
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: List[Dict[str, Any]]


class IndicatorRequest(BaseModel):
    """Indicator calculation request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    indicators: List[str] = Field(..., description="List of indicators to calculate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class IndicatorResponse(BaseResponse):
    """Indicator calculation response model."""
    symbol: str
    timeframe: str
    indicators: Dict[str, List[float]]


class OptimizationRequest(BaseModel):
    """Optimization request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    strategy: StrategyType = Field(..., description="Strategy type")
    parameter_ranges: Dict[str, List[Any]] = Field(..., description="Parameter ranges to optimize")
    optimization_metric: OptimizationMetric = Field(..., description="Metric to optimize")
    initial_capital: float = Field(10000.0, description="Initial capital")
    commission: float = Field(0.001, description="Commission rate")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class OptimizationResult(BaseModel):
    """Single optimization result."""
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    rank: int


class OptimizationResponse(BaseResponse):
    """Optimization response model."""
    symbol: str
    timeframe: str
    strategy: str
    optimization_metric: str
    results: List[OptimizationResult]
    best_parameters: Dict[str, Any]
    best_metrics: BacktestMetrics


class StrategyComparisonRequest(BaseModel):
    """Strategy comparison request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    strategies: List[Dict[str, Any]] = Field(..., description="List of strategies to compare")
    initial_capital: float = Field(10000.0, description="Initial capital")
    commission: float = Field(0.001, description="Commission rate")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class StrategyComparison(BaseModel):
    """Strategy comparison result."""
    strategy: str
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    rank: int


class StrategyComparisonResponse(BaseResponse):
    """Strategy comparison response model."""
    symbol: str
    timeframe: str
    comparisons: List[StrategyComparison]
    best_strategy: Dict[str, Any]


# Results models
class BacktestListItem(BaseModel):
    """Backtest list item model."""
    backtest_id: str
    symbol: str
    timeframe: str
    strategy: str
    created_at: datetime
    metrics: BacktestMetrics


class BacktestListResponse(BaseResponse):
    """Backtest list response model."""
    backtests: List[BacktestListItem]
    total_count: int


class StrategyRanking(BaseModel):
    """Strategy ranking model."""
    strategy: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    rank: int


class StrategyRankingsResponse(BaseResponse):
    """Strategy rankings response model."""
    rankings: List[StrategyRanking]
    metric_used: str


class BestStrategyResponse(BaseResponse):
    """Best strategy response model."""
    symbol: str
    timeframe: str
    strategy: str
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    backtest_id: str


# Real-time data models
class RealtimeDataPoint(BaseModel):
    """Real-time market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


class IndicatorValue(BaseModel):
    """Technical indicator value."""
    name: str
    value: Optional[float] = None
    values: Optional[Dict[str, float]] = None  # For multi-value indicators like MACD
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class TradingSignal(BaseModel):
    """Trading signal model."""
    symbol: str
    strategy: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # Signal strength 0-1
    price: float
    timestamp: datetime
    indicators: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RealtimeDataResponse(BaseResponse):
    """Real-time data response model."""
    data: List[RealtimeDataPoint]
    count: int


class IndicatorResponse(BaseResponse):
    """Indicator values response model."""
    symbol: str
    indicators: List[IndicatorValue]


class SignalResponse(BaseResponse):
    """Trading signals response model."""
    symbol: str
    signals: List[TradingSignal]


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str  # 'price', 'indicator', 'signal', 'alert'
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime


# Configuration management models
class AssetConfigModel(BaseModel):
    """Asset configuration model."""
    symbol: str
    enabled: bool
    data_provider: str
    timeframes: List[str]
    priority: int = 1
    last_updated: Optional[datetime] = None


class IndicatorConfigModel(BaseModel):
    """Indicator configuration model."""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    update_frequency_seconds: int = 1
    ttl_seconds: int = 300


class StrategyConfigModel(BaseModel):
    """Strategy configuration model."""
    name: str
    enabled: bool
    parameters: Dict[str, Any]
    required_indicators: List[str]
    update_frequency_seconds: int = 1
    ttl_seconds: int = 600


class SystemConfigModel(BaseModel):
    """System configuration model."""
    max_concurrent_calculations: int = 100
    batch_calculation_size: int = 10
    data_update_interval_seconds: int = 1
    calculation_workers: int = 4
    redis_streams_maxlen: int = 10000
    enable_monitoring: bool = True
    enable_alerts: bool = True


class WatchlistRequest(BaseModel):
    """Watchlist update request model."""
    assets: List[AssetConfigModel]


class WatchlistResponse(BaseResponse):
    """Watchlist response model."""
    assets: List[AssetConfigModel]


class AddSymbolRequest(BaseModel):
    """Add symbol request model."""
    symbol: str
    data_provider: Optional[str] = None
    timeframes: Optional[List[str]] = None
    priority: int = 1

    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class ConfigurationSummaryResponse(BaseResponse):
    """Configuration summary response model."""
    watchlist: Dict[str, Any]
    indicators: Dict[str, Any]
    strategies: Dict[str, Any]
    system: Optional[Dict[str, Any]]
    last_updated: str


class SystemHealthResponse(BaseResponse):
    """System health response model."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    uptime_seconds: Optional[float] = None
    components: Dict[str, str]  # component_name -> status
    statistics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    component: str
    events_processed: int
    events_per_second: float
    errors: int
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_update: datetime


class SystemStatsResponse(BaseResponse):
    """System statistics response model."""
    uptime_seconds: float
    running: bool
    configuration: Dict[str, Any]
    ingestion: Optional[Dict[str, Any]] = None
    calculation: Optional[Dict[str, Any]] = None
    performance: List[PerformanceMetrics]


class AlertModel(BaseModel):
    """Alert model."""
    id: str
    type: str  # 'error', 'warning', 'info'
    component: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    metadata: Optional[Dict[str, Any]] = None


class AlertsResponse(BaseResponse):
    """Alerts response model."""
    alerts: List[AlertModel]
    total_count: int
    unacknowledged_count: int


# Data query models
class HistoricalDataQuery(BaseModel):
    """Historical data query model."""
    symbol: str
    indicator: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 1000
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class HistoricalDataResponse(BaseResponse):
    """Historical data response model."""
    symbol: str
    indicator: Optional[str] = None
    data: List[Dict[str, Any]]
    total_count: int
    query_params: Dict[str, Any]


class BulkExportRequest(BaseModel):
    """Bulk data export request model."""
    symbols: List[str]
    indicators: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: str = "json"  # json, csv, parquet
    
    @validator('symbols')
    def validate_symbols(cls, v):
        return [symbol.upper().strip() for symbol in v]


class BulkExportResponse(BaseResponse):
    """Bulk export response model."""
    export_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    records_count: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# Validation response models
class ValidationResult(BaseModel):
    """Configuration validation result."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]