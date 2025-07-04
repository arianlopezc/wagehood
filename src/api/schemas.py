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