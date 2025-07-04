"""API module exports."""

from .app import app
from .dependencies import get_data_service, get_backtest_service, get_analysis_service
from .schemas import (
    BacktestRequest,
    BacktestResponse,
    DataUploadRequest,
    IndicatorRequest,
    OptimizationRequest,
    StrategyComparisonRequest,
    ErrorResponse,
)

__all__ = [
    "app",
    "get_data_service",
    "get_backtest_service",
    "get_analysis_service",
    "BacktestRequest",
    "BacktestResponse",
    "DataUploadRequest",
    "IndicatorRequest",
    "OptimizationRequest",
    "StrategyComparisonRequest",
    "ErrorResponse",
]