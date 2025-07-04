"""Dependency injection setup for FastAPI."""

from typing import Annotated
from fastapi import Depends, HTTPException, Request

from ..services.data_service import DataService
from ..services.backtest_service import BacktestService
from ..services.analysis_service import AnalysisService


async def get_data_service(request: Request) -> DataService:
    """Get data service instance from app state."""
    if not hasattr(request.app.state, 'data_service'):
        raise HTTPException(
            status_code=503,
            detail="Data service not available"
        )
    return request.app.state.data_service


async def get_backtest_service(request: Request) -> BacktestService:
    """Get backtest service instance from app state."""
    if not hasattr(request.app.state, 'backtest_service'):
        raise HTTPException(
            status_code=503,
            detail="Backtest service not available"
        )
    return request.app.state.backtest_service


async def get_analysis_service(request: Request) -> AnalysisService:
    """Get analysis service instance from app state."""
    if not hasattr(request.app.state, 'analysis_service'):
        raise HTTPException(
            status_code=503,
            detail="Analysis service not available"
        )
    return request.app.state.analysis_service


# Type aliases for cleaner dependency injection
DataServiceDep = Annotated[DataService, Depends(get_data_service)]
BacktestServiceDep = Annotated[BacktestService, Depends(get_backtest_service)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]