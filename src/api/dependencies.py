"""Dependency injection setup for FastAPI."""

from typing import Annotated
from fastapi import Depends, HTTPException, Request

from ..services.data_service import DataService
from ..services.backtest_service import BacktestService
from ..services.analysis_service import AnalysisService
from ..realtime.stream_processor import StreamProcessor
from ..realtime.config_manager import ConfigManager


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


async def get_stream_processor(request: Request) -> StreamProcessor:
    """Get stream processor instance from app state."""
    if not hasattr(request.app.state, 'stream_processor'):
        raise HTTPException(
            status_code=503,
            detail="Stream processor not available"
        )
    return request.app.state.stream_processor


async def get_config_manager(request: Request) -> ConfigManager:
    """Get config manager instance from app state."""
    if not hasattr(request.app.state, 'config_manager'):
        raise HTTPException(
            status_code=503,
            detail="Config manager not available"
        )
    return request.app.state.config_manager


# Mock user authentication (replace with actual authentication)
async def get_current_user(request: Request) -> dict:
    """Get current user from request (mock implementation)."""
    # In a real implementation, this would validate JWT tokens, API keys, etc.
    return {"user_id": "mock_user", "permissions": ["read", "write"]}


# Type aliases for cleaner dependency injection
DataServiceDep = Annotated[DataService, Depends(get_data_service)]
BacktestServiceDep = Annotated[BacktestService, Depends(get_backtest_service)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
StreamProcessorDep = Annotated[StreamProcessor, Depends(get_stream_processor)]
ConfigManagerDep = Annotated[ConfigManager, Depends(get_config_manager)]
CurrentUserDep = Annotated[dict, Depends(get_current_user)]