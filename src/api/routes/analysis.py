"""Analysis endpoints for backtesting and strategy optimization."""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
import asyncio

from ..dependencies import BacktestServiceDep, AnalysisServiceDep, DataServiceDep
from ..schemas import (
    BacktestRequest,
    BacktestResponse,
    IndicatorRequest,
    IndicatorResponse,
    OptimizationRequest,
    OptimizationResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    BaseResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    backtest_service: BacktestServiceDep,
    data_service: DataServiceDep,
) -> BacktestResponse:
    """
    Run a backtest for a specific strategy and symbol.
    
    This endpoint executes a complete backtest using the provided strategy
    and parameters, returning detailed results including metrics and trades.
    """
    try:
        logger.info(f"Running backtest for {request.symbol} ({request.timeframe}) using {request.strategy}")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Run backtest using service
        result = await backtest_service.run_backtest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            parameters=request.parameters.parameters,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        logger.info(f"Backtest completed with {result['metrics']['total_trades']} trades")
        
        return BacktestResponse(
            backtest_id=result['backtest_id'],
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            strategy=request.strategy.value,
            parameters=request.parameters.parameters,
            metrics=result['metrics'],
            trades=result['trades'],
            equity_curve=result['equity_curve']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run backtest: {str(e)}"
        )


@router.post("/backtest/async")
async def run_backtest_async(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    backtest_service: BacktestServiceDep,
    data_service: DataServiceDep,
) -> Dict[str, Any]:
    """
    Run a backtest asynchronously in the background.
    
    Returns a job ID that can be used to check the status and retrieve results.
    """
    try:
        logger.info(f"Starting async backtest for {request.symbol} ({request.timeframe})")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Start backtest in background
        job_id = await backtest_service.run_backtest_async(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            parameters=request.parameters.parameters,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        logger.info(f"Async backtest started with job ID: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Backtest started successfully",
            "status_url": f"/analysis/backtest/status/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async backtest: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start backtest: {str(e)}"
        )


@router.get("/backtest/status/{job_id}")
async def get_backtest_status(
    job_id: str,
    backtest_service: BacktestServiceDep,
) -> Dict[str, Any]:
    """
    Get the status of an asynchronous backtest job.
    """
    try:
        status = await backtest_service.get_backtest_status(job_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest job {job_id} not found"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            **status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get backtest status: {str(e)}"
        )


@router.post("/indicators", response_model=IndicatorResponse)
async def calculate_indicators(
    request: IndicatorRequest,
    analysis_service: AnalysisServiceDep,
    data_service: DataServiceDep,
) -> IndicatorResponse:
    """
    Calculate technical indicators for a symbol and timeframe.
    
    Supported indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - BB (Bollinger Bands)
    - ATR (Average True Range)
    - Stochastic
    """
    try:
        logger.info(f"Calculating indicators for {request.symbol} ({request.timeframe})")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Calculate indicators using service
        indicators = await analysis_service.calculate_indicators(
            symbol=request.symbol,
            timeframe=request.timeframe,
            indicators=request.indicators,
            parameters=request.parameters
        )
        
        logger.info(f"Calculated {len(indicators)} indicators")
        
        return IndicatorResponse(
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            indicators=indicators
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate indicators: {str(e)}"
        )


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_strategy(
    request: OptimizationRequest,
    analysis_service: AnalysisServiceDep,
    data_service: DataServiceDep,
) -> OptimizationResponse:
    """
    Optimize strategy parameters using grid search or other optimization methods.
    
    This endpoint finds the best parameter combinations for a given strategy
    by testing multiple parameter sets and ranking them by the specified metric.
    """
    try:
        logger.info(f"Optimizing {request.strategy} for {request.symbol} ({request.timeframe})")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Run optimization using service
        result = await analysis_service.optimize_strategy(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            parameter_ranges=request.parameter_ranges,
            optimization_metric=request.optimization_metric,
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        logger.info(f"Optimization completed with {len(result['results'])} parameter combinations")
        
        return OptimizationResponse(
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            strategy=request.strategy.value,
            optimization_metric=request.optimization_metric.value,
            results=result['results'],
            best_parameters=result['best_parameters'],
            best_metrics=result['best_metrics']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize strategy: {str(e)}"
        )


@router.post("/optimize/async")
async def optimize_strategy_async(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    analysis_service: AnalysisServiceDep,
    data_service: DataServiceDep,
) -> Dict[str, Any]:
    """
    Run strategy optimization asynchronously in the background.
    
    Returns a job ID that can be used to check the status and retrieve results.
    """
    try:
        logger.info(f"Starting async optimization for {request.symbol} ({request.timeframe})")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Start optimization in background
        job_id = await analysis_service.optimize_strategy_async(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            parameter_ranges=request.parameter_ranges,
            optimization_metric=request.optimization_metric,
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        logger.info(f"Async optimization started with job ID: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Optimization started successfully",
            "status_url": f"/analysis/optimize/status/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start optimization: {str(e)}"
        )


@router.get("/optimize/status/{job_id}")
async def get_optimization_status(
    job_id: str,
    analysis_service: AnalysisServiceDep,
) -> Dict[str, Any]:
    """
    Get the status of an asynchronous optimization job.
    """
    try:
        status = await analysis_service.get_optimization_status(job_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization job {job_id} not found"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            **status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization status: {str(e)}"
        )


@router.post("/compare", response_model=StrategyComparisonResponse)
async def compare_strategies(
    request: StrategyComparisonRequest,
    analysis_service: AnalysisServiceDep,
    data_service: DataServiceDep,
) -> StrategyComparisonResponse:
    """
    Compare multiple strategies on the same symbol and timeframe.
    
    This endpoint runs backtests for multiple strategies and returns
    a comparison of their performance metrics.
    """
    try:
        logger.info(f"Comparing {len(request.strategies)} strategies for {request.symbol} ({request.timeframe})")
        
        # Validate that data exists
        data_info = await data_service.get_data_info(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        if not data_info:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.symbol} ({request.timeframe})"
            )
        
        # Run strategy comparison using service
        result = await analysis_service.compare_strategies(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategies=request.strategies,
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        logger.info(f"Strategy comparison completed")
        
        return StrategyComparisonResponse(
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            comparisons=result['comparisons'],
            best_strategy=result['best_strategy']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare strategies: {str(e)}"
        )


@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """
    Get list of available strategies and their parameter requirements.
    """
    try:
        strategies = {
            "sma_crossover": {
                "name": "SMA Crossover",
                "description": "Simple Moving Average crossover strategy",
                "parameters": {
                    "fast_period": {"type": "int", "default": 10, "min": 1, "max": 100},
                    "slow_period": {"type": "int", "default": 20, "min": 1, "max": 200}
                }
            },
            "ema_crossover": {
                "name": "EMA Crossover",
                "description": "Exponential Moving Average crossover strategy",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 1, "max": 100},
                    "slow_period": {"type": "int", "default": 26, "min": 1, "max": 200}
                }
            },
            "rsi_oversold": {
                "name": "RSI Oversold/Overbought",
                "description": "RSI-based oversold/overbought strategy",
                "parameters": {
                    "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 50},
                    "oversold_level": {"type": "float", "default": 30.0, "min": 10.0, "max": 40.0},
                    "overbought_level": {"type": "float", "default": 70.0, "min": 60.0, "max": 90.0}
                }
            },
            "macd_signal": {
                "name": "MACD Signal",
                "description": "MACD signal line crossover strategy",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 1, "max": 50},
                    "slow_period": {"type": "int", "default": 26, "min": 1, "max": 100},
                    "signal_period": {"type": "int", "default": 9, "min": 1, "max": 50}
                }
            },
            "bollinger_bands": {
                "name": "Bollinger Bands",
                "description": "Bollinger Bands mean reversion strategy",
                "parameters": {
                    "period": {"type": "int", "default": 20, "min": 5, "max": 100},
                    "std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0}
                }
            }
        }
        
        return {
            "success": True,
            "strategies": strategies
        }
        
    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available strategies: {str(e)}"
        )