"""Results management endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from ..dependencies import BacktestServiceDep, AnalysisServiceDep
from ..schemas import (
    BacktestResponse,
    BacktestListResponse,
    StrategyRankingsResponse,
    BestStrategyResponse,
    BaseResponse,
    TimeFrame,
    OptimizationMetric,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_results(
    backtest_id: str,
    backtest_service: BacktestServiceDep,
) -> BacktestResponse:
    """
    Get detailed results for a specific backtest.
    
    Returns complete backtest results including metrics, trades, and equity curve.
    """
    try:
        logger.info(f"Getting backtest results for ID: {backtest_id}")
        
        # Get backtest results using service
        result = await backtest_service.get_backtest_results(backtest_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest {backtest_id} not found"
            )
        
        logger.info(f"Retrieved backtest results with {len(result['trades'])} trades")
        
        return BacktestResponse(
            backtest_id=backtest_id,
            symbol=result['symbol'],
            timeframe=result['timeframe'],
            strategy=result['strategy'],
            parameters=result['parameters'],
            metrics=result['metrics'],
            trades=result['trades'],
            equity_curve=result['equity_curve']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get backtest results: {str(e)}"
        )


@router.get("/", response_model=BacktestListResponse)
async def list_backtest_results(
    backtest_service: BacktestServiceDep,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    limit: Optional[int] = Query(50, description="Maximum number of results"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
) -> BacktestListResponse:
    """
    List backtest results with optional filtering.
    
    Returns a paginated list of backtest results with basic information.
    """
    try:
        logger.info("Getting backtest results list")
        
        # Get backtest list using service
        results = await backtest_service.list_backtest_results(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"Retrieved {len(results['backtests'])} backtest results")
        
        return BacktestListResponse(
            backtests=results['backtests'],
            total_count=results['total_count']
        )
        
    except Exception as e:
        logger.error(f"Error listing backtest results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list backtest results: {str(e)}"
        )


@router.get("/rankings", response_model=StrategyRankingsResponse)
async def get_strategy_rankings(
    analysis_service: AnalysisServiceDep,
    metric: OptimizationMetric = Query(
        OptimizationMetric.SHARPE_RATIO,
        description="Metric to rank strategies by"
    ),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
    limit: Optional[int] = Query(20, description="Maximum number of results"),
) -> StrategyRankingsResponse:
    """
    Get strategy rankings based on performance metrics.
    
    Returns strategies ranked by the specified metric across all backtests.
    """
    try:
        logger.info(f"Getting strategy rankings by {metric}")
        
        # Get strategy rankings using service
        rankings = await analysis_service.get_strategy_rankings(
            metric=metric,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(rankings)} strategy rankings")
        
        return StrategyRankingsResponse(
            rankings=rankings,
            metric_used=metric.value
        )
        
    except Exception as e:
        logger.error(f"Error getting strategy rankings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get strategy rankings: {str(e)}"
        )


@router.get("/best-strategy/{symbol}/{timeframe}", response_model=BestStrategyResponse)
async def get_best_strategy(
    symbol: str,
    timeframe: TimeFrame,
    analysis_service: AnalysisServiceDep,
    metric: OptimizationMetric = Query(
        OptimizationMetric.SHARPE_RATIO,
        description="Metric to determine best strategy"
    ),
) -> BestStrategyResponse:
    """
    Get the best performing strategy for a specific symbol and timeframe.
    
    Returns the strategy with the highest score for the specified metric.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting best strategy for {symbol} ({timeframe}) by {metric}")
        
        # Get best strategy using service
        best_strategy = await analysis_service.get_best_strategy(
            symbol=symbol,
            timeframe=timeframe,
            metric=metric
        )
        
        if not best_strategy:
            raise HTTPException(
                status_code=404,
                detail=f"No strategies found for {symbol} ({timeframe})"
            )
        
        logger.info(f"Best strategy: {best_strategy['strategy']}")
        
        return BestStrategyResponse(
            symbol=symbol,
            timeframe=timeframe.value,
            strategy=best_strategy['strategy'],
            parameters=best_strategy['parameters'],
            metrics=best_strategy['metrics'],
            backtest_id=best_strategy['backtest_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best strategy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get best strategy: {str(e)}"
        )


@router.delete("/{backtest_id}", response_model=BaseResponse)
async def delete_backtest_results(
    backtest_id: str,
    backtest_service: BacktestServiceDep,
) -> BaseResponse:
    """
    Delete a specific backtest and its results.
    
    This permanently removes the backtest data and cannot be undone.
    """
    try:
        logger.info(f"Deleting backtest results for ID: {backtest_id}")
        
        # Check if backtest exists
        result = await backtest_service.get_backtest_results(backtest_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # Delete backtest using service
        await backtest_service.delete_backtest_results(backtest_id)
        
        logger.info(f"Successfully deleted backtest {backtest_id}")
        
        return BaseResponse()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backtest results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete backtest results: {str(e)}"
        )


@router.delete("/", response_model=BaseResponse)
async def delete_multiple_backtest_results(
    backtest_service: BacktestServiceDep,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    older_than_days: Optional[int] = Query(None, description="Delete results older than N days"),
) -> BaseResponse:
    """
    Delete multiple backtest results based on filters.
    
    This permanently removes matching backtest data and cannot be undone.
    Use with caution.
    """
    try:
        logger.info("Deleting multiple backtest results")
        
        # Delete backtests using service
        deleted_count = await backtest_service.delete_multiple_backtest_results(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            older_than_days=older_than_days
        )
        
        logger.info(f"Successfully deleted {deleted_count} backtest results")
        
        return BaseResponse()
        
    except Exception as e:
        logger.error(f"Error deleting multiple backtest results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete backtest results: {str(e)}"
        )


@router.get("/export/{backtest_id}")
async def export_backtest_results(
    backtest_id: str,
    backtest_service: BacktestServiceDep,
    format: str = Query("json", description="Export format (json, csv, xlsx)"),
) -> dict:
    """
    Export backtest results in various formats.
    
    Supported formats: json, csv, xlsx
    """
    try:
        logger.info(f"Exporting backtest results for ID: {backtest_id} in {format} format")
        
        # Check if backtest exists
        result = await backtest_service.get_backtest_results(backtest_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest {backtest_id} not found"
            )
        
        # Export results using service
        export_data = await backtest_service.export_backtest_results(
            backtest_id=backtest_id,
            format=format
        )
        
        logger.info(f"Successfully exported backtest {backtest_id}")
        
        return {
            "success": True,
            "backtest_id": backtest_id,
            "format": format,
            "data": export_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting backtest results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export backtest results: {str(e)}"
        )


@router.get("/summary")
async def get_results_summary(
    backtest_service: BacktestServiceDep,
) -> dict:
    """
    Get summary statistics across all backtest results.
    
    Returns aggregate statistics and insights across all stored backtests.
    """
    try:
        logger.info("Getting results summary")
        
        # Get summary using service
        summary = await backtest_service.get_results_summary()
        
        logger.info("Retrieved results summary")
        
        return {
            "success": True,
            **summary
        }
        
    except Exception as e:
        logger.error(f"Error getting results summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get results summary: {str(e)}"
        )