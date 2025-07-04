"""Data management endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from ..dependencies import DataServiceDep
from ..schemas import (
    DataUploadRequest,
    DataUploadResponse,
    MarketDataResponse,
    SymbolsResponse,
    BaseResponse,
    TimeFrame,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DataUploadResponse)
async def upload_market_data(
    request: DataUploadRequest,
    data_service: DataServiceDep,
) -> DataUploadResponse:
    """
    Upload market data for a symbol and timeframe.
    
    This endpoint accepts market data in the following format:
    - timestamp: ISO format datetime
    - open, high, low, close: price values
    - volume: optional volume data
    """
    try:
        logger.info(f"Uploading data for {request.symbol} ({request.timeframe})")
        
        # Validate data format
        if not request.data:
            raise HTTPException(
                status_code=400,
                detail="No data provided"
            )
        
        # Upload data using service
        result = await data_service.upload_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            data=request.data
        )
        
        logger.info(f"Successfully uploaded {result['records_count']} records")
        
        return DataUploadResponse(
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            records_count=result['records_count'],
            message=f"Successfully uploaded {result['records_count']} records"
        )
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload data: {str(e)}"
        )


@router.get("/{symbol}/{timeframe}", response_model=MarketDataResponse)
async def get_historical_data(
    symbol: str,
    timeframe: TimeFrame,
    data_service: DataServiceDep,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: Optional[int] = Query(None, description="Maximum number of records"),
) -> MarketDataResponse:
    """
    Get historical market data for a symbol and timeframe.
    
    Parameters:
    - symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
    - timeframe: Data timeframe (1min, 5min, 15min, 30min, 1hour, 4hour, daily, weekly, monthly)
    - start_date: Optional start date filter
    - end_date: Optional end date filter
    - limit: Optional limit on number of records returned
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting historical data for {symbol} ({timeframe})")
        
        # Get data using service
        data = await data_service.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol} ({timeframe})"
            )
        
        logger.info(f"Retrieved {len(data)} records")
        
        return MarketDataResponse(
            symbol=symbol,
            timeframe=timeframe.value,
            data=data,
            total_records=len(data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get historical data: {str(e)}"
        )


@router.get("/symbols", response_model=SymbolsResponse)
async def get_available_symbols(
    data_service: DataServiceDep,
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
) -> SymbolsResponse:
    """
    Get list of available trading symbols.
    
    Parameters:
    - timeframe: Optional filter by timeframe
    """
    try:
        logger.info("Getting available symbols")
        
        # Get symbols using service
        symbols = await data_service.get_available_symbols(timeframe=timeframe)
        
        logger.info(f"Found {len(symbols)} symbols")
        
        return SymbolsResponse(symbols=symbols)
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get symbols: {str(e)}"
        )


@router.delete("/{symbol}", response_model=BaseResponse)
async def clear_symbol_data(
    symbol: str,
    data_service: DataServiceDep,
    timeframe: Optional[TimeFrame] = Query(None, description="Specific timeframe to clear"),
) -> BaseResponse:
    """
    Clear market data for a symbol.
    
    Parameters:
    - symbol: Trading symbol to clear
    - timeframe: Optional specific timeframe to clear (if not provided, all timeframes cleared)
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Clearing data for {symbol}" + (f" ({timeframe})" if timeframe else ""))
        
        # Check if symbol exists
        symbols = await data_service.get_available_symbols()
        if symbol not in symbols:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found"
            )
        
        # Clear data using service
        result = await data_service.clear_data(symbol=symbol, timeframe=timeframe)
        
        message = f"Successfully cleared data for {symbol}"
        if timeframe:
            message += f" ({timeframe})"
        
        logger.info(message)
        
        return BaseResponse()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear data: {str(e)}"
        )


@router.get("/{symbol}/{timeframe}/info")
async def get_data_info(
    symbol: str,
    timeframe: TimeFrame,
    data_service: DataServiceDep,
) -> dict:
    """
    Get information about available data for a symbol and timeframe.
    
    Returns metadata such as date range, record count, etc.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting data info for {symbol} ({timeframe})")
        
        # Get data info using service
        info = await data_service.get_data_info(symbol=symbol, timeframe=timeframe)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol} ({timeframe})"
            )
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe.value,
            **info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data info: {str(e)}"
        )