"""
Real-time Market Data API Routes

This module provides comprehensive FastAPI endpoints for the real-time market data
processing system, including real-time data streams, configuration management,
system monitoring, and data query capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.websockets import WebSocketState

from ..schemas import (
    # Real-time data schemas
    RealtimeDataResponse, RealtimeDataPoint, IndicatorResponse, IndicatorValue,
    SignalResponse, TradingSignal, WebSocketMessage,
    # Configuration schemas
    WatchlistRequest, WatchlistResponse, AddSymbolRequest, AssetConfigModel,
    IndicatorConfigModel, StrategyConfigModel, SystemConfigModel,
    ConfigurationSummaryResponse, ValidationResult,
    # Monitoring schemas
    SystemHealthResponse, SystemStatsResponse, AlertsResponse, AlertModel,
    PerformanceMetrics,
    # Data query schemas
    HistoricalDataQuery, HistoricalDataResponse, BulkExportRequest, BulkExportResponse,
    # Base schemas
    BaseResponse, ErrorResponse
)
from ..dependencies import get_current_user
from ...realtime.stream_processor import StreamProcessor
from ...realtime.config_manager import ConfigManager, AssetConfig, IndicatorConfig, StrategyConfig, SystemConfig
from ...storage.cache import cache_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global stream processor instance - will be initialized during app startup
stream_processor: Optional[StreamProcessor] = None
config_manager: Optional[ConfigManager] = None

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # connection_id -> [symbols]
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.subscriptions[connection_id] = []
        logger.info(f"WebSocket connection established: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_message(self, connection_id: str, message: WebSocketMessage):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message.json())
                except Exception as e:
                    logger.error(f"Error sending message to {connection_id}: {e}")
                    self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, symbol: str, message: WebSocketMessage):
        """Broadcast a message to all connections subscribed to a symbol."""
        disconnected_connections = []
        
        for connection_id, subscribed_symbols in self.subscriptions.items():
            if symbol in subscribed_symbols:
                try:
                    await self.send_message(connection_id, message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    def subscribe(self, connection_id: str, symbols: List[str]):
        """Subscribe a connection to symbols."""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].extend(symbols)
            # Remove duplicates
            self.subscriptions[connection_id] = list(set(self.subscriptions[connection_id]))
    
    def unsubscribe(self, connection_id: str, symbols: List[str]):
        """Unsubscribe a connection from symbols."""
        if connection_id in self.subscriptions:
            for symbol in symbols:
                if symbol in self.subscriptions[connection_id]:
                    self.subscriptions[connection_id].remove(symbol)

# Global connection manager
connection_manager = ConnectionManager()


def get_stream_processor() -> StreamProcessor:
    """Get the global stream processor instance."""
    global stream_processor
    if stream_processor is None:
        stream_processor = StreamProcessor()
    return stream_processor


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager


# =============================================================================
# 1. REAL-TIME DATA ENDPOINTS
# =============================================================================

@router.get("/data/latest/{symbol}", response_model=RealtimeDataResponse)
async def get_latest_data(
    symbol: str,
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    Get the latest real-time data for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL', 'SPY')
        
    Returns:
        Latest real-time data point for the symbol
    """
    try:
        symbol = symbol.upper()
        
        # Get latest results from stream processor
        results = processor.get_latest_results(symbol)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No real-time data found for symbol {symbol}"
            )
        
        # Convert to RealtimeDataPoint
        data_points = []
        if "price_data" in results:
            price_data = results["price_data"]
            data_point = RealtimeDataPoint(
                symbol=symbol,
                timestamp=datetime.fromisoformat(price_data.get("timestamp", datetime.now().isoformat())),
                price=price_data.get("close", 0.0),
                volume=price_data.get("volume"),
                bid=price_data.get("bid"),
                ask=price_data.get("ask"),
                spread=price_data.get("spread")
            )
            data_points.append(data_point)
        
        return RealtimeDataResponse(
            data=data_points,
            count=len(data_points)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}", response_model=IndicatorResponse)
async def get_latest_indicators(
    symbol: str,
    indicators: Optional[List[str]] = Query(None, description="Specific indicators to retrieve"),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    Get the latest indicator values for a specific symbol.
    
    Args:
        symbol: Trading symbol
        indicators: Optional list of specific indicators to retrieve
        
    Returns:
        Latest indicator values for the symbol
    """
    try:
        symbol = symbol.upper()
        
        # Get latest results from stream processor
        results = processor.get_latest_results(symbol)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No indicator data found for symbol {symbol}"
            )
        
        # Convert to IndicatorValue objects
        indicator_values = []
        
        if "indicators" in results:
            for indicator_name, indicator_data in results["indicators"].items():
                # Filter by requested indicators if specified
                if indicators and indicator_name not in indicators:
                    continue
                
                if isinstance(indicator_data, dict):
                    # Multi-value indicator (e.g., MACD)
                    indicator_value = IndicatorValue(
                        name=indicator_name,
                        values=indicator_data.get("values", {}),
                        timestamp=datetime.fromisoformat(indicator_data.get("timestamp", datetime.now().isoformat())),
                        metadata=indicator_data.get("metadata")
                    )
                else:
                    # Single-value indicator
                    indicator_value = IndicatorValue(
                        name=indicator_name,
                        value=indicator_data,
                        timestamp=datetime.now(),
                        metadata=None
                    )
                
                indicator_values.append(indicator_value)
        
        return IndicatorResponse(
            symbol=symbol,
            indicators=indicator_values
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{symbol}", response_model=SignalResponse)
async def get_latest_signals(
    symbol: str,
    strategy: Optional[str] = Query(None, description="Specific strategy to retrieve signals for"),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    Get the latest trading signals for a specific symbol.
    
    Args:
        symbol: Trading symbol
        strategy: Optional specific strategy to filter signals
        
    Returns:
        Latest trading signals for the symbol
    """
    try:
        symbol = symbol.upper()
        
        # Get latest results from stream processor
        results = processor.get_latest_results(symbol)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No signal data found for symbol {symbol}"
            )
        
        # Convert to TradingSignal objects
        signals = []
        
        if "signals" in results:
            for signal_data in results["signals"]:
                # Filter by strategy if specified
                if strategy and signal_data.get("strategy") != strategy:
                    continue
                
                signal = TradingSignal(
                    symbol=symbol,
                    strategy=signal_data.get("strategy", "unknown"),
                    signal_type=signal_data.get("signal_type", "HOLD"),
                    strength=signal_data.get("strength", 0.0),
                    price=signal_data.get("price", 0.0),
                    timestamp=datetime.fromisoformat(signal_data.get("timestamp", datetime.now().isoformat())),
                    indicators=signal_data.get("indicators"),
                    metadata=signal_data.get("metadata")
                )
                signals.append(signal)
        
        return SignalResponse(
            symbol=symbol,
            signals=signals
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    WebSocket endpoint for real-time data streaming.
    
    Clients can subscribe to specific symbols and receive real-time updates
    for price data, indicators, and trading signals.
    
    Message format:
    - Subscribe: {"action": "subscribe", "symbols": ["AAPL", "SPY"]}
    - Unsubscribe: {"action": "unsubscribe", "symbols": ["AAPL"]}
    """
    await connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            action = message.get("action")
            symbols = message.get("symbols", [])
            
            if action == "subscribe":
                connection_manager.subscribe(connection_id, symbols)
                await websocket.send_text(json.dumps({
                    "type": "confirmation",
                    "message": f"Subscribed to {len(symbols)} symbols",
                    "symbols": symbols
                }))
                
            elif action == "unsubscribe":
                connection_manager.unsubscribe(connection_id, symbols)
                await websocket.send_text(json.dumps({
                    "type": "confirmation",
                    "message": f"Unsubscribed from {len(symbols)} symbols",
                    "symbols": symbols
                }))
                
            elif action == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        connection_manager.disconnect(connection_id)


# =============================================================================
# 2. CONFIGURATION ENDPOINTS
# =============================================================================

@router.get("/config/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Get the current watchlist configuration.
    
    Returns:
        Current watchlist with all configured assets
    """
    try:
        watchlist = config_mgr.get_watchlist()
        
        # Convert to API models
        assets = []
        for asset_config in watchlist:
            asset_model = AssetConfigModel(
                symbol=asset_config.symbol,
                enabled=asset_config.enabled,
                data_provider=asset_config.data_provider,
                timeframes=asset_config.timeframes,
                priority=asset_config.priority,
                last_updated=asset_config.last_updated
            )
            assets.append(asset_model)
        
        return WatchlistResponse(assets=assets)
        
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/watchlist", response_model=WatchlistResponse)
async def update_watchlist(
    request: WatchlistRequest,
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Update the watchlist configuration.
    
    Args:
        request: Watchlist update request with new asset configurations
        
    Returns:
        Updated watchlist configuration
    """
    try:
        # Convert API models to config objects
        asset_configs = []
        for asset_model in request.assets:
            asset_config = AssetConfig(
                symbol=asset_model.symbol,
                enabled=asset_model.enabled,
                data_provider=asset_model.data_provider,
                timeframes=asset_model.timeframes,
                priority=asset_model.priority,
                last_updated=asset_model.last_updated
            )
            asset_configs.append(asset_config)
        
        # Update watchlist
        success = config_mgr.update_watchlist(asset_configs)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update watchlist"
            )
        
        # Return updated watchlist
        return await get_watchlist(config_mgr)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/watchlist/add", response_model=BaseResponse)
async def add_symbol_to_watchlist(
    request: AddSymbolRequest,
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Add a symbol to the watchlist.
    
    Args:
        request: Add symbol request with symbol and configuration
        
    Returns:
        Success response
    """
    try:
        success = config_mgr.add_symbol(
            symbol=request.symbol,
            data_provider=request.data_provider,
            timeframes=request.timeframes,
            priority=request.priority
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to add symbol {request.symbol} to watchlist"
            )
        
        return BaseResponse(
            success=True,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding symbol {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/config/watchlist/{symbol}", response_model=BaseResponse)
async def remove_symbol_from_watchlist(
    symbol: str,
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Remove a symbol from the watchlist.
    
    Args:
        symbol: Trading symbol to remove
        
    Returns:
        Success response
    """
    try:
        symbol = symbol.upper()
        success = config_mgr.remove_symbol(symbol)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found in watchlist"
            )
        
        return BaseResponse(
            success=True,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/indicators", response_model=List[IndicatorConfigModel])
async def get_indicator_configs(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Get current indicator configurations.
    
    Returns:
        List of indicator configurations
    """
    try:
        indicators = config_mgr.get_indicator_configs()
        
        # Convert to API models
        indicator_models = []
        for indicator in indicators:
            model = IndicatorConfigModel(
                name=indicator.name,
                enabled=indicator.enabled,
                parameters=indicator.parameters,
                update_frequency_seconds=indicator.update_frequency_seconds,
                ttl_seconds=indicator.ttl_seconds
            )
            indicator_models.append(model)
        
        return indicator_models
        
    except Exception as e:
        logger.error(f"Error getting indicator configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/indicators", response_model=List[IndicatorConfigModel])
async def update_indicator_configs(
    indicators: List[IndicatorConfigModel],
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Update indicator configurations.
    
    Args:
        indicators: List of indicator configurations
        
    Returns:
        Updated indicator configurations
    """
    try:
        # Convert API models to config objects
        indicator_configs = []
        for indicator_model in indicators:
            config = IndicatorConfig(
                name=indicator_model.name,
                enabled=indicator_model.enabled,
                parameters=indicator_model.parameters,
                update_frequency_seconds=indicator_model.update_frequency_seconds,
                ttl_seconds=indicator_model.ttl_seconds
            )
            indicator_configs.append(config)
        
        # Update configurations
        success = config_mgr.update_indicator_configs(indicator_configs)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update indicator configurations"
            )
        
        # Return updated configurations
        return await get_indicator_configs(config_mgr)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating indicator configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/strategies", response_model=List[StrategyConfigModel])
async def get_strategy_configs(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Get current strategy configurations.
    
    Returns:
        List of strategy configurations
    """
    try:
        strategies = config_mgr.get_strategy_configs()
        
        # Convert to API models
        strategy_models = []
        for strategy in strategies:
            model = StrategyConfigModel(
                name=strategy.name,
                enabled=strategy.enabled,
                parameters=strategy.parameters,
                required_indicators=strategy.required_indicators,
                update_frequency_seconds=strategy.update_frequency_seconds,
                ttl_seconds=strategy.ttl_seconds
            )
            strategy_models.append(model)
        
        return strategy_models
        
    except Exception as e:
        logger.error(f"Error getting strategy configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/strategies", response_model=List[StrategyConfigModel])
async def update_strategy_configs(
    strategies: List[StrategyConfigModel],
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Update strategy configurations.
    
    Args:
        strategies: List of strategy configurations
        
    Returns:
        Updated strategy configurations
    """
    try:
        # Convert API models to config objects
        strategy_configs = []
        for strategy_model in strategies:
            config = StrategyConfig(
                name=strategy_model.name,
                enabled=strategy_model.enabled,
                parameters=strategy_model.parameters,
                required_indicators=strategy_model.required_indicators,
                update_frequency_seconds=strategy_model.update_frequency_seconds,
                ttl_seconds=strategy_model.ttl_seconds
            )
            strategy_configs.append(config)
        
        # Update configurations
        success = config_mgr.update_strategy_configs(strategy_configs)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update strategy configurations"
            )
        
        # Return updated configurations
        return await get_strategy_configs(config_mgr)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/system", response_model=SystemConfigModel)
async def get_system_config(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Get current system configuration.
    
    Returns:
        System configuration
    """
    try:
        system_config = config_mgr.get_system_config()
        
        if not system_config:
            raise HTTPException(
                status_code=404,
                detail="System configuration not found"
            )
        
        return SystemConfigModel(
            max_concurrent_calculations=system_config.max_concurrent_calculations,
            batch_calculation_size=system_config.batch_calculation_size,
            data_update_interval_seconds=system_config.data_update_interval_seconds,
            calculation_workers=system_config.calculation_workers,
            redis_streams_maxlen=system_config.redis_streams_maxlen,
            enable_monitoring=system_config.enable_monitoring,
            enable_alerts=system_config.enable_alerts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/system", response_model=SystemConfigModel)
async def update_system_config(
    config: SystemConfigModel,
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Update system configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Updated system configuration
    """
    try:
        # Convert API model to config object
        system_config = SystemConfig(
            max_concurrent_calculations=config.max_concurrent_calculations,
            batch_calculation_size=config.batch_calculation_size,
            data_update_interval_seconds=config.data_update_interval_seconds,
            calculation_workers=config.calculation_workers,
            redis_streams_maxlen=config.redis_streams_maxlen,
            enable_monitoring=config.enable_monitoring,
            enable_alerts=config.enable_alerts
        )
        
        # Update configuration
        success = config_mgr.update_system_config(system_config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update system configuration"
            )
        
        # Return updated configuration
        return await get_system_config(config_mgr)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/summary", response_model=ConfigurationSummaryResponse)
async def get_configuration_summary(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Get a summary of all current configurations.
    
    Returns:
        Configuration summary
    """
    try:
        summary = config_mgr.get_configuration_summary()
        
        return ConfigurationSummaryResponse(
            watchlist=summary.get("watchlist", {}),
            indicators=summary.get("indicators", {}),
            strategies=summary.get("strategies", {}),
            system=summary.get("system"),
            last_updated=summary.get("last_updated", datetime.now().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Error getting configuration summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/validate", response_model=ValidationResult)
async def validate_configuration(
    config_mgr: ConfigManager = Depends(get_config_manager)
):
    """
    Validate the current configuration for potential issues.
    
    Returns:
        Validation result with warnings and errors
    """
    try:
        validation_result = config_mgr.validate_configuration()
        
        return ValidationResult(
            is_valid=validation_result.get("is_valid", False),
            warnings=validation_result.get("warnings", []),
            errors=validation_result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 3. MONITORING ENDPOINTS
# =============================================================================

@router.get("/monitor/health", response_model=SystemHealthResponse)
async def get_system_health(
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    Get comprehensive system health information.
    
    Returns:
        System health status including component health and statistics
    """
    try:
        # Get system stats
        stats = await processor.get_system_stats()
        
        # Determine overall health status
        status = "healthy"
        components = {}
        alerts = []
        
        # Check ingestion service
        if "ingestion" in stats:
            ingestion_stats = stats["ingestion"]
            error_rate = ingestion_stats.get("errors", 0)
            
            if error_rate > 10:
                components["ingestion"] = "unhealthy"
                status = "degraded"
                alerts.append({
                    "type": "error",
                    "message": f"High error rate in ingestion service: {error_rate} errors"
                })
            elif error_rate > 5:
                components["ingestion"] = "degraded"
                if status == "healthy":
                    status = "degraded"
                alerts.append({
                    "type": "warning",
                    "message": f"Moderate error rate in ingestion service: {error_rate} errors"
                })
            else:
                components["ingestion"] = "healthy"
        
        # Check calculation engine
        if "calculation" in stats:
            calculation_stats = stats["calculation"]
            error_rate = calculation_stats.get("errors", 0)
            
            if error_rate > 5:
                components["calculation"] = "unhealthy"
                status = "degraded"
                alerts.append({
                    "type": "error",
                    "message": f"High error rate in calculation engine: {error_rate} errors"
                })
            elif error_rate > 2:
                components["calculation"] = "degraded"
                if status == "healthy":
                    status = "degraded"
                alerts.append({
                    "type": "warning",
                    "message": f"Moderate error rate in calculation engine: {error_rate} errors"
                })
            else:
                components["calculation"] = "healthy"
        
        # Calculate uptime
        uptime_seconds = None
        if "timestamp" in stats:
            try:
                start_time = datetime.fromisoformat(stats["timestamp"])
                uptime_seconds = (datetime.now() - start_time).total_seconds()
            except Exception:
                pass
        
        return SystemHealthResponse(
            status=status,
            uptime_seconds=uptime_seconds,
            components=components,
            statistics=stats,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """
    Get detailed system statistics and performance metrics.
    
    Returns:
        Comprehensive system statistics
    """
    try:
        stats = await processor.get_system_stats()
        
        # Calculate uptime
        uptime_seconds = 0.0
        if "timestamp" in stats:
            try:
                start_time = datetime.fromisoformat(stats["timestamp"])
                uptime_seconds = (datetime.now() - start_time).total_seconds()
            except Exception:
                pass
        
        # Create performance metrics
        performance_metrics = []
        
        # Ingestion performance
        if "ingestion" in stats:
            ingestion_stats = stats["ingestion"]
            performance_metrics.append(PerformanceMetrics(
                component="ingestion",
                events_processed=ingestion_stats.get("events_published", 0),
                events_per_second=ingestion_stats.get("events_per_second", 0.0),
                errors=ingestion_stats.get("errors", 0),
                error_rate=ingestion_stats.get("error_rate", 0.0),
                memory_usage_mb=ingestion_stats.get("memory_usage_mb", 0.0),
                cpu_usage_percent=ingestion_stats.get("cpu_usage_percent", 0.0),
                last_update=datetime.now()
            ))
        
        # Calculation performance
        if "calculation" in stats:
            calculation_stats = stats["calculation"]
            performance_metrics.append(PerformanceMetrics(
                component="calculation",
                events_processed=calculation_stats.get("calculations_performed", 0),
                events_per_second=calculation_stats.get("calculations_per_second", 0.0),
                errors=calculation_stats.get("errors", 0),
                error_rate=calculation_stats.get("error_rate", 0.0),
                memory_usage_mb=calculation_stats.get("memory_usage_mb", 0.0),
                cpu_usage_percent=calculation_stats.get("cpu_usage_percent", 0.0),
                last_update=datetime.now()
            ))
        
        return SystemStatsResponse(
            uptime_seconds=uptime_seconds,
            running=stats.get("running", False),
            configuration=stats.get("configuration", {}),
            ingestion=stats.get("ingestion"),
            calculation=stats.get("calculation"),
            performance=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/alerts", response_model=AlertsResponse)
async def get_alerts(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    alert_type: Optional[str] = Query(None, regex="^(error|warning|info)$"),
    component: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None)
):
    """
    Get system alerts and notifications.
    
    Args:
        limit: Maximum number of alerts to return
        offset: Number of alerts to skip
        alert_type: Filter by alert type (error, warning, info)
        component: Filter by component name
        acknowledged: Filter by acknowledgment status
        
    Returns:
        List of system alerts
    """
    try:
        # This would typically query from a database or Redis
        # For now, we'll return mock data
        
        alerts = []
        
        # Mock alert data
        mock_alerts = [
            {
                "id": str(uuid4()),
                "type": "warning",
                "component": "ingestion",
                "message": "Data ingestion rate below expected threshold",
                "timestamp": datetime.now() - timedelta(minutes=5),
                "acknowledged": False,
                "metadata": {"threshold": 100, "actual": 75}
            },
            {
                "id": str(uuid4()),
                "type": "info",
                "component": "calculation",
                "message": "Successfully processed 1000 calculations",
                "timestamp": datetime.now() - timedelta(minutes=10),
                "acknowledged": True,
                "metadata": {"count": 1000}
            }
        ]
        
        # Apply filters
        filtered_alerts = []
        for alert_data in mock_alerts:
            if alert_type and alert_data["type"] != alert_type:
                continue
            if component and alert_data["component"] != component:
                continue
            if acknowledged is not None and alert_data["acknowledged"] != acknowledged:
                continue
            
            alert = AlertModel(
                id=alert_data["id"],
                type=alert_data["type"],
                component=alert_data["component"],
                message=alert_data["message"],
                timestamp=alert_data["timestamp"],
                acknowledged=alert_data["acknowledged"],
                metadata=alert_data.get("metadata")
            )
            filtered_alerts.append(alert)
        
        # Apply pagination
        total_count = len(filtered_alerts)
        paginated_alerts = filtered_alerts[offset:offset + limit]
        
        # Count unacknowledged alerts
        unacknowledged_count = len([a for a in filtered_alerts if not a.acknowledged])
        
        return AlertsResponse(
            alerts=paginated_alerts,
            total_count=total_count,
            unacknowledged_count=unacknowledged_count
        )
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 4. DATA QUERY ENDPOINTS
# =============================================================================

@router.get("/data/historical/{symbol}", response_model=HistoricalDataResponse)
async def get_historical_data(
    symbol: str,
    indicator: Optional[str] = Query(None, description="Specific indicator to retrieve"),
    start_date: Optional[datetime] = Query(None, description="Start date for data query"),
    end_date: Optional[datetime] = Query(None, description="End date for data query"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of records to return")
):
    """
    Get historical indicator data for a symbol.
    
    Args:
        symbol: Trading symbol
        indicator: Specific indicator to retrieve
        start_date: Start date for data query
        end_date: End date for data query
        limit: Maximum number of records to return
        
    Returns:
        Historical data for the specified parameters
    """
    try:
        symbol = symbol.upper()
        
        # Build query parameters
        query_params = {
            "symbol": symbol,
            "indicator": indicator,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "limit": limit
        }
        
        # This would typically query from a database
        # For now, we'll return mock data
        
        historical_data = []
        
        # Generate mock historical data
        end_time = end_date or datetime.now()
        start_time = start_date or (end_time - timedelta(days=30))
        
        current_time = start_time
        while current_time <= end_time and len(historical_data) < limit:
            data_point = {
                "timestamp": current_time.isoformat(),
                "value": 100 + (current_time.timestamp() % 1000) / 10,  # Mock value
                "metadata": {"source": "mock_data"}
            }
            historical_data.append(data_point)
            current_time += timedelta(minutes=1)
        
        return HistoricalDataResponse(
            symbol=symbol,
            indicator=indicator,
            data=historical_data,
            total_count=len(historical_data),
            query_params=query_params
        )
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/export", response_model=BulkExportResponse)
async def create_bulk_export(
    request: BulkExportRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a bulk data export job.
    
    Args:
        request: Bulk export request with symbols and parameters
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Export job information
    """
    try:
        export_id = str(uuid4())
        
        # Create export job
        export_job = BulkExportResponse(
            export_id=export_id,
            status="pending",
            created_at=datetime.now()
        )
        
        # Add background task to process export
        background_tasks.add_task(
            process_bulk_export,
            export_id,
            request.symbols,
            request.indicators,
            request.strategies,
            request.start_date,
            request.end_date,
            request.format
        )
        
        return export_job
        
    except Exception as e:
        logger.error(f"Error creating bulk export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/export/{export_id}", response_model=BulkExportResponse)
async def get_export_status(export_id: str):
    """
    Get the status of a bulk export job.
    
    Args:
        export_id: Export job ID
        
    Returns:
        Export job status and information
    """
    try:
        # This would typically query from a database
        # For now, we'll return mock data
        
        return BulkExportResponse(
            export_id=export_id,
            status="completed",
            download_url=f"/api/realtime/data/export/{export_id}/download",
            file_size_bytes=1024 * 1024,  # 1MB
            records_count=10000,
            created_at=datetime.now() - timedelta(minutes=5),
            completed_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting export status for {export_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/export/{export_id}/download")
async def download_export(export_id: str):
    """
    Download the results of a bulk export job.
    
    Args:
        export_id: Export job ID
        
    Returns:
        Export file download
    """
    try:
        # This would typically stream the actual export file
        # For now, we'll return mock data
        
        def generate_mock_export():
            yield '{"export_id": "' + export_id + '", '
            yield '"data": ['
            for i in range(100):
                yield f'{{"timestamp": "{datetime.now().isoformat()}", "value": {i}}}'
                if i < 99:
                    yield ','
            yield ']}'
        
        return StreamingResponse(
            generate_mock_export(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=export_{export_id}.json"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading export {export_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def process_bulk_export(
    export_id: str,
    symbols: List[str],
    indicators: Optional[List[str]],
    strategies: Optional[List[str]],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    format: str
):
    """
    Background task to process bulk export requests.
    
    Args:
        export_id: Export job ID
        symbols: List of symbols to export
        indicators: List of indicators to include
        strategies: List of strategies to include
        start_date: Start date for export
        end_date: End date for export
        format: Export format (json, csv, parquet)
    """
    try:
        logger.info(f"Processing bulk export {export_id} for {len(symbols)} symbols")
        
        # Simulate export processing
        await asyncio.sleep(5)  # Simulate processing time
        
        # Update export status in database/cache
        # This would typically update the export job status to "completed"
        
        logger.info(f"Bulk export {export_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing bulk export {export_id}: {e}")
        # Update export status to "failed"