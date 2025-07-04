"""Main FastAPI application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging
import traceback
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes import data, analysis, results, realtime
from .schemas import ErrorResponse
from ..services.data_service import DataService
from ..services.backtest_service import BacktestService
from ..services.analysis_service import AnalysisService
from ..realtime.stream_processor import StreamProcessor
from ..realtime.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting trading system API...")
    
    # Initialize services
    try:
        data_service = DataService()
        backtest_service = BacktestService()
        analysis_service = AnalysisService()
        
        # Initialize real-time services
        config_manager = ConfigManager()
        stream_processor = StreamProcessor()
        
        # Store services in app state
        app.state.data_service = data_service
        app.state.backtest_service = backtest_service
        app.state.analysis_service = analysis_service
        app.state.config_manager = config_manager
        app.state.stream_processor = stream_processor
        
        logger.info("Services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    finally:
        logger.info("Shutting down trading system API...")
        # Cleanup services if needed
        if hasattr(app.state, 'data_service'):
            logger.info("Cleaning up data service...")
        if hasattr(app.state, 'backtest_service'):
            logger.info("Cleaning up backtest service...")
        if hasattr(app.state, 'analysis_service'):
            logger.info("Cleaning up analysis service...")
        if hasattr(app.state, 'stream_processor'):
            logger.info("Shutting down stream processor...")
            await app.state.stream_processor.shutdown()
        if hasattr(app.state, 'config_manager'):
            logger.info("Cleaning up config manager...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Trading System API",
    description="Advanced trading system with backtesting and analysis capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url),
        ).dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            message="Validation error",
            detail=exc.errors(),
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url),
        ).dict(),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions."""
    logger.error(f"Starlette error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat(),
            path=str(request.url),
        ).dict(),
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check including service status."""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {},
    }
    
    # Check data service
    try:
        data_service = app.state.data_service
        # Simple check - could be more sophisticated
        health_info["services"]["data"] = "healthy"
    except Exception as e:
        health_info["services"]["data"] = f"unhealthy: {e}"
        health_info["status"] = "degraded"
    
    # Check backtest service
    try:
        backtest_service = app.state.backtest_service
        health_info["services"]["backtest"] = "healthy"
    except Exception as e:
        health_info["services"]["backtest"] = f"unhealthy: {e}"
        health_info["status"] = "degraded"
    
    # Check analysis service
    try:
        analysis_service = app.state.analysis_service
        health_info["services"]["analysis"] = "healthy"
    except Exception as e:
        health_info["services"]["analysis"] = f"unhealthy: {e}"
        health_info["status"] = "degraded"
    
    # Check real-time services
    try:
        config_manager = app.state.config_manager
        health_info["services"]["config_manager"] = "healthy"
    except Exception as e:
        health_info["services"]["config_manager"] = f"unhealthy: {e}"
        health_info["status"] = "degraded"
    
    try:
        stream_processor = app.state.stream_processor
        health_info["services"]["stream_processor"] = "healthy"
    except Exception as e:
        health_info["services"]["stream_processor"] = f"unhealthy: {e}"
        health_info["status"] = "degraded"
    
    return health_info


# Include routers
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(results.router, prefix="/results", tags=["Results"])
app.include_router(realtime.router, prefix="/realtime", tags=["Real-time"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Trading System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }