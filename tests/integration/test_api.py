"""
Integration tests for API endpoints.

Tests the FastAPI application and all endpoints with realistic scenarios.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import httpx
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import (
    MarketDataRequest, BacktestRequest, AnalysisRequest,
    StrategyComparisonRequest, ParameterOptimizationRequest
)
from src.core.models import MarketData, TimeFrame, BacktestResult, PerformanceMetrics
from src.strategies.ma_crossover import MovingAverageCrossover
from src.data.mock_generator import MockDataGenerator


class TestAPIHealthChecks:
    """Test API health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert data["message"] == "Trading System API"
    
    def test_basic_health_check(self):
        """Test basic health check."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_detailed_health_check(self):
        """Test detailed health check."""
        client = TestClient(app)
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data
        
        # Should include service status
        services = data["services"]
        expected_services = ["data", "backtest", "analysis"]
        
        for service in expected_services:
            assert service in services


class TestDataEndpoints:
    """Test data-related endpoints."""
    
    def test_get_symbols(self):
        """Test get symbols endpoint."""
        client = TestClient(app)
        
        with patch.object(app.state, 'data_service') as mock_service:
            mock_service.get_symbols.return_value = ["AAPL", "GOOGL", "MSFT"]
            
            response = client.get("/data/symbols")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "symbols" in data
            assert isinstance(data["symbols"], list)
            assert "AAPL" in data["symbols"]
    
    def test_get_market_data(self):
        """Test get market data endpoint."""
        client = TestClient(app)
        
        # Mock market data
        mock_data_generator = MockDataGenerator(seed=42)
        data_points = mock_data_generator.generate_trending_data(periods=50)
        
        mock_market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        with patch.object(app.state, 'data_service') as mock_service:
            mock_service.get_market_data.return_value = mock_market_data
            
            response = client.get("/data/market-data/AAPL?timeframe=1d&periods=50")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "symbol" in data
            assert "timeframe" in data
            assert "data" in data
            assert data["symbol"] == "AAPL"
            assert len(data["data"]) == 50
    
    def test_get_market_data_invalid_symbol(self):
        """Test get market data with invalid symbol."""
        client = TestClient(app)
        
        with patch.object(app.state, 'data_service') as mock_service:
            mock_service.get_market_data.return_value = None
            
            response = client.get("/data/market-data/INVALID?timeframe=1d&periods=50")
            
            assert response.status_code == 404
    
    def test_market_status(self):
        """Test market status endpoint."""
        client = TestClient(app)
        
        with patch.object(app.state, 'data_service') as mock_service:
            mock_service.is_market_open.return_value = True
            
            response = client.get("/data/market-status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "is_open" in data
            assert "timestamp" in data
            assert data["is_open"] is True
    
    def test_generate_mock_data(self):
        """Test mock data generation endpoint."""
        client = TestClient(app)
        
        request_data = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 100,
            "pattern": "trending",
            "trend_strength": 0.02,
            "volatility": 0.15
        }
        
        response = client.post("/data/generate-mock", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "symbol" in data
        assert "data" in data
        assert data["symbol"] == "AAPL"
        assert len(data["data"]) == 100


class TestAnalysisEndpoints:
    """Test analysis-related endpoints."""
    
    def test_run_backtest(self):
        """Test run backtest endpoint."""
        client = TestClient(app)
        
        # Mock backtest result
        mock_performance = PerformanceMetrics(
            total_trades=10, winning_trades=6, losing_trades=4, win_rate=0.6,
            total_pnl=1500.0, total_return_pct=15.0, max_drawdown=500.0,
            max_drawdown_pct=5.0, sharpe_ratio=1.2, sortino_ratio=1.5,
            profit_factor=1.8, avg_win=350.0, avg_loss=-200.0,
            largest_win=800.0, largest_loss=-400.0, avg_trade_duration_hours=24.0,
            max_consecutive_wins=3, max_consecutive_losses=2
        )
        
        mock_result = BacktestResult(
            strategy_name="MovingAverageCrossover",
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=10000.0,
            final_capital=11500.0,
            trades=[],
            equity_curve=[10000.0, 11500.0],
            performance_metrics=mock_performance,
            signals=[]
        )
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.run_backtest.return_value = mock_result
            
            request_data = {
                "strategy_name": "MovingAverageCrossover",
                "strategy_parameters": {"short_period": 20, "long_period": 50},
                "symbol": "AAPL",
                "timeframe": "1d",
                "periods": 100,
                "initial_capital": 10000.0,
                "commission_rate": 0.001
            }
            
            response = client.post("/analysis/backtest", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "strategy_name" in data
            assert "performance_metrics" in data
            assert data["strategy_name"] == "MovingAverageCrossover"
            assert data["performance_metrics"]["total_return_pct"] == 15.0
    
    def test_compare_strategies(self):
        """Test strategy comparison endpoint."""
        client = TestClient(app)
        
        # Mock comparison result
        mock_comparison = {
            "strategies": ["MovingAverageCrossover", "RSITrend"],
            "performance_comparison": {
                "MovingAverageCrossover": {"total_return_pct": 15.0, "sharpe_ratio": 1.2},
                "RSITrend": {"total_return_pct": 12.0, "sharpe_ratio": 1.0}
            },
            "best_strategy": "MovingAverageCrossover",
            "comparison_metrics": ["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
        }
        
        with patch.object(app.state, 'analysis_service') as mock_service:
            mock_service.compare_strategies.return_value = mock_comparison
            
            request_data = {
                "strategies": [
                    {
                        "name": "MovingAverageCrossover",
                        "parameters": {"short_period": 20, "long_period": 50}
                    },
                    {
                        "name": "RSITrend",
                        "parameters": {"rsi_period": 14, "rsi_oversold": 30}
                    }
                ],
                "symbol": "AAPL",
                "timeframe": "1d",
                "periods": 100,
                "initial_capital": 10000.0
            }
            
            response = client.post("/analysis/compare-strategies", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "strategies" in data
            assert "best_strategy" in data
            assert data["best_strategy"] == "MovingAverageCrossover"
    
    def test_optimize_parameters(self):
        """Test parameter optimization endpoint."""
        client = TestClient(app)
        
        # Mock optimization result
        mock_optimization = {
            "strategy_name": "MovingAverageCrossover",
            "best_parameters": {"short_period": 25, "long_period": 55},
            "best_score": 1.45,
            "optimization_metric": "sharpe_ratio",
            "tested_combinations": 12,
            "optimization_results": [
                {"parameters": {"short_period": 20, "long_period": 50}, "score": 1.2},
                {"parameters": {"short_period": 25, "long_period": 55}, "score": 1.45}
            ]
        }
        
        with patch.object(app.state, 'analysis_service') as mock_service:
            mock_service.optimize_parameters.return_value = mock_optimization
            
            request_data = {
                "strategy_name": "MovingAverageCrossover",
                "parameter_ranges": {
                    "short_period": [20, 25, 30],
                    "long_period": [50, 55, 60]
                },
                "symbol": "AAPL",
                "timeframe": "1d",
                "periods": 200,
                "optimization_metric": "sharpe_ratio",
                "initial_capital": 10000.0
            }
            
            response = client.post("/analysis/optimize-parameters", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "best_parameters" in data
            assert "best_score" in data
            assert data["best_score"] == 1.45
    
    def test_analyze_strategy(self):
        """Test strategy analysis endpoint."""
        client = TestClient(app)
        
        # Mock analysis result
        mock_analysis = {
            "strategy_name": "MovingAverageCrossover",
            "analysis_type": "comprehensive",
            "risk_analysis": {
                "max_drawdown_pct": 8.5,
                "volatility": 0.15,
                "beta": 1.1,
                "value_at_risk": 2.3
            },
            "performance_analysis": {
                "total_return_pct": 15.0,
                "annualized_return": 18.2,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.6
            },
            "trade_analysis": {
                "total_trades": 24,
                "avg_trade_duration_days": 5.2,
                "win_rate": 0.625,
                "profit_factor": 1.8
            }
        }
        
        with patch.object(app.state, 'analysis_service') as mock_service:
            mock_service.analyze_strategy.return_value = mock_analysis
            
            request_data = {
                "strategy_name": "MovingAverageCrossover",
                "strategy_parameters": {"short_period": 20, "long_period": 50},
                "symbol": "AAPL",
                "timeframe": "1d",
                "periods": 200,
                "analysis_type": "comprehensive"
            }
            
            response = client.post("/analysis/analyze-strategy", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "strategy_name" in data
            assert "risk_analysis" in data
            assert "performance_analysis" in data
            assert data["strategy_name"] == "MovingAverageCrossover"


class TestResultsEndpoints:
    """Test results-related endpoints."""
    
    def test_get_backtest_results(self):
        """Test get backtest results endpoint."""
        client = TestClient(app)
        
        # Mock results
        mock_results = [
            {
                "id": "result_1",
                "strategy_name": "MovingAverageCrossover",
                "symbol": "AAPL",
                "total_return_pct": 15.0,
                "sharpe_ratio": 1.2,
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "result_2",
                "strategy_name": "RSITrend",
                "symbol": "AAPL",
                "total_return_pct": 12.0,
                "sharpe_ratio": 1.0,
                "created_at": datetime.now().isoformat()
            }
        ]
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.get_results.return_value = mock_results
            
            response = client.get("/results/backtest")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["strategy_name"] == "MovingAverageCrossover"
    
    def test_get_backtest_result_by_id(self):
        """Test get specific backtest result."""
        client = TestClient(app)
        
        # Mock specific result
        mock_result = {
            "id": "result_1",
            "strategy_name": "MovingAverageCrossover",
            "symbol": "AAPL",
            "performance_metrics": {
                "total_return_pct": 15.0,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": 8.5
            },
            "trades": [],
            "equity_curve": [10000.0, 10500.0, 11000.0, 11500.0],
            "created_at": datetime.now().isoformat()
        }
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.get_result.return_value = mock_result
            
            response = client.get("/results/backtest/result_1")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["id"] == "result_1"
            assert data["strategy_name"] == "MovingAverageCrossover"
            assert "performance_metrics" in data
    
    def test_delete_backtest_result(self):
        """Test delete backtest result."""
        client = TestClient(app)
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.delete_result.return_value = True
            
            response = client.delete("/results/backtest/result_1")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "Result deleted successfully"
    
    def test_export_results(self):
        """Test export results endpoint."""
        client = TestClient(app)
        
        # Mock export data
        mock_export_data = {
            "format": "json",
            "results": [
                {
                    "strategy_name": "MovingAverageCrossover",
                    "total_return_pct": 15.0,
                    "sharpe_ratio": 1.2
                }
            ],
            "exported_at": datetime.now().isoformat()
        }
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.export_results.return_value = mock_export_data
            
            response = client.get("/results/export?format=json")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert data["format"] == "json"


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_validation_error(self):
        """Test request validation error."""
        client = TestClient(app)
        
        # Invalid request data
        request_data = {
            "strategy_name": "",  # Empty strategy name
            "symbol": "AAPL",
            "periods": -1  # Invalid periods
        }
        
        response = client.post("/analysis/backtest", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        
        assert "message" in data
        assert "detail" in data
    
    def test_not_found_error(self):
        """Test 404 error handling."""
        client = TestClient(app)
        
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_internal_server_error(self):
        """Test 500 error handling."""
        client = TestClient(app)
        
        with patch.object(app.state, 'data_service') as mock_service:
            mock_service.get_symbols.side_effect = Exception("Internal error")
            
            response = client.get("/data/symbols")
            
            assert response.status_code == 500
            data = response.json()
            
            assert "message" in data
            assert data["message"] == "Internal server error"
    
    def test_service_unavailable(self):
        """Test service unavailable error."""
        client = TestClient(app)
        
        # Mock service not available
        with patch.object(app.state, 'backtest_service', None):
            request_data = {
                "strategy_name": "MovingAverageCrossover",
                "symbol": "AAPL",
                "periods": 100
            }
            
            response = client.post("/analysis/backtest", json=request_data)
            
            # Should handle gracefully
            assert response.status_code in [500, 503]
    
    def test_invalid_json(self):
        """Test invalid JSON handling."""
        client = TestClient(app)
        
        response = client.post(
            "/analysis/backtest",
            data="invalid json data",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        client = TestClient(app)
        
        # Missing required fields
        request_data = {
            "symbol": "AAPL"
            # Missing strategy_name, periods, etc.
        }
        
        response = client.post("/analysis/backtest", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        
        assert "detail" in data


class TestAPIPerformance:
    """Test API performance."""
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        client = TestClient(app)
        
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)
        
        # Should complete quickly
        assert end_time - start_time < 5.0
    
    def test_large_request_handling(self):
        """Test handling large requests."""
        client = TestClient(app)
        
        # Large request data
        request_data = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 1000,  # Large dataset
            "pattern": "trending"
        }
        
        response = client.post("/data/generate-mock", json=request_data)
        
        # Should handle large requests
        assert response.status_code in [200, 202, 413]  # 413 if too large
    
    def test_response_time(self):
        """Test API response times."""
        client = TestClient(app)
        
        import time
        
        # Measure response time for health check
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond quickly
    
    def test_memory_usage(self, memory_monitor):
        """Test API memory usage."""
        client = TestClient(app)
        
        initial_memory = memory_monitor.get_current_usage()
        
        # Make multiple requests
        for i in range(20):
            response = client.get("/health")
            assert response.status_code == 200
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not leak excessive memory
        assert memory_increase < 50  # MB


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint."""
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check API info
        info = schema["info"]
        assert info["title"] == "Trading System API"
        assert "version" in info
    
    def test_docs_endpoint(self):
        """Test documentation endpoint."""
        client = TestClient(app)
        
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self):
        """Test ReDoc documentation endpoint."""
        client = TestClient(app)
        
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIAuthentication:
    """Test API authentication (if implemented)."""
    
    def test_unauthenticated_access(self):
        """Test unauthenticated access to public endpoints."""
        client = TestClient(app)
        
        # Public endpoints should be accessible
        public_endpoints = [
            "/",
            "/health",
            "/docs",
            "/openapi.json"
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
    
    def test_cors_headers(self):
        """Test CORS headers."""
        client = TestClient(app)
        
        # Preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should include CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestAPIIntegrationScenarios:
    """Test realistic API integration scenarios."""
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow through API."""
        client = TestClient(app)
        
        # Step 1: Get available symbols
        response = client.get("/data/symbols")
        assert response.status_code == 200
        
        # Step 2: Generate mock data
        mock_data_request = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 100,
            "pattern": "trending"
        }
        response = client.post("/data/generate-mock", json=mock_data_request)
        assert response.status_code == 200
        
        # Step 3: Run backtest
        backtest_request = {
            "strategy_name": "MovingAverageCrossover",
            "strategy_parameters": {"short_period": 20, "long_period": 50},
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 100,
            "initial_capital": 10000.0
        }
        
        with patch.object(app.state, 'backtest_service') as mock_service:
            # Mock successful backtest
            mock_performance = PerformanceMetrics(
                total_trades=5, winning_trades=3, losing_trades=2, win_rate=0.6,
                total_pnl=800.0, total_return_pct=8.0, max_drawdown=200.0,
                max_drawdown_pct=2.0, sharpe_ratio=1.1, sortino_ratio=1.3,
                profit_factor=1.5, avg_win=300.0, avg_loss=-150.0,
                largest_win=500.0, largest_loss=-250.0, avg_trade_duration_hours=48.0,
                max_consecutive_wins=2, max_consecutive_losses=1
            )
            
            mock_result = BacktestResult(
                strategy_name="MovingAverageCrossover",
                symbol="AAPL",
                start_date=datetime.now() - timedelta(days=100),
                end_date=datetime.now(),
                initial_capital=10000.0,
                final_capital=10800.0,
                trades=[],
                equity_curve=[10000.0, 10800.0],
                performance_metrics=mock_performance,
                signals=[]
            )
            mock_service.run_backtest.return_value = mock_result
            
            response = client.post("/analysis/backtest", json=backtest_request)
            assert response.status_code == 200
        
        # Step 4: Get results
        with patch.object(app.state, 'backtest_service') as mock_service:
            mock_service.get_results.return_value = [{"id": "result_1"}]
            response = client.get("/results/backtest")
            assert response.status_code == 200
    
    def test_strategy_comparison_workflow(self):
        """Test strategy comparison workflow."""
        client = TestClient(app)
        
        comparison_request = {
            "strategies": [
                {
                    "name": "MovingAverageCrossover",
                    "parameters": {"short_period": 20, "long_period": 50}
                },
                {
                    "name": "RSITrend",
                    "parameters": {"rsi_period": 14}
                }
            ],
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 100,
            "initial_capital": 10000.0
        }
        
        mock_comparison = {
            "strategies": ["MovingAverageCrossover", "RSITrend"],
            "best_strategy": "MovingAverageCrossover",
            "performance_comparison": {
                "MovingAverageCrossover": {"total_return_pct": 12.0},
                "RSITrend": {"total_return_pct": 8.0}
            }
        }
        
        with patch.object(app.state, 'analysis_service') as mock_service:
            mock_service.compare_strategies.return_value = mock_comparison
            
            response = client.post("/analysis/compare-strategies", json=comparison_request)
            assert response.status_code == 200
            
            data = response.json()
            assert data["best_strategy"] == "MovingAverageCrossover"
    
    def test_parameter_optimization_workflow(self):
        """Test parameter optimization workflow."""
        client = TestClient(app)
        
        optimization_request = {
            "strategy_name": "MovingAverageCrossover",
            "parameter_ranges": {
                "short_period": [15, 20, 25],
                "long_period": [45, 50, 55]
            },
            "symbol": "AAPL",
            "timeframe": "1d",
            "periods": 200,
            "optimization_metric": "sharpe_ratio"
        }
        
        mock_optimization = {
            "best_parameters": {"short_period": 20, "long_period": 50},
            "best_score": 1.35,
            "tested_combinations": 9
        }
        
        with patch.object(app.state, 'analysis_service') as mock_service:
            mock_service.optimize_parameters.return_value = mock_optimization
            
            response = client.post("/analysis/optimize-parameters", json=optimization_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "best_parameters" in data
            assert data["best_score"] == 1.35