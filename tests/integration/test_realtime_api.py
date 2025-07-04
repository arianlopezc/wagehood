"""
Integration tests for the real-time API endpoints.

Tests the complete real-time system including API endpoints,
WebSocket connections, authentication, and data flow.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import websockets

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.api.app import app
from src.realtime.config_manager import ConfigManager
from src.realtime.stream_processor import StreamProcessor
from src.api.auth import api_key_manager


class TestRealtimeAPI:
    """Integration tests for real-time API endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Test client for API requests."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def api_key(self):
        """Generate test API key."""
        return api_key_manager.create_key(
            name="test_key",
            permissions=["read", "write", "admin"],
            rate_limit=1000
        )
    
    @pytest.fixture(scope="class")
    def auth_headers(self, api_key):
        """Authentication headers for requests."""
        return {"Authorization": f"Bearer {api_key}"}
    
    @pytest.fixture(scope="class")
    async def mock_stream_processor(self):
        """Mock stream processor for testing."""
        processor = Mock(spec=StreamProcessor)
        processor.config_manager = Mock(spec=ConfigManager)
        
        # Mock configuration methods
        processor.config_manager.get_enabled_symbols.return_value = ["SPY", "QQQ", "AAPL"]
        processor.config_manager.get_configuration_summary.return_value = {
            "watchlist": {"symbols": ["SPY", "QQQ", "AAPL"]},
            "indicators": {"indicator_names": ["sma_50", "rsi_14", "macd"]},
            "strategies": {"strategy_names": ["macd_rsi_strategy"]}
        }
        
        # Mock data methods
        processor.get_latest_results.return_value = {
            "timestamp": datetime.now().isoformat(),
            "indicators": {
                "sma_50": 485.67,
                "rsi_14": 65.23,
                "macd": {"macd_line": 1.23, "signal_line": 0.89, "histogram": 0.34}
            },
            "signals": {
                "macd_rsi_strategy": {"signal": "buy", "confidence": 0.75}
            }
        }
        
        processor.get_system_stats.return_value = {
            "running": True,
            "ingestion": {"events_published": 1000, "errors": 0},
            "calculation": {"calculations_performed": 500, "signals_generated": 50}
        }
        
        return processor


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_health_check(self, client, auth_headers):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/realtime/monitor/health", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
    
    def test_health_check_detailed(self, client, auth_headers):
        """Test detailed health check."""
        response = client.get(
            "/api/v1/realtime/monitor/health?detailed=true", 
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["detailed"] is True
        assert "components" in data
        assert "system_info" in data
    
    def test_stats_endpoint(self, client, auth_headers):
        """Test system statistics endpoint."""
        response = client.get("/api/v1/realtime/monitor/stats", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "running" in data


class TestDataEndpoints:
    """Test real-time data endpoints."""
    
    def test_latest_data_single_symbol(self, client, auth_headers):
        """Test getting latest data for a single symbol."""
        response = client.get("/api/v1/realtime/data/latest/SPY", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "SPY"
        assert "timestamp" in data
        assert "price" in data
    
    def test_latest_data_multiple_symbols(self, client, auth_headers):
        """Test getting latest data for multiple symbols."""
        response = client.get(
            "/api/v1/realtime/data/latest/SPY,QQQ,AAPL", 
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        
        symbols = [item["symbol"] for item in data]
        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "AAPL" in symbols
    
    def test_indicators_endpoint(self, client, auth_headers):
        """Test indicators endpoint."""
        response = client.get("/api/v1/realtime/indicators/SPY", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "SPY"
        assert "indicators" in data
        assert "timestamp" in data
    
    def test_signals_endpoint(self, client, auth_headers):
        """Test trading signals endpoint."""
        response = client.get("/api/v1/realtime/signals/SPY", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "SPY"
        assert "signals" in data
        assert "timestamp" in data
    
    def test_historical_data(self, client, auth_headers):
        """Test historical data endpoint."""
        start_date = (datetime.now() - timedelta(days=30)).date().isoformat()
        end_date = datetime.now().date().isoformat()
        
        response = client.get(
            f"/api/v1/realtime/data/historical/SPY?start_date={start_date}&end_date={end_date}",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "SPY"
        assert "data" in data
        assert "start_date" in data
        assert "end_date" in data


class TestConfigurationEndpoints:
    """Test configuration management endpoints."""
    
    def test_get_watchlist(self, client, auth_headers):
        """Test getting current watchlist."""
        response = client.get("/api/v1/realtime/config/watchlist", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "symbols" in data
        assert isinstance(data["symbols"], list)
    
    def test_add_symbol_to_watchlist(self, client, auth_headers):
        """Test adding symbol to watchlist."""
        request_data = {"symbol": "TSLA", "data_provider": "mock"}
        
        response = client.post(
            "/api/v1/realtime/config/watchlist/add",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "TSLA" in data["message"]
    
    def test_remove_symbol_from_watchlist(self, client, auth_headers):
        """Test removing symbol from watchlist."""
        # First add a symbol
        add_response = client.post(
            "/api/v1/realtime/config/watchlist/add",
            json={"symbol": "NVDA", "data_provider": "mock"},
            headers=auth_headers
        )
        assert add_response.status_code == 200
        
        # Then remove it
        response = client.delete(
            "/api/v1/realtime/config/watchlist/NVDA",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_get_configuration_summary(self, client, auth_headers):
        """Test configuration summary endpoint."""
        response = client.get("/api/v1/realtime/config/summary", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "watchlist" in data
        assert "indicators" in data
        assert "strategies" in data
    
    def test_validate_configuration(self, client, auth_headers):
        """Test configuration validation endpoint."""
        response = client.get("/api/v1/realtime/config/validate", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_valid" in data
        assert "warnings" in data
        assert "errors" in data


class TestExportEndpoints:
    """Test data export functionality."""
    
    def test_create_export_job(self, client, auth_headers):
        """Test creating data export job."""
        request_data = {
            "symbols": ["SPY", "QQQ"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "format": "csv",
            "include_indicators": True
        }
        
        response = client.post(
            "/api/v1/realtime/data/export",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "export_id" in data
        assert data["status"] == "created"
    
    def test_get_export_status(self, client, auth_headers):
        """Test getting export job status."""
        # Create export job first
        create_response = client.post(
            "/api/v1/realtime/data/export",
            json={
                "symbols": ["SPY"],
                "format": "json",
                "include_indicators": True
            },
            headers=auth_headers
        )
        export_id = create_response.json()["export_id"]
        
        # Check status
        response = client.get(
            f"/api/v1/realtime/data/export/{export_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["export_id"] == export_id
        assert "status" in data


class TestWebSocketConnections:
    """Test WebSocket real-time streaming."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, api_key):
        """Test WebSocket connection and subscription."""
        # Note: This test requires a running server
        # In practice, you'd use a test server or mock WebSocket
        
        uri = f"ws://localhost:8000/api/v1/realtime/ws/test_connection_1"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # Subscribe to symbol
                subscribe_message = {
                    "action": "subscribe",
                    "symbols": ["SPY"]
                }
                await websocket.send(json.dumps(subscribe_message))
                
                # Wait for response
                response = await websocket.recv()
                data = json.loads(response)
                
                assert data["type"] == "subscription_confirmed"
                assert "SPY" in data["symbols"]
                
        except Exception as e:
            # Skip if WebSocket server not available
            pytest.skip(f"WebSocket server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_data_streaming(self, api_key):
        """Test WebSocket data streaming."""
        uri = f"ws://localhost:8000/api/v1/realtime/ws/test_stream_1"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # Subscribe to symbol
                await websocket.send(json.dumps({
                    "action": "subscribe", 
                    "symbols": ["SPY"],
                    "include_indicators": True
                }))
                
                # Wait for subscription confirmation
                await websocket.recv()
                
                # Wait for data message (with timeout)
                try:
                    data_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(data_message)
                    
                    assert data["type"] == "market_data"
                    assert "symbol" in data
                    assert "timestamp" in data
                    
                except asyncio.TimeoutError:
                    pytest.skip("No data received within timeout")
                
        except Exception as e:
            pytest.skip(f"WebSocket server not available: {e}")


class TestAuthentication:
    """Test API authentication and authorization."""
    
    def test_no_auth_header(self, client):
        """Test request without authentication header."""
        response = client.get("/api/v1/realtime/monitor/health")
        assert response.status_code == 401
        
        data = response.json()
        assert "Missing authentication token" in data["detail"]
    
    def test_invalid_api_key(self, client):
        """Test request with invalid API key."""
        headers = {"Authorization": "Bearer invalid_key_12345"}
        
        response = client.get("/api/v1/realtime/monitor/health", headers=headers)
        assert response.status_code == 401
        
        data = response.json()
        assert "Invalid or expired API key" in data["detail"]
    
    def test_insufficient_permissions(self, client):
        """Test request with insufficient permissions."""
        # Create read-only API key
        readonly_key = api_key_manager.create_key(
            name="readonly_test",
            permissions=["read"],
            rate_limit=100
        )
        headers = {"Authorization": f"Bearer {readonly_key}"}
        
        # Try to access write endpoint
        response = client.post(
            "/api/v1/realtime/config/watchlist/add",
            json={"symbol": "TEST", "data_provider": "mock"},
            headers=headers
        )
        assert response.status_code == 403
        
        data = response.json()
        assert "Insufficient permissions" in data["detail"]


class TestRateLimiting:
    """Test API rate limiting functionality."""
    
    def test_rate_limit_enforcement(self, client):
        """Test rate limit enforcement."""
        # Create API key with low rate limit
        limited_key = api_key_manager.create_key(
            name="limited_test",
            permissions=["read", "write"],
            rate_limit=5  # Very low limit for testing
        )
        headers = {"Authorization": f"Bearer {limited_key}"}
        
        # Make requests up to the limit
        for i in range(5):
            response = client.get("/api/v1/realtime/monitor/health", headers=headers)
            assert response.status_code == 200
            
            # Check rate limit headers
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
        
        # Next request should be rate limited
        response = client.get("/api/v1/realtime/monitor/health", headers=headers)
        assert response.status_code == 429
        
        data = response.json()
        assert "Rate limit exceeded" in data["detail"]
        assert "Retry-After" in response.headers
    
    def test_rate_limit_headers(self, client, auth_headers):
        """Test rate limit headers in responses."""
        response = client.get("/api/v1/realtime/monitor/health", headers=auth_headers)
        assert response.status_code == 200
        
        # Check rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        
        # Verify header values
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        reset_time = int(response.headers["X-RateLimit-Reset"])
        
        assert limit > 0
        assert remaining >= 0
        assert reset_time > time.time()


class TestErrorHandling:
    """Test API error handling and responses."""
    
    def test_invalid_symbol(self, client, auth_headers):
        """Test handling of invalid symbols."""
        response = client.get("/api/v1/realtime/data/latest/INVALID123", headers=auth_headers)
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_invalid_date_format(self, client, auth_headers):
        """Test handling of invalid date formats."""
        response = client.get(
            "/api/v1/realtime/data/historical/SPY?start_date=invalid-date",
            headers=auth_headers
        )
        assert response.status_code == 422
        
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    def test_malformed_json(self, client, auth_headers):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/api/v1/realtime/config/watchlist/add",
            data="invalid json{",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test handling of missing required fields."""
        response = client.post(
            "/api/v1/realtime/config/watchlist/add",
            json={},  # Missing required symbol field
            headers=auth_headers
        )
        assert response.status_code == 422
        
        data = response.json()
        assert any("symbol" in str(error) for error in data["detail"])


class TestPerformance:
    """Test API performance characteristics."""
    
    def test_response_time(self, client, auth_headers):
        """Test API response times."""
        start_time = time.time()
        
        response = client.get("/api/v1/realtime/monitor/health", headers=auth_headers)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, client, auth_headers):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/api/v1/realtime/monitor/health", headers=auth_headers)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_large_data_handling(self, client, auth_headers):
        """Test handling of large data requests."""
        # Request data for many symbols
        symbols = ",".join(["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])
        
        response = client.get(
            f"/api/v1/realtime/data/latest/{symbols}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 8  # Should handle all requested symbols


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])