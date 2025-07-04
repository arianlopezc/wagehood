#!/usr/bin/env python3
"""
Test script for the Trading System API.

This script demonstrates how to interact with the API endpoints.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi.testclient import TestClient
from src.api.app import app

# Create test client
client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    
    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed\n")


def test_detailed_health_check():
    """Test the detailed health check endpoint."""
    print("Testing detailed health check endpoint...")
    
    response = client.get("/health/detailed")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Detailed health check passed\n")


def test_root_endpoint():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert "Trading System API" in response.json()["message"]
    print("✓ Root endpoint test passed\n")


def test_data_endpoints():
    """Test data management endpoints."""
    print("Testing data endpoints...")
    
    # Test get symbols (should be empty initially)
    response = client.get("/data/symbols")
    print(f"Get symbols - Status: {response.status_code}")
    print(f"Symbols: {response.json()}")
    
    # Test upload data
    sample_data = generate_sample_data()
    upload_request = {
        "symbol": "AAPL",
        "timeframe": "daily",
        "data": sample_data
    }
    
    response = client.post("/data/upload", json=upload_request)
    print(f"Upload data - Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Upload response: {response.json()}")
        print("✓ Data upload test passed")
    else:
        print(f"Upload failed: {response.json()}")
    
    # Test get historical data
    response = client.get("/data/AAPL/daily")
    print(f"Get historical data - Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Retrieved {data['total_records']} records")
        print("✓ Historical data test passed")
    else:
        print(f"Get historical data failed: {response.json()}")
    
    print()


def test_analysis_endpoints():
    """Test analysis endpoints."""
    print("Testing analysis endpoints...")
    
    # Test get available strategies
    response = client.get("/analysis/strategies")
    print(f"Get strategies - Status: {response.status_code}")
    if response.status_code == 200:
        strategies = response.json()
        print(f"Available strategies: {list(strategies['strategies'].keys())}")
        print("✓ Get strategies test passed")
    else:
        print(f"Get strategies failed: {response.json()}")
    
    # Test backtest (this might fail due to missing services)
    backtest_request = {
        "symbol": "AAPL",
        "timeframe": "daily",
        "strategy": "sma_crossover",
        "parameters": {
            "parameters": {
                "fast_period": 10,
                "slow_period": 20
            }
        },
        "initial_capital": 10000.0,
        "commission": 0.001
    }
    
    response = client.post("/analysis/backtest", json=backtest_request)
    print(f"Backtest - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Backtest ID: {result['backtest_id']}")
        print("✓ Backtest test passed")
    else:
        print(f"Backtest failed: {response.json()}")
    
    print()


def test_results_endpoints():
    """Test results endpoints."""
    print("Testing results endpoints...")
    
    # Test list results
    response = client.get("/results/")
    print(f"List results - Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"Total results: {results['total_count']}")
        print("✓ List results test passed")
    else:
        print(f"List results failed: {response.json()}")
    
    # Test get rankings
    response = client.get("/results/rankings")
    print(f"Get rankings - Status: {response.status_code}")
    if response.status_code == 200:
        rankings = response.json()
        print(f"Rankings count: {len(rankings['rankings'])}")
        print("✓ Rankings test passed")
    else:
        print(f"Rankings failed: {response.json()}")
    
    print()


def generate_sample_data():
    """Generate sample OHLCV data for testing."""
    data = []
    base_price = 150.0
    
    for i in range(100):
        timestamp = datetime.now() - timedelta(days=100-i)
        
        # Simple random walk
        change = (i % 10 - 5) * 0.5
        base_price += change
        
        open_price = base_price
        high_price = base_price + abs(change) + 1.0
        low_price = base_price - abs(change) - 1.0
        close_price = base_price + change * 0.5
        volume = 1000000 + (i % 100) * 10000
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        })
    
    return data


def main():
    """Run all tests."""
    print("=== Trading System API Test Suite ===\n")
    
    try:
        test_health_check()
        test_detailed_health_check()
        test_root_endpoint()
        test_data_endpoints()
        test_analysis_endpoints()
        test_results_endpoints()
        
        print("=== All tests completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()