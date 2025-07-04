#!/usr/bin/env python3
"""
Real-time Trading API CLI Client

This script provides a command-line interface for interacting with the
real-time trading API. It can be used for testing, monitoring, and
automated trading operations.
"""

import asyncio
import json
import sys
from typing import Dict, List, Optional
import argparse
import websockets
import requests
from datetime import datetime, timedelta

# Default API base URL
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_WS_BASE = "ws://localhost:8000"


class RealtimeAPIClient:
    """Client for interacting with the real-time trading API."""
    
    def __init__(self, base_url: str = DEFAULT_API_BASE):
        """Initialize the API client."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "RealtimeAPIClient/1.0"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    def get_latest_data(self, symbol: str) -> Dict:
        """Get latest real-time data for a symbol."""
        response = self._make_request("GET", f"/realtime/data/latest/{symbol}")
        return response.json()
    
    def get_indicators(self, symbol: str, indicators: Optional[List[str]] = None) -> Dict:
        """Get latest indicators for a symbol."""
        params = {}
        if indicators:
            params["indicators"] = indicators
        
        response = self._make_request("GET", f"/realtime/indicators/{symbol}", params=params)
        return response.json()
    
    def get_signals(self, symbol: str, strategy: Optional[str] = None) -> Dict:
        """Get latest trading signals for a symbol."""
        params = {}
        if strategy:
            params["strategy"] = strategy
        
        response = self._make_request("GET", f"/realtime/signals/{symbol}", params=params)
        return response.json()
    
    def get_watchlist(self) -> Dict:
        """Get current watchlist."""
        response = self._make_request("GET", "/realtime/config/watchlist")
        return response.json()
    
    def add_symbol(self, symbol: str, **kwargs) -> Dict:
        """Add a symbol to the watchlist."""
        data = {"symbol": symbol, **kwargs}
        response = self._make_request("POST", "/realtime/config/watchlist/add", json=data)
        return response.json()
    
    def remove_symbol(self, symbol: str) -> Dict:
        """Remove a symbol from the watchlist."""
        response = self._make_request("DELETE", f"/realtime/config/watchlist/{symbol}")
        return response.json()
    
    def get_system_health(self) -> Dict:
        """Get system health status."""
        response = self._make_request("GET", "/realtime/monitor/health")
        return response.json()
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        response = self._make_request("GET", "/realtime/monitor/stats")
        return response.json()
    
    def get_alerts(self, **kwargs) -> Dict:
        """Get system alerts."""
        response = self._make_request("GET", "/realtime/monitor/alerts", params=kwargs)
        return response.json()
    
    def get_historical_data(self, symbol: str, **kwargs) -> Dict:
        """Get historical data for a symbol."""
        response = self._make_request("GET", f"/realtime/data/historical/{symbol}", params=kwargs)
        return response.json()
    
    def create_export(self, symbols: List[str], **kwargs) -> Dict:
        """Create a bulk data export."""
        data = {"symbols": symbols, **kwargs}
        response = self._make_request("POST", "/realtime/data/export", json=data)
        return response.json()
    
    def get_export_status(self, export_id: str) -> Dict:
        """Get export job status."""
        response = self._make_request("GET", f"/realtime/data/export/{export_id}")
        return response.json()


class WebSocketClient:
    """WebSocket client for real-time data streaming."""
    
    def __init__(self, base_url: str = DEFAULT_WS_BASE):
        """Initialize the WebSocket client."""
        self.base_url = base_url.rstrip("/")
        self.websocket = None
        self.connection_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def connect(self):
        """Connect to the WebSocket endpoint."""
        uri = f"{self.base_url}/realtime/ws/{self.connection_id}"
        self.websocket = await websockets.connect(uri)
        print(f"Connected to WebSocket: {uri}")
    
    async def disconnect(self):
        """Disconnect from the WebSocket."""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from WebSocket")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to real-time data for symbols."""
        message = {
            "action": "subscribe",
            "symbols": symbols
        }
        await self.websocket.send(json.dumps(message))
        print(f"Subscribed to symbols: {symbols}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols."""
        message = {
            "action": "unsubscribe",
            "symbols": symbols
        }
        await self.websocket.send(json.dumps(message))
        print(f"Unsubscribed from symbols: {symbols}")
    
    async def listen(self, duration: Optional[int] = None):
        """Listen for real-time messages."""
        print("Listening for real-time messages... (Press Ctrl+C to stop)")
        start_time = datetime.now()
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if data.get("type") == "confirmation":
                    print(f"[{timestamp}] CONFIRMATION: {data.get('message')}")
                elif data.get("type") == "pong":
                    print(f"[{timestamp}] PONG: {data.get('timestamp')}")
                else:
                    print(f"[{timestamp}] {data.get('type', 'DATA').upper()}: {data.get('symbol', 'N/A')} - {json.dumps(data.get('data', {}))}")
                
                # Check duration limit
                if duration and (datetime.now() - start_time).seconds >= duration:
                    print(f"Reached duration limit of {duration} seconds")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except KeyboardInterrupt:
            print("\nStopping WebSocket listener...")


def print_json(data: Dict, title: str = "Response"):
    """Pretty print JSON data."""
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, default=str))
    print("=" * (len(title) + 8))


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Real-time Trading API CLI Client")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--ws-base-url", default=DEFAULT_WS_BASE, help="WebSocket base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data commands
    data_parser = subparsers.add_parser("data", help="Real-time data commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    
    latest_parser = data_subparsers.add_parser("latest", help="Get latest data")
    latest_parser.add_argument("symbol", help="Trading symbol")
    
    indicators_parser = data_subparsers.add_parser("indicators", help="Get indicators")
    indicators_parser.add_argument("symbol", help="Trading symbol")
    indicators_parser.add_argument("--indicators", nargs="+", help="Specific indicators")
    
    signals_parser = data_subparsers.add_parser("signals", help="Get trading signals")
    signals_parser.add_argument("symbol", help="Trading symbol")
    signals_parser.add_argument("--strategy", help="Specific strategy")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration commands")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    watchlist_parser = config_subparsers.add_parser("watchlist", help="Get watchlist")
    
    add_parser = config_subparsers.add_parser("add", help="Add symbol to watchlist")
    add_parser.add_argument("symbol", help="Trading symbol")
    add_parser.add_argument("--priority", type=int, default=1, help="Priority level")
    
    remove_parser = config_subparsers.add_parser("remove", help="Remove symbol from watchlist")
    remove_parser.add_argument("symbol", help="Trading symbol")
    
    # Monitor commands
    monitor_parser = subparsers.add_parser("monitor", help="Monitoring commands")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_command")
    
    health_parser = monitor_subparsers.add_parser("health", help="Get system health")
    stats_parser = monitor_subparsers.add_parser("stats", help="Get system statistics")
    alerts_parser = monitor_subparsers.add_parser("alerts", help="Get system alerts")
    
    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream real-time data")
    stream_parser.add_argument("symbols", nargs="+", help="Symbols to subscribe to")
    stream_parser.add_argument("--duration", type=int, help="Duration in seconds")
    
    # Export commands
    export_parser = subparsers.add_parser("export", help="Data export commands")
    export_subparsers = export_parser.add_subparsers(dest="export_command")
    
    create_export_parser = export_subparsers.add_parser("create", help="Create export job")
    create_export_parser.add_argument("symbols", nargs="+", help="Symbols to export")
    create_export_parser.add_argument("--format", default="json", help="Export format")
    
    status_export_parser = export_subparsers.add_parser("status", help="Get export status")
    status_export_parser.add_argument("export_id", help="Export job ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize API client
        client = RealtimeAPIClient(args.base_url)
        
        # Handle data commands
        if args.command == "data":
            if args.data_command == "latest":
                data = client.get_latest_data(args.symbol)
                print_json(data, f"Latest Data - {args.symbol}")
            
            elif args.data_command == "indicators":
                data = client.get_indicators(args.symbol, args.indicators)
                print_json(data, f"Indicators - {args.symbol}")
            
            elif args.data_command == "signals":
                data = client.get_signals(args.symbol, args.strategy)
                print_json(data, f"Signals - {args.symbol}")
        
        # Handle config commands
        elif args.command == "config":
            if args.config_command == "watchlist":
                data = client.get_watchlist()
                print_json(data, "Watchlist")
            
            elif args.config_command == "add":
                data = client.add_symbol(args.symbol, priority=args.priority)
                print_json(data, f"Add Symbol - {args.symbol}")
            
            elif args.config_command == "remove":
                data = client.remove_symbol(args.symbol)
                print_json(data, f"Remove Symbol - {args.symbol}")
        
        # Handle monitor commands
        elif args.command == "monitor":
            if args.monitor_command == "health":
                data = client.get_system_health()
                print_json(data, "System Health")
            
            elif args.monitor_command == "stats":
                data = client.get_system_stats()
                print_json(data, "System Statistics")
            
            elif args.monitor_command == "alerts":
                data = client.get_alerts()
                print_json(data, "System Alerts")
        
        # Handle stream command
        elif args.command == "stream":
            ws_client = WebSocketClient(args.ws_base_url)
            await ws_client.connect()
            
            try:
                await ws_client.subscribe(args.symbols)
                await ws_client.listen(args.duration)
            finally:
                await ws_client.disconnect()
        
        # Handle export commands
        elif args.command == "export":
            if args.export_command == "create":
                data = client.create_export(args.symbols, format=args.format)
                print_json(data, "Export Job Created")
            
            elif args.export_command == "status":
                data = client.get_export_status(args.export_id)
                print_json(data, f"Export Status - {args.export_id}")
    
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())