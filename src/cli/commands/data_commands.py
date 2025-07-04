"""
Data Commands Module

This module provides comprehensive data-related commands for the Wagehood CLI,
including real-time data queries, streaming, historical data, and export functionality.
"""

import asyncio
import json
import websockets
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import csv
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation, log_api_call
from ..config import CLIConfig


class DataAPIClient:
    """Client for data API operations."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize data API client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_api_headers())
        self.logger = CLILogger("data_client")
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest real-time data for a symbol."""
        url = self.config.get_api_url(f"realtime/data/latest/{symbol}")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_indicators(self, symbol: str, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get latest indicators for a symbol."""
        url = self.config.get_api_url(f"realtime/indicators/{symbol}")
        params = {}
        if indicators:
            params['indicators'] = indicators
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, params=params, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_signals(self, symbol: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Get latest trading signals for a symbol."""
        url = self.config.get_api_url(f"realtime/signals/{symbol}")
        params = {}
        if strategy:
            params['strategy'] = strategy
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, params=params, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_historical_data(self, symbol: str, indicator: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 1000) -> Dict[str, Any]:
        """Get historical data for a symbol."""
        url = self.config.get_api_url(f"realtime/data/historical/{symbol}")
        params = {'limit': limit}
        
        if indicator:
            params['indicator'] = indicator
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, params=params, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def create_export(self, symbols: List[str], indicators: Optional[List[str]] = None,
                     strategies: Optional[List[str]] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     format: str = 'json') -> Dict[str, Any]:
        """Create a bulk data export."""
        url = self.config.get_api_url("realtime/data/export")
        data = {
            'symbols': symbols,
            'format': format
        }
        
        if indicators:
            data['indicators'] = indicators
        if strategies:
            data['strategies'] = strategies
        if start_date:
            data['start_date'] = start_date.isoformat()
        if end_date:
            data['end_date'] = end_date.isoformat()
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get export job status."""
        url = self.config.get_api_url(f"realtime/data/export/{export_id}")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def download_export(self, export_id: str, output_file: str) -> None:
        """Download export data to file."""
        url = self.config.get_api_url(f"realtime/data/export/{export_id}/download")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout, stream=True)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


class WebSocketStreamer:
    """WebSocket client for real-time data streaming."""
    
    def __init__(self, config: CLIConfig, formatter: OutputFormatter):
        """
        Initialize WebSocket streamer.
        
        Args:
            config: CLI configuration
            formatter: Output formatter
        """
        self.config = config
        self.formatter = formatter
        self.logger = CLILogger("ws_streamer")
        self.websocket = None
        self.connection_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def connect(self) -> None:
        """Connect to WebSocket endpoint."""
        uri = self.config.get_ws_url(f"realtime/ws/{self.connection_id}")
        
        try:
            self.websocket = await websockets.connect(uri)
            self.logger.info(f"Connected to WebSocket: {uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from WebSocket")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time data for symbols."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        message = {
            "action": "subscribe",
            "symbols": symbols
        }
        
        await self.websocket.send(json.dumps(message))
        self.logger.info(f"Subscribed to symbols: {symbols}")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        message = {
            "action": "unsubscribe",
            "symbols": symbols
        }
        
        await self.websocket.send(json.dumps(message))
        self.logger.info(f"Unsubscribed from symbols: {symbols}")
    
    async def listen(self, duration: Optional[int] = None, 
                    output_file: Optional[str] = None) -> None:
        """
        Listen for real-time messages.
        
        Args:
            duration: Duration in seconds to listen
            output_file: Optional file to save messages
        """
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        self.formatter.print_info("Listening for real-time messages... (Press Ctrl+C to stop)")
        
        start_time = datetime.now()
        messages = []
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Store message if output file specified
                    if output_file:
                        messages.append({
                            'timestamp': datetime.now().isoformat(),
                            'data': data
                        })
                    
                    # Format and display message
                    self._display_message(data, timestamp)
                    
                    # Check duration limit
                    if duration and (datetime.now() - start_time).seconds >= duration:
                        self.formatter.print_info(f"Reached duration limit of {duration} seconds")
                        break
                
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON message: {message}")
                
        except websockets.exceptions.ConnectionClosed:
            self.formatter.print_warning("WebSocket connection closed")
        except KeyboardInterrupt:
            self.formatter.print_info("Stopping WebSocket listener...")
        finally:
            # Save messages to file if specified
            if output_file and messages:
                self._save_messages(messages, output_file)
    
    def _display_message(self, data: Dict[str, Any], timestamp: str) -> None:
        """Display WebSocket message."""
        message_type = data.get('type', 'DATA').upper()
        
        if message_type == 'CONFIRMATION':
            self.formatter.print_success(f"[{timestamp}] {data.get('message', 'Confirmed')}")
        elif message_type == 'PONG':
            self.formatter.print_info(f"[{timestamp}] Server responded: {data.get('timestamp', 'pong')}")
        elif message_type == 'ERROR':
            self.formatter.print_error(f"[{timestamp}] {data.get('message', 'Unknown error')}")
        else:
            # Regular data message
            symbol = data.get('symbol', 'N/A')
            message_data = data.get('data', {})
            
            # Create a formatted display
            if self.config.output_format == 'table':
                self._display_data_table(symbol, message_data, timestamp)
            else:
                self.formatter.print_data(data, f"[{timestamp}] {symbol}")
    
    def _display_data_table(self, symbol: str, data: Dict[str, Any], timestamp: str) -> None:
        """Display data as a formatted table."""
        table = Table(title=f"{symbol} - {timestamp}", box=None, show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            else:
                value_str = str(value)
            
            table.add_row(key, value_str)
        
        self.formatter.console.print(table)
    
    def _save_messages(self, messages: List[Dict[str, Any]], output_file: str) -> None:
        """Save messages to file."""
        try:
            file_path = Path(output_file)
            
            if file_path.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(messages, f, indent=2)
            elif file_path.suffix.lower() == '.csv':
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'type', 'symbol', 'data'])
                    for msg in messages:
                        data = msg['data']
                        writer.writerow([
                            msg['timestamp'],
                            data.get('type', ''),
                            data.get('symbol', ''),
                            json.dumps(data.get('data', {}))
                        ])
            else:
                # Default to JSON
                with open(output_file, 'w') as f:
                    json.dump(messages, f, indent=2)
            
            self.formatter.print_success(f"Saved {len(messages)} messages to {output_file}")
            
        except Exception as e:
            self.formatter.print_error(f"Failed to save messages: {e}")


# Click command group
@click.group(name='data')
@click.pass_context
def data_commands(ctx):
    """Data-related commands for querying and streaming market data."""
    pass


@data_commands.command()
@click.argument('symbol')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def latest(ctx, symbol, output_format):
    """
    Get latest real-time data for a symbol.
    
    SYMBOL: Trading symbol (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py data latest AAPL
        wagehood_cli.py data latest SPY --format json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Fetching latest data...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_latest_data(symbol.upper())
            progress.update(task, completed=1)
        
        formatter.print_data(data, f"Latest Data - {symbol.upper()}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@data_commands.command()
@click.argument('symbol')
@click.option('--indicators', '-i', multiple=True,
              help='Specific indicators to retrieve (can be used multiple times)')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def indicators(ctx, symbol, indicators, output_format):
    """
    Get latest indicators for a symbol.
    
    SYMBOL: Trading symbol (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py data indicators AAPL
        wagehood_cli.py data indicators SPY -i sma_20 -i rsi
        wagehood_cli.py data indicators QQQ --indicators macd --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Fetching indicators...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_indicators(symbol.upper(), list(indicators) if indicators else None)
            progress.update(task, completed=1)
        
        formatter.print_data(data, f"Indicators - {symbol.upper()}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@data_commands.command()
@click.argument('symbol')
@click.option('--strategy', '-s', help='Specific strategy to retrieve signals for')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def signals(ctx, symbol, strategy, output_format):
    """
    Get latest trading signals for a symbol.
    
    SYMBOL: Trading symbol (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py data signals AAPL
        wagehood_cli.py data signals SPY --strategy ma_crossover
        wagehood_cli.py data signals QQQ -s rsi_trend --format json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Fetching signals...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_signals(symbol.upper(), strategy)
            progress.update(task, completed=1)
        
        formatter.print_data(data, f"Trading Signals - {symbol.upper()}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@data_commands.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--duration', '-d', type=int, help='Duration in seconds to stream')
@click.option('--output-file', '-o', type=click.Path(), help='Save messages to file')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def stream(ctx, symbols, duration, output_file, output_format):
    """
    Stream real-time data for symbols.
    
    SYMBOLS: One or more trading symbols (e.g., AAPL SPY QQQ)
    
    Examples:
        wagehood_cli.py data stream AAPL
        wagehood_cli.py data stream AAPL SPY --duration 60
        wagehood_cli.py data stream QQQ --output-file stream_data.json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    async def stream_data():
        streamer = WebSocketStreamer(config, formatter)
        
        try:
            await streamer.connect()
            await streamer.subscribe([s.upper() for s in symbols])
            await streamer.listen(duration, output_file)
        except KeyboardInterrupt:
            formatter.print_info("Stream interrupted by user")
        except Exception as e:
            formatter.print_error(f"Streaming error: {e}")
        finally:
            await streamer.disconnect()
    
    try:
        asyncio.run(stream_data())
    except KeyboardInterrupt:
        formatter.print_info("Stream cancelled by user")
    except Exception as e:
        formatter.print_error(f"Failed to start stream: {e}")
        ctx.exit(1)


@data_commands.command()
@click.argument('symbol')
@click.option('--indicator', '-i', help='Specific indicator to retrieve')
@click.option('--start-date', type=click.DateTime(),
              help='Start date for data query (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(),
              help='End date for data query (YYYY-MM-DD)')
@click.option('--limit', '-l', type=int, default=1000,
              help='Maximum number of records to return')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def historical(ctx, symbol, indicator, start_date, end_date, limit, output_format):
    """
    Get historical data for a symbol.
    
    SYMBOL: Trading symbol (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py data historical AAPL
        wagehood_cli.py data historical SPY --indicator sma_20
        wagehood_cli.py data historical QQQ --start-date 2024-01-01 --limit 500
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Fetching historical data...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_historical_data(
                symbol.upper(), indicator, start_date, end_date, limit
            )
            progress.update(task, completed=1)
        
        formatter.print_data(data, f"Historical Data - {symbol.upper()}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@data_commands.group()
@click.pass_context
def export(ctx):
    """Data export commands."""
    pass


@export.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--indicators', '-i', multiple=True,
              help='Indicators to include (can be used multiple times)')
@click.option('--strategies', '-s', multiple=True,
              help='Strategies to include (can be used multiple times)')
@click.option('--start-date', type=click.DateTime(),
              help='Start date for export (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(),
              help='End date for export (YYYY-MM-DD)')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'csv', 'parquet'], case_sensitive=False),
              help='Export format')
@click.pass_context
def create(ctx, symbols, indicators, strategies, start_date, end_date, format):
    """
    Create a bulk data export job.
    
    SYMBOLS: One or more trading symbols to export
    
    Examples:
        wagehood_cli.py data export create AAPL SPY
        wagehood_cli.py data export create QQQ -i sma_20 -i rsi --format csv
        wagehood_cli.py data export create AAPL --start-date 2024-01-01 --end-date 2024-12-31
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Creating export job...") as progress:
            task = progress.add_task("Processing...", total=None)
            
            export_data = client.create_export(
                symbols=[s.upper() for s in symbols],
                indicators=list(indicators) if indicators else None,
                strategies=list(strategies) if strategies else None,
                start_date=start_date,
                end_date=end_date,
                format=format
            )
            
            progress.update(task, completed=1)
        
        formatter.print_data(export_data, "Export Job Created")
        
        # Show helpful information
        export_id = export_data.get('export_id')
        if export_id:
            formatter.print_info(f"Export ID: {export_id}")
            formatter.print_info(f"Check status with: wagehood_cli.py data export status {export_id}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@export.command()
@click.argument('export_id')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def status(ctx, export_id, output_format):
    """
    Get the status of an export job.
    
    EXPORT_ID: The ID of the export job to check
    
    Examples:
        wagehood_cli.py data export status abc123
        wagehood_cli.py data export status abc123 --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = DataAPIClient(config)
        
        with formatter.create_progress_bar("Checking export status...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_export_status(export_id)
            progress.update(task, completed=1)
        
        formatter.print_data(data, f"Export Status - {export_id}")
        
        # Show helpful information
        status = data.get('status')
        if status == 'completed':
            formatter.print_success("Export is ready for download")
            formatter.print_info(f"Download with: wagehood_cli.py data export download {export_id}")
        elif status == 'failed':
            formatter.print_error("Export failed")
        elif status == 'pending':
            formatter.print_info("Export is still processing...")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@export.command()
@click.argument('export_id')
@click.option('--output-file', '-o', type=click.Path(),
              help='Output file path (default: export_<id>.<format>)')
@click.pass_context
def download(ctx, export_id, output_file):
    """
    Download an export job's data.
    
    EXPORT_ID: The ID of the export job to download
    
    Examples:
        wagehood_cli.py data export download abc123
        wagehood_cli.py data export download abc123 -o my_export.json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = DataAPIClient(config)
        
        # If no output file specified, use default name
        if not output_file:
            output_file = f"export_{export_id}.json"
        
        with formatter.create_progress_bar("Downloading export...") as progress:
            task = progress.add_task("Downloading...", total=None)
            client.download_export(export_id, output_file)
            progress.update(task, completed=1)
        
        formatter.print_success(f"Export downloaded to: {output_file}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)