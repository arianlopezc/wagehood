"""
Utility Functions for Wagehood CLI

This module provides utility functions including API client wrapper, WebSocket client,
error handling, and other common utilities used across the CLI.
"""

import asyncio
import json
import time
import requests
import websockets
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import backoff

from .utils.logging import CLILogger
from .config import CLIConfig


class APIClient:
    """Enhanced API client with retry logic and caching."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize API client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_api_headers())
        self.logger = CLILogger("api_client")
        self._cache = {}
        self._cache_ttl = config.cache_ttl if config.cache_enabled else 0
    
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3,
        max_time=30
    )
    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an API request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response object
        """
        url = self.config.get_api_url(endpoint)
        
        # Set timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.config.api_timeout
        
        # Log request
        self.logger.log_api_request(method, url)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Log successful response
            self.logger.log_api_request(method, url, response.status_code)
            
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.log_api_error(method, url, e)
            raise
    
    def get(self, endpoint: str, params: Optional[Dict] = None, 
            use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Make GET request with optional caching.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use cache
            **kwargs: Additional request arguments
            
        Returns:
            Response data
        """
        # Check cache
        if use_cache and self._cache_ttl > 0:
            cache_key = f"{endpoint}:{json.dumps(params or {})}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Make request
        response = self.request('GET', endpoint, params=params, **kwargs)
        data = response.json()
        
        # Cache response
        if use_cache and self._cache_ttl > 0:
            self._set_cached(cache_key, data)
        
        return data
    
    def post(self, endpoint: str, data: Optional[Dict] = None,
             json_data: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            **kwargs: Additional request arguments
            
        Returns:
            Response data
        """
        if json_data:
            kwargs['json'] = json_data
        elif data:
            kwargs['data'] = data
        
        response = self.request('POST', endpoint, **kwargs)
        return response.json()
    
    def put(self, endpoint: str, data: Optional[Dict] = None,
            json_data: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Make PUT request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            **kwargs: Additional request arguments
            
        Returns:
            Response data
        """
        if json_data:
            kwargs['json'] = json_data
        elif data:
            kwargs['data'] = data
        
        response = self.request('PUT', endpoint, **kwargs)
        return response.json()
    
    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make DELETE request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response data
        """
        response = self.request('DELETE', endpoint, **kwargs)
        return response.json()
    
    def stream(self, endpoint: str, params: Optional[Dict] = None, **kwargs):
        """
        Stream data from endpoint.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional request arguments
            
        Yields:
            Response lines
        """
        kwargs['stream'] = True
        response = self.request('GET', endpoint, params=params, **kwargs)
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def download(self, endpoint: str, output_path: str, **kwargs) -> None:
        """
        Download file from endpoint.
        
        Args:
            endpoint: API endpoint
            output_path: Path to save file
            **kwargs: Additional request arguments
        """
        kwargs['stream'] = True
        response = self.request('GET', endpoint, **kwargs)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        self.logger.log_file_operation("download", output_path)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self._cache_ttl:
                self.logger.log_cache_operation("get", key, hit=True)
                return entry['data']
            else:
                # Expired
                del self._cache[key]
        
        self.logger.log_cache_operation("get", key, hit=False)
        return None
    
    def _set_cached(self, key: str, data: Any) -> None:
        """Set cache value."""
        self._cache[key] = {
            'timestamp': time.time(),
            'data': data
        }
        self.logger.log_cache_operation("set", key)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.logger.info("Cache cleared")


class WebSocketClient:
    """WebSocket client for real-time data streaming."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize WebSocket client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.logger = CLILogger("websocket_client")
        self.websocket = None
        self.connection_id = None
        self.running = False
    
    async def connect(self, connection_id: Optional[str] = None) -> None:
        """
        Connect to WebSocket endpoint.
        
        Args:
            connection_id: Optional connection ID
        """
        self.connection_id = connection_id or f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        uri = self.config.get_ws_url(f"realtime/ws/{self.connection_id}")
        
        try:
            self.websocket = await websockets.connect(uri)
            self.running = True
            self.logger.log_websocket_event("connected", {"uri": uri})
        except Exception as e:
            self.logger.log_websocket_event("connection_failed", {"error": str(e)})
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.logger.log_websocket_event("disconnected")
    
    async def send(self, message: Dict[str, Any]) -> None:
        """
        Send message to WebSocket.
        
        Args:
            message: Message to send
        """
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        await self.websocket.send(json.dumps(message))
        self.logger.log_websocket_event("message_sent", message)
    
    async def receive(self) -> Dict[str, Any]:
        """
        Receive message from WebSocket.
        
        Returns:
            Received message
        """
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        message = await self.websocket.recv()
        data = json.loads(message)
        self.logger.log_websocket_event("message_received", data)
        return data
    
    async def subscribe(self, symbols: List[str], streams: Optional[List[str]] = None) -> None:
        """
        Subscribe to symbol streams.
        
        Args:
            symbols: List of symbols to subscribe
            streams: Optional list of stream types
        """
        message = {
            "action": "subscribe",
            "symbols": symbols
        }
        
        if streams:
            message["streams"] = streams
        
        await self.send(message)
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        message = {
            "action": "unsubscribe",
            "symbols": symbols
        }
        
        await self.send(message)
    
    async def ping(self) -> None:
        """Send ping message."""
        await self.send({"action": "ping"})
    
    async def listen(self, callback: Callable[[Dict[str, Any]], None],
                    error_callback: Optional[Callable[[Exception], None]] = None) -> None:
        """
        Listen for messages and call callback.
        
        Args:
            callback: Function to call with received messages
            error_callback: Optional function to call on errors
        """
        while self.running:
            try:
                message = await self.receive()
                callback(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.log_websocket_event("connection_closed")
                break
            except Exception as e:
                self.logger.log_websocket_event("error", {"error": str(e)})
                if error_callback:
                    error_callback(e)
                else:
                    raise
    
    async def stream_with_timeout(self, duration: int, 
                                 callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Stream messages for a specific duration.
        
        Args:
            duration: Duration in seconds
            callback: Function to call with received messages
        """
        end_time = datetime.now() + timedelta(seconds=duration)
        
        while self.running and datetime.now() < end_time:
            try:
                # Use wait_for to implement timeout on receive
                message = await asyncio.wait_for(self.receive(), timeout=1.0)
                callback(message)
            except asyncio.TimeoutError:
                # Check if we should continue
                continue
            except websockets.exceptions.ConnectionClosed:
                self.logger.log_websocket_event("connection_closed")
                break


class RetryManager:
    """Manages retry logic for operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retries
            backoff_factor: Backoff multiplication factor
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = CLILogger("retry_manager")
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    self.logger.log_retry_attempt(attempt + 1, self.max_retries, e)
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    self.logger.error(f"All retry attempts failed: {e}")
        
        raise last_error
    
    def async_retry(self, async_func: Callable) -> Callable:
        """
        Decorator for retrying async functions.
        
        Args:
            async_func: Async function to wrap
            
        Returns:
            Wrapped function
        """
        @wraps(async_func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await async_func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if attempt < self.max_retries:
                        wait_time = self.backoff_factor ** attempt
                        self.logger.log_retry_attempt(attempt + 1, self.max_retries, e)
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(f"All retry attempts failed: {e}")
            
            raise last_error
        
        return wrapper


class BatchProcessor:
    """Process items in batches."""
    
    def __init__(self, batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
        """
        self.batch_size = batch_size
        self.logger = CLILogger("batch_processor")
    
    def process_batches(self, items: List[Any], 
                       process_func: Callable[[List[Any]], Any],
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            process_func: Function to process each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        results = []
        total_items = len(items)
        processed = 0
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            try:
                batch_result = process_func(batch)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                
                processed += len(batch)
                
                if progress_callback:
                    progress_callback(processed, total_items)
                
                self.logger.debug(f"Processed batch {i//self.batch_size + 1}: {len(batch)} items")
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                raise
        
        return results
    
    async def async_process_batches(self, items: List[Any],
                                   async_process_func: Callable[[List[Any]], Any],
                                   concurrency: int = 5) -> List[Any]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: Items to process
            async_process_func: Async function to process each batch
            concurrency: Number of concurrent batches
            
        Returns:
            List of results
        """
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await async_process_func(batch)
        
        # Create batch tasks
        tasks = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            task = asyncio.create_task(process_batch_with_semaphore(batch))
            tasks.append(task)
        
        # Wait for all tasks
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results


class DataValidator:
    """Validates data according to schemas."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = CLILogger("data_validator")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate trading symbol.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if valid
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic symbol validation
        if not symbol.isalnum():
            return False
        
        if len(symbol) > 10:
            return False
        
        return True
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if valid
        """
        if start_date >= end_date:
            self.logger.log_validation_error("date_range", 
                                           f"{start_date} - {end_date}",
                                           "Start date must be before end date")
            return False
        
        # Check if range is reasonable (e.g., not more than 5 years)
        max_days = 365 * 5
        if (end_date - start_date).days > max_days:
            self.logger.log_validation_error("date_range",
                                           f"{start_date} - {end_date}",
                                           f"Date range exceeds maximum of {max_days} days")
            return False
        
        return True
    
    def validate_config_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration data against schema.
        
        Args:
            data: Data to validate
            schema: Validation schema
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        field_types = schema.get('types', {})
        for field, expected_type in field_types.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if actual_type != expected_type:
                    errors.append(f"Field '{field}' should be {expected_type}, got {actual_type}")
        
        # Check field values
        field_values = schema.get('values', {})
        for field, allowed_values in field_values.items():
            if field in data and data[field] not in allowed_values:
                errors.append(f"Field '{field}' has invalid value: {data[field]}")
        
        # Check numeric ranges
        field_ranges = schema.get('ranges', {})
        for field, (min_val, max_val) in field_ranges.items():
            if field in data:
                value = data[field]
                if not (min_val <= value <= max_val):
                    errors.append(f"Field '{field}' out of range: {value} (expected {min_val}-{max_val})")
        
        return errors


# Utility functions
def parse_date_string(date_str: str) -> datetime:
    """
    Parse date string in various formats.
    
    Args:
        date_str: Date string
        
    Returns:
        Parsed datetime
    """
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")


def format_table_data(data: List[Dict[str, Any]], 
                     columns: Optional[List[str]] = None) -> List[List[str]]:
    """
    Format data for table display.
    
    Args:
        data: List of dictionaries
        columns: Optional list of column names
        
    Returns:
        List of rows for table
    """
    if not data:
        return []
    
    # Get columns if not specified
    if not columns:
        columns = list(data[0].keys())
    
    # Create rows
    rows = []
    for item in data:
        row = []
        for col in columns:
            value = item.get(col, "")
            # Format value
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, datetime):
                row.append(value.strftime("%Y-%m-%d %H:%M:%S"))
            elif isinstance(value, (dict, list)):
                row.append(json.dumps(value))
            else:
                row.append(str(value))
        rows.append(row)
    
    return rows


def calculate_rate_limit_delay(remaining: int, reset_time: int) -> float:
    """
    Calculate delay for rate limiting.
    
    Args:
        remaining: Remaining requests
        reset_time: Reset timestamp
        
    Returns:
        Delay in seconds
    """
    if remaining > 0:
        return 0.0
    
    # Calculate time until reset
    current_time = time.time()
    delay = max(0, reset_time - current_time)
    
    # Add small buffer
    return delay + 0.1


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:max_length - len(ext) - 1] + '.' + ext if ext else name[:max_length]
    
    return filename


def create_backup_filename(original: str) -> str:
    """
    Create backup filename with timestamp.
    
    Args:
        original: Original filename
        
    Returns:
        Backup filename
    """
    path = Path(original)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return str(path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}")


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid
    """
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_file_hash(filepath: str, algorithm: str = 'sha256') -> str:
    """
    Calculate file hash.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm
        
    Returns:
        File hash
    """
    import hashlib
    
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


# Async utility functions
async def gather_with_concurrency(n: int, *coros) -> List[Any]:
    """
    Gather coroutines with concurrency limit.
    
    Args:
        n: Concurrency limit
        *coros: Coroutines to run
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(sem_coro(c) for c in coros))


def create_async_timer(interval: float, callback: Callable) -> asyncio.Task:
    """
    Create async timer that calls callback periodically.
    
    Args:
        interval: Timer interval in seconds
        callback: Callback function
        
    Returns:
        Timer task
    """
    async def timer_loop():
        while True:
            await asyncio.sleep(interval)
            await callback() if asyncio.iscoroutinefunction(callback) else callback()
    
    return asyncio.create_task(timer_loop())