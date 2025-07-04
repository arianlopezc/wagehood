"""
Logging Utilities for Wagehood CLI

This module provides comprehensive logging setup and utilities for the CLI,
including rich console logging, file logging, and log management.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


def setup_logging(level: str = "INFO", 
                  log_file: Optional[str] = None,
                  console: Optional[Console] = None,
                  enable_rich_traceback: bool = True) -> None:
    """
    Set up comprehensive logging for the CLI.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Rich console instance
        enable_rich_traceback: Whether to enable rich traceback formatting
    """
    # Install rich traceback handler
    if enable_rich_traceback:
        install(show_locals=level == "DEBUG")
    
    # Get log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create console handler with rich formatting
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=enable_rich_traceback,
        show_path=level == "DEBUG",
        show_time=True,
        show_level=True,
        markup=True
    )
    console_handler.setLevel(log_level)
    
    # Create console formatter
    console_formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = create_file_handler(log_file, log_level)
        root_logger.addHandler(file_handler)
    
    # Set up specific logger levels
    setup_logger_levels(level)


def create_file_handler(log_file: str, level: int) -> logging.Handler:
    """
    Create a file handler for logging.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured file handler
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    file_handler.setLevel(level)
    
    # Create detailed formatter for file
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    return file_handler


def setup_logger_levels(level: str) -> None:
    """
    Set up specific logger levels for different components.
    
    Args:
        level: Base logging level
    """
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # Set CLI-specific logger levels
    if level == "DEBUG":
        logging.getLogger("cli").setLevel(logging.DEBUG)
        logging.getLogger("wagehood").setLevel(logging.DEBUG)
    else:
        logging.getLogger("cli").setLevel(logging.INFO)
        logging.getLogger("wagehood").setLevel(logging.INFO)


class CLILogger:
    """Enhanced logger for CLI operations."""
    
    def __init__(self, name: str = "cli"):
        """
        Initialize CLI logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.start_time = datetime.now()
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    def log_command_start(self, command: str, args: Dict[str, Any]) -> None:
        """Log command start."""
        self.info(f"Starting command: {command}")
        if args:
            self.debug(f"Command arguments: {args}")
    
    def log_command_end(self, command: str, success: bool = True) -> None:
        """Log command end."""
        duration = (datetime.now() - self.start_time).total_seconds()
        status = "completed" if success else "failed"
        self.info(f"Command {command} {status} in {duration:.2f}s")
    
    def log_api_request(self, method: str, url: str, status_code: Optional[int] = None) -> None:
        """Log API request."""
        message = f"API {method} {url}"
        if status_code:
            message += f" -> {status_code}"
        self.debug(message)
    
    def log_api_error(self, method: str, url: str, error: Exception) -> None:
        """Log API error."""
        self.error(f"API {method} {url} failed: {error}")
    
    def log_data_operation(self, operation: str, count: int, duration: float) -> None:
        """Log data operation."""
        rate = count / duration if duration > 0 else 0
        self.info(f"{operation}: {count} items in {duration:.2f}s ({rate:.1f} items/s)")
    
    def log_file_operation(self, operation: str, filename: str, size: Optional[int] = None) -> None:
        """Log file operation."""
        message = f"File {operation}: {filename}"
        if size is not None:
            message += f" ({size} bytes)"
        self.info(message)
    
    def log_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Log configuration change."""
        self.info(f"Configuration changed: {key} = {old_value} -> {new_value}")
    
    def log_performance_metric(self, metric: str, value: float, unit: str = "") -> None:
        """Log performance metric."""
        self.info(f"Performance metric: {metric} = {value}{unit}")
    
    def log_system_info(self, info: Dict[str, Any]) -> None:
        """Log system information."""
        self.info("System information:")
        for key, value in info.items():
            self.info(f"  {key}: {value}")
    
    def log_websocket_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log WebSocket event."""
        message = f"WebSocket {event}"
        if details:
            message += f": {details}"
        self.debug(message)
    
    def log_cache_operation(self, operation: str, key: str, hit: bool = True) -> None:
        """Log cache operation."""
        status = "hit" if hit else "miss"
        self.debug(f"Cache {operation} {key}: {status}")
    
    def log_validation_error(self, field: str, value: Any, error: str) -> None:
        """Log validation error."""
        self.error(f"Validation error for {field} = {value}: {error}")
    
    def log_retry_attempt(self, attempt: int, max_attempts: int, error: Exception) -> None:
        """Log retry attempt."""
        self.warning(f"Retry attempt {attempt}/{max_attempts}: {error}")
    
    def log_background_task(self, task: str, status: str) -> None:
        """Log background task status."""
        self.info(f"Background task {task}: {status}")


class ContextLogger:
    """Context manager for logging operations."""
    
    def __init__(self, logger: CLILogger, operation: str, **kwargs):
        """
        Initialize context logger.
        
        Args:
            logger: CLI logger instance
            operation: Operation name
            **kwargs: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
        self.logger.info(f"Starting {self.operation}" + (f" ({context_str})" if context_str else ""))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(f"{self.operation} failed after {duration:.2f}s: {exc_val}")
        else:
            self.logger.info(f"{self.operation} completed in {duration:.2f}s")


class LogCollector:
    """Collects and manages log entries."""
    
    def __init__(self, max_entries: int = 1000):
        """
        Initialize log collector.
        
        Args:
            max_entries: Maximum number of log entries to keep
        """
        self.max_entries = max_entries
        self.entries = []
        self.handler = None
    
    def start_collecting(self, logger_name: str = "cli") -> None:
        """Start collecting log entries."""
        self.handler = LogCollectorHandler(self)
        logger = logging.getLogger(logger_name)
        logger.addHandler(self.handler)
    
    def stop_collecting(self, logger_name: str = "cli") -> None:
        """Stop collecting log entries."""
        if self.handler:
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self.handler)
            self.handler = None
    
    def add_entry(self, record: logging.LogRecord) -> None:
        """Add log entry."""
        entry = {
            'timestamp': datetime.fromtimestamp(record.created),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        self.entries.append(entry)
        
        # Keep only the most recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_entries(self, level: Optional[str] = None, 
                   module: Optional[str] = None,
                   limit: Optional[int] = None) -> list:
        """
        Get log entries with optional filtering.
        
        Args:
            level: Filter by log level
            module: Filter by module name
            limit: Limit number of entries
            
        Returns:
            List of log entries
        """
        filtered_entries = self.entries
        
        if level:
            filtered_entries = [e for e in filtered_entries if e['level'] == level]
        
        if module:
            filtered_entries = [e for e in filtered_entries if e['module'] == module]
        
        if limit:
            filtered_entries = filtered_entries[-limit:]
        
        return filtered_entries
    
    def get_error_entries(self, limit: int = 50) -> list:
        """Get recent error entries."""
        return self.get_entries(level='ERROR', limit=limit)
    
    def get_warning_entries(self, limit: int = 50) -> list:
        """Get recent warning entries."""
        return self.get_entries(level='WARNING', limit=limit)
    
    def clear_entries(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
    
    def export_entries(self, filename: str, format: str = 'json') -> None:
        """
        Export log entries to file.
        
        Args:
            filename: Output filename
            format: Export format (json, csv)
        """
        if format.lower() == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(self.entries, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                if self.entries:
                    writer = csv.DictWriter(f, fieldnames=self.entries[0].keys())
                    writer.writeheader()
                    writer.writerows(self.entries)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class LogCollectorHandler(logging.Handler):
    """Handler that collects log records."""
    
    def __init__(self, collector: LogCollector):
        """
        Initialize handler.
        
        Args:
            collector: Log collector instance
        """
        super().__init__()
        self.collector = collector
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to collector."""
        self.collector.add_entry(record)


def get_log_file_path(name: str = "wagehood-cli") -> str:
    """
    Get default log file path.
    
    Args:
        name: Base name for log file
        
    Returns:
        Log file path
    """
    # Use user's home directory for logs
    log_dir = Path.home() / ".wagehood" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return str(log_dir / f"{name}.log")


def setup_debug_logging(console: Optional[Console] = None) -> CLILogger:
    """
    Set up debug logging for development.
    
    Args:
        console: Rich console instance
        
    Returns:
        CLI logger instance
    """
    setup_logging(
        level="DEBUG",
        log_file=get_log_file_path("debug"),
        console=console,
        enable_rich_traceback=True
    )
    
    return CLILogger("debug")


def setup_production_logging(console: Optional[Console] = None) -> CLILogger:
    """
    Set up production logging.
    
    Args:
        console: Rich console instance
        
    Returns:
        CLI logger instance
    """
    setup_logging(
        level="INFO",
        log_file=get_log_file_path(),
        console=console,
        enable_rich_traceback=False
    )
    
    return CLILogger("cli")


# Context managers for common logging patterns
def log_operation(logger: CLILogger, operation: str, **kwargs):
    """Context manager for logging operations."""
    return ContextLogger(logger, operation, **kwargs)


def log_api_call(logger: CLILogger, method: str, url: str):
    """Context manager for logging API calls."""
    return ContextLogger(logger, f"API {method}", url=url)


def log_file_operation(logger: CLILogger, operation: str, filename: str):
    """Context manager for logging file operations."""
    return ContextLogger(logger, f"File {operation}", filename=filename)


def log_data_processing(logger: CLILogger, operation: str, count: int):
    """Context manager for logging data processing."""
    return ContextLogger(logger, f"Data {operation}", count=count)