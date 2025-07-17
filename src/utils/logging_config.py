"""
Centralized Logging Configuration with Rotation

Provides log rotation, size limits, and retention policies to prevent disk space issues.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_rotating_logger(
    name: str,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5,              # Keep 5 backup files
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with rotating file handler to prevent disk space issues.
    
    Args:
        name: Logger name
        log_file: Path to log file (default: ~/.wagehood/{name}.log)
        max_bytes: Maximum size per log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        level: Logging level (default: INFO)
        console_output: Whether to also log to console (default: True)
        
    Returns:
        Configured logger with rotation
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Create log directory
    if log_file is None:
        log_dir = Path.home() / '.wagehood'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'{name}.log'
    else:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


def setup_trigger_logging(
    console_output: bool = False,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Set up logging for the trigger analysis system with rotation.
    
    Args:
        console_output: Whether to output to console (default: False for production)
        max_file_size_mb: Maximum size per log file in MB (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    max_bytes = max_file_size_mb * 1024 * 1024
    log_dir = Path.home() / '.wagehood'
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # Set up main application logger
    app_logger = setup_rotating_logger(
        name='triggers',
        log_file=log_dir / 'triggers.log',
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_output=console_output
    )
    
    # Set up error logger for critical issues
    error_logger = setup_rotating_logger(
        name='trigger_errors',
        log_file=log_dir / 'trigger_error.log',
        max_bytes=max_bytes,
        backup_count=backup_count,
        level=logging.WARNING,
        console_output=console_output
    )
    
    # Configure specific loggers to use our rotating handlers
    for logger_name in [
        'src.strategies',
        'src.data.providers',
        'src.notifications',
        'signals'  # Signal detection logger
    ]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(app_logger.handlers[0])  # File handler
        if console_output and len(app_logger.handlers) > 1:
            logger.addHandler(app_logger.handlers[1])  # Console handler
        logger.propagate = False  # Don't propagate to root
    
    return app_logger


def cleanup_old_logs(log_dir: Optional[Path] = None, days_to_keep: int = 7) -> None:
    """
    Clean up old log files beyond the backup count.
    
    Args:
        log_dir: Directory containing log files (default: ~/.wagehood)
        days_to_keep: Number of days to keep log files (default: 7)
    """
    if log_dir is None:
        log_dir = Path.home() / '.wagehood'
    
    if not log_dir.exists():
        return
    
    import time
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    # Clean up old log files
    for log_file in log_dir.glob('*.log*'):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Cleaned up old log file: {log_file}")
        except Exception as e:
            print(f"Error cleaning up {log_file}: {e}")


def get_log_file_stats(log_dir: Optional[Path] = None) -> dict:
    """
    Get statistics about log files.
    
    Args:
        log_dir: Directory containing log files (default: ~/.wagehood)
        
    Returns:
        Dictionary with log file statistics
    """
    if log_dir is None:
        log_dir = Path.home() / '.wagehood'
    
    if not log_dir.exists():
        return {}
    
    stats = {}
    total_size = 0
    
    for log_file in log_dir.glob('*.log*'):
        try:
            size = log_file.stat().st_size
            stats[log_file.name] = {
                'size_mb': round(size / (1024 * 1024), 2),
                'size_bytes': size
            }
            total_size += size
        except Exception as e:
            stats[log_file.name] = {'error': str(e)}
    
    stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
    stats['total_files'] = len([f for f in log_dir.glob('*.log*')])
    
    return stats


if __name__ == "__main__":
    # Test the logging configuration
    setup_trigger_logging(console_output=True, max_file_size_mb=1, backup_count=3)
    
    logger = logging.getLogger('triggers')
    
    # Test log rotation by generating many log entries
    for i in range(1000):
        logger.info(f"Test log entry {i} - testing log rotation functionality")
    
    print("Log rotation test completed")
    print("Log file stats:", get_log_file_stats())