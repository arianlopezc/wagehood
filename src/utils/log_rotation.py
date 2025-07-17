#!/usr/bin/env python3
"""
Log Rotation Utilities

Provides log rotation functionality to prevent log files from growing too large.
"""

import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LogRotator:
    """Handles log file rotation with compression."""
    
    def __init__(self, 
                 log_file: Path,
                 max_size_mb: float = 10,
                 max_backups: int = 5,
                 compress: bool = True):
        """
        Initialize log rotator.
        
        Args:
            log_file: Path to the log file
            max_size_mb: Maximum size in MB before rotation
            max_backups: Number of backup files to keep
            compress: Whether to compress rotated files
        """
        self.log_file = Path(log_file)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_backups = max_backups
        self.compress = compress
    
    def should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if not self.log_file.exists():
            return False
        
        size = self.log_file.stat().st_size
        return size >= self.max_size_bytes
    
    def rotate(self):
        """Rotate the log file."""
        if not self.log_file.exists():
            return
        
        # Generate timestamp for rotated file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Shift existing backups
        self._shift_backups()
        
        # Move current log to .1
        if self.compress:
            rotated_path = self.log_file.with_suffix(f'.log.1.gz')
            # Compress the current log file
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(rotated_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove original
            self.log_file.unlink()
        else:
            rotated_path = self.log_file.with_suffix(f'.log.1')
            shutil.move(str(self.log_file), str(rotated_path))
        
        logger.info(f"Rotated {self.log_file} to {rotated_path}")
    
    def _shift_backups(self):
        """Shift backup files to make room for new rotation."""
        # Find existing backups
        backups = []
        for i in range(1, self.max_backups + 1):
            if self.compress:
                backup_path = self.log_file.with_suffix(f'.log.{i}.gz')
            else:
                backup_path = self.log_file.with_suffix(f'.log.{i}')
            
            if backup_path.exists():
                backups.append((i, backup_path))
        
        # Sort in reverse order (highest number first)
        backups.sort(reverse=True)
        
        # Shift each backup
        for num, backup_path in backups:
            if num >= self.max_backups:
                # Delete oldest backup
                backup_path.unlink()
                logger.info(f"Deleted old backup: {backup_path}")
            else:
                # Shift to next number
                if self.compress:
                    new_path = self.log_file.with_suffix(f'.log.{num + 1}.gz')
                else:
                    new_path = self.log_file.with_suffix(f'.log.{num + 1}')
                
                shutil.move(str(backup_path), str(new_path))
    
    def cleanup_old_backups(self):
        """Remove backups beyond max_backups limit."""
        for i in range(self.max_backups + 1, 100):  # Check up to 100
            if self.compress:
                backup_path = self.log_file.with_suffix(f'.log.{i}.gz')
            else:
                backup_path = self.log_file.with_suffix(f'.log.{i}')
            
            if backup_path.exists():
                backup_path.unlink()
                logger.info(f"Cleaned up old backup: {backup_path}")
            else:
                break  # No more backups


class RotatingFileHandler(logging.FileHandler):
    """
    A file handler that rotates logs based on size.
    
    This is a simple implementation that checks rotation on each emit.
    For production use, consider using logging.handlers.RotatingFileHandler.
    """
    
    def __init__(self, filename, max_bytes=10*1024*1024, backup_count=5):
        super().__init__(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotator = LogRotator(
            Path(filename),
            max_size_mb=max_bytes / (1024 * 1024),
            max_backups=backup_count,
            compress=True
        )
    
    def emit(self, record):
        """Emit a record, rotating if necessary."""
        # Check if rotation is needed before emitting
        if self.rotator.should_rotate():
            self.close()  # Close the current file
            self.rotator.rotate()  # Rotate it
            self._open()  # Reopen for writing
        
        # Emit the record
        super().emit(record)


def setup_rotating_logger(name: str, 
                         log_file: Path,
                         max_size_mb: float = 10,
                         max_backups: int = 5,
                         level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with rotating file handler.
    
    Args:
        name: Logger name
        log_file: Path to log file
        max_size_mb: Maximum size before rotation
        max_backups: Number of backups to keep
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create rotating file handler
    handler = RotatingFileHandler(
        str(log_file),
        max_bytes=int(max_size_mb * 1024 * 1024),
        backup_count=max_backups
    )
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(handler)
    
    return logger


def rotate_log_if_needed(log_file: Path, max_size_mb: float = 10, max_backups: int = 5):
    """
    Check and rotate a log file if needed.
    
    This can be called periodically or at startup to ensure logs don't grow too large.
    """
    rotator = LogRotator(log_file, max_size_mb, max_backups)
    
    if rotator.should_rotate():
        rotator.rotate()
        return True
    
    return False