#!/usr/bin/env python3
"""
Comprehensive Test Framework Logging System

Provides structured, multi-level logging for the comprehensive test framework
with file rotation, test session isolation, and detailed error tracking.
"""

import logging
import os
import sys
import json
import gzip
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TextIO
from pathlib import Path
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextlib import contextmanager
import traceback


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: str
    logger_name: str
    message: str
    test_suite: Optional[str] = None
    test_name: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    traceback_info: Optional[str] = None


class TestSessionHandler(logging.Handler):
    """Custom logging handler for test session isolation."""

    def __init__(self, session_id: str, base_dir: str):
        super().__init__()
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / f"session_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Session-specific log file
        self.log_file = self.session_dir / "session.log"
        self.file_handler = None

        # In-memory log storage for session analysis
        self.log_entries = []
        self.lock = threading.Lock()

        self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup file handler for session logs."""
        self.file_handler = logging.FileHandler(
            self.log_file, mode="a", encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(formatter)

    def emit(self, record):
        """Emit a log record."""
        try:
            # Write to file
            if self.file_handler:
                self.file_handler.emit(record)

            # Store in memory for analysis
            with self.lock:
                log_entry = LogEntry(
                    timestamp=datetime.fromtimestamp(record.created),
                    level=record.levelname,
                    logger_name=record.name,
                    message=record.getMessage(),
                    session_id=self.session_id,
                    extra_data=getattr(record, "extra_data", {}),
                    traceback_info=record.exc_text if record.exc_info else None,
                )

                # Extract test context if available
                if hasattr(record, "test_suite"):
                    log_entry.test_suite = record.test_suite
                if hasattr(record, "test_name"):
                    log_entry.test_name = record.test_name

                self.log_entries.append(log_entry)

        except Exception:
            self.handleError(record)

    def get_session_logs(self) -> List[LogEntry]:
        """Get all log entries for this session."""
        with self.lock:
            return self.log_entries.copy()

    def close(self):
        """Close the handler and cleanup resources."""
        if self.file_handler:
            self.file_handler.close()
        super().close()


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra context
        if hasattr(record, "test_suite"):
            log_entry["test_suite"] = record.test_suite
        if hasattr(record, "test_name"):
            log_entry["test_name"] = record.test_name
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "extra_data"):
            log_entry["extra_data"] = record.extra_data

        return json.dumps(log_entry)


class TestLogger:
    """Comprehensive test framework logging manager."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize test logger."""
        self.base_dir = Path(base_dir or self._get_default_log_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Session management
        self.current_session_id = None
        self.session_handler = None
        self.active_loggers = {}

        # Archival settings
        self.archive_dir = self.base_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)

        # Setup root logger configuration
        self._setup_root_logger()

        # Performance tracking
        self.log_stats = {
            "entries_logged": 0,
            "errors_logged": 0,
            "warnings_logged": 0,
            "sessions_created": 0,
        }

    def _get_default_log_dir(self) -> str:
        """Get default logging directory."""
        return os.path.join(os.path.dirname(__file__), "..", "logs")

    def _setup_root_logger(self):
        """Setup root logger configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler for comprehensive logging
        main_log_file = self.base_dir / "comprehensive_tests.log"
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # JSON handler for structured logging
        json_log_file = self.base_dir / "comprehensive_tests.jsonl"
        json_handler = RotatingFileHandler(
            json_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3,
            encoding="utf-8",
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)

        # Error handler for critical issues
        error_log_file = self.base_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

    def setup(self):
        """Setup test logger."""
        logging.info("Test logging system initialized")

        # Create new session
        self.start_new_session()

        # Log system information
        self._log_system_info()

    def start_new_session(self) -> str:
        """Start a new test session."""
        # End previous session if active
        if self.session_handler:
            self.end_current_session()

        # Create new session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_id

        # Setup session handler
        self.session_handler = TestSessionHandler(session_id, str(self.base_dir))

        # Add session handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.session_handler)

        self.log_stats["sessions_created"] += 1

        logging.info(f"Started new test session: {session_id}")
        return session_id

    def end_current_session(self):
        """End the current test session."""
        if not self.session_handler:
            return

        session_id = self.current_session_id

        # Generate session summary
        self._generate_session_summary()

        # Remove session handler
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.session_handler)

        # Close and cleanup
        self.session_handler.close()
        self.session_handler = None
        self.current_session_id = None

        logging.info(f"Ended test session: {session_id}")

    def get_logger(self, name: str, test_suite: Optional[str] = None) -> logging.Logger:
        """Get a logger with test context."""
        logger = logging.getLogger(name)

        # Add test context to logger
        if test_suite:
            logger = TestContextLogger(logger, test_suite, self.current_session_id)

        self.active_loggers[name] = logger
        return logger

    def _log_system_info(self):
        """Log system information for debugging."""
        import platform
        import psutil

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
        }

        logging.info(f"System information: {json.dumps(system_info, indent=2)}")

    def _generate_session_summary(self):
        """Generate summary for the current session."""
        if not self.session_handler:
            return

        session_logs = self.session_handler.get_session_logs()

        # Calculate session statistics
        total_entries = len(session_logs)
        error_count = len([log for log in session_logs if log.level == "ERROR"])
        warning_count = len([log for log in session_logs if log.level == "WARNING"])
        info_count = len([log for log in session_logs if log.level == "INFO"])
        debug_count = len([log for log in session_logs if log.level == "DEBUG"])

        # Test suite breakdown
        test_suites = set(log.test_suite for log in session_logs if log.test_suite)

        summary = {
            "session_id": self.current_session_id,
            "total_log_entries": total_entries,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "debug_count": debug_count,
            "test_suites_involved": list(test_suites),
            "session_start": (
                session_logs[0].timestamp.isoformat() if session_logs else None
            ),
            "session_end": (
                session_logs[-1].timestamp.isoformat() if session_logs else None
            ),
        }

        # Save session summary
        summary_file = self.session_handler.session_dir / "session_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Session summary: {json.dumps(summary, indent=2, default=str)}")

        # Update global stats
        self.log_stats["entries_logged"] += total_entries
        self.log_stats["errors_logged"] += error_count
        self.log_stats["warnings_logged"] += warning_count

    def archive_logs(self, days_to_keep: int = 30):
        """Archive old log files."""
        logging.info(f"Archiving logs older than {days_to_keep} days")

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        archived_count = 0

        # Archive session directories
        for session_dir in self.base_dir.glob("session_*"):
            if session_dir.is_dir():
                try:
                    # Check session date from directory name
                    session_date_str = session_dir.name.replace("session_", "")
                    session_date = datetime.strptime(session_date_str[:8], "%Y%m%d")

                    if session_date < cutoff_date:
                        # Create compressed archive
                        archive_file = self.archive_dir / f"{session_dir.name}.tar.gz"

                        import tarfile

                        with tarfile.open(archive_file, "w:gz") as tar:
                            tar.add(session_dir, arcname=session_dir.name)

                        # Remove original directory
                        import shutil

                        shutil.rmtree(session_dir)

                        archived_count += 1
                        logging.info(f"Archived session: {session_dir.name}")

                except (ValueError, OSError) as e:
                    logging.warning(f"Failed to archive {session_dir}: {e}")

        logging.info(f"Archived {archived_count} old log sessions")

    def get_recent_errors(self, hours: int = 24) -> List[LogEntry]:
        """Get recent error log entries."""
        if not self.session_handler:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        session_logs = self.session_handler.get_session_logs()

        return [
            log
            for log in session_logs
            if log.level == "ERROR" and log.timestamp > cutoff_time
        ]

    def get_test_suite_logs(self, test_suite: str) -> List[LogEntry]:
        """Get all logs for a specific test suite."""
        if not self.session_handler:
            return []

        session_logs = self.session_handler.get_session_logs()
        return [log for log in session_logs if log.test_suite == test_suite]

    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        return {
            **self.log_stats,
            "current_session": self.current_session_id,
            "active_loggers": len(self.active_loggers),
            "base_directory": str(self.base_dir),
            "disk_usage_mb": sum(
                f.stat().st_size for f in self.base_dir.rglob("*") if f.is_file()
            )
            / (1024 * 1024),
        }

    @contextmanager
    def test_context(self, test_suite: str, test_name: str):
        """Context manager for test-specific logging."""
        # Create test-specific logger
        test_logger = self.get_logger(f"{test_suite}.{test_name}", test_suite)

        try:
            test_logger.info(f"Starting test: {test_name}")
            yield test_logger
        except Exception as e:
            test_logger.error(f"Test failed with exception: {e}", exc_info=True)
            raise
        finally:
            test_logger.info(f"Completed test: {test_name}")


class TestContextLogger:
    """Logger wrapper that adds test context to log records."""

    def __init__(
        self, logger: logging.Logger, test_suite: str, session_id: Optional[str] = None
    ):
        self.logger = logger
        self.test_suite = test_suite
        self.session_id = session_id

    def _add_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add test context to log record extra data."""
        context = {"test_suite": self.test_suite, "session_id": self.session_id}

        if extra:
            context.update(extra)

        return context

    def debug(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs["extra"] = self._add_context(kwargs.get("extra"))
        self.logger.exception(msg, *args, **kwargs)


# Global test logger instance
_test_logger_instance = None


def get_test_logger() -> TestLogger:
    """Get the global test logger instance."""
    global _test_logger_instance
    if _test_logger_instance is None:
        _test_logger_instance = TestLogger()
    return _test_logger_instance


def setup_test_logging(base_dir: Optional[str] = None) -> TestLogger:
    """Setup and return the global test logger."""
    global _test_logger_instance
    _test_logger_instance = TestLogger(base_dir)
    _test_logger_instance.setup()
    return _test_logger_instance
