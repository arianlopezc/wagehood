"""
Distributed job processing system for Wagehood.

This module provides Redis-based distributed job management with features:
- Job deduplication based on parameter hashing
- Worker coordination and heartbeat monitoring
- TTL-based automatic cleanup
- Multi-instance safe operations
"""

from .distributed import (
    DistributedJobManager,
    JobDeduplicator,
    WorkerManager,
    DistributedWorker,
    JobProgressTracker,
    DeadWorkerDetector,
)
from .models import JobStatus, JobType, JobMetadata, JobResult
from .job_processor import JobProcessor

__all__ = [
    "DistributedJobManager",
    "JobDeduplicator",
    "WorkerManager",
    "DistributedWorker",
    "JobProgressTracker",
    "DeadWorkerDetector",
    "JobStatus",
    "JobType",
    "JobMetadata",
    "JobResult",
    "JobProcessor",
]
