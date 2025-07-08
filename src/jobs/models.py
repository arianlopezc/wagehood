"""
Data models for distributed job processing system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import uuid


class JobStatus(Enum):
    """Job processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEDUPLICATED = "deduplicated"
    EXPIRED = "expired"


class JobType(Enum):
    """Types of jobs that can be processed."""
    HISTORICAL_ANALYSIS = "historical_analysis"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    BACKTEST = "backtest"
    INDICATOR_CALCULATION = "indicator_calculation"


@dataclass
class JobMetadata:
    """Metadata for a distributed job."""
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        # Convert enums to strings
        data['job_type'] = self.job_type.value
        data['status'] = self.status.value
        # Convert datetimes to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadata':
        """Create instance from dictionary."""
        # Convert strings back to enums
        data['job_type'] = JobType(data['job_type'])
        data['status'] = JobStatus(data['status'])
        # Convert ISO strings back to datetimes
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


@dataclass
class JobParameters:
    """Parameters for job execution."""
    symbol: str
    timeframe: str
    strategy: str
    start_date: datetime
    end_date: datetime
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        # Convert datetimes to ISO strings
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobParameters':
        """Create instance from dictionary."""
        # Convert ISO strings back to datetimes
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['end_date'] = datetime.fromisoformat(data['end_date'])
        return cls(**data)
    
    def normalize_for_hash(self) -> Dict[str, Any]:
        """Create normalized version for consistent hashing."""
        return {
            'symbol': self.symbol.upper(),
            'timeframe': self.timeframe,
            'strategy': self.strategy,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'parameters': self._normalize_parameters(self.parameters or {})
        }
    
    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for consistent hashing."""
        if not params:
            return {}
        
        # Sort keys and ensure consistent types
        normalized = {}
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, float):
                # Round floats to avoid precision issues
                normalized[key] = round(value, 8)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_parameters(value)
            else:
                normalized[key] = value
        return normalized


@dataclass
class JobResult:
    """Result data from job execution."""
    job_id: str
    status: JobStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobResult':
        """Create instance from dictionary."""
        data['status'] = JobStatus(data['status'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class WorkerInfo:
    """Information about a distributed worker."""
    worker_id: str
    started_at: datetime
    last_heartbeat: datetime
    current_job_id: Optional[str] = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerInfo':
        """Create instance from dictionary."""
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        return cls(**data)


# TTL settings for different data types
TTL_SETTINGS = {
    'job_meta': 86400,           # 24 hours
    'job_result': 3600,          # 1 hour
    'job_progress': 1800,        # 30 minutes
    'job_log': 3600,             # 1 hour
    'deduplication': 3600,       # 1 hour
    'worker_heartbeat': 300,     # 5 minutes
    'job_lock': 30,              # 30 seconds
    'dedup_lock': 60             # 1 minute
}