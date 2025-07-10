"""
Job models and data structures for the job processing system.
"""

import json
import time
import uuid
import re
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRequest:
    """Job request model."""
    symbol: str
    strategy: str
    timeframe: str
    start_date: str
    end_date: str
    
    def __post_init__(self):
        # Validate symbol
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if not re.match(r'^[A-Za-z]{1,10}$', self.symbol):
            raise ValueError(f"Invalid symbol: {self.symbol}. Must be 1-10 letters only")
        self.symbol = self.symbol.upper()
        
        # Validate dates
        try:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            if start >= end:
                raise ValueError("start_date must be before end_date")
            # Validate reasonable date range (not more than 5 years)
            if (end - start).days > 1825:
                raise ValueError("Date range cannot exceed 5 years")
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
            raise
        
        # Validate timeframe
        if self.timeframe not in ['1h', '1d']:
            raise ValueError(f"Invalid timeframe: {self.timeframe}. Must be '1h' or '1d'")
        
        # Validate strategy
        valid_strategies = [
            'rsi_trend', 'bollinger_breakout', 'macd_rsi', 
            'sr_breakout'
        ]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")


@dataclass
class Job:
    """Job model with full lifecycle information."""
    id: str
    request: JobRequest
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @classmethod
    def create(cls, request: JobRequest) -> 'Job':
        """Create a new job with pending status."""
        return cls(
            id=str(uuid.uuid4()),
            request=request,
            status=JobStatus.PENDING,
            created_at=time.time()
        )
    
    def start_processing(self, worker_id: str) -> None:
        """Mark job as processing."""
        self.status = JobStatus.PROCESSING
        self.started_at = time.time()
        self.worker_id = worker_id
    
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark job as completed with result."""
        self.status = JobStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark job as failed with error."""
        self.status = JobStatus.FAILED
        self.completed_at = time.time()
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['request'] = asdict(self.request)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary with validation."""
        try:
            # Make a copy to avoid modifying original
            data = data.copy()
            
            # Validate and extract request
            request_data = data.get('request')
            if not request_data:
                raise ValueError("Missing 'request' field in job data")
            request = JobRequest(**request_data)
            
            # Validate required fields
            required_fields = ['id', 'status', 'created_at']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate job ID format to prevent path traversal
            job_id = data.get('id', '')
            if not re.match(r'^[a-f0-9\-]{36}$', job_id):
                raise ValueError(f"Invalid job ID format: {job_id}")
            
            # Convert status safely
            status_value = data.get('status')
            try:
                data['status'] = JobStatus(status_value)
            except ValueError:
                raise ValueError(f"Invalid job status: {status_value}")
            
            data['request'] = request
            return cls(**data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize job: {e}")
    
    def get_duration(self) -> Optional[float]:
        """Get job processing duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def get_age(self) -> float:
        """Get job age since creation in seconds."""
        return time.time() - self.created_at


@dataclass
class JobResult:
    """Job result model for displaying results."""
    job_id: str
    symbol: str
    strategy: str
    timeframe: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    signals: List[Dict[str, Any]]
    error: Optional[str]
    
    @classmethod
    def from_job(cls, job: Job) -> 'JobResult':
        """Create result from job."""
        return cls(
            job_id=job.id,
            symbol=job.request.symbol,
            strategy=job.request.strategy,
            timeframe=job.request.timeframe,
            status=job.status,
            created_at=datetime.fromtimestamp(job.created_at),
            completed_at=datetime.fromtimestamp(job.completed_at) if job.completed_at else None,
            duration=job.get_duration(),
            signals=job.result.get('signals', []) if job.result else [],
            error=job.error
        )