"""
Job queue management system using file-based storage.
"""

import json
import os
import time
import fcntl
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from threading import RLock
from contextlib import contextmanager

from .models import Job, JobStatus, JobRequest


class JobQueue:
    """Thread-safe job queue using file-based storage."""
    
    def __init__(self, queue_dir: str = ".jobs"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different job states
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        
        for dir_path in [self.pending_dir, self.processing_dir, self.completed_dir, self.failed_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self._lock = RLock()  # Use reentrant lock
    
    @contextmanager
    def _file_lock(self, file_path: Path, mode: str = 'r'):
        """Proper file locking context manager."""
        lock_path = file_path.with_suffix('.lock')
        lock_file = None
        
        try:
            # Create lock file
            lock_file = open(lock_path, 'w')
            
            # Acquire exclusive lock with timeout
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if time.time() - start_time > 5:  # 5 second timeout
                        raise TimeoutError(f"Failed to acquire lock for {file_path}")
                    time.sleep(0.01)
            
            # Open actual file
            if file_path.exists() or mode == 'w':
                with open(file_path, mode) as f:
                    yield f
            else:
                yield None
                
        finally:
            if lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    lock_path.unlink(missing_ok=True)
                except:
                    pass
    
    def _get_job_file_path(self, job_id: str, status: JobStatus) -> Path:
        """Get the file path for a job based on its status."""
        status_dirs = {
            JobStatus.PENDING: self.pending_dir,
            JobStatus.PROCESSING: self.processing_dir,
            JobStatus.COMPLETED: self.completed_dir,
            JobStatus.FAILED: self.failed_dir
        }
        return status_dirs[status] / f"{job_id}.json"
    
    def _write_job_file(self, job: Job) -> None:
        """Write job to file with atomic operation."""
        file_path = self._get_job_file_path(job.id, job.status)
        
        # Use system temp directory for atomic writes
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=file_path.parent,
            delete=False,
            suffix='.tmp',
            prefix=f'{job.id}_'
        ) as tmp_file:
            json.dump(job.to_dict(), tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Atomic rename
            tmp_path.replace(file_path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise e
    
    def _read_job_file(self, file_path: Path) -> Optional[Job]:
        """Read job from file with proper locking."""
        try:
            with self._file_lock(file_path, 'r') as f:
                if f:
                    data = json.load(f)
                    return Job.from_dict(data)
            return None
        except (json.JSONDecodeError, ValueError, IOError):
            # Log error if needed, but continue
            return None
    
    def _move_job_file(self, job_id: str, old_status: JobStatus, new_status: JobStatus) -> None:
        """Move job file from one status directory to another."""
        old_path = self._get_job_file_path(job_id, old_status)
        new_path = self._get_job_file_path(job_id, new_status)
        
        if old_path.exists():
            old_path.rename(new_path)
    
    def submit_job(self, request: JobRequest) -> Job:
        """Submit a new job to the queue."""
        with self._lock:
            job = Job.create(request)
            self._write_job_file(job)
            return job
    
    def get_next_job(self, worker_id: str) -> Optional[Job]:
        """Get the next pending job and mark it as processing."""
        with self._lock:
            # Use os.scandir for better performance
            pending_entries = []
            try:
                with os.scandir(self.pending_dir) as entries:
                    for entry in entries:
                        if entry.name.endswith('.json') and not entry.name.startswith('.'):
                            pending_entries.append((entry.name, entry.stat().st_mtime))
            except OSError:
                return None
            
            # Sort by modification time (oldest first)
            pending_entries.sort(key=lambda x: x[1])
            
            for filename, _ in pending_entries:
                file_path = self.pending_dir / filename
                
                try:
                    job = self._read_job_file(file_path)
                    if job and job.status == JobStatus.PENDING:
                        # Update job status
                        job.start_processing(worker_id)
                        
                        # Move file atomically
                        new_path = self._get_job_file_path(job.id, JobStatus.PROCESSING)
                        try:
                            file_path.rename(new_path)
                        except FileNotFoundError:
                            # Another worker got it first
                            continue
                        
                        # Write updated job
                        self._write_job_file(job)
                        return job
                except Exception:
                    # Skip problematic files
                    continue
            
            return None
    
    def complete_job(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark a job as completed with result."""
        with self._lock:
            processing_path = self._get_job_file_path(job_id, JobStatus.PROCESSING)
            job = self._read_job_file(processing_path)
            
            if job and job.status == JobStatus.PROCESSING:
                job.complete(result)
                self._move_job_file(job_id, JobStatus.PROCESSING, JobStatus.COMPLETED)
                self._write_job_file(job)
                return True
            
            return False
    
    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as failed with error."""
        with self._lock:
            processing_path = self._get_job_file_path(job_id, JobStatus.PROCESSING)
            job = self._read_job_file(processing_path)
            
            if job and job.status == JobStatus.PROCESSING:
                job.fail(error)
                self._move_job_file(job_id, JobStatus.PROCESSING, JobStatus.FAILED)
                self._write_job_file(job)
                return True
            
            return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID, searching all status directories."""
        for status in JobStatus:
            file_path = self._get_job_file_path(job_id, status)
            if file_path.exists():
                return self._read_job_file(file_path)
        return None
    
    def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """Get all jobs with a specific status."""
        jobs = []
        status_dir = self._get_job_file_path("dummy", status).parent
        
        for file_path in status_dir.glob("*.json"):
            job = self._read_job_file(file_path)
            if job and job.status == status:
                jobs.append(job)
        
        return sorted(jobs, key=lambda j: j.created_at)
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {}
        for status in JobStatus:
            status_dir = self._get_job_file_path("dummy", status).parent
            stats[status.value] = len(list(status_dir.glob("*.json")))
        return stats
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up completed and failed jobs older than max_age_hours."""
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        cleaned_count = 0
        
        with self._lock:
            for status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                status_dir = self._get_job_file_path("dummy", status).parent
                
                # Batch delete operations for better performance
                files_to_delete = []
                
                try:
                    with os.scandir(status_dir) as entries:
                        for entry in entries:
                            if entry.name.endswith('.json') and not entry.name.startswith('.'):
                                # Check age without reading file content
                                if current_time - entry.stat().st_mtime > max_age_seconds:
                                    files_to_delete.append(status_dir / entry.name)
                except OSError:
                    continue
                
                # Delete files
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        # Also clean up any stale lock files
                        lock_path = file_path.with_suffix('.lock')
                        lock_path.unlink(missing_ok=True)
                    except:
                        pass
        
        return cleaned_count