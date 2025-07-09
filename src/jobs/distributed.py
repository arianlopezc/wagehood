"""
Distributed job management system using Redis.

This provides the core distributed job architecture to replace in-memory job tracking
in AnalysisService with Redis-based coordination, deduplication, and worker management.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..storage.cache import cache_manager
from .models import (
    JobStatus,
    JobType,
    JobMetadata,
    JobParameters,
    JobResult,
    WorkerInfo,
    TTL_SETTINGS,
)

logger = logging.getLogger(__name__)


class JobDeduplicator:
    """Handles job parameter hashing and deduplication logic."""

    def __init__(self):
        self.cache = cache_manager

    def generate_job_hash(self, params: JobParameters) -> str:
        """Generate deterministic hash for job deduplication."""
        normalized = params.normalize_for_hash()
        param_string = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(param_string.encode()).hexdigest()[:16]

    async def check_existing_job(self, param_hash: str) -> Optional[str]:
        """Check if a job with these parameters already exists."""
        try:
            # Use Redis directly for dedup keys since they're stored as simple strings
            dedup_key = f"wagehood:jobs:dedup:{param_hash}"
            redis_client = self.cache._redis_client

            if not redis_client:
                return None

            existing_job_id = redis_client.get(dedup_key)

            if existing_job_id:
                # Decode the job ID
                job_id = (
                    existing_job_id.decode()
                    if isinstance(existing_job_id, bytes)
                    else existing_job_id
                )

                # Verify the job still exists and is valid
                meta_key = f"{job_id}:meta"
                job_meta = self.cache.get("jobs", meta_key)

                if job_meta and job_meta.get("status") in [
                    "pending",
                    "processing",
                    "completed",
                ]:
                    return job_id
                else:
                    # Clean up stale deduplication entry
                    redis_client.delete(dedup_key)

            return None
        except Exception:
            # Fallback to original method
            dedup_key = f"dedup:{param_hash}"
            existing_job_id = self.cache.get("jobs", dedup_key)
            return existing_job_id if existing_job_id else None

    def register_job_for_dedup(self, param_hash: str, job_id: str) -> bool:
        """Register job for deduplication."""
        try:
            # Use Redis directly for consistency with check_existing_job
            dedup_key = f"wagehood:jobs:dedup:{param_hash}"
            redis_client = self.cache._redis_client

            if not redis_client:
                # Fallback to cache manager
                dedup_key = f"dedup:{param_hash}"
                return self.cache.set(
                    "jobs", dedup_key, job_id, TTL_SETTINGS["deduplication"]
                )

            # Store as simple string with TTL
            result = redis_client.setex(
                dedup_key, TTL_SETTINGS["deduplication"], job_id
            )
            return bool(result)
        except Exception:
            # Fallback to cache manager
            dedup_key = f"dedup:{param_hash}"
            return self.cache.set(
                "jobs", dedup_key, job_id, TTL_SETTINGS["deduplication"]
            )


class DistributedJobManager:
    """Redis-based distributed job management."""

    def __init__(self):
        self.cache = cache_manager
        self.deduplicator = JobDeduplicator()
        self._redis_client = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis client for atomic operations."""
        if hasattr(self.cache, "_redis_client") and self.cache._redis_client:
            self._redis_client = self.cache._redis_client
        else:
            logger.warning("Redis client not available for atomic operations")

    async def create_job(
        self,
        job_type: JobType,
        symbol: str,
        timeframe: str,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Create a new distributed job with deduplication."""

        # Create job parameters for hashing
        job_params = JobParameters(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
        )

        # Check for existing job
        param_hash = self.deduplicator.generate_job_hash(job_params)
        existing_job_id = await self.deduplicator.check_existing_job(param_hash)

        if existing_job_id:
            logger.info(f"Found existing job {existing_job_id} for parameters")
            return existing_job_id

        # Create new job
        job_id = str(uuid.uuid4())
        current_time = datetime.utcnow()

        # Create job metadata
        metadata = JobMetadata(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=current_time,
            priority=priority,
        )

        # Store job data atomically
        success = await self._store_job_atomically(
            job_id, metadata, job_params, param_hash, priority
        )

        if success:
            logger.info(f"Created distributed job {job_id} for {symbol} {strategy}")
            return job_id
        else:
            raise Exception("Failed to create distributed job")

    async def _store_job_atomically(
        self,
        job_id: str,
        metadata: JobMetadata,
        params: JobParameters,
        param_hash: str,
        priority: int,
    ) -> bool:
        """Store job data atomically using Redis transaction."""
        if not self._redis_client:
            # Fallback to non-atomic storage
            return self._store_job_non_atomic(
                job_id, metadata, params, param_hash, priority
            )

        try:
            pipe = self._redis_client.pipeline()
            pipe.multi()

            # Store job metadata
            meta_key = self.cache._make_key("jobs", f"{job_id}:meta")
            pipe.setex(
                meta_key,
                TTL_SETTINGS["job_meta"],
                json.dumps(metadata.to_dict()).encode(),
            )

            # Store job parameters
            params_key = self.cache._make_key("jobs", f"{job_id}:params")
            pipe.setex(
                params_key,
                TTL_SETTINGS["job_meta"],
                json.dumps(params.to_dict()).encode(),
            )

            # Add to pending queue (sorted set with priority)
            pending_key = self.cache._make_key("jobs", "historical:pending")
            pipe.zadd(pending_key, {job_id: priority})

            # Register for deduplication
            dedup_key = self.cache._make_key("jobs", f"dedup:{param_hash}")
            pipe.setex(dedup_key, TTL_SETTINGS["deduplication"], job_id.encode())

            # Execute atomically
            pipe.execute()
            logger.debug(f"Stored job {job_id} atomically")
            return True

        except Exception as e:
            logger.error(f"Failed to store job {job_id} atomically: {e}")
            return False

    def _store_job_non_atomic(
        self,
        job_id: str,
        metadata: JobMetadata,
        params: JobParameters,
        param_hash: str,
        priority: int,
    ) -> bool:
        """Fallback non-atomic storage."""
        try:
            # Store metadata
            meta_key = f"{job_id}:meta"
            self.cache.set(
                "jobs", meta_key, metadata.to_dict(), TTL_SETTINGS["job_meta"]
            )

            # Store parameters
            params_key = f"{job_id}:params"
            self.cache.set(
                "jobs", params_key, params.to_dict(), TTL_SETTINGS["job_meta"]
            )

            # Add to pending (simulate with regular key)
            pending_key = f"pending:{job_id}"
            self.cache.set("jobs", pending_key, priority, TTL_SETTINGS["job_meta"])

            # Register deduplication
            self.deduplicator.register_job_for_dedup(param_hash, job_id)

            return True
        except Exception as e:
            logger.error(f"Failed to store job {job_id}: {e}")
            return False

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status and metadata."""
        try:
            # Get metadata
            meta_key = f"{job_id}:meta"
            meta_data = self.cache.get("jobs", meta_key)
            if not meta_data:
                return None

            # Get parameters
            params_key = f"{job_id}:params"
            params_data = self.cache.get("jobs", params_key)

            # Get result if completed
            result_key = f"{job_id}:result"
            result_data = self.cache.get("jobs", result_key)

            # Get progress
            progress_key = f"{job_id}:progress"
            progress_data = self.cache.get("jobs", progress_key)

            # Combine all data
            status = {
                "job_id": job_id,
                "metadata": meta_data,
                "parameters": params_data,
                "result": result_data,
                "progress": progress_data,
            }

            # Convert to legacy format for backward compatibility
            if meta_data:
                legacy_status = {
                    "status": meta_data.get("status", "unknown"),
                    "started_at": meta_data.get("created_at"),
                    "completed_at": meta_data.get("completed_at"),
                    "worker_id": meta_data.get("worker_id"),
                    "progress": (
                        progress_data.get("percentage", 0) if progress_data else 0
                    ),
                }

                if params_data:
                    legacy_status.update(
                        {
                            "symbol": params_data.get("symbol"),
                            "timeframe": params_data.get("timeframe"),
                            "strategy": params_data.get("strategy"),
                            "start_date": params_data.get("start_date"),
                            "end_date": params_data.get("end_date"),
                            "parameters": params_data.get("parameters", {}),
                        }
                    )

                if result_data:
                    legacy_status["result"] = result_data.get("data")
                    if result_data.get("error_message"):
                        legacy_status["error"] = result_data.get("error_message")

                # Also check for error in metadata
                if meta_data.get("error_message"):
                    legacy_status["error"] = meta_data.get("error_message")

                return legacy_status

            return status

        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        worker_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update job status."""
        try:
            meta_key = f"{job_id}:meta"
            meta_data = self.cache.get("jobs", meta_key)

            if not meta_data:
                logger.warning(f"Job {job_id} metadata not found for status update")
                return False

            # Update metadata
            meta_data["status"] = status.value
            if worker_id:
                meta_data["worker_id"] = worker_id
            if status == JobStatus.PROCESSING:
                meta_data["started_at"] = datetime.utcnow().isoformat()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                meta_data["completed_at"] = datetime.utcnow().isoformat()
            if error_message:
                meta_data["error_message"] = error_message

            # Store updated metadata
            success = self.cache.set(
                "jobs", meta_key, meta_data, TTL_SETTINGS["job_meta"]
            )

            if success:
                logger.debug(f"Updated job {job_id} status to {status.value}")

            return success

        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            return False

    async def store_job_result(self, job_id: str, result_data: Dict[str, Any]) -> bool:
        """Store job execution result."""
        try:
            result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                data=result_data,
                completed_at=datetime.utcnow(),
            )

            result_key = f"{job_id}:result"
            success = self.cache.set(
                "jobs", result_key, result.to_dict(), TTL_SETTINGS["job_result"]
            )

            if success:
                # Update job status to completed
                await self.update_job_status(job_id, JobStatus.COMPLETED)
                logger.debug(f"Stored result for job {job_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to store result for job {job_id}: {e}")
            return False


class WorkerManager:
    """Manages distributed workers and coordination."""

    def __init__(self):
        self.cache = cache_manager
        self._redis_client = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis client."""
        if hasattr(self.cache, "_redis_client") and self.cache._redis_client:
            self._redis_client = self.cache._redis_client

    async def register_worker(self, worker_id: str) -> bool:
        """Register a new worker using atomic operation to prevent conflicts."""
        try:
            current_time = datetime.utcnow()
            worker_info = WorkerInfo(
                worker_id=worker_id,
                started_at=current_time,
                last_heartbeat=current_time,
            )

            worker_key = f"worker:{worker_id}"

            # Use atomic SET NX operation to prevent race conditions
            # Only succeeds if worker doesn't already exist
            if self._redis_client:
                # Use direct Redis client for atomic SET NX with TTL
                success = self._redis_client.set(
                    self.cache._make_key("jobs", worker_key),
                    json.dumps(worker_info.to_dict()).encode(),
                    nx=True,  # Only set if not exists
                    ex=TTL_SETTINGS["worker_heartbeat"],  # Set TTL
                )

                if success:
                    logger.info(f"Registered worker {worker_id} (atomic)")
                else:
                    logger.warning(
                        f"Worker {worker_id} already registered by another instance"
                    )

                return bool(success)
            else:
                # Fallback to non-atomic for non-Redis setups
                success = self.cache.set(
                    "jobs",
                    worker_key,
                    worker_info.to_dict(),
                    TTL_SETTINGS["worker_heartbeat"],
                )

                if success:
                    logger.info(f"Registered worker {worker_id} (non-atomic fallback)")

                return success

        except Exception as e:
            logger.error(f"Failed to register worker {worker_id}: {e}")
            return False

    async def heartbeat(
        self, worker_id: str, current_job_id: Optional[str] = None
    ) -> bool:
        """Update worker heartbeat."""
        try:
            worker_key = f"worker:{worker_id}"
            worker_data = self.cache.get("jobs", worker_key)

            if worker_data:
                worker_data["last_heartbeat"] = datetime.utcnow().isoformat()
                if current_job_id:
                    worker_data["current_job_id"] = current_job_id

                success = self.cache.set(
                    "jobs", worker_key, worker_data, TTL_SETTINGS["worker_heartbeat"]
                )
                return success

            return False

        except Exception as e:
            logger.error(f"Failed to update heartbeat for worker {worker_id}: {e}")
            return False

    async def get_next_job(self, worker_id: str) -> Optional[str]:
        """Get next job for worker to process."""
        if not self._redis_client:
            # Fallback for non-Redis setup
            return await self._get_next_job_fallback()

        try:
            # Use Lua script for atomic job claiming
            lua_script = """
            local pending_key = KEYS[1]
            local processing_key = KEYS[2]
            local worker_id = ARGV[1]
            local current_time = ARGV[2]
            
            -- Get highest priority job
            local job = redis.call('ZPOPMAX', pending_key)
            if #job == 0 then
                return nil
            end
            
            local job_id = job[1]
            
            -- Move to processing
            redis.call('ZADD', processing_key, current_time, job_id)
            
            return job_id
            """

            pending_key = self.cache._make_key("jobs", "historical:pending")
            processing_key = self.cache._make_key("jobs", "historical:processing")
            current_time = str(int(time.time()))

            job_id = self._redis_client.eval(
                lua_script, 2, pending_key, processing_key, worker_id, current_time
            )

            if job_id:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                logger.debug(f"Worker {worker_id} claimed job {job_id}")

            return job_id

        except Exception as e:
            logger.error(f"Failed to get next job for worker {worker_id}: {e}")
            return None

    async def _get_next_job_fallback(self) -> Optional[str]:
        """Fallback job claiming for non-Redis setups."""
        # Simple implementation - just return None for now
        # In production, this should not be used
        return None


class DistributedWorker:
    """Individual worker that processes jobs."""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.job_manager = DistributedJobManager()
        self.worker_manager = WorkerManager()
        self._running = False
        self._heartbeat_task = None

    async def start(self):
        """Start the worker."""
        self._running = True
        await self.worker_manager.register_worker(self.worker_id)

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Worker {self.worker_id} started")

    async def stop(self):
        """Stop the worker."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Worker {self.worker_id} stopped")

    async def _heartbeat_loop(self):
        """Periodic heartbeat to maintain worker registration."""
        while self._running:
            try:
                await self.worker_manager.heartbeat(self.worker_id)
                await asyncio.sleep(60)  # Heartbeat every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error for worker {self.worker_id}: {e}")
                await asyncio.sleep(60)

    async def process_job(self, job_id: str) -> bool:
        """Process a specific job."""
        try:
            # Update job status to processing
            await self.job_manager.update_job_status(
                job_id, JobStatus.PROCESSING, self.worker_id
            )

            # Get job parameters
            status = await self.job_manager.get_job_status(job_id)
            if not status:
                logger.error(f"Job {job_id} not found")
                return False

            # Process the job (placeholder - will be implemented by specific worker)
            result = await self._execute_job(status)

            if result:
                # Store result
                await self.job_manager.store_job_result(job_id, result)
                logger.info(f"Worker {self.worker_id} completed job {job_id}")
                return True
            else:
                # Mark as failed
                await self.job_manager.update_job_status(
                    job_id, JobStatus.FAILED, self.worker_id, "Job execution failed"
                )
                return False

        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to process job {job_id}: {e}")
            await self.job_manager.update_job_status(
                job_id, JobStatus.FAILED, self.worker_id, str(e)
            )
            return False

    async def _execute_job(
        self, job_status: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute the actual job logic (to be overridden)."""
        # This is a placeholder - actual implementation will depend on job type
        logger.warning(f"Job execution not implemented for worker {self.worker_id}")
        return None


class JobProgressTracker:
    """Tracks job progress and provides updates."""

    def __init__(self):
        self.cache = cache_manager

    async def update_progress(
        self, job_id: str, percentage: float, message: Optional[str] = None
    ) -> bool:
        """Update job progress."""
        try:
            progress_data = {
                "job_id": job_id,
                "percentage": max(0, min(100, percentage)),
                "message": message,
                "updated_at": datetime.utcnow().isoformat(),
            }

            progress_key = f"{job_id}:progress"
            success = self.cache.set(
                "jobs", progress_key, progress_data, TTL_SETTINGS["job_progress"]
            )

            if success:
                logger.debug(f"Updated progress for job {job_id}: {percentage}%")

            return success

        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id}: {e}")
            return False


class DeadWorkerDetector:
    """Detects and cleans up dead workers."""

    def __init__(self):
        self.cache = cache_manager
        self._redis_client = None
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis client."""
        if hasattr(self.cache, "_redis_client") and self.cache._redis_client:
            self._redis_client = self.cache._redis_client

    async def cleanup_dead_workers(self, timeout_seconds: int = 300) -> int:
        """Clean up dead workers and requeue their jobs using atomic operations."""
        if not self._redis_client:
            logger.warning("Redis not available for dead worker cleanup")
            return 0

        try:
            current_time = int(time.time())
            timeout_threshold = current_time - timeout_seconds

            # Lua script for atomic dead worker cleanup and job requeuing
            lua_script = """
            local worker_pattern = KEYS[1]
            local processing_queue = KEYS[2]
            local pending_queue = KEYS[3]
            local timeout_threshold = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            
            local dead_workers = {}
            local requeued_jobs = 0
            
            -- Find all worker keys
            local worker_keys = redis.call('KEYS', worker_pattern)
            
            for _, worker_key in ipairs(worker_keys) do
                local worker_data = redis.call('GET', worker_key)
                if worker_data then
                    local success, worker_info = pcall(cjson.decode, worker_data)
                    if success and worker_info.last_heartbeat then
                        local last_heartbeat = worker_info.last_heartbeat
                        -- Parse ISO timestamp to unix timestamp (simplified)
                        local heartbeat_time = 0
                        if type(last_heartbeat) == "string" then
                            -- For safety, assume old heartbeat if can't parse
                            heartbeat_time = timeout_threshold - 1
                        else
                            heartbeat_time = tonumber(last_heartbeat) or (timeout_threshold - 1)
                        end
                        
                        if heartbeat_time < timeout_threshold then
                            -- Worker is dead
                            local worker_id = worker_info.worker_id
                            table.insert(dead_workers, worker_id)
                            
                            -- Find and requeue jobs from this dead worker
                            local processing_jobs = redis.call('ZRANGE', processing_queue, 0, -1, 'WITHSCORES')
                            for i = 1, #processing_jobs, 2 do
                                local job_id = processing_jobs[i]
                                local job_time = tonumber(processing_jobs[i + 1])
                                
                                -- Check if job belongs to this dead worker
                                local job_meta_key = 'wagehood:jobs:' .. job_id .. ':meta'
                                local job_meta = redis.call('GET', job_meta_key)
                                if job_meta then
                                    local job_success, job_info = pcall(cjson.decode, job_meta)
                                    if job_success and job_info.worker_id == worker_id then
                                        -- Requeue job to pending
                                        redis.call('ZREM', processing_queue, job_id)
                                        redis.call('ZADD', pending_queue, current_time, job_id)
                                        
                                        -- Update job status back to pending
                                        job_info.status = 'pending'
                                        job_info.worker_id = nil
                                        job_info.started_at = nil
                                        redis.call('SET', job_meta_key, cjson.encode(job_info))
                                        
                                        requeued_jobs = requeued_jobs + 1
                                    end
                                end
                            end
                            
                            -- Remove dead worker registration
                            redis.call('DEL', worker_key)
                        end
                    end
                end
            end
            
            return {#dead_workers, requeued_jobs}
            """

            # Execute atomic cleanup
            worker_pattern = self.cache._make_key("jobs", "worker:*")
            processing_key = self.cache._make_key("jobs", "historical:processing")
            pending_key = self.cache._make_key("jobs", "historical:pending")

            result = self._redis_client.eval(
                lua_script,
                3,
                worker_pattern,
                processing_key,
                pending_key,
                str(timeout_threshold),
                str(current_time),
            )

            if result and len(result) >= 2:
                dead_workers_count = int(result[0])
                requeued_jobs_count = int(result[1])

                if dead_workers_count > 0 or requeued_jobs_count > 0:
                    logger.info(
                        f"Cleaned up {dead_workers_count} dead workers, requeued {requeued_jobs_count} jobs"
                    )

                return requeued_jobs_count
            else:
                logger.debug("Dead worker cleanup completed - no dead workers found")
                return 0

        except Exception as e:
            logger.error(f"Failed to cleanup dead workers: {e}")
            return 0
