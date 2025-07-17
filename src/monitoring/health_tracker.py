#!/usr/bin/env python3
"""
Health Tracking and Monitoring System

Tracks the health of the cron jobs and analysis system, providing:
- Heartbeat monitoring
- Success/failure metrics
- Data fetching verification
- Signal detection tracking
- Automatic recovery mechanisms
"""

import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil
import logging

logger = logging.getLogger(__name__)


class HealthTracker:
    """Tracks system health and provides monitoring capabilities."""
    
    def __init__(self):
        self.health_dir = Path.home() / '.wagehood' / 'health'
        self.health_dir.mkdir(parents=True, exist_ok=True)
        
        self.heartbeat_file = self.health_dir / 'heartbeat.json'
        self.metrics_file = self.health_dir / 'metrics.json'
        self.alerts_file = self.health_dir / 'alerts.json'
        
        # Initialize or load existing metrics
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or initialize new ones."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize metrics structure
        return {
            "1h": {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "timeouts": 0,
                "data_fetch_success": 0,
                "data_fetch_failures": 0,
                "signals_detected": 0,
                "last_success": None,
                "last_failure": None,
                "average_execution_time": 0,
                "iterations_per_minute": []
            },
            "1d": {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "timeouts": 0,
                "data_fetch_success": 0,
                "data_fetch_failures": 0,
                "signals_detected": 0,
                "last_success": None,
                "last_failure": None,
                "average_execution_time": 0,
                "iterations_per_minute": []
            }
        }
    
    def record_heartbeat(self, job_type: str, iteration: int, status: str = "running"):
        """Record a heartbeat for the job."""
        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "job_type": job_type,
            "iteration": iteration,
            "status": status,
            "pid": os.getpid()
        }
        
        # Read existing heartbeats
        heartbeats = {}
        if self.heartbeat_file.exists():
            try:
                with open(self.heartbeat_file, 'r') as f:
                    heartbeats = json.load(f)
            except:
                heartbeats = {}
        
        # Update heartbeat for this job type
        if job_type not in heartbeats:
            heartbeats[job_type] = []
        
        # Keep only last 60 heartbeats (10 minutes worth)
        heartbeats[job_type].append(heartbeat_data)
        heartbeats[job_type] = heartbeats[job_type][-60:]
        
        # Save heartbeats
        with open(self.heartbeat_file, 'w') as f:
            json.dump(heartbeats, f, indent=2)
    
    def record_execution(self, job_type: str, success: bool, execution_time: float,
                        data_fetched: bool = False, signals_found: int = 0,
                        error_message: Optional[str] = None):
        """Record execution metrics."""
        metrics = self.metrics[job_type]
        
        # Update counters
        metrics["total_runs"] += 1
        if success:
            metrics["successful_runs"] += 1
            metrics["last_success"] = datetime.now().isoformat()
        else:
            metrics["failed_runs"] += 1
            metrics["last_failure"] = datetime.now().isoformat()
            
            # Check if it was a timeout
            if error_message and "timeout" in error_message.lower():
                metrics["timeouts"] += 1
        
        # Update data fetch metrics
        if data_fetched:
            metrics["data_fetch_success"] += 1
        else:
            metrics["data_fetch_failures"] += 1
        
        # Update signal metrics
        metrics["signals_detected"] += signals_found
        
        # Update average execution time
        current_avg = metrics["average_execution_time"]
        total_runs = metrics["total_runs"]
        metrics["average_execution_time"] = (
            (current_avg * (total_runs - 1) + execution_time) / total_runs
        )
        
        # Save metrics
        self._save_metrics()
    
    def record_minute_iterations(self, job_type: str, iterations: int):
        """Record how many iterations completed in the last minute."""
        metrics = self.metrics[job_type]
        
        # Keep rolling window of last 60 minutes
        metrics["iterations_per_minute"].append({
            "timestamp": datetime.now().isoformat(),
            "iterations": iterations
        })
        metrics["iterations_per_minute"] = metrics["iterations_per_minute"][-60:]
        
        self._save_metrics()
    
    def check_health(self) -> Dict[str, Any]:
        """Check overall system health and return status."""
        health_status = {
            "healthy": True,
            "issues": [],
            "warnings": []
        }
        
        # Check heartbeats
        if self.heartbeat_file.exists():
            try:
                with open(self.heartbeat_file, 'r') as f:
                    heartbeats = json.load(f)
                
                for job_type in ["1h", "1d"]:
                    if job_type in heartbeats and heartbeats[job_type]:
                        last_heartbeat = heartbeats[job_type][-1]
                        last_time = datetime.fromisoformat(last_heartbeat["timestamp"])
                        
                        # Check if we've had a heartbeat in the last 2 minutes
                        if datetime.now() - last_time > timedelta(minutes=2):
                            health_status["healthy"] = False
                            health_status["issues"].append(
                                f"{job_type} job hasn't reported in {(datetime.now() - last_time).seconds}s"
                            )
                    else:
                        health_status["warnings"].append(f"No heartbeat data for {job_type} job")
            except Exception as e:
                health_status["warnings"].append(f"Error reading heartbeat file: {e}")
        
        # Check metrics
        for job_type in ["1h", "1d"]:
            metrics = self.metrics[job_type]
            
            # Check success rate
            if metrics["total_runs"] > 10:
                success_rate = metrics["successful_runs"] / metrics["total_runs"]
                if success_rate < 0.8:
                    health_status["warnings"].append(
                        f"{job_type} job success rate is low: {success_rate:.1%}"
                    )
                if success_rate < 0.5:
                    health_status["healthy"] = False
                    health_status["issues"].append(
                        f"{job_type} job failing frequently: {success_rate:.1%} success rate"
                    )
            
            # Check timeout rate
            if metrics["total_runs"] > 0:
                timeout_rate = metrics["timeouts"] / metrics["total_runs"]
                if timeout_rate > 0.1:
                    health_status["warnings"].append(
                        f"{job_type} job has high timeout rate: {timeout_rate:.1%}"
                    )
            
            # Check iterations per minute
            if metrics["iterations_per_minute"]:
                recent_iterations = [x["iterations"] for x in metrics["iterations_per_minute"][-10:]]
                avg_iterations = sum(recent_iterations) / len(recent_iterations)
                if avg_iterations < 5:
                    health_status["warnings"].append(
                        f"{job_type} job averaging only {avg_iterations:.1f} iterations per minute"
                    )
        
        return health_status
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of system health."""
        lines = ["=== System Health Summary ===\n"]
        
        health = self.check_health()
        lines.append(f"Overall Status: {'✅ HEALTHY' if health['healthy'] else '❌ UNHEALTHY'}")
        
        if health['issues']:
            lines.append("\n🚨 Critical Issues:")
            for issue in health['issues']:
                lines.append(f"  - {issue}")
        
        if health['warnings']:
            lines.append("\n⚠️  Warnings:")
            for warning in health['warnings']:
                lines.append(f"  - {warning}")
        
        lines.append("\n📊 Metrics Summary:")
        for job_type in ["1h", "1d"]:
            metrics = self.metrics[job_type]
            if metrics["total_runs"] > 0:
                success_rate = metrics["successful_runs"] / metrics["total_runs"]
                lines.append(f"\n{job_type} Analysis:")
                lines.append(f"  - Total runs: {metrics['total_runs']}")
                lines.append(f"  - Success rate: {success_rate:.1%}")
                lines.append(f"  - Timeouts: {metrics['timeouts']}")
                lines.append(f"  - Avg execution time: {metrics['average_execution_time']:.1f}s")
                lines.append(f"  - Signals detected: {metrics['signals_detected']}")
                
                if metrics["iterations_per_minute"]:
                    recent = [x["iterations"] for x in metrics["iterations_per_minute"][-10:]]
                    avg_iter = sum(recent) / len(recent)
                    lines.append(f"  - Avg iterations/minute: {avg_iter:.1f}")
        
        return "\n".join(lines)


class CronHealthMonitor:
    """Monitors cron job health and can trigger recovery actions."""
    
    def __init__(self):
        self.tracker = HealthTracker()
        self.recovery_script = Path(__file__).parent.parent.parent / "setup_cron_jobs.py"
        
    def check_and_recover(self) -> bool:
        """Check health and attempt recovery if needed."""
        health = self.tracker.check_health()
        
        if not health["healthy"]:
            logger.error(f"System unhealthy: {health['issues']}")
            
            # Check if cron jobs are even running
            cron_running = self._check_cron_processes()
            
            if not cron_running:
                logger.error("Cron jobs appear to be dead - attempting recovery")
                return self._attempt_recovery()
        
        return health["healthy"]
    
    def _check_cron_processes(self) -> bool:
        """Check if cron wrapper processes are running."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('cron_wrapper' in str(arg) for arg in cmdline):
                    return True
            except:
                continue
        return False
    
    def _attempt_recovery(self) -> bool:
        """Attempt to recover by reinstalling cron jobs."""
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(self.recovery_script), 'setup'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Successfully reinstalled cron jobs")
                return True
            else:
                logger.error(f"Failed to reinstall cron jobs: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            return False


# Helper function for cron wrappers to use
def report_health(job_type: str, iteration: int, success: bool, 
                 execution_time: float, error: Optional[str] = None):
    """Simple helper for cron wrappers to report health."""
    tracker = HealthTracker()
    tracker.record_heartbeat(job_type, iteration)
    tracker.record_execution(job_type, success, execution_time, 
                           data_fetched=success, error_message=error)