#!/usr/bin/env python3
"""
Cron Wrapper for 1-Day Analysis Trigger

This script manages the execution of the 1-day analysis trigger with:
- Process management to prevent overlapping runs
- Timeout detection with automatic process termination
- Discord alerts for infrastructure issues
- Robust error handling and logging
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.notifications.discord_client import DiscordNotificationSender
from src.notifications.routing import create_default_config_manager
from src.notifications.models import NotificationMessage
from src.monitoring.health_tracker import HealthTracker
import json

# Configuration
TIMEOUT_SECONDS = 60
SCRIPT_NAME = "trigger_1d_analysis.py"
LOCK_FILE = Path.home() / '.wagehood' / 'cron_1d.lock'
PID_FILE = Path.home() / '.wagehood' / 'cron_1d.pid'
LOG_FILE = Path.home() / '.wagehood' / 'cron_1d.log'
TIMEOUT_TRACKING_FILE = Path.home() / '.wagehood' / 'cron_1d_timeout_tracking.json'

# Ensure directories exist
LOCK_FILE.parent.mkdir(exist_ok=True)

# Set up logging with rotation
from src.utils.log_rotation import setup_rotating_logger, rotate_log_if_needed

# Rotate existing log if needed at startup
rotate_log_if_needed(LOG_FILE, max_size_mb=10, max_backups=5)

# Set up logger with rotation (10MB max, keep 5 backups)
logger = setup_rotating_logger(
    __name__,
    LOG_FILE,
    max_size_mb=10,
    max_backups=5
)


class CronJobManager:
    """Manages cron job execution with timeout detection and process management."""
    
    def __init__(self):
        self.discord_sender = None
        self.project_root = Path(__file__).parent
        self.script_path = self.project_root / SCRIPT_NAME
        self.timeout_tracking = self._load_timeout_tracking()
        
        # Initialize Discord for infrastructure alerts
        try:
            config_manager = create_default_config_manager()
            if config_manager.is_configured():
                self.discord_sender = DiscordNotificationSender(config_manager.get_all_configs())
        except Exception as e:
            logger.error(f"Failed to initialize Discord sender: {e}")
    
    def is_locked(self) -> bool:
        """Check if another instance is already running."""
        if not LOCK_FILE.exists():
            return False
        
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                
                # Check if it's still our script
                if SCRIPT_NAME in ' '.join(process.cmdline()):
                    return True
                else:
                    # PID exists but not our script - remove stale lock
                    logger.warning(f"Removing stale lock file - PID {pid} is not our script")
                    LOCK_FILE.unlink()
                    return False
            else:
                # Process doesn't exist - remove stale lock
                logger.warning(f"Removing stale lock file - PID {pid} no longer exists")
                LOCK_FILE.unlink()
                return False
                
        except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
            # Invalid lock file - remove it
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
            return False
    
    def create_lock(self, pid: int):
        """Create lock file with process PID."""
        try:
            with open(LOCK_FILE, 'w') as f:
                f.write(str(pid))
            logger.debug(f"Created lock file with PID {pid}")
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
    
    def remove_lock(self):
        """Remove lock file."""
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
                logger.debug("Removed lock file")
        except Exception as e:
            logger.error(f"Failed to remove lock file: {e}")
    
    def kill_stuck_process(self, pid: int) -> bool:
        """Kill a stuck process and its children."""
        try:
            if not psutil.pid_exists(pid):
                logger.warning(f"Process {pid} no longer exists")
                return True
            
            process = psutil.Process(pid)
            
            # Kill all children first
            children = process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                    logger.info(f"Terminated child process {child.pid}")
                except psutil.NoSuchProcess:
                    pass
            
            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                    logger.warning(f"Force killed child process {p.pid}")
                except psutil.NoSuchProcess:
                    pass
            
            # Kill the main process
            process.terminate()
            
            # Wait for graceful termination
            try:
                process.wait(timeout=5)
                logger.info(f"Process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if it doesn't terminate
                process.kill()
                logger.warning(f"Force killed process {pid}")
                return True
                
        except psutil.NoSuchProcess:
            logger.info(f"Process {pid} already terminated")
            return True
        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")
            return False
    
    async def send_infrastructure_alert(self, message: str):
        """Send infrastructure alert to Discord."""
        if not self.discord_sender:
            logger.warning("Discord sender not available for infrastructure alert")
            return
        
        try:
            notification = NotificationMessage.create_service_notification(
                content=f"🚨 **1-Day Analysis Cron Job Issue**\n\n{message}\n\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                priority=1  # High priority for infrastructure alerts
            )
            
            await self.discord_sender.send_notification(notification)
            logger.info("Infrastructure alert sent to Discord")
            
        except Exception as e:
            logger.error(f"Failed to send infrastructure alert: {e}")
    
    def run_with_timeout(self) -> bool:
        """Run the analysis script with timeout monitoring."""
        if not self.script_path.exists():
            logger.error(f"Script not found: {self.script_path}")
            return False
        
        start_time = time.time()
        logger.info(f"Starting 1-day analysis at {datetime.now()}")
        
        try:
            # Start the process
            # Don't capture stdout/stderr to avoid pipe buffer deadlock
            # The trigger script already logs to file
            process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                cwd=self.project_root
            )
            
            # Create lock with the subprocess PID
            self.create_lock(process.pid)
            
            # Monitor for timeout
            while process.poll() is None:
                elapsed = time.time() - start_time
                
                if elapsed > TIMEOUT_SECONDS:
                    logger.error(f"Process {process.pid} exceeded {TIMEOUT_SECONDS}s timeout")
                    
                    # Kill the stuck process
                    killed = self.kill_stuck_process(process.pid)
                    
                    # Record timeout
                    self._record_timeout()
                    
                    # Send infrastructure alert immediately
                    import asyncio
                    asyncio.run(self.send_infrastructure_alert(
                        f"⚠️ TIMEOUT: 1-day analysis exceeded {TIMEOUT_SECONDS}s timeout\n\n"
                        f"Timeout details:\n"
                        f"• Process exceeded {TIMEOUT_SECONDS}s timeout\n"
                        f"• PID: {process.pid} {'(terminated)' if killed else '(failed to terminate)'}\n"
                        f"• Elapsed time: {elapsed:.1f}s\n\n"
                        f"This indicates a performance issue that needs investigation."
                    ))
                    
                    self.remove_lock()
                    return False
                
                time.sleep(0.5)  # Check every 500ms
            
            # Process completed normally
            execution_time = time.time() - start_time
            return_code = process.returncode
            
            # Log results
            if return_code == 0:
                logger.info(f"1-day analysis completed successfully in {execution_time:.1f}s")
                # Reset timeout tracking on successful run
                self._reset_timeout_tracking()
            else:
                logger.error(f"1-day analysis failed with code {return_code} after {execution_time:.1f}s")
                
                # Send alert for persistent failures
                if execution_time > 5:  # Only alert for non-quick failures
                    import asyncio
                    asyncio.run(self.send_infrastructure_alert(
                        f"1-day analysis failed with exit code {return_code}.\n"
                        f"Execution time: {execution_time:.1f}s"
                    ))
            
            self.remove_lock()
            return return_code == 0
            
        except Exception as e:
            logger.error(f"Error running analysis script: {e}")
            self.remove_lock()
            
            # This is a different type of failure, always alert
            import asyncio
            asyncio.run(self.send_infrastructure_alert(
                f"Failed to execute 1-day analysis script.\n"
                f"Error: {str(e)}"
            ))
            return False
    
    def _load_timeout_tracking(self) -> dict:
        """Load timeout tracking data."""
        if TIMEOUT_TRACKING_FILE.exists():
            try:
                with open(TIMEOUT_TRACKING_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load timeout tracking: {e}")
        return {'consecutive_timeouts': 0, 'last_timeout': None}
    
    def _save_timeout_tracking(self):
        """Save timeout tracking data."""
        try:
            with open(TIMEOUT_TRACKING_FILE, 'w') as f:
                json.dump(self.timeout_tracking, f)
        except Exception as e:
            logger.error(f"Failed to save timeout tracking: {e}")
    
    def _record_timeout(self):
        """Record a timeout occurrence."""
        self.timeout_tracking['consecutive_timeouts'] += 1
        self.timeout_tracking['last_timeout'] = datetime.now().isoformat()
        self._save_timeout_tracking()
        
        logger.warning(f"Timeout recorded - consecutive count: {self.timeout_tracking['consecutive_timeouts']}")
        
        # No longer checking for multiple timeouts
        return True
    
    def _reset_timeout_tracking(self):
        """Reset timeout tracking after successful run."""
        if self.timeout_tracking['consecutive_timeouts'] > 0:
            logger.info(f"Resetting timeout counter (was {self.timeout_tracking['consecutive_timeouts']}")
            self.timeout_tracking['consecutive_timeouts'] = 0
            self.timeout_tracking['last_timeout'] = None
            self._save_timeout_tracking()
    
    async def cleanup_and_close(self):
        """Clean up resources."""
        if self.discord_sender:
            await self.discord_sender.close()


def main():
    """Main cron job entry point."""
    manager = CronJobManager()
    
    try:
        # Track the start of this minute's execution
        minute_start = time.time()
        iteration = 0
        
        # Run for up to 60 seconds (with some buffer)
        while time.time() - minute_start < 58:  # Stop 2 seconds before the minute ends
            iteration_start = time.time()
            
            # Calculate which 10-second slot we should be in
            elapsed_in_minute = iteration_start - minute_start
            expected_iteration = int(elapsed_in_minute / 10)
            
            # If we've done 6 iterations, we're done
            if expected_iteration >= 6:
                break
            
            # Log if we're behind schedule
            if expected_iteration > iteration:
                logger.warning(f"Behind schedule - should be on iteration {expected_iteration + 1}, but on {iteration + 1}")
                iteration = expected_iteration
            
            # Check if another instance is running
            if manager.is_locked():
                logger.info(f"Another instance is already running - skipping iteration {iteration + 1}")
            else:
                # Run the analysis with timeout monitoring
                logger.info(f"Starting iteration {iteration + 1} at {elapsed_in_minute:.1f}s into the minute")
                success = manager.run_with_timeout()
                if not success:
                    logger.error(f"Analysis failed on iteration {iteration + 1}")
            
            # Calculate next run time (next 10-second boundary)
            next_run_time = minute_start + ((iteration + 1) * 10)
            current_time = time.time()
            
            # If we're past the next boundary, move to the appropriate slot
            if current_time >= next_run_time:
                # We've overrun - figure out which slot we should be in now
                new_iteration = int((current_time - minute_start) / 10)
                if new_iteration > iteration:
                    logger.info(f"Execution overran - jumping from iteration {iteration + 1} to {new_iteration + 1}")
                    iteration = new_iteration
                else:
                    iteration += 1
            else:
                # Wait until the next 10-second boundary
                sleep_time = next_run_time - current_time
                if sleep_time > 0 and iteration < 5:  # Don't sleep after the last iteration
                    logger.info(f"Sleeping for {sleep_time:.1f}s until next run")
                    time.sleep(sleep_time)
                iteration += 1
            
            # Safety check - don't run more than 6 times
            if iteration >= 6:
                break
        
        logger.info(f"Completed {iteration} iterations in {time.time() - minute_start:.1f}s")
        return 0
        
    except Exception as e:
        logger.error(f"Cron job failed: {e}")
        return 1
        
    finally:
        import asyncio
        asyncio.run(manager.cleanup_and_close())


if __name__ == "__main__":
    sys.exit(main())