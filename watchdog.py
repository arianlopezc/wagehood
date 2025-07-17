#!/usr/bin/env python3
"""
Watchdog Script for Cron Jobs

This lightweight script runs every 5 minutes to ensure cron jobs are healthy.
If they're not running or haven't been active, it attempts to restart them.
Also performs log rotation to prevent disk space issues.
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path for log rotation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.log_rotation import rotate_log_if_needed

# Set up logging
log_file = Path.home() / '.wagehood' / 'watchdog.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_log_activity(job_type: str, max_age_minutes: int = 3) -> bool:
    """Check if a log file has been updated recently."""
    log_file = Path.home() / '.wagehood' / f'cron_{job_type}.log'
    
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return False
    
    # Check last modification time
    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
    age = datetime.now() - mtime
    
    if age > timedelta(minutes=max_age_minutes):
        logger.warning(f"{job_type} log inactive for {age.total_seconds():.0f} seconds")
        return False
    
    return True


def check_cron_installed() -> dict:
    """Check if cron jobs are installed."""
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            content = result.stdout
            return {
                'installed': True,
                '1h': 'cron_wrapper_1h.py' in content,
                '1d': 'cron_wrapper_1d.py' in content
            }
    except Exception as e:
        logger.error(f"Error checking crontab: {e}")
    
    return {'installed': False, '1h': False, '1d': False}


def reinstall_cron_jobs():
    """Reinstall cron jobs."""
    logger.info("Attempting to reinstall cron jobs...")
    
    script_path = Path(__file__).parent / 'setup_cron_jobs.py'
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), 'setup'],
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
        logger.error(f"Error reinstalling cron jobs: {e}")
        return False


def main():
    """Main watchdog logic."""
    logger.info("=== Watchdog check starting ===")
    
    # First, rotate logs if needed
    logs_to_rotate = [
        ('cron_1h.log', 10, 5),
        ('cron_1d.log', 10, 5),
        ('notification_workers.log', 20, 3),
        ('watchdog.log', 5, 3),
    ]
    
    for log_name, max_size_mb, max_backups in logs_to_rotate:
        log_path = Path.home() / '.wagehood' / log_name
        if log_path.exists():
            if rotate_log_if_needed(log_path, max_size_mb, max_backups):
                logger.info(f"Rotated {log_name} (exceeded {max_size_mb}MB)")
    
    issues_found = False
    
    # Check if logs are active
    log_status = {
        '1h': check_log_activity('1h'),
        '1d': check_log_activity('1d')
    }
    
    # Check if cron jobs are installed
    cron_status = check_cron_installed()
    
    # Determine if we need to take action
    if not cron_status['installed']:
        logger.error("Cron jobs not installed at all!")
        issues_found = True
    else:
        if not cron_status['1h']:
            logger.error("1-hour cron job missing from crontab!")
            issues_found = True
        if not cron_status['1d']:
            logger.error("1-day cron job missing from crontab!")
            issues_found = True
    
    # Check log activity
    if not log_status['1h']:
        logger.error("1-hour job appears to be dead (log inactive)")
        issues_found = True
    if not log_status['1d']:
        logger.error("1-day job appears to be dead (log inactive)")
        issues_found = True
    
    # Take action if needed
    if issues_found:
        logger.warning("Issues detected - attempting recovery")
        
        # Always try to reinstall if there are issues
        if reinstall_cron_jobs():
            logger.info("Recovery completed - cron jobs reinstalled")
            
            # Create a marker file to track recovery attempts
            recovery_file = Path.home() / '.wagehood' / 'last_recovery.txt'
            with open(recovery_file, 'w') as f:
                f.write(f"Recovery performed at {datetime.now().isoformat()}\n")
                f.write(f"Issues: {log_status}, {cron_status}\n")
        else:
            logger.error("Recovery failed!")
    else:
        logger.info("All systems healthy - no action needed")
    
    logger.info("=== Watchdog check complete ===\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Watchdog failed with error: {e}")