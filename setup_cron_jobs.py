#!/usr/bin/env python3
"""
Cron Job Setup and Management Script

This script sets up, manages, and monitors the cron jobs for the trigger analysis system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CronJobManager:
    """Manages cron jobs for the trigger analysis system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        
        # Cron job configurations
        self.cron_jobs = {
            "1h": {
                "script": self.project_root / "cron_wrapper_1h.py",
                "schedule": "*/10 * * * * *",  # Every 10 seconds (if supported by cron)
                "description": "1-hour analysis trigger (every 10 seconds)"
            },
            "1d": {
                "script": self.project_root / "cron_wrapper_1d.py", 
                "schedule": "*/10 * * * * *",  # Every 10 seconds (if supported by cron)
                "description": "1-day analysis trigger (every 10 seconds)"
            },
            "watchdog": {
                "script": self.project_root / "watchdog.py",
                "schedule": "*/5 * * * *",  # Every 5 minutes
                "description": "Watchdog health monitor (every 5 minutes)"
            }
        }
    
    def get_current_crontab(self) -> str:
        """Get current crontab content."""
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else ""
        except Exception as e:
            logger.error(f"Failed to get current crontab: {e}")
            return ""
    
    def remove_existing_jobs(self) -> bool:
        """Remove existing wagehood cron jobs."""
        try:
            current_crontab = self.get_current_crontab()
            
            # Filter out existing wagehood jobs
            lines = current_crontab.split('\n')
            new_lines = []
            
            for line in lines:
                # Skip wagehood-related lines
                if ('cron_wrapper_1h.py' in line or 
                    'cron_wrapper_1d.py' in line or
                    'trigger_1h_analysis.py' in line or
                    'trigger_1d_analysis.py' in line or
                    'watchdog.py' in line):
                    logger.info(f"Removing existing job: {line.strip()}")
                    continue
                
                if line.strip():  # Keep non-empty lines
                    new_lines.append(line)
            
            # Install updated crontab
            new_crontab = '\n'.join(new_lines)
            if new_crontab and not new_crontab.endswith('\n'):
                new_crontab += '\n'
            
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(new_crontab)
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to remove existing jobs: {e}")
            return False
    
    def install_jobs(self) -> bool:
        """Install new cron jobs."""
        try:
            current_crontab = self.get_current_crontab()
            
            # Add new jobs
            new_lines = []
            if current_crontab.strip():
                new_lines.extend(current_crontab.strip().split('\n'))
            
            # Add comment header
            new_lines.append("")
            new_lines.append("# Wagehood Trigger Analysis Cron Jobs")
            
            for job_name, config in self.cron_jobs.items():
                if not config["script"].exists():
                    logger.error(f"Script not found: {config['script']}")
                    return False
                
                # For systems that don't support seconds in cron, use every minute
                # Most systems don't support seconds, so we'll use every minute for now
                if "*/10 * * * * *" in config["schedule"]:
                    # Fallback to every minute if seconds not supported
                    schedule = "* * * * *"
                    logger.warning(f"Using fallback schedule (every minute) for {job_name} job")
                else:
                    schedule = config["schedule"]
                
                cron_line = (
                    f"{schedule} cd {self.project_root} && "
                    f"{self.python_executable} {config['script']}"
                )
                
                new_lines.append(f"# {config['description']}")
                new_lines.append(cron_line)
            
            # Install new crontab
            new_crontab = '\n'.join(new_lines) + '\n'
            
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(new_crontab)
            
            if process.returncode == 0:
                logger.info("Cron jobs installed successfully")
                return True
            else:
                logger.error("Failed to install cron jobs")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install cron jobs: {e}")
            return False
    
    def setup_cron_jobs(self) -> bool:
        """Set up cron jobs (remove existing and install new)."""
        logger.info("Setting up cron jobs for trigger analysis system...")
        
        # Check if scripts exist
        for job_name, config in self.cron_jobs.items():
            if not config["script"].exists():
                logger.error(f"Required script not found: {config['script']}")
                return False
        
        # Remove existing jobs
        if not self.remove_existing_jobs():
            logger.error("Failed to remove existing cron jobs")
            return False
        
        # Install new jobs
        if not self.install_jobs():
            logger.error("Failed to install new cron jobs")
            return False
        
        logger.info("Cron jobs setup completed successfully!")
        return True
    
    def remove_cron_jobs(self) -> bool:
        """Remove all wagehood cron jobs."""
        logger.info("Removing wagehood cron jobs...")
        
        if self.remove_existing_jobs():
            logger.info("Cron jobs removed successfully!")
            return True
        else:
            logger.error("Failed to remove cron jobs")
            return False
    
    def status_cron_jobs(self) -> None:
        """Show status of cron jobs."""
        logger.info("Checking cron job status...")
        
        current_crontab = self.get_current_crontab()
        
        # Check for wagehood jobs
        wagehood_jobs = []
        for line in current_crontab.split('\n'):
            if ('cron_wrapper_1h.py' in line or 
                'cron_wrapper_1d.py' in line or
                'watchdog.py' in line):
                wagehood_jobs.append(line.strip())
        
        if wagehood_jobs:
            print("\n📊 Active Wagehood Cron Jobs:")
            for i, job in enumerate(wagehood_jobs, 1):
                print(f"  {i}. {job}")
        else:
            print("\n❌ No Wagehood cron jobs found")
        
        # Check if scripts exist
        print("\n📁 Script Status:")
        for job_name, config in self.cron_jobs.items():
            exists = "✅" if config["script"].exists() else "❌"
            print(f"  {job_name}: {exists} {config['script']}")
        
        # Check log files
        print("\n📝 Log Files:")
        log_dir = Path.home() / '.wagehood'
        for job_name in self.cron_jobs.keys():
            log_file = log_dir / f"cron_{job_name}.log"
            exists = "✅" if log_file.exists() else "❌"
            size = f"({log_file.stat().st_size} bytes)" if log_file.exists() else ""
            print(f"  {job_name}: {exists} {log_file} {size}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Wagehood cron jobs")
    parser.add_argument('action', choices=['setup', 'remove', 'status'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    manager = CronJobManager()
    
    if args.action == 'setup':
        success = manager.setup_cron_jobs()
        sys.exit(0 if success else 1)
    elif args.action == 'remove':
        success = manager.remove_cron_jobs() 
        sys.exit(0 if success else 1)
    elif args.action == 'status':
        manager.status_cron_jobs()
        sys.exit(0)


if __name__ == "__main__":
    main()