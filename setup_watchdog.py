#!/usr/bin/env python3
"""
Setup Watchdog Cron Job

This script sets up the watchdog to run every 5 minutes to monitor system health.
"""

import subprocess
import sys
from pathlib import Path


def setup_watchdog():
    """Install watchdog cron job."""
    watchdog_script = Path(__file__).parent / 'watchdog.py'
    python_executable = sys.executable
    
    # Get current crontab
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
    except:
        current_crontab = ""
    
    # Check if watchdog already installed
    if 'watchdog.py' in current_crontab:
        print("Watchdog already installed in crontab")
        return True
    
    # Add watchdog to crontab
    new_crontab = current_crontab.strip()
    if new_crontab:
        new_crontab += "\n\n"
    
    new_crontab += "# Wagehood Watchdog - monitors system health\n"
    new_crontab += f"*/5 * * * * {python_executable} {watchdog_script}\n"
    
    # Install new crontab
    try:
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(new_crontab)
        
        if process.returncode == 0:
            print("✅ Watchdog cron job installed successfully!")
            print("   It will run every 5 minutes to monitor system health")
            return True
        else:
            print("❌ Failed to install watchdog cron job")
            return False
            
    except Exception as e:
        print(f"❌ Error installing watchdog: {e}")
        return False


if __name__ == "__main__":
    setup_watchdog()