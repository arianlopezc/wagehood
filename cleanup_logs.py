#!/usr/bin/env python3
"""
Log Cleanup Script

Cleans up old and unused log files to save disk space.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.log_rotation import LogRotator, rotate_log_if_needed


def format_size(bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"


def cleanup_old_logs(dry_run=False):
    """Clean up old and unused log files."""
    log_dir = Path.home() / '.wagehood'
    
    # Define log files that should be rotated
    active_logs = {
        'cron_1h.log': {'max_size_mb': 10, 'max_backups': 5},
        'cron_1d.log': {'max_size_mb': 10, 'max_backups': 5},
        'notification_workers.log': {'max_size_mb': 20, 'max_backups': 3},
        'notifications_error.log': {'max_size_mb': 10, 'max_backups': 3},
        'watchdog.log': {'max_size_mb': 5, 'max_backups': 3},
    }
    
    # Define old/unused logs that can be deleted or archived
    old_logs_to_remove = [
        'streaming_fixed.log',  # 40MB - old test log
        'streaming_new.log',    # 32MB - old test log
        'streaming_output.log', # 5.8MB - old test log
        'streaming_output_new.log', # 1.5MB - old test log
        'streaming.log.2',      # 10MB - old rotation
        'streaming.log.3',      # 10MB - old rotation
        'streaming.log.4',      # 10MB - old rotation
        'streaming.log.5',      # 10MB - old rotation
        'streaming_error.log.2', # Old rotation
        'streaming_error.log.3', # Old rotation
        'streaming_error.log.4', # Old rotation
        'streaming_error.log.5', # Old rotation
    ]
    
    # Zero-byte logs that can be removed
    zero_byte_logs = ['workers.log', 'notifications.log', 'eod_scheduler.log']
    
    total_saved = 0
    
    print("=== Log Cleanup Report ===\n")
    
    # 1. Rotate active logs if needed
    print("📊 Checking active logs for rotation:")
    for log_name, config in active_logs.items():
        log_path = log_dir / log_name
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"  • {log_name}: {format_size(size)}")
            
            if not dry_run:
                rotated = rotate_log_if_needed(
                    log_path, 
                    max_size_mb=config['max_size_mb'],
                    max_backups=config['max_backups']
                )
                if rotated:
                    print(f"    ✅ Rotated!")
    
    # 2. Remove old/unused logs
    print("\n🗑️  Old/unused logs to remove:")
    for log_name in old_logs_to_remove:
        log_path = log_dir / log_name
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"  • {log_name}: {format_size(size)}")
            total_saved += size
            
            if not dry_run:
                log_path.unlink()
                print(f"    ✅ Deleted!")
    
    # 3. Remove zero-byte logs
    print("\n🗑️  Zero-byte logs to remove:")
    for log_name in zero_byte_logs:
        log_path = log_dir / log_name
        if log_path.exists() and log_path.stat().st_size == 0:
            print(f"  • {log_name}: 0 bytes")
            
            if not dry_run:
                log_path.unlink()
                print(f"    ✅ Deleted!")
    
    # 4. Check for other large logs
    print("\n⚠️  Other large log files:")
    for log_path in log_dir.glob('*.log*'):
        if log_path.name not in active_logs and log_path.name not in old_logs_to_remove:
            size = log_path.stat().st_size
            if size > 1024 * 1024:  # > 1MB
                age = datetime.now() - datetime.fromtimestamp(log_path.stat().st_mtime)
                print(f"  • {log_path.name}: {format_size(size)} (modified {age.days} days ago)")
    
    print(f"\n💾 Total space to be freed: {format_size(total_saved)}")
    
    if dry_run:
        print("\n⚠️  This was a dry run. Use --execute to actually clean up.")
    else:
        print("\n✅ Cleanup completed!")


def setup_log_rotation_configs():
    """Set up log rotation for all active services."""
    configs = {
        'notification_workers': {
            'config_file': Path.home() / '.wagehood' / 'log_rotation.conf',
            'content': '''# Log rotation configuration for wagehood services

# Notification workers
/home/*/.wagehood/notification_workers.log {
    size 10M
    rotate 5
    compress
    delaycompress
    missingok
    notifempty
}

# Cron job logs
/home/*/.wagehood/cron_*.log {
    size 10M
    rotate 5
    compress
    delaycompress
    missingok
    notifempty
}

# Error logs
/home/*/.wagehood/*_error.log {
    size 5M
    rotate 3
    compress
    delaycompress
    missingok
    notifempty
}
'''
        }
    }
    
    print("Log rotation configuration saved to ~/.wagehood/log_rotation.conf")
    print("To use with system logrotate, add to /etc/logrotate.d/")


def main():
    parser = argparse.ArgumentParser(description="Clean up wagehood log files")
    parser.add_argument('--execute', action='store_true',
                       help='Actually perform cleanup (default is dry run)')
    parser.add_argument('--setup-rotation', action='store_true',
                       help='Set up log rotation configuration')
    
    args = parser.parse_args()
    
    if args.setup_rotation:
        setup_log_rotation_configs()
    else:
        cleanup_old_logs(dry_run=not args.execute)


if __name__ == "__main__":
    main()