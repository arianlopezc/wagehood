#!/usr/bin/env python3
"""
Log Management Utility

Provides commands to manage, monitor, and clean up log files.
"""

import sys
import argparse
from pathlib import Path
from src.utils.logging_config import get_log_file_stats, cleanup_old_logs

def show_log_stats():
    """Show current log file statistics."""
    stats = get_log_file_stats()
    
    if not stats:
        print("No log files found.")
        return
    
    print("Log File Statistics:")
    print("=" * 50)
    
    total_size = stats.pop('total_size_mb', 0)
    total_files = stats.pop('total_files', 0)
    
    for filename, file_stats in stats.items():
        if 'error' in file_stats:
            print(f"❌ {filename}: Error - {file_stats['error']}")
        else:
            size_mb = file_stats['size_mb']
            status = "🔴 LARGE" if size_mb > 50 else "🟡 MEDIUM" if size_mb > 10 else "🟢 OK"
            print(f"{status} {filename}: {size_mb} MB")
    
    print("-" * 50)
    print(f"📊 Total: {total_files} files, {total_size} MB")
    
    if total_size > 100:
        print("⚠️  WARNING: Log files using significant disk space!")
        print("💡 Consider running: python manage_logs.py --cleanup")

def cleanup_logs(days: int = 7):
    """Clean up old log files."""
    print(f"Cleaning up log files older than {days} days...")
    cleanup_old_logs(days_to_keep=days)
    print("✅ Cleanup completed!")
    show_log_stats()

def monitor_logs():
    """Monitor log file growth in real-time."""
    import time
    
    print("🔍 Monitoring log file growth (Ctrl+C to stop)...")
    print("Checking every 30 seconds...")
    
    try:
        while True:
            stats = get_log_file_stats()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n[{current_time}] Current sizes:")
            for filename, file_stats in stats.items():
                if filename not in ['total_size_mb', 'total_files'] and 'size_mb' in file_stats:
                    size_mb = file_stats['size_mb']
                    print(f"  {filename}: {size_mb} MB")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped.")

def tail_logs(lines: int = 50):
    """Show the last N lines from active log files."""
    log_dir = Path.home() / '.wagehood'
    
    if not log_dir.exists():
        print("No log directory found.")
        return
    
    active_logs = ['streaming.log', 'streaming_error.log']
    
    for log_name in active_logs:
        log_file = log_dir / log_name
        if log_file.exists():
            print(f"\n📄 Last {lines} lines from {log_name}:")
            print("=" * 60)
            
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    
                    for line in recent_lines:
                        print(line.rstrip())
                        
            except Exception as e:
                print(f"❌ Error reading {log_name}: {e}")
        else:
            print(f"📄 {log_name}: Not found")

def rotate_now():
    """Force log rotation for current log files."""
    print("🔄 Forcing log rotation...")
    
    # This will be handled by the RotatingFileHandler when it detects size limits
    # For immediate rotation, we can rename current files
    log_dir = Path.home() / '.wagehood'
    
    if not log_dir.exists():
        print("No log directory found.")
        return
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for log_file in log_dir.glob('*.log'):
        if log_file.is_file():
            backup_name = f"{log_file.stem}_{timestamp}.log"
            backup_path = log_dir / backup_name
            
            try:
                log_file.rename(backup_path)
                print(f"✅ Rotated {log_file.name} → {backup_name}")
            except Exception as e:
                print(f"❌ Error rotating {log_file.name}: {e}")
    
    print("🔄 Log rotation completed. New logs will be created on next write.")

def main():
    parser = argparse.ArgumentParser(description="Manage Wagehood log files")
    parser.add_argument('--stats', action='store_true', help='Show log file statistics')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', nargs='?', const=7, 
                        help='Clean up log files older than DAYS (default: 7)')
    parser.add_argument('--monitor', action='store_true', help='Monitor log file growth')
    parser.add_argument('--tail', type=int, metavar='LINES', nargs='?', const=50,
                        help='Show last LINES from log files (default: 50)')
    parser.add_argument('--rotate', action='store_true', help='Force log rotation now')
    
    args = parser.parse_args()
    
    if args.stats:
        show_log_stats()
    elif args.cleanup is not None:
        cleanup_logs(args.cleanup)
    elif args.monitor:
        monitor_logs()
    elif args.tail is not None:
        tail_logs(args.tail)
    elif args.rotate:
        rotate_now()
    else:
        # Default: show stats
        show_log_stats()

if __name__ == "__main__":
    main()