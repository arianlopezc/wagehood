#!/usr/bin/env python3
"""
List jobs script.

This script lists all jobs in the queue system.
"""

import argparse
from datetime import datetime
from src.jobs.queue import JobQueue
from src.jobs.models import JobStatus


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text to console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def list_jobs(limit: int = 20) -> None:
    """List recent jobs from all statuses."""
    queue = JobQueue()
    
    print_colored(f"\n📝 JOB LIST (last {limit}):", "cyan")
    print_colored("="*80, "cyan")
    
    all_jobs = []
    for status in JobStatus:
        jobs = queue.get_jobs_by_status(status)
        all_jobs.extend(jobs)
    
    # Sort by creation time, most recent first
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)
    
    if not all_jobs:
        print_colored("   No jobs found", "yellow")
        return
    
    # Header
    print_colored(f"{'#':>3} | {'Job ID':^36} | {'Symbol':^6} | {'Strategy':^15} | {'Status':^10} | {'Created':^12} | {'Duration':^8}", "white")
    print_colored("-"*80, "white")
    
    for i, job in enumerate(all_jobs[:limit], 1):
        status_color = {
            JobStatus.PENDING: "yellow",
            JobStatus.PROCESSING: "blue", 
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red"
        }.get(job.status, "white")
        
        created_str = ""
        if job.created_at:
            created_str = datetime.fromtimestamp(job.created_at).strftime('%m-%d %H:%M')
        
        duration_str = "-"
        if job.get_duration():
            duration_str = f"{job.get_duration():.1f}s"
        
        print_colored(
            f"{i:3d} | {job.id} | {job.request.symbol:^6} | {job.request.strategy:^15} | "
            f"{job.status.value.upper():^10} | {created_str:^12} | {duration_str:^8}", 
            status_color
        )
    
    # Summary stats
    print_colored("-"*80, "white")
    stats = queue.get_queue_stats()
    print_colored(
        f"Total: {sum(stats.values())} | "
        f"Pending: {stats.get('pending', 0)} | "
        f"Processing: {stats.get('processing', 0)} | "
        f"Completed: {stats.get('completed', 0)} | "
        f"Failed: {stats.get('failed', 0)}", 
        "cyan"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="List jobs in the queue system")
    parser.add_argument("--limit", type=int, default=20, help="Number of jobs to show (default: 20)")
    
    args = parser.parse_args()
    
    list_jobs(args.limit)


if __name__ == "__main__":
    main()