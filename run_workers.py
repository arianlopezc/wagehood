#!/usr/bin/env python3
"""
Worker runner script for the job processing system.

This script starts worker processes that will pick up and execute jobs from the queue.
"""

import asyncio
import sys
from src.workers.worker import WorkerManager


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


async def main():
    """Main entry point for running workers."""
    print_colored("🔧 Starting Wagehood Job Workers...", "cyan")
    print_colored("Press Ctrl+C to stop workers", "yellow")
    
    try:
        manager = WorkerManager(num_workers=2)
        await manager.start()
    except KeyboardInterrupt:
        print_colored("\\n🛑 Workers stopped by user", "yellow")
    except Exception as e:
        print_colored(f"\\n❌ Error running workers: {str(e)}", "red")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())