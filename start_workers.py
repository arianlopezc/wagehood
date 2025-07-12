#!/usr/bin/env python3
"""Worker launcher script that handles imports correctly."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file."""
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment at module level
load_env_file()

# Now import and run the worker manager
from src.workers.worker import WorkerManager
import asyncio

async def main():
    """Main entry point for starting workers."""
    manager = WorkerManager(num_workers=2)
    await manager.start()

if __name__ == "__main__":
    asyncio.run(main())