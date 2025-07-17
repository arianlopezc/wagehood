#!/usr/bin/env python3
"""
Start notification workers for Discord notification service.

This script starts the notification service worker that processes
the notification queue and sends messages to Discord channels.
"""

import asyncio
import os
import sys
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

from src.notifications.worker import NotificationService
from src.utils.logging_config import setup_rotating_logger

# Set up logging
logger = setup_rotating_logger(
    'notification_workers',
    log_file=os.path.expanduser('~/.wagehood/notification_workers.log'),
    console_output=True,
    level=logging.DEBUG  # Enable debug logging
)


async def run_notification_service():
    """Run the notification service with automatic recovery."""
    logger.info("Starting notification service...")
    
    # Track consecutive failures
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while consecutive_failures < max_consecutive_failures:
        service = None
        try:
            # Create notification service
            service = NotificationService()
            
            # Set up signal handlers for graceful shutdown
            shutdown_requested = False
            
            async def shutdown(signame):
                nonlocal shutdown_requested
                logger.info(f"Received {signame}, shutting down notification service...")
                shutdown_requested = True
                if service:
                    await service.stop()
            
            # Register signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda s, f: asyncio.create_task(shutdown(signal.Signals(s).name)))
            
            # Start the service
            logger.info("About to start notification service...")
            await service.start()
            logger.info("Notification service started successfully")
            
            # Reset failure counter on successful start
            consecutive_failures = 0
            
            # Keep the service running and monitor health
            last_healthy_time = asyncio.get_event_loop().time()
            unhealthy_count = 0
            
            while not shutdown_requested:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                try:
                    status = await service.get_status()
                    is_running = status.get('running', False)
                    is_healthy = status.get('healthy', False)
                    
                    logger.debug(f"Service status - Running: {is_running}, Healthy: {is_healthy}, "
                               f"Queue stats: {status.get('queue_health', {}).get('stats', {})}")
                    
                    if is_running and is_healthy:
                        last_healthy_time = asyncio.get_event_loop().time()
                        unhealthy_count = 0
                    else:
                        unhealthy_count += 1
                        logger.warning(f"Service unhealthy (count: {unhealthy_count})")
                        
                        # If unhealthy for too long, restart
                        if unhealthy_count > 6:  # 60 seconds of unhealthy state
                            logger.error("Service unhealthy for too long, restarting...")
                            break
                            
                except Exception as e:
                    logger.error(f"Error checking service status: {e}")
                    unhealthy_count += 1
                    
            if shutdown_requested:
                logger.info("Shutdown requested, exiting...")
                break
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Error in notification service (failure {consecutive_failures}/{max_consecutive_failures}): {e}", 
                        exc_info=True)
            
            if consecutive_failures < max_consecutive_failures:
                wait_time = min(60, 10 * consecutive_failures)  # Exponential backoff up to 60s
                logger.info(f"Restarting service in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.critical("Too many consecutive failures, giving up")
                sys.exit(1)
        finally:
            # Ensure cleanup
            if service:
                try:
                    await service.stop()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
                    
    logger.info("Notification service stopped")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("WAGEHOOD NOTIFICATION SERVICE")
    logger.info("=" * 60)
    
    # Check if Discord webhooks are configured
    webhook_vars = [
        'DISCORD_WEBHOOK_INFRA',
        'DISCORD_WEBHOOK_MACD_RSI',
        'DISCORD_WEBHOOK_SUPPORT_RESISTANCE',
        'DISCORD_WEBHOOK_RSI_TREND',
        'DISCORD_WEBHOOK_BOLLINGER'
    ]
    
    configured_webhooks = sum(1 for var in webhook_vars if os.getenv(var))
    
    if configured_webhooks == 0:
        logger.error("No Discord webhook URLs configured!")
        logger.error("Please configure webhook URLs in .env file or environment variables:")
        for var in webhook_vars:
            logger.error(f"  {var}")
        sys.exit(1)
    elif configured_webhooks < len(webhook_vars):
        logger.warning(f"Only {configured_webhooks}/{len(webhook_vars)} Discord webhooks configured")
        logger.warning("Some notification channels may not work")
    else:
        logger.info(f"All {configured_webhooks} Discord webhooks configured")
    
    # Save PID for service management
    pid_file = Path.home() / '.wagehood' / 'notification_workers.pid'
    pid_file.parent.mkdir(exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    
    try:
        # Run the async service
        asyncio.run(run_notification_service())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()


if __name__ == '__main__':
    main()