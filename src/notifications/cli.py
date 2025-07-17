"""
CLI management for notification service.
"""

import asyncio
import click
import sys
from typing import Dict, Any
import json

from .worker import NotificationService
from .routing import create_default_config_manager
from .models import NotificationMessage


@click.group()
def notifications():
    """Notification service management commands."""
    pass


@notifications.command()
@click.option('--duration', '-d', default=30, help='Test duration in seconds')
@click.option('--send-real', is_flag=True, help='Actually send test notifications to Discord (default: dry run)')
def test(duration: int, send_real: bool):
    """Test notification service connectivity (dry run by default)."""
    
    async def run_test():
        if not send_real:
            click.echo("🧪 Testing notification service (DRY RUN - no real notifications)")
            click.echo("💡 Use --send-real flag to actually send test notifications")
        else:
            click.echo("🧪 Testing notification service (SENDING REAL NOTIFICATIONS)")
        
        try:
            # Create service
            service = NotificationService()
            await service.start()
            
            # Test channel connectivity
            click.echo("\n📡 Testing Discord connectivity...")
            results = await service.test_channels()
            
            if results:
                click.echo("✅ Channel test results:")
                for channel, success in results.items():
                    status = "✅ OK" if success else "❌ FAILED"
                    click.echo(f"  {channel}: {status}")
            else:
                click.echo("⚠️  No channels configured for testing")
            
            # Send test notifications if channels are working
            working_channels = sum(results.values()) if results else 0
            if working_channels > 0 and send_real:
                click.echo(f"\n📨 Sending test notifications...")
                
                # Test service notification
                success = await service.send_service_notification(
                    "🧪 Test service notification from CLI",
                    priority=2
                )
                
                if success:
                    click.echo("✅ Service notification sent")
                else:
                    click.echo("❌ Service notification failed")
                
                # Test signal notification (if strategy channels configured)
                signal_sent = await service.send_signal(
                    "AAPL", "macd_rsi", "1d", 
                    "🧪 Test signal notification from CLI"
                )
                
                if signal_sent:
                    click.echo("✅ Signal notification sent")
                else:
                    click.echo("❌ Signal notification failed")
                
                # Wait for processing
                click.echo(f"⏳ Waiting {duration} seconds for processing...")
                await asyncio.sleep(duration)
            elif working_channels > 0 and not send_real:
                click.echo(f"\n📨 Would send test notifications to {working_channels} channels (dry run)")
                click.echo("  - Would send service notification to 'infra' channel")
                click.echo("  - Would send test signal to 'macd-rsi' channel")
            
            # Show final status
            status = await service.get_status()
            if status.get('healthy'):
                click.echo("\n✅ Notification service is healthy")
            else:
                click.echo("\n❌ Notification service has issues")
                if 'error' in status:
                    click.echo(f"Error: {status['error']}")
            
            await service.stop()
            
        except Exception as e:
            click.echo(f"❌ Test failed: {e}")
            sys.exit(1)
    
    asyncio.run(run_test())


@notifications.command()
def status():
    """Check notification service status (read-only, no notifications sent)."""
    
    async def check_status():
        try:
            # Import necessary components for status checking
            from .models import NotificationDatabase
            from .queue import NotificationQueueManager
            
            # Check configuration without starting service
            config_manager = create_default_config_manager()
            
            click.echo("📊 Notification Service Status")
            click.echo("=" * 40)
            
            # Configuration status
            click.echo(f"Channels configured: {len(config_manager.configs)}")
            missing = config_manager.get_missing_channels()
            if missing:
                click.echo(f"Missing channels: {', '.join(missing)}")
            
            validation = config_manager.validate_configuration()
            if validation['errors']:
                click.echo("❌ Configuration errors:")
                for error in validation['errors']:
                    click.echo(f"  - {error}")
            
            if validation['warnings']:
                click.echo("⚠️  Configuration warnings:")
                for warning in validation['warnings']:
                    click.echo(f"  - {warning}")
            
            # Get queue status without starting worker
            try:
                db = NotificationDatabase()
                queue_manager = NotificationQueueManager()
                
                # Get queue statistics directly
                queue_stats = await queue_manager.get_queue_stats()
                
                click.echo(f"\nRuntime Status:")
                click.echo(f"  Healthy: {'✅' if config_manager.is_configured() else '❌'}")
                click.echo(f"  Running: {'✅' if queue_stats else '❌'}")
                
                # Get processing statistics from database
                stats = db.get_processing_stats()
                if stats:
                    click.echo(f"  Messages processed: {stats.get('total_processed', 0)}")
                    click.echo(f"  Messages sent: {stats.get('total_sent', 0)}")
                    click.echo(f"  Messages failed: {stats.get('total_failed', 0)}")
                
                # Show queue status
                click.echo(f"  Queue pending: {queue_stats.get('pending', 0)}")
                click.echo(f"  Queue processing: {queue_stats.get('processing', 0)}")
                
            except Exception as e:
                click.echo(f"⚠️  Could not get runtime status: {e}")
                click.echo("     (This is normal if the notification service has never been started)")
            
        except Exception as e:
            click.echo(f"❌ Error checking status: {e}")
            sys.exit(1)
    
    asyncio.run(check_status())


@notifications.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol')
@click.option('--strategy', '-st', required=True, 
              type=click.Choice(['macd_rsi', 'sr_breakout', 'rsi_trend', 'bollinger_breakout']),
              help='Strategy name')
@click.option('--timeframe', '-t', required=True,
              type=click.Choice(['1h', '1d']), help='Timeframe')
@click.option('--content', '-c', required=True, help='Notification content')
def send_signal(symbol: str, strategy: str, timeframe: str, content: str):
    """Send a test signal notification."""
    
    async def send_notification():
        try:
            service = NotificationService()
            await service.start()
            
            click.echo(f"📨 Sending signal notification...")
            click.echo(f"  Symbol: {symbol}")
            click.echo(f"  Strategy: {strategy}")
            click.echo(f"  Timeframe: {timeframe}")
            click.echo(f"  Content: {content}")
            
            success = await service.send_signal(symbol, strategy, timeframe, content)
            
            if success:
                click.echo("✅ Signal notification sent successfully")
            else:
                click.echo("❌ Failed to send signal notification")
                sys.exit(1)
            
            # Wait a moment for processing
            await asyncio.sleep(3)
            
            # Show status
            status = await service.get_status()
            stats = status.get('stats', {})
            click.echo(f"Messages processed: {stats.get('messages_processed', 0)}")
            
            await service.stop()
            
        except Exception as e:
            click.echo(f"❌ Error: {e}")
            sys.exit(1)
    
    asyncio.run(send_notification())


@notifications.command()
@click.option('--content', '-c', required=True, help='Service notification content')
@click.option('--priority', '-p', default=2, type=click.Choice(['1', '2', '3']),
              help='Priority: 1=high, 2=normal, 3=low')
def send_service(content: str, priority: int):
    """Send a test service notification."""
    
    async def send_notification():
        try:
            service = NotificationService()
            await service.start()
            
            click.echo(f"📨 Sending service notification...")
            click.echo(f"  Content: {content}")
            click.echo(f"  Priority: {priority}")
            
            success = await service.send_service_notification(content, int(priority))
            
            if success:
                click.echo("✅ Service notification sent successfully")
            else:
                click.echo("❌ Failed to send service notification")
                sys.exit(1)
            
            # Wait a moment for processing
            await asyncio.sleep(3)
            
            await service.stop()
            
        except Exception as e:
            click.echo(f"❌ Error: {e}")
            sys.exit(1)
    
    asyncio.run(send_notification())


@notifications.command()
def config():
    """Show notification configuration."""
    try:
        config_manager = create_default_config_manager()
        
        click.echo("🔧 Notification Configuration")
        click.echo("=" * 40)
        
        if config_manager.configs:
            for channel_name, config in config_manager.configs.items():
                click.echo(f"\n📺 Channel: {channel_name}")
                click.echo(f"  Enabled: {'✅' if config.enabled else '❌'}")
                click.echo(f"  Strategy: {config.strategy or 'N/A'}")
                click.echo(f"  Timeframe: {config.timeframe or 'N/A'}")
                click.echo(f"  Webhook: {config.webhook_url[:50]}..." if len(config.webhook_url) > 50 else f"  Webhook: {config.webhook_url}")
        else:
            click.echo("❌ No channels configured")
            click.echo("\nRequired environment variables:")
            env_vars = [
                'DISCORD_WEBHOOK_INFRA',
                'DISCORD_WEBHOOK_MACD_RSI',
                'DISCORD_WEBHOOK_SUPPORT_RESISTANCE', 
                'DISCORD_WEBHOOK_RSI_TREND',
                'DISCORD_WEBHOOK_BOLLINGER'
            ]
            for var in env_vars:
                click.echo(f"  {var}")
        
        # Show routing rules
        click.echo(f"\n🔀 Routing Rules:")
        from .models import CHANNEL_ROUTING
        for (strategy, timeframe), channel in CHANNEL_ROUTING.items():
            if strategy != 'service':
                click.echo(f"  {strategy} ({timeframe}) → {channel}")
            else:
                click.echo(f"  service → {channel}")
        
    except Exception as e:
        click.echo(f"❌ Error showing config: {e}")
        sys.exit(1)


if __name__ == '__main__':
    notifications()