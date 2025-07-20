"""Wagehood CLI main entry point."""

import asyncio
import os
import sys
import click
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ConfigManager


def setup_env():
    """Set up environment variables from CLI configuration."""
    config_manager = ConfigManager()
    env_vars = config_manager.get_env_vars()
    
    for key, value in env_vars.items():
        os.environ[key] = value


@click.group()
@click.pass_context
def cli(ctx):
    """Wagehood - Trading Strategy Analysis CLI"""
    # Ensure configuration is available
    config_manager = ConfigManager()
    ctx.obj = config_manager
    
    # Set up environment variables for all commands
    setup_env()


@cli.command()
@click.pass_obj
def configure(config_manager: ConfigManager):
    """Configure API credentials and symbols."""
    click.echo("🔧 Wagehood Configuration")
    click.echo("=" * 40)
    
    # Get API credentials
    api_key = click.prompt("Enter your Alpaca API key", hide_input=True)
    secret_key = click.prompt("Enter your Alpaca secret key", hide_input=True)
    
    # Get stock symbols
    symbols_input = click.prompt(
        "Enter supported stock symbols (comma-separated)",
        default="AAPL,GOOGL,MSFT,AMZN,TSLA"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Get crypto symbols
    crypto_symbols_input = click.prompt(
        "Enter supported crypto symbols (comma-separated, e.g., BTC/USD,ETH/USD)",
        default="BTC/USD,ETH/USD,SOL/USD,XRP/USD"
    )
    crypto_symbols = [s.strip().upper() for s in crypto_symbols_input.split(',') if s.strip()]
    
    # Save configuration
    config_manager.update({
        'api_key': api_key,
        'secret_key': secret_key,
        'symbols': symbols,
        'crypto_symbols': crypto_symbols
    })
    
    click.echo(f"\n✅ Configuration saved successfully!")
    click.echo(f"📊 Tracking {len(symbols)} stock symbols: {', '.join(symbols[:5])}" + 
               (" ..." if len(symbols) > 5 else ""))
    click.echo(f"🪙 Tracking {len(crypto_symbols)} crypto symbols: {', '.join(crypto_symbols[:5])}" + 
               (" ..." if len(crypto_symbols) > 5 else ""))


@cli.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol (e.g., AAPL)')
@click.option('--strategy', '-st', required=True, 
              type=click.Choice(['rsi_trend', 'bollinger_breakout', 'macd_rsi', 'sr_breakout']),
              help='Trading strategy')
@click.option('--timeframe', '-t', required=True,
              type=click.Choice(['1h', '1d']),
              help='Timeframe for analysis')
@click.option('--start-date', '-sd', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-ed', required=True, help='End date (YYYY-MM-DD)')
@click.pass_obj
def submit(config_manager: ConfigManager, symbol: str, strategy: str, 
           timeframe: str, start_date: str, end_date: str):
    """Submit a new analysis job."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    try:
        # Parse and validate dates
        from datetime import datetime
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            click.echo(f"❌ Invalid date format. Use YYYY-MM-DD format: {str(e)}", err=True)
            sys.exit(1)
        
        # Validate date range for timeframe
        date_range = end_dt - start_dt
        
        if timeframe == "1h":
            # For hourly data, limit to 26 days to stay well within Alpaca's 30-day limit
            if date_range.days > 26:
                click.echo(f"❌ Date range too large for hourly timeframe.", err=True)
                click.echo(f"   Requested: {date_range.days} days", err=True)
                click.echo(f"   Maximum allowed: 26 days", err=True)
                click.echo(f"   Reason: Alpaca API limits hourly data to 30 days", err=True)
                click.echo(f"💡 Try using a recent 26-day period or switch to daily timeframe", err=True)
                sys.exit(1)
        elif timeframe == "1d":
            # For daily data, limit to 2 years
            if date_range.days > 730:
                click.echo(f"❌ Date range too large for daily timeframe.", err=True)
                click.echo(f"   Requested: {date_range.days} days", err=True)
                click.echo(f"   Maximum allowed: 730 days (2 years)", err=True)
                sys.exit(1)
        
        # Import the backtesting runner
        from run_backtesting import main as run_backtesting
        
        # Display analysis parameters
        click.echo(f"📊 Starting analysis: {symbol.upper()} / {strategy} / {timeframe}")
        click.echo(f"📅 Period: {start_date} to {end_date}")
        click.echo(f"⏳ Calculating backtesting...\n")
        
        # Run backtesting directly
        result = asyncio.run(run_backtesting(
            symbol=symbol.upper(),
            strategy=strategy,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        ))
        
        # Display results
        if result['status'] == 'completed':
            click.echo(f"✅ Analysis completed successfully!")
            
            signals = result.get('signals', [])
            signal_count = len(signals)
            
            click.echo(f"\n📊 Results:")
            click.echo(f"   Signals found: {signal_count}")
            click.echo(f"   Completed at: {result.get('completed_at', 'unknown')}")
            
            if signals:
                click.echo(f"\n📈 Signals:")
                for i, signal in enumerate(signals[:10], 1):  # Show first 10 signals
                    if isinstance(signal, dict):
                        timestamp = signal.get('timestamp', 'N/A')
                        signal_type = signal.get('signal_type', 'N/A')
                        price = signal.get('price', 'N/A')
                        confidence = signal.get('confidence', 'N/A')
                        
                        # Format timestamp if it's a datetime object
                        if hasattr(timestamp, 'isoformat'):
                            timestamp = timestamp.isoformat()
                        
                        # Format price and confidence as floats
                        if isinstance(price, (int, float)):
                            price = f"{price:.2f}"
                        if isinstance(confidence, (int, float)):
                            confidence = f"{confidence:.2f}"
                            
                        click.echo(f"   {i}. {timestamp} | {signal_type} | Price: {price} | Confidence: {confidence}")
                    else:
                        click.echo(f"   {i}. {signal}")
                
                if signal_count > 10:
                    click.echo(f"   ... and {signal_count - 10} more signals")
            else:
                click.echo(f"\n📭 No signals generated for this period")
                
        else:
            # Failed
            click.echo(f"❌ Analysis failed!")
            if result.get('error'):
                click.echo(f"   Error: {result['error']}")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error submitting job: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--status', '-s', help='Filter by status (pending/processing/completed/failed)')
@click.option('--limit', '-l', default=10, help='Number of jobs to show')
@click.pass_obj
def jobs(config_manager: ConfigManager, status: Optional[str], limit: int):
    """List all jobs (deprecated - job system has been removed)."""
    click.echo("⚠️  The job system has been deprecated and removed.")
    click.echo("   Backtesting now runs directly without job queuing.")
    click.echo("   Use 'wagehood submit' to run backtesting analysis directly.")


@cli.group()
def symbols():
    """Manage tracked symbols."""
    pass


@symbols.command('list')
@click.pass_obj
def list_symbols(config_manager: ConfigManager):
    """List all tracked symbols."""
    symbols = config_manager.get('symbols', [])
    crypto_symbols = config_manager.get('crypto_symbols', [])
    
    if not symbols and not crypto_symbols:
        click.echo("📭 No symbols configured.")
        return
    
    if symbols:
        click.echo(f"📊 Tracking {len(symbols)} stock symbols:")
        for i, symbol in enumerate(symbols, 1):
            click.echo(f"  {i}. {symbol}")
    
    if crypto_symbols:
        click.echo(f"\n🪙 Tracking {len(crypto_symbols)} crypto symbols:")
        for i, symbol in enumerate(crypto_symbols, 1):
            click.echo(f"  {i}. {symbol}")


@symbols.command('add')
@click.argument('symbol')
@click.option('--crypto', is_flag=True, help='Add as crypto symbol (e.g., BTC/USD)')
@click.pass_obj
def add_symbol(config_manager: ConfigManager, symbol: str, crypto: bool):
    """Add a new symbol to track."""
    symbol = symbol.upper()
    
    if crypto:
        symbols = config_manager.get('crypto_symbols', [])
        symbol_type = "crypto"
        emoji = "🪙"
    else:
        symbols = config_manager.get('symbols', [])
        symbol_type = "stock"
        emoji = "📊"
    
    if symbol in symbols:
        click.echo(f"⚠️  {emoji} {symbol} is already being tracked as a {symbol_type} symbol.")
        return
    
    symbols.append(symbol)
    if crypto:
        config_manager.set('crypto_symbols', symbols)
    else:
        config_manager.set('symbols', symbols)
    click.echo(f"✅ {emoji} Added {symbol} to tracked {symbol_type} symbols.")


@symbols.command('remove')
@click.argument('symbol')
@click.option('--crypto', is_flag=True, help='Remove from crypto symbols')
@click.pass_obj
def remove_symbol(config_manager: ConfigManager, symbol: str, crypto: bool):
    """Remove a symbol from tracking."""
    symbol = symbol.upper()
    
    if crypto:
        symbols = config_manager.get('crypto_symbols', [])
        symbol_type = "crypto"
        emoji = "🪙"
    else:
        symbols = config_manager.get('symbols', [])
        symbol_type = "stock"
        emoji = "📊"
    
    if symbol not in symbols:
        click.echo(f"⚠️  {emoji} {symbol} is not being tracked as a {symbol_type} symbol.")
        return
    
    symbols.remove(symbol)
    if crypto:
        config_manager.set('crypto_symbols', symbols)
    else:
        config_manager.set('symbols', symbols)
    click.echo(f"✅ {emoji} Removed {symbol} from tracked {symbol_type} symbols.")


@cli.command()
@click.pass_obj
def test(config_manager: ConfigManager):
    """Run all integration tests."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo("🧪 Running integration tests...")
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    test_script = project_root / "run_all_tests.py"
    
    if not test_script.exists():
        click.echo(f"❌ Test script not found at {test_script}")
        return
    
    try:
        # Run tests
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            click.echo("\n✅ All tests passed!")
        else:
            click.echo(f"\n❌ Tests failed with exit code: {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error running tests: {str(e)}", err=True)
        sys.exit(1)




# Trigger commands for cron-based analysis system

@cli.group()
def triggers():
    """Trigger analysis management."""
    pass


@triggers.command('test-1h')
@click.pass_obj
def test_1h_triggers(config_manager: ConfigManager):
    """Test 1-hour analysis trigger."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo("🧪 Testing 1-hour analysis trigger...")
    
    # Find and run the trigger script
    project_root = Path(__file__).parent.parent.parent
    trigger_script = project_root / "trigger_1h_analysis.py"
    
    if not trigger_script.exists():
        click.echo(f"❌ Trigger script not found at {trigger_script}")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, str(trigger_script)],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            click.echo("\n✅ 1-hour analysis trigger test completed!")
        else:
            click.echo(f"\n❌ Test failed with exit code: {result.returncode}")
            
    except Exception as e:
        click.echo(f"❌ Error running 1-hour trigger test: {str(e)}", err=True)


@triggers.command('test-1d')
@click.pass_obj
def test_1d_triggers(config_manager: ConfigManager):
    """Test 1-day analysis trigger."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo("🧪 Testing 1-day analysis trigger...")
    
    # Find and run the trigger script
    project_root = Path(__file__).parent.parent.parent
    trigger_script = project_root / "trigger_1d_analysis.py"
    
    if not trigger_script.exists():
        click.echo(f"❌ Trigger script not found at {trigger_script}")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, str(trigger_script)],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            click.echo("\n✅ 1-day analysis trigger test completed!")
        else:
            click.echo(f"\n❌ Test failed with exit code: {result.returncode}")
            
    except Exception as e:
        click.echo(f"❌ Error running 1-day trigger test: {str(e)}", err=True)


@triggers.command('status')
@click.pass_obj
def triggers_status(config_manager: ConfigManager):
    """Check trigger analysis status."""
    click.echo("📊 Trigger Analysis Status")
    click.echo("=" * 40)
    
    # Check for cron jobs (future implementation)
    click.echo("⚠️  Cron job management will be implemented in the install script.")
    click.echo("   For now, run triggers manually or use the test commands.")
    
    # Check if trigger scripts exist
    project_root = Path(__file__).parent.parent.parent
    trigger_1h = project_root / "trigger_1h_analysis.py"
    trigger_1d = project_root / "trigger_1d_analysis.py"
    
    click.echo(f"\n📁 Trigger Scripts:")
    click.echo(f"   1-hour: {'✅' if trigger_1h.exists() else '❌'} {trigger_1h}")
    click.echo(f"   1-day:  {'✅' if trigger_1d.exists() else '❌'} {trigger_1d}")
    
    # Check signal history files
    history_1h = Path.home() / '.wagehood' / 'signal_history.json'
    history_1d = Path.home() / '.wagehood' / 'signal_history_1d.json'
    
    click.echo(f"\n📝 Signal History:")
    click.echo(f"   1-hour: {'✅' if history_1h.exists() else '❌'} {history_1h}")
    click.echo(f"   1-day:  {'✅' if history_1d.exists() else '❌'} {history_1d}")



@cli.command('sync-env')
@click.pass_obj
def sync_env(config_manager: ConfigManager):
    """Sync .env file to shell configurations (zshrc, bashrc, etc)."""
    click.echo("🔧 Syncing .env to shell configuration...")
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    sync_script = project_root / 'sync_env_to_shell.sh'
    
    if not sync_script.exists():
        click.echo("❌ sync_env_to_shell.sh not found")
        return
    
    try:
        # Make sure script is executable
        sync_script.chmod(0o755)
        
        # Run the sync script
        result = subprocess.run(['bash', str(sync_script)], 
                              capture_output=True, 
                              text=True,
                              cwd=project_root)
        
        if result.returncode == 0:
            click.echo(result.stdout)
            click.echo("\n✅ Environment variables synced successfully!")
            click.echo("📋 To use in current terminal:")
            click.echo("   source ~/.zshrc  (for zsh)")
            click.echo("   source ~/.bashrc (for bash)")
        else:
            click.echo(f"❌ Failed to sync environment: {result.stderr}")
            
    except Exception as e:
        click.echo(f"❌ Error syncing environment: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def summary():
    """End-of-day summary generation."""
    pass


@summary.command('generate')
@click.option('--symbols', '-s', help='Comma-separated symbols (default: all supported)')
@click.option('--discord', '-d', is_flag=True, help='Send summary to Discord webhook')
@click.pass_obj
def generate_summary(config_manager: ConfigManager, symbols: Optional[str], discord: bool):
    """Generate trading signal summary immediately."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    try:
        click.echo("📊 Generating today's trading signal summary...")
        
        # Check Discord webhook if --discord flag is used
        if discord:
            webhook_url = os.getenv('DISCORD_WEBHOOK_EOD_SUMMARY')
            if not webhook_url:
                click.echo("❌ DISCORD_WEBHOOK_EOD_SUMMARY not configured in .env file")
                click.echo("   Please configure the webhook URL to use --discord flag")
                return
            click.echo("📨 Discord notifications enabled")
        
        # Parse symbols if provided
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
            click.echo(f"📈 Analyzing symbols: {', '.join(symbol_list)}")
        else:
            click.echo("📈 Analyzing all supported symbols")
        
        # Import and run the summary script
        from run_summary import main as run_summary
        
        async def generate():
            # Pass the discord flag to enable/disable Discord notifications
            result = await run_summary(symbols=symbol_list, scheduled=discord)
            
            # Display results
            click.echo(f"\n✅ Summary generation completed!")
            click.echo(f"⏰ Generated at: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"📊 Symbols processed: {result.symbols_processed}")
            click.echo(f"🎯 Symbols with signals: {result.symbols_with_signals}")
            click.echo(f"📈 Total signals: {result.total_signals} ({result.buy_signals} BUY, {result.sell_signals} SELL)")
            click.echo(f"⏱️  Execution time: {result.execution_duration_seconds:.2f} seconds")
            
            if result.errors:
                click.echo(f"\n⚠️  {len(result.errors)} errors occurred:")
                for error in result.errors[:3]:
                    click.echo(f"   • {error}")
                if len(result.errors) > 3:
                    click.echo(f"   ... and {len(result.errors) - 3} more errors")
            
            # Show top signals
            if result.signal_summaries:
                top_signals = [s for s in result.signal_summaries if s.signal_count > 0][:5]
                if top_signals:
                    click.echo(f"\n🎯 Top opportunities:")
                    for summary in top_signals:
                        signal_type = "📈 BUY" if summary.buy_signals > summary.sell_signals else "📉 SELL"
                        strategy_names = [s.replace('_', ' ').title() for s in summary.strategies]
                        strategy_info = f" ({', '.join(strategy_names)})" if strategy_names else ""
                        click.echo(f"   • {summary.symbol}: {signal_type} ({summary.signal_count} signals, {summary.avg_confidence:.0%} avg confidence){strategy_info}")
            
            # Show Discord notification status
            if discord:
                click.echo(f"\n📨 Summary sent to Discord webhook")
        
        # Run the async function
        asyncio.run(generate())
        
    except Exception as e:
        click.echo(f"❌ Error generating summary: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def schedule():
    """Daily summary scheduling management."""
    pass


@schedule.command('setup')
@click.pass_obj
def setup_schedule(config_manager: ConfigManager):
    """Set up daily summary to run at 5pm ET."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    try:
        # Check if Discord webhook is configured
        webhook_url = os.getenv('DISCORD_WEBHOOK_EOD_SUMMARY')
        if not webhook_url:
            click.echo("❌ DISCORD_WEBHOOK_EOD_SUMMARY environment variable not configured.")
            click.echo("   Please set this variable before setting up the schedule.")
            return
        
        # Find project root and scheduled script
        project_root = Path(__file__).parent.parent.parent
        scheduled_script = project_root / "run_scheduled_summary.py"
        
        if not scheduled_script.exists():
            click.echo(f"❌ Scheduled script not found at {scheduled_script}")
            return
        
        # Create cron job for 5pm ET daily (17:00 in Eastern Time)
        # Note: This assumes the system timezone is set to Eastern Time
        # For production, you might want to use a more robust timezone handling
        cron_command = f"0 17 * * * cd {project_root} && {sys.executable} {scheduled_script}"
        
        click.echo("📅 Setting up daily summary schedule...")
        click.echo(f"   Schedule: Daily at 5:00 PM ET")
        click.echo(f"   Command: {cron_command}")
        
        # Check if cron job already exists
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        if 'run_scheduled_summary.py' in current_crontab:
            click.echo("⚠️  A scheduled summary job already exists.")
            if not click.confirm("Do you want to replace it?"):
                click.echo("❌ Schedule setup cancelled.")
                return
            
            # Remove existing entry
            updated_crontab = '\n'.join([
                line for line in current_crontab.split('\n') 
                if 'run_scheduled_summary.py' not in line and line.strip()
            ])
        else:
            updated_crontab = current_crontab.strip()
        
        # Add new cron job
        if updated_crontab:
            updated_crontab += '\n'
        updated_crontab += cron_command + '\n'
        
        # Install new crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(updated_crontab)
        
        if process.returncode == 0:
            click.echo("✅ Daily summary schedule configured successfully!")
            click.echo("📊 The summary will run daily at 5:00 PM ET and send results to Discord.")
            click.echo("💡 Use 'wagehood schedule status' to check the schedule.")
        else:
            click.echo("❌ Failed to set up cron job. Please check your crontab configuration.")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error setting up schedule: {str(e)}", err=True)
        sys.exit(1)


@schedule.command('status')
@click.pass_obj
def schedule_status(config_manager: ConfigManager):
    """Check the status of the daily summary schedule."""
    try:
        # Check if cron job exists
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo("❌ No crontab found or unable to access crontab.")
            return
        
        current_crontab = result.stdout
        summary_jobs = [line for line in current_crontab.split('\n') if 'run_scheduled_summary.py' in line]
        
        if summary_jobs:
            click.echo("✅ Daily summary schedule is active:")
            for job in summary_jobs:
                click.echo(f"   {job}")
            
            # Check if Discord webhook is configured
            webhook_url = os.getenv('DISCORD_WEBHOOK_EOD_SUMMARY')
            if webhook_url:
                click.echo("✅ Discord webhook is configured.")
            else:
                click.echo("⚠️  Discord webhook (DISCORD_WEBHOOK_EOD_SUMMARY) is not configured.")
            
            # Check log file
            log_file = Path.home() / '.wagehood' / 'logs' / 'scheduled_summary.log'
            if log_file.exists():
                click.echo(f"📋 Log file: {log_file}")
                # Show last few lines of log
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            click.echo("📊 Recent log entries:")
                            for line in lines[-3:]:
                                click.echo(f"   {line.strip()}")
                except:
                    pass
        else:
            click.echo("❌ No daily summary schedule found.")
            click.echo("💡 Use 'wagehood schedule setup' to configure the daily schedule.")
            
    except Exception as e:
        click.echo(f"❌ Error checking schedule status: {str(e)}", err=True)


@schedule.command('remove')
@click.pass_obj
def remove_schedule(config_manager: ConfigManager):
    """Remove the daily summary schedule."""
    try:
        # Check if cron job exists
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            click.echo("❌ No crontab found or unable to access crontab.")
            return
        
        current_crontab = result.stdout
        summary_jobs = [line for line in current_crontab.split('\n') if 'run_scheduled_summary.py' in line]
        
        if not summary_jobs:
            click.echo("❌ No daily summary schedule found to remove.")
            return
        
        click.echo("🗑️  Found daily summary schedule:")
        for job in summary_jobs:
            click.echo(f"   {job}")
        
        if not click.confirm("Are you sure you want to remove the schedule?"):
            click.echo("❌ Schedule removal cancelled.")
            return
        
        # Remove the cron job
        updated_crontab = '\n'.join([
            line for line in current_crontab.split('\n') 
            if 'run_scheduled_summary.py' not in line and line.strip()
        ])
        
        # Install updated crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(updated_crontab)
        
        if process.returncode == 0:
            click.echo("✅ Daily summary schedule removed successfully!")
        else:
            click.echo("❌ Failed to remove cron job. Please check your crontab configuration.")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error removing schedule: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def notifications():
    """Discord notification service management."""
    pass


@notifications.command('test')
@click.option('--duration', '-d', default=30, help='Test duration in seconds')
@click.option('--send-real', is_flag=True, help='Actually send test notifications to Discord (default: dry run)')
@click.pass_obj
def test_notifications(config_manager: ConfigManager, duration: int, send_real: bool):
    """Test Discord notification service (dry run by default)."""
    try:
        from src.notifications.cli import test
        
        # Create a mock context for the notification CLI
        import click
        ctx = click.Context(test)
        ctx.params = {'duration': duration, 'send_real': send_real}
        
        # Run the notification test
        test.invoke(ctx)
        
    except ImportError:
        click.echo("❌ Notification service not available")
    except Exception as e:
        click.echo(f"❌ Error testing notifications: {e}")


@notifications.command('status')
@click.pass_obj
def notification_status(config_manager: ConfigManager):
    """Check notification service status."""
    try:
        from src.notifications.cli import status
        
        # Run the notification status check
        import click
        ctx = click.Context(status)
        status.invoke(ctx)
        
    except ImportError:
        click.echo("❌ Notification service not available")
    except Exception as e:
        click.echo(f"❌ Error checking notification status: {e}")


@notifications.command('config')
@click.pass_obj
def notification_config(config_manager: ConfigManager):
    """Show notification configuration."""
    try:
        from src.notifications.cli import config
        
        # Run the notification config display
        import click
        ctx = click.Context(config)
        config.invoke(ctx)
        
    except ImportError:
        click.echo("❌ Notification service not available")
    except Exception as e:
        click.echo(f"❌ Error showing notification config: {e}")


@notifications.command('start')
@click.pass_obj
def start_notifications(config_manager: ConfigManager):
    """Start notification workers in background."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo("🚀 Starting notification workers...")
    
    # Check if notification workers are already running
    pid_file = Path.home() / '.wagehood' / 'notification_workers.pid'
    if pid_file.exists():
        click.echo("⚠️  Notification workers may already be running. Use 'wagehood notifications restart' to restart.")
        return
    
    # Start notification workers in background
    project_root = Path(__file__).parent.parent.parent
    notification_script = project_root / "start_notification_workers.py"
    
    try:
        # Start notification process
        import subprocess
        process = subprocess.Popen(
            [sys.executable, str(notification_script)],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Save PID
        pid_file.write_text(str(process.pid))
        
        click.echo(f"✅ Notification workers started with PID: {process.pid}")
        click.echo("📊 Discord notifications will be sent when signals are detected")
        
    except Exception as e:
        click.echo(f"❌ Error starting notification workers: {str(e)}", err=True)
        sys.exit(1)


@notifications.command('stop')
@click.pass_obj
def stop_notifications(config_manager: ConfigManager):
    """Stop notification workers."""
    pid_file = Path.home() / '.wagehood' / 'notification_workers.pid'
    
    if not pid_file.exists():
        click.echo("⚠️  No notification workers are running.")
        return
    
    try:
        pid = int(pid_file.read_text().strip())
        
        # Stop the process
        os.kill(pid, 15)  # SIGTERM
        
        # Remove PID file
        pid_file.unlink()
        
        click.echo(f"✅ Stopped notification workers (PID: {pid})")
        
    except ProcessLookupError:
        # Process already stopped
        pid_file.unlink()
        click.echo("⚠️  Notification workers were not running.")
    except Exception as e:
        click.echo(f"❌ Error stopping notification workers: {str(e)}", err=True)
        sys.exit(1)


@notifications.command('restart')
@click.pass_context
def restart_notifications(ctx):
    """Restart notification workers."""
    # Stop notification workers
    ctx.invoke(stop_notifications)
    
    # Wait a moment
    import time
    time.sleep(1)
    
    # Start notification workers
    ctx.invoke(start_notifications)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()