#!/usr/bin/env python3
"""
Wagehood CLI - Command Line Interface for Real-time Trading System

A comprehensive CLI tool for interacting with the Wagehood real-time market data
processing API. Provides complete system management including installation, 
configuration, data retrieval, monitoring, and administrative tasks.

Usage:
    wagehood [OPTIONS] COMMAND [ARGS]...

Quick Start:
    # Install and setup the system
    wagehood install setup

    # Check system status
    wagehood install status

    # Start services
    wagehood install start

Data Operations:
    # Get latest data for a symbol
    wagehood data latest SPY

    # Stream real-time data
    wagehood data stream SPY QQQ --duration 60

    # Export historical data
    wagehood data export --symbols SPY,QQQ --start 2024-01-01 --format csv

Configuration:
    # Add symbols to watchlist
    wagehood config watchlist add AAPL TSLA NVDA

    # Update configuration
    wagehood install configure

    # Show current configuration
    wagehood config show

System Management:
    # Check system health
    wagehood monitor health

    # Restart services
    wagehood install restart

    # Stop services
    wagehood install stop

Service Management:
    # Install auto-start service
    wagehood service install

    # Check service status
    wagehood service status

    # Enable/disable auto-start
    wagehood service enable
    wagehood service disable
"""

import click
import sys
import os
from pathlib import Path

# Try to import from installed package first, fallback to local development
try:
    from src.cli.commands import (
        data_commands,
        config_commands,
        monitor_commands,
        admin_commands,
        install_commands,
        service_commands
    )
    from src.cli.config import CLIConfig, ConfigManager
    from src.cli.utils.logging import setup_logging
    from src.cli.utils.output import OutputFormatter
except ImportError:
    # Add src to path for local development
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from src.cli.commands import (
        data_commands,
        config_commands,
        monitor_commands,
        admin_commands,
        install_commands,
        service_commands
    )
    from src.cli.config import CLIConfig, ConfigManager
    from src.cli.utils.logging import setup_logging
    from src.cli.utils.output import OutputFormatter
from rich.console import Console

# CLI Configuration
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('--api-url', 
              default='http://localhost:8000', 
              envvar='WAGEHOOD_API_URL',
              help='API base URL')
@click.option('--config', '-c',
              type=click.Path(),
              default='~/.wagehood/cli_config.yaml',
              help='Configuration file path')
@click.option('--output-format', '-f',
              type=click.Choice(['json', 'table', 'csv']),
              default='table',
              help='Output format')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose logging')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress non-error output')
@click.option('--no-color',
              is_flag=True,
              help='Disable colored output')
@click.version_option(version='1.0.0', prog_name='wagehood')
@click.pass_context
def cli(ctx, api_url, config, output_format, verbose, quiet, no_color):
    """
    Wagehood CLI - Real-time Trading System Interface
    
    A comprehensive command-line tool for the Wagehood real-time market data
    processing system. Features complete system management including installation,
    configuration, data access, monitoring, and service management.
    
    Get started by running: wagehood install setup
    """
    # Create console
    console = Console(no_color=no_color)
    
    # Setup logging
    log_level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    setup_logging(level=log_level, console=console)
    
    # Load configuration
    config_path = os.path.expanduser(config) if config else None
    cli_config = CLIConfig()
    if config_path:
        cli_config.config_file = config_path
    
    # Try to load existing config
    try:
        cli_config.load_config(config_path)
    except Exception:
        # Use defaults if config doesn't exist
        pass
    
    # Override with command line options
    if api_url:
        cli_config.api_url = api_url
    if output_format:
        cli_config.output_format = output_format
    if no_color:
        cli_config.no_color = no_color
    
    # Create output formatter
    formatter = OutputFormatter(console)
    formatter.set_format(cli_config.output_format)
    formatter.set_color(not cli_config.no_color)
    
    # Store config and formatter in context
    ctx.obj = {
        'config': cli_config,
        'formatter': formatter,
        'console': console
    }


# Register command groups
cli.add_command(install_commands)
cli.add_command(service_commands)
cli.add_command(data_commands)
cli.add_command(config_commands)
cli.add_command(monitor_commands)
cli.add_command(admin_commands)


@cli.command()
@click.pass_context
def info(ctx):
    """Display CLI and API information"""
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    info_data = {
        'CLI Version': '1.0.0',
        'Environment': config.environment,
        'API URL': config.api_url,
        'WebSocket URL': config.ws_url,
        'Config File': config.config_file,
        'Output Format': config.output_format,
        'Log Level': config.log_level,
        'Cache Enabled': config.cache_enabled,
        'Color Output': not config.no_color
    }
    
    formatter.print_header("Wagehood CLI Information")
    formatter.print_key_value_pairs(info_data)


@cli.command()
@click.pass_context
def getting_started(ctx):
    """Display getting started guide and setup instructions"""
    console = ctx.obj['console']
    
    from rich.panel import Panel
    from rich.text import Text
    
    console.print(Panel.fit(
        Text.from_markup(
            "[bold green]Welcome to Wagehood CLI![/bold green]\n\n"
            "[bold]Getting Started Guide[/bold]\n\n"
            "[yellow]1. First-time Setup[/yellow]\n"
            "   Run the interactive installation wizard:\n"
            "   [cyan]wagehood install setup[/cyan]\n\n"
            "[yellow]2. Check System Status[/yellow]\n"
            "   Verify your installation and configuration:\n"
            "   [cyan]wagehood install status[/cyan]\n\n"
            "[yellow]3. Start Services[/yellow]\n"
            "   Launch the API server and real-time processor:\n"
            "   [cyan]wagehood install start[/cyan]\n\n"
            "[yellow]4. Explore Data[/yellow]\n"
            "   Get latest market data:\n"
            "   [cyan]wagehood data latest SPY[/cyan]\n\n"
            "[yellow]5. Configure Watchlist[/yellow]\n"
            "   Add symbols to track:\n"
            "   [cyan]wagehood config watchlist add AAPL TSLA[/cyan]\n\n"
            "[bold]Common Commands[/bold]\n"
            "• [cyan]wagehood install --help[/cyan] - Installation and service management\n"
            "• [cyan]wagehood service --help[/cyan] - Auto-start service management\n"
            "• [cyan]wagehood config --help[/cyan] - Configuration management\n"
            "• [cyan]wagehood data --help[/cyan] - Data operations\n"
            "• [cyan]wagehood monitor --help[/cyan] - System monitoring\n"
            "• [cyan]wagehood --help[/cyan] - Show all available commands\n\n"
            "[bold]Prerequisites[/bold]\n"
            "• Alpaca Markets account (free paper trading)\n"
            "• Redis server (for real-time data)\n"
            "• Python 3.8+ with required packages\n\n"
            "[dim]For detailed documentation, visit: https://github.com/your-repo/wagehood[/dim]"
        ),
        title="🚀 Wagehood CLI",
        padding=(1, 2)
    ))


@cli.command()
@click.option('--shell', 
              type=click.Choice(['bash', 'zsh', 'fish']),
              help='Shell type for completion')
def completion(shell):
    """Generate shell completion script"""
    if shell == 'bash':
        click.echo('eval "$(_WAGEHOOD_COMPLETE=source_bash wagehood)"')
    elif shell == 'zsh':
        click.echo('eval "$(_WAGEHOOD_COMPLETE=source_zsh wagehood)"')
    elif shell == 'fish':
        click.echo('eval "$(_WAGEHOOD_COMPLETE=source_fish wagehood)"')
    else:
        click.echo('Please specify shell type with --shell option')


def main():
    """Main entry point"""
    try:
        cli(prog_name='wagehood')
    except KeyboardInterrupt:
        click.echo('\nInterrupted by user', err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()