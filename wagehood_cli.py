#!/usr/bin/env python3
"""
Wagehood CLI - Command Line Interface for Real-time Trading System

A comprehensive CLI tool for interacting with the Wagehood real-time market data
processing API. Provides commands for data retrieval, configuration management,
system monitoring, and administrative tasks.

Usage:
    wagehood [OPTIONS] COMMAND [ARGS]...

Examples:
    # Get latest data for a symbol
    wagehood data latest SPY

    # Stream real-time data
    wagehood data stream SPY QQQ --duration 60

    # Add symbols to watchlist
    wagehood config watchlist add AAPL TSLA NVDA

    # Check system health
    wagehood monitor health

    # Export historical data
    wagehood data export --symbols SPY,QQQ --start 2024-01-01 --format csv
"""

import click
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.commands import (
    data_commands,
    config_commands,
    monitor_commands,
    admin_commands
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
    
    A comprehensive command-line tool for interacting with the Wagehood
    real-time market data processing system. Manage configurations,
    monitor system health, and access real-time market data.
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