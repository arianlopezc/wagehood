"""
Configuration Commands Module

This module provides comprehensive configuration management commands for the Wagehood CLI,
including watchlist management, indicator/strategy configuration, and system settings.
"""

import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation, log_api_call
from ..config import CLIConfig, ConfigManager


class ConfigAPIClient:
    """Client for configuration API operations."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize configuration API client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_api_headers())
        self.logger = CLILogger("config_client")
    
    def get_watchlist(self) -> Dict[str, Any]:
        """Get current watchlist."""
        url = self.config.get_api_url("realtime/config/watchlist")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def update_watchlist(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update watchlist."""
        url = self.config.get_api_url("realtime/config/watchlist")
        data = {"assets": assets}
        
        with log_api_call(self.logger, "PUT", url):
            response = self.session.put(url, json=data, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def add_symbol(self, symbol: str, data_provider: str = "mock", 
                   timeframes: List[str] = None, priority: int = 1) -> Dict[str, Any]:
        """Add symbol to watchlist."""
        url = self.config.get_api_url("realtime/config/watchlist/add")
        data = {
            "symbol": symbol,
            "data_provider": data_provider,
            "timeframes": timeframes or ["1m"],
            "priority": priority
        }
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def remove_symbol(self, symbol: str) -> Dict[str, Any]:
        """Remove symbol from watchlist."""
        url = self.config.get_api_url(f"realtime/config/watchlist/{symbol}")
        
        with log_api_call(self.logger, "DELETE", url):
            response = self.session.delete(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_indicators(self) -> List[Dict[str, Any]]:
        """Get indicator configurations."""
        url = self.config.get_api_url("realtime/config/indicators")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def update_indicators(self, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update indicator configurations."""
        url = self.config.get_api_url("realtime/config/indicators")
        
        with log_api_call(self.logger, "PUT", url):
            response = self.session.put(url, json=indicators, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get strategy configurations."""
        url = self.config.get_api_url("realtime/config/strategies")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def update_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update strategy configurations."""
        url = self.config.get_api_url("realtime/config/strategies")
        
        with log_api_call(self.logger, "PUT", url):
            response = self.session.put(url, json=strategies, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        url = self.config.get_api_url("realtime/config/system")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def update_system_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration."""
        url = self.config.get_api_url("realtime/config/system")
        
        with log_api_call(self.logger, "PUT", url):
            response = self.session.put(url, json=config, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        url = self.config.get_api_url("realtime/config/summary")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        url = self.config.get_api_url("realtime/config/validate")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()


# Click command group
@click.group(name='config')
@click.pass_context
def config_commands(ctx):
    """Configuration management commands."""
    pass


@config_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, output_format):
    """
    Show current CLI configuration.
    
    Examples:
        wagehood_cli.py config show
        wagehood_cli.py config show --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    # Display CLI configuration
    config_data = config.to_dict()
    formatter.print_data(config_data, "CLI Configuration")
    
    # Show configuration file location
    formatter.print_info(f"Configuration file: {config.config_file}")


@config_commands.command()
@click.option('--api-url', help='API base URL')
@click.option('--ws-url', help='WebSocket base URL')
@click.option('--output-format', type=click.Choice(['json', 'table', 'csv', 'yaml']),
              help='Default output format')
@click.option('--timeout', type=int, help='API timeout in seconds')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Logging level')
@click.option('--no-color/--color', default=None, help='Enable/disable colored output')
@click.pass_context
def set(ctx, api_url, ws_url, output_format, timeout, log_level, no_color):
    """
    Set CLI configuration values.
    
    Examples:
        wagehood_cli.py config set --api-url http://localhost:8000
        wagehood_cli.py config set --output-format table --log-level DEBUG
        wagehood_cli.py config set --no-color
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    changes = {}
    
    if api_url is not None:
        old_value = config.api_url
        config.api_url = api_url
        changes['api_url'] = (old_value, api_url)
    
    if ws_url is not None:
        old_value = config.ws_url
        config.ws_url = ws_url
        changes['ws_url'] = (old_value, ws_url)
    
    if output_format is not None:
        old_value = config.output_format
        config.output_format = output_format
        changes['output_format'] = (old_value, output_format)
    
    if timeout is not None:
        old_value = config.api_timeout
        config.api_timeout = timeout
        changes['api_timeout'] = (old_value, timeout)
    
    if log_level is not None:
        old_value = config.log_level
        config.log_level = log_level
        changes['log_level'] = (old_value, log_level)
    
    if no_color is not None:
        old_value = config.no_color
        config.no_color = no_color
        changes['no_color'] = (old_value, no_color)
    
    if not changes:
        formatter.print_warning("No configuration changes specified")
        return
    
    try:
        config.save_config()
        formatter.print_success("Configuration updated successfully!")
        
        # Show changes
        for key, (old_val, new_val) in changes.items():
            formatter.print_info(f"{key}: {old_val} → {new_val}")
        
    except Exception as e:
        formatter.print_error(f"Failed to save configuration: {e}")
        ctx.exit(1)


@config_commands.command()
@click.pass_context
def reset(ctx):
    """
    Reset CLI configuration to defaults.
    
    Examples:
        wagehood_cli.py config reset
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if not Confirm.ask("Are you sure you want to reset configuration to defaults?"):
        formatter.print_info("Configuration reset cancelled")
        return
    
    try:
        config.reset_to_defaults()
        config.save_config()
        formatter.print_success("Configuration reset to defaults successfully!")
        
    except Exception as e:
        formatter.print_error(f"Failed to reset configuration: {e}")
        ctx.exit(1)


@config_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def validate(ctx, output_format):
    """
    Validate CLI configuration.
    
    Examples:
        wagehood_cli.py config validate
        wagehood_cli.py config validate --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    # Validate CLI configuration
    validation_result = config.validate()
    formatter.print_data(validation_result, "Configuration Validation")
    
    if validation_result['valid']:
        formatter.print_success("Configuration is valid!")
    else:
        formatter.print_error("Configuration has issues that need to be fixed")
        if validation_result['issues']:
            formatter.print_info("Issues found:")
            for issue in validation_result['issues']:
                formatter.print_error(f"  • {issue}")
    
    if validation_result['warnings']:
        formatter.print_warning("Warnings:")
        for warning in validation_result['warnings']:
            formatter.print_warning(f"  • {warning}")


@config_commands.command()
@click.argument('file_path', type=click.Path())
@click.option('--format', type=click.Choice(['json', 'yaml']), default='yaml',
              help='Export format')
@click.pass_context
def export(ctx, file_path, format):
    """
    Export CLI configuration to file.
    
    FILE_PATH: Path to export file
    
    Examples:
        wagehood_cli.py config export config.yaml
        wagehood_cli.py config export config.json --format json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        config_manager = ConfigManager()
        config_manager.config = config
        config_manager.export_config(file_path, format)
        formatter.print_success(f"Configuration exported to {file_path}")
        
    except Exception as e:
        formatter.print_error(f"Failed to export configuration: {e}")
        ctx.exit(1)


@config_commands.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def import_config(ctx, file_path):
    """
    Import CLI configuration from file.
    
    FILE_PATH: Path to configuration file
    
    Examples:
        wagehood_cli.py config import config.yaml
        wagehood_cli.py config import backup-config.json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        config_manager = ConfigManager()
        config_manager.config = config
        config_manager.import_config(file_path)
        formatter.print_success(f"Configuration imported from {file_path}")
        
    except Exception as e:
        formatter.print_error(f"Failed to import configuration: {e}")
        ctx.exit(1)


# Watchlist commands
@config_commands.group()
@click.pass_context
def watchlist(ctx):
    """Watchlist management commands."""
    pass


@watchlist.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, output_format):
    """
    Show current watchlist.
    
    Examples:
        wagehood_cli.py config watchlist show
        wagehood_cli.py config watchlist show --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Fetching watchlist...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_watchlist()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Current Watchlist")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@watchlist.command()
@click.argument('symbol')
@click.option('--provider', default='mock', help='Data provider')
@click.option('--timeframes', multiple=True, default=['1m'],
              help='Timeframes to track (can be used multiple times)')
@click.option('--priority', type=int, default=1, help='Priority level (1-10)')
@click.pass_context
def add(ctx, symbol, provider, timeframes, priority):
    """
    Add symbol to watchlist.
    
    SYMBOL: Trading symbol to add (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py config watchlist add AAPL
        wagehood_cli.py config watchlist add SPY --priority 5
        wagehood_cli.py config watchlist add QQQ --timeframes 1m --timeframes 5m
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Adding symbol...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.add_symbol(
                symbol.upper(), provider, list(timeframes), priority
            )
            progress.update(task, completed=1)
        
        formatter.print_success(f"Added {symbol.upper()} to watchlist")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@watchlist.command()
@click.argument('symbol')
@click.option('--confirm/--no-confirm', default=True,
              help='Confirm before removing')
@click.pass_context
def remove(ctx, symbol, confirm):
    """
    Remove symbol from watchlist.
    
    SYMBOL: Trading symbol to remove (e.g., AAPL, SPY, QQQ)
    
    Examples:
        wagehood_cli.py config watchlist remove AAPL
        wagehood_cli.py config watchlist remove SPY --no-confirm
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    symbol = symbol.upper()
    
    if confirm and not Confirm.ask(f"Remove {symbol} from watchlist?"):
        formatter.print_info("Operation cancelled")
        return
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Removing symbol...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.remove_symbol(symbol)
            progress.update(task, completed=1)
        
        formatter.print_success(f"Removed {symbol} from watchlist")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


# Indicators commands
@config_commands.group()
@click.pass_context
def indicators(ctx):
    """Indicator configuration commands."""
    pass


@indicators.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, output_format):
    """
    Show indicator configurations.
    
    Examples:
        wagehood_cli.py config indicators show
        wagehood_cli.py config indicators show --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Fetching indicators...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_indicators()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Indicator Configurations")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@indicators.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def update(ctx, config_file):
    """
    Update indicator configurations from file.
    
    CONFIG_FILE: Path to JSON/YAML file with indicator configurations
    
    Examples:
        wagehood_cli.py config indicators update indicators.json
        wagehood_cli.py config indicators update indicators.yaml
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        # Load configuration from file
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                indicators_config = yaml.safe_load(f)
            else:
                indicators_config = json.load(f)
        
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Updating indicators...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.update_indicators(indicators_config)
            progress.update(task, completed=1)
        
        formatter.print_success("Indicator configurations updated successfully")
        formatter.print_data(data, "Updated Indicators")
        
    except Exception as e:
        formatter.print_error(f"Failed to update indicators: {e}")
        ctx.exit(1)


# Strategies commands
@config_commands.group()
@click.pass_context
def strategies(ctx):
    """Strategy configuration commands."""
    pass


@strategies.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, output_format):
    """
    Show strategy configurations.
    
    Examples:
        wagehood_cli.py config strategies show
        wagehood_cli.py config strategies show --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Fetching strategies...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_strategies()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Strategy Configurations")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@strategies.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def update(ctx, config_file):
    """
    Update strategy configurations from file.
    
    CONFIG_FILE: Path to JSON/YAML file with strategy configurations
    
    Examples:
        wagehood_cli.py config strategies update strategies.json
        wagehood_cli.py config strategies update strategies.yaml
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        # Load configuration from file
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                strategies_config = yaml.safe_load(f)
            else:
                strategies_config = json.load(f)
        
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Updating strategies...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.update_strategies(strategies_config)
            progress.update(task, completed=1)
        
        formatter.print_success("Strategy configurations updated successfully")
        formatter.print_data(data, "Updated Strategies")
        
    except Exception as e:
        formatter.print_error(f"Failed to update strategies: {e}")
        ctx.exit(1)


# System commands
@config_commands.group()
@click.pass_context
def system(ctx):
    """System configuration commands."""
    pass


@system.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, output_format):
    """
    Show system configuration.
    
    Examples:
        wagehood_cli.py config system show
        wagehood_cli.py config system show --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Fetching system config...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_system_config()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "System Configuration")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@system.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def update(ctx, config_file):
    """
    Update system configuration from file.
    
    CONFIG_FILE: Path to JSON/YAML file with system configuration
    
    Examples:
        wagehood_cli.py config system update system.json
        wagehood_cli.py config system update system.yaml
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        # Load configuration from file
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                system_config = yaml.safe_load(f)
            else:
                system_config = json.load(f)
        
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Updating system config...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.update_system_config(system_config)
            progress.update(task, completed=1)
        
        formatter.print_success("System configuration updated successfully")
        formatter.print_data(data, "Updated System Configuration")
        
    except Exception as e:
        formatter.print_error(f"Failed to update system configuration: {e}")
        ctx.exit(1)


@config_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def summary(ctx, output_format):
    """
    Show configuration summary.
    
    Examples:
        wagehood_cli.py config summary
        wagehood_cli.py config summary --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Fetching config summary...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_config_summary()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Configuration Summary")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@config_commands.command(name='validate-remote')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def validate_remote(ctx, output_format):
    """
    Validate remote system configuration.
    
    Examples:
        wagehood_cli.py config validate-remote
        wagehood_cli.py config validate-remote --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = ConfigAPIClient(config)
        
        with formatter.create_progress_bar("Validating configuration...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.validate_config()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Configuration Validation")
        
        if data.get('is_valid'):
            formatter.print_success("Remote configuration is valid!")
        else:
            formatter.print_error("Remote configuration has issues")
            
            if data.get('errors'):
                formatter.print_error("Errors:")
                for error in data['errors']:
                    formatter.print_error(f"  • {error}")
            
            if data.get('warnings'):
                formatter.print_warning("Warnings:")
                for warning in data['warnings']:
                    formatter.print_warning(f"  • {warning}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)