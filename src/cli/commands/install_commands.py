"""
Installation and System Management Commands Module

This module provides comprehensive installation, configuration, and system management 
commands for the Wagehood CLI, including interactive setup, validation, and service management.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml

import click
import requests
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation
from ..config import CLIConfig, ConfigManager


class InstallationManager:
    """Manager for installation and configuration operations."""
    
    def __init__(self, console: Console):
        """
        Initialize installation manager.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.logger = CLILogger("installation")
        self.config_dir = Path.home() / ".wagehood"
        self.env_file = Path.cwd() / ".env"
        
    def ensure_config_directory(self) -> None:
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Configuration directory: {self.config_dir}")
    
    def load_env_example(self) -> Dict[str, str]:
        """Load the .env.example file for reference."""
        env_example_path = Path.cwd() / ".env.example"
        env_vars = {}
        
        if env_example_path.exists():
            with open(env_example_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        return env_vars
    
    def validate_alpaca_credentials(self, api_key: str, secret_key: str, paper_trading: bool = True) -> Tuple[bool, str]:
        """
        Validate Alpaca API credentials.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Whether to use paper trading
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
            
            api = tradeapi.REST(
                api_key,
                secret_key,
                base_url=base_url
            )
            
            # Test connection by getting account info
            account = api.get_account()
            
            if account.status == 'ACTIVE':
                return True, f"Account active with ${account.cash} cash"
            else:
                return False, f"Account status: {account.status}"
                
        except ImportError:
            return False, "alpaca-trade-api package not installed. Run: pip install alpaca-trade-api"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def test_redis_connection(self, host: str, port: int, db: int = 0) -> Tuple[bool, str]:
        """
        Test Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            
        Returns:
            Tuple of (is_connected, error_message)
        """
        try:
            import redis
            
            r = redis.Redis(host=host, port=port, db=db, socket_timeout=5)
            r.ping()
            info = r.info()
            return True, f"Connected to Redis {info['redis_version']}"
            
        except ImportError:
            return False, "redis package not installed. Run: pip install redis"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def test_api_connectivity(self, api_url: str) -> Tuple[bool, str]:
        """
        Test API connectivity.
        
        Args:
            api_url: API base URL
            
        Returns:
            Tuple of (is_connected, status_message)
        """
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                return True, "API is healthy"
            else:
                return False, f"API returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"API not accessible: {str(e)}"
    
    def create_env_file(self, config: Dict[str, str]) -> None:
        """
        Create .env file with configuration.
        
        Args:
            config: Configuration dictionary
        """
        env_content = []
        env_content.append("# Wagehood Configuration")
        env_content.append("# Generated by Wagehood CLI installation")
        env_content.append("")
        
        # Group related settings
        sections = {
            "Alpaca API Configuration": [
                "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_LIVE_API_KEY", 
                "ALPACA_LIVE_SECRET_KEY", "ALPACA_DATA_FEED", "ALPACA_PAPER_TRADING", 
                "ALPACA_MAX_RETRIES", "ALPACA_RETRY_DELAY"
            ],
            "Watchlist Configuration": ["WATCHLIST_SYMBOLS"],
            "Redis Configuration": [
                "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_STREAMS_MAXLEN"
            ],
            "System Configuration": [
                "DATA_UPDATE_INTERVAL", "CALCULATION_WORKERS", 
                "MAX_CONCURRENT_CALCULATIONS", "BATCH_CALCULATION_SIZE"
            ],
            "Logging Configuration": ["LOG_LEVEL", "LOG_FORMAT"],
            "API Configuration": ["API_HOST", "API_PORT", "API_WORKERS"],
            "Rate Limiting": ["MAX_EVENTS_PER_SECOND", "API_RATE_LIMIT"]
        }
        
        for section_name, keys in sections.items():
            env_content.append(f"# {section_name}")
            for key in keys:
                if key in config:
                    env_content.append(f"{key}={config[key]}")
            env_content.append("")
        
        # Add any remaining keys
        for key, value in config.items():
            if not any(key in section_keys for section_keys in sections.values()):
                env_content.append(f"{key}={value}")
        
        with open(self.env_file, 'w') as f:
            f.write('\n'.join(env_content))
        
        self.logger.info(f"Created .env file at {self.env_file}")
    
    def create_cli_config(self, api_url: str, ws_url: str, output_format: str = "table") -> CLIConfig:
        """
        Create CLI configuration.
        
        Args:
            api_url: API URL
            ws_url: WebSocket URL
            output_format: Default output format
            
        Returns:
            CLIConfig instance
        """
        config = CLIConfig()
        config.api_url = api_url
        config.ws_url = ws_url
        config.output_format = output_format
        config.config_file = str(self.config_dir / "config.yaml")
        
        config.save_config()
        return config


class SystemStatusChecker:
    """System status and health checker."""
    
    def __init__(self, console: Console):
        """
        Initialize status checker.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.logger = CLILogger("status_checker")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required Python packages."""
        dependencies = {
            'click': False,
            'rich': False,
            'requests': False,
            'pyyaml': False,
            'pandas': False,
            'redis': False,
            'alpaca-trade-api': False,
            'websockets': False
        }
        
        for package in dependencies:
            try:
                if package == 'alpaca-trade-api':
                    import alpaca_trade_api
                elif package == 'pyyaml':
                    import yaml
                else:
                    __import__(package)
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False
        
        return dependencies
    
    def check_configuration_files(self) -> Dict[str, bool]:
        """Check if configuration files exist."""
        config_dir = Path.home() / ".wagehood"
        
        files = {
            '.env': Path.cwd() / ".env",
            'CLI config': config_dir / "config.yaml",
            'Config directory': config_dir
        }
        
        status = {}
        for name, path in files.items():
            status[name] = path.exists()
        
        return status
    
    def check_environment_variables(self) -> Dict[str, bool]:
        """Check required environment variables."""
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'REDIS_HOST',
            'REDIS_PORT',
            'WATCHLIST_SYMBOLS'
        ]
        
        status = {}
        for var in required_vars:
            status[var] = bool(os.getenv(var))
        
        return status
    
    def check_services(self, config: CLIConfig) -> Dict[str, Tuple[bool, str]]:
        """Check external services connectivity."""
        services = {}
        
        # Check API
        try:
            response = requests.get(f"{config.api_url}/health", timeout=5)
            services['API'] = (response.status_code == 200, f"Status: {response.status_code}")
        except Exception as e:
            services['API'] = (False, f"Error: {str(e)}")
        
        # Check Redis
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            r = redis.Redis(host=redis_host, port=redis_port)
            r.ping()
            services['Redis'] = (True, "Connected")
        except Exception as e:
            services['Redis'] = (False, f"Error: {str(e)}")
        
        # Check Alpaca (if configured)
        if os.getenv('ALPACA_API_KEY'):
            try:
                import alpaca_trade_api as tradeapi
                paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true'
                base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
                
                api = tradeapi.REST(
                    os.getenv('ALPACA_API_KEY'),
                    os.getenv('ALPACA_SECRET_KEY'),
                    base_url=base_url
                )
                account = api.get_account()
                services['Alpaca'] = (True, f"Account: {account.status}")
            except Exception as e:
                services['Alpaca'] = (False, f"Error: {str(e)}")
        
        return services


class ServiceManager:
    """Service management for starting/stopping system components."""
    
    def __init__(self, console: Console):
        """
        Initialize service manager.
        
        Args:
            console: Rich console instance
        """
        self.console = console
        self.logger = CLILogger("service_manager")
        self.project_root = Path.cwd()
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> bool:
        """Start the API server."""
        try:
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "src.api.app:app",
                "--host", host,
                "--port", str(port),
                "--workers", str(workers),
                "--reload"
            ]
            
            self.logger.info(f"Starting API server: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, cwd=self.project_root)
            
            # Give it a moment to start
            time.sleep(2)
            
            if process.poll() is None:
                self.logger.info(f"API server started with PID {process.pid}")
                return True
            else:
                self.logger.error("API server failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_realtime_processor(self) -> bool:
        """Start the real-time data processor."""
        try:
            # Use the global wagehood command if available, fallback to direct script
            try:
                # Try using the global command first
                cmd = ["wagehood", "admin", "run-realtime"]
                self.logger.info(f"Starting real-time processor via global command: {' '.join(cmd)}")
                process = subprocess.Popen(cmd)
            except FileNotFoundError:
                # Fallback to direct script execution for development
                cmd = [sys.executable, "run_realtime.py"]
                self.logger.info(f"Starting real-time processor via script: {' '.join(cmd)}")
                process = subprocess.Popen(cmd, cwd=self.project_root)
            
            # Give it a moment to start
            time.sleep(2)
            
            if process.poll() is None:
                self.logger.info(f"Real-time processor started with PID {process.pid}")
                return True
            else:
                self.logger.error("Real-time processor failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start real-time processor: {e}")
            return False
    
    def stop_services(self) -> bool:
        """Stop all running services."""
        try:
            # Find and kill processes
            processes_killed = 0
            
            # Kill uvicorn processes
            try:
                subprocess.run(["pkill", "-f", "uvicorn.*src.api.app"], check=False)
                processes_killed += 1
            except:
                pass
            
            # Kill real-time processor
            try:
                # Kill both global command and direct script processes
                subprocess.run(["pkill", "-f", "wagehood.*admin.*run-realtime"], check=False)
                subprocess.run(["pkill", "-f", "run_realtime.py"], check=False)
                processes_killed += 1
            except:
                pass
            
            self.logger.info(f"Stopped {processes_killed} processes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop services: {e}")
            return False


# Click command group
@click.group(name='install')
def install_commands():
    """Installation and system management commands."""
    pass


@install_commands.command()
@click.option('--force', is_flag=True, help='Force reinstallation')
@click.option('--skip-validation', is_flag=True, help='Skip API validation')
@click.pass_context
def setup(ctx, force, skip_validation):
    """
    Interactive installation and setup wizard.
    
    This command guides you through the complete setup process including:
    - API key configuration
    - Service configuration
    - Environment setup
    - Connectivity testing
    
    Examples:
        wagehood install setup
        wagehood install setup --force
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    # Create installation manager
    installer = InstallationManager(console)
    
    with log_operation(installer.logger, "Interactive Installation"):
        
        # Check if already installed
        if not force and installer.env_file.exists():
            if not Confirm.ask("Installation detected. Continue anyway?"):
                formatter.print_info("Installation cancelled")
                return
        
        console.print(Panel.fit(
            "[bold green]Wagehood CLI Installation Wizard[/bold green]\n"
            "This wizard will help you set up the Wagehood trading system.",
            title="🚀 Welcome"
        ))
        
        # Ensure config directory exists
        installer.ensure_config_directory()
        
        # Load existing env example for defaults
        env_defaults = installer.load_env_example()
        config = {}
        
        # 1. Alpaca Configuration
        console.print("\n[bold blue]1. Alpaca Markets Configuration[/bold blue]")
        console.print("Get your API keys from: https://app.alpaca.markets/")
        
        api_key = Prompt.ask(
            "Enter your Alpaca API key",
            default=env_defaults.get('ALPACA_API_KEY', ''),
            password=True
        )
        secret_key = Prompt.ask(
            "Enter your Alpaca secret key",
            default=env_defaults.get('ALPACA_SECRET_KEY', ''),
            password=True
        )
        
        paper_trading = Confirm.ask("Use paper trading?", default=True)
        
        config.update({
            'ALPACA_API_KEY': api_key,
            'ALPACA_SECRET_KEY': secret_key,
            'ALPACA_PAPER_TRADING': str(paper_trading).lower(),
            'ALPACA_DATA_FEED': env_defaults.get('ALPACA_DATA_FEED', 'iex'),
            'ALPACA_MAX_RETRIES': env_defaults.get('ALPACA_MAX_RETRIES', '3'),
            'ALPACA_RETRY_DELAY': env_defaults.get('ALPACA_RETRY_DELAY', '1.0')
        })
        
        # Validate Alpaca credentials
        if not skip_validation and api_key and secret_key:
            with console.status("[spinner]Validating Alpaca credentials..."):
                valid, message = installer.validate_alpaca_credentials(api_key, secret_key, paper_trading)
                if valid:
                    console.print(f"✅ {message}")
                else:
                    console.print(f"❌ {message}")
                    if not Confirm.ask("Continue with invalid credentials?"):
                        return
        
        # 2. Watchlist Configuration
        console.print("\n[bold blue]2. Watchlist Configuration[/bold blue]")
        default_symbols = env_defaults.get('WATCHLIST_SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,SPY,QQQ,IWM')
        
        watchlist = Prompt.ask(
            "Enter symbols to watch (comma-separated)",
            default=default_symbols
        )
        config['WATCHLIST_SYMBOLS'] = watchlist
        
        # 3. Redis Configuration
        console.print("\n[bold blue]3. Redis Configuration[/bold blue]")
        redis_host = Prompt.ask("Redis host", default=env_defaults.get('REDIS_HOST', 'localhost'))
        redis_port = IntPrompt.ask("Redis port", default=int(env_defaults.get('REDIS_PORT', 6379)))
        redis_db = IntPrompt.ask("Redis database", default=int(env_defaults.get('REDIS_DB', 0)))
        
        config.update({
            'REDIS_HOST': redis_host,
            'REDIS_PORT': str(redis_port),
            'REDIS_DB': str(redis_db),
            'REDIS_STREAMS_MAXLEN': env_defaults.get('REDIS_STREAMS_MAXLEN', '10000')
        })
        
        # Test Redis connection
        if not skip_validation:
            with console.status("[spinner]Testing Redis connection..."):
                connected, message = installer.test_redis_connection(redis_host, redis_port, redis_db)
                if connected:
                    console.print(f"✅ {message}")
                else:
                    console.print(f"❌ {message}")
                    if not Confirm.ask("Continue with Redis connection issues?"):
                        return
        
        # 4. System Configuration
        console.print("\n[bold blue]4. System Configuration[/bold blue]")
        
        api_host = Prompt.ask("API host", default=env_defaults.get('API_HOST', '0.0.0.0'))
        api_port = IntPrompt.ask("API port", default=int(env_defaults.get('API_PORT', 8000)))
        
        config.update({
            'API_HOST': api_host,
            'API_PORT': str(api_port),
            'API_WORKERS': env_defaults.get('API_WORKERS', '1'),
            'DATA_UPDATE_INTERVAL': env_defaults.get('DATA_UPDATE_INTERVAL', '1'),
            'CALCULATION_WORKERS': env_defaults.get('CALCULATION_WORKERS', '4'),
            'MAX_CONCURRENT_CALCULATIONS': env_defaults.get('MAX_CONCURRENT_CALCULATIONS', '100'),
            'BATCH_CALCULATION_SIZE': env_defaults.get('BATCH_CALCULATION_SIZE', '10'),
            'LOG_LEVEL': env_defaults.get('LOG_LEVEL', 'INFO'),
            'LOG_FORMAT': env_defaults.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'MAX_EVENTS_PER_SECOND': env_defaults.get('MAX_EVENTS_PER_SECOND', '1000'),
            'API_RATE_LIMIT': env_defaults.get('API_RATE_LIMIT', '200')
        })
        
        # 5. CLI Configuration
        console.print("\n[bold blue]5. CLI Configuration[/bold blue]")
        
        output_format = click.prompt(
            "Default output format",
            type=click.Choice(['table', 'json', 'csv', 'yaml']),
            default='table'
        )
        
        api_url = f"http://{api_host}:{api_port}"
        ws_url = f"ws://{api_host}:{api_port}"
        
        # Create configuration files
        console.print("\n[bold green]Creating configuration files...[/bold green]")
        
        # Create .env file
        installer.create_env_file(config)
        console.print("✅ Created .env file")
        
        # Create CLI config
        cli_config = installer.create_cli_config(api_url, ws_url, output_format)
        console.print("✅ Created CLI configuration")
        
        # Test API connectivity
        if not skip_validation:
            console.print("\n[bold blue]Testing API connectivity...[/bold blue]")
            with console.status("[spinner]Checking API..."):
                connected, message = installer.test_api_connectivity(api_url)
                if connected:
                    console.print(f"✅ {message}")
                else:
                    console.print(f"❌ {message}")
                    console.print("Note: API may not be running yet. Use 'wagehood install start' to start services.")
        
        # Installation complete
        console.print(Panel.fit(
            "[bold green]Installation Complete![/bold green]\n\n"
            "Configuration files created:\n"
            f"• .env file: {installer.env_file}\n"
            f"• CLI config: {cli_config.config_file}\n\n"
            "Next steps:\n"
            "• Run 'wagehood install status' to check system health\n"
            "• Run 'wagehood install start' to start services\n"
            "• Run 'wagehood --help' to see available commands",
            title="🎉 Success"
        ))


@install_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def status(ctx, output_format):
    """
    Check system status and configuration health.
    
    This command performs comprehensive health checks including:
    - Dependencies verification
    - Configuration files
    - Environment variables
    - Service connectivity
    
    Examples:
        wagehood install status
        wagehood install status --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    console = ctx.obj['console']
    
    if output_format:
        formatter.set_format(output_format)
    
    # Create status checker
    checker = SystemStatusChecker(console)
    
    with log_operation(checker.logger, "System Status Check"):
        
        console.print(Panel.fit(
            "[bold blue]System Status Check[/bold blue]",
            title="🔍 Health Check"
        ))
        
        # Check dependencies
        console.print("\n[bold]Dependencies[/bold]")
        deps = checker.check_dependencies()
        
        deps_table = Table(show_header=True, header_style="bold magenta")
        deps_table.add_column("Package", style="dim")
        deps_table.add_column("Status", justify="center")
        
        for package, installed in deps.items():
            status_text = "✅ Installed" if installed else "❌ Missing"
            status_style = "green" if installed else "red"
            deps_table.add_row(package, Text(status_text, style=status_style))
        
        console.print(deps_table)
        
        # Check configuration files
        console.print("\n[bold]Configuration Files[/bold]")
        files = checker.check_configuration_files()
        
        files_table = Table(show_header=True, header_style="bold magenta")
        files_table.add_column("File", style="dim")
        files_table.add_column("Status", justify="center")
        
        for name, exists in files.items():
            status_text = "✅ Exists" if exists else "❌ Missing"
            status_style = "green" if exists else "red"
            files_table.add_row(name, Text(status_text, style=status_style))
        
        console.print(files_table)
        
        # Check environment variables
        console.print("\n[bold]Environment Variables[/bold]")
        env_vars = checker.check_environment_variables()
        
        env_table = Table(show_header=True, header_style="bold magenta")
        env_table.add_column("Variable", style="dim")
        env_table.add_column("Status", justify="center")
        
        for var, set_val in env_vars.items():
            status_text = "✅ Set" if set_val else "❌ Missing"
            status_style = "green" if set_val else "red"
            env_table.add_row(var, Text(status_text, style=status_style))
        
        console.print(env_table)
        
        # Check services
        console.print("\n[bold]Services[/bold]")
        services = checker.check_services(config)
        
        services_table = Table(show_header=True, header_style="bold magenta")
        services_table.add_column("Service", style="dim")
        services_table.add_column("Status", justify="center")
        services_table.add_column("Details", style="dim")
        
        for service, (is_healthy, details) in services.items():
            status_text = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            status_style = "green" if is_healthy else "red"
            services_table.add_row(service, Text(status_text, style=status_style), details)
        
        console.print(services_table)
        
        # Overall status
        all_deps_ok = all(deps.values())
        all_files_ok = all(files.values())
        all_env_ok = all(env_vars.values())
        all_services_ok = all(status for status, _ in services.values())
        
        overall_status = all_deps_ok and all_files_ok and all_env_ok
        
        console.print(f"\n[bold]Overall Status[/bold]")
        if overall_status:
            console.print("✅ [green]System is properly configured[/green]")
        else:
            console.print("❌ [red]System has configuration issues[/red]")
            console.print("\nTo fix issues, run: [bold]wagehood install setup --force[/bold]")


@install_commands.command()
@click.option('--api-only', is_flag=True, help='Start only the API server')
@click.option('--realtime-only', is_flag=True, help='Start only the real-time processor')
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--workers', default=1, help='Number of API workers')
@click.pass_context
def start(ctx, api_only, realtime_only, host, port, workers):
    """
    Start system services.
    
    Examples:
        wagehood install start
        wagehood install start --api-only
        wagehood install start --realtime-only
        wagehood install start --port 8080
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    # Create service manager
    manager = ServiceManager(console)
    
    with log_operation(manager.logger, "Starting Services"):
        
        console.print(Panel.fit(
            "[bold green]Starting Wagehood Services[/bold green]",
            title="🚀 Service Manager"
        ))
        
        services_started = []
        
        if not realtime_only:
            console.print("\n[bold blue]Starting API Server...[/bold blue]")
            if manager.start_api_server(host, port, workers):
                services_started.append("API Server")
                console.print("✅ API Server started")
                console.print(f"   📡 Available at: http://{host}:{port}")
            else:
                console.print("❌ Failed to start API Server")
        
        if not api_only:
            console.print("\n[bold blue]Starting Real-time Processor...[/bold blue]")
            if manager.start_realtime_processor():
                services_started.append("Real-time Processor")
                console.print("✅ Real-time Processor started")
            else:
                console.print("❌ Failed to start Real-time Processor")
        
        if services_started:
            console.print(f"\n✅ [green]Started {len(services_started)} service(s)[/green]")
            for service in services_started:
                console.print(f"   • {service}")
            
            console.print("\n[dim]Use 'wagehood install stop' to stop services[/dim]")
            console.print("[dim]Use 'wagehood install status' to check health[/dim]")
        else:
            console.print("\n❌ [red]No services were started[/red]")


@install_commands.command()
@click.pass_context
def stop(ctx):
    """
    Stop all system services.
    
    Examples:
        wagehood install stop
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    # Create service manager
    manager = ServiceManager(console)
    
    with log_operation(manager.logger, "Stopping Services"):
        
        console.print(Panel.fit(
            "[bold yellow]Stopping Wagehood Services[/bold yellow]",
            title="🛑 Service Manager"
        ))
        
        if manager.stop_services():
            console.print("✅ [green]Services stopped successfully[/green]")
        else:
            console.print("❌ [red]Failed to stop some services[/red]")


@install_commands.command()
@click.pass_context
def restart(ctx):
    """
    Restart all system services.
    
    Examples:
        wagehood install restart
    """
    console = ctx.obj['console']
    
    console.print(Panel.fit(
        "[bold blue]Restarting Wagehood Services[/bold blue]",
        title="🔄 Service Manager"
    ))
    
    # Stop services first
    ctx.invoke(stop)
    
    # Wait a moment
    time.sleep(2)
    
    # Start services
    ctx.invoke(start)


@install_commands.command()
@click.option('--reset-env', is_flag=True, help='Reset environment configuration')
@click.option('--reset-cli', is_flag=True, help='Reset CLI configuration')
@click.pass_context
def configure(ctx, reset_env, reset_cli):
    """
    Interactive configuration update wizard.
    
    Examples:
        wagehood install configure
        wagehood install configure --reset-env
        wagehood install configure --reset-cli
    """
    console = ctx.obj['console']
    config = ctx.obj['config']
    
    # Create installation manager
    installer = InstallationManager(console)
    
    with log_operation(installer.logger, "Configuration Update"):
        
        console.print(Panel.fit(
            "[bold blue]Configuration Update Wizard[/bold blue]\n"
            "Update your Wagehood system configuration.",
            title="⚙️ Configure"
        ))
        
        if reset_env or not installer.env_file.exists():
            console.print("\n[bold yellow]Environment Configuration[/bold yellow]")
            
            # Load current or default values
            env_defaults = installer.load_env_example()
            
            # Quick configuration prompts
            api_key = Prompt.ask("Alpaca API key", password=True)
            secret_key = Prompt.ask("Alpaca secret key", password=True)
            paper_trading = Confirm.ask("Use paper trading?", default=True)
            
            watchlist = Prompt.ask(
                "Watchlist symbols (comma-separated)",
                default=env_defaults.get('WATCHLIST_SYMBOLS', 'AAPL,SPY,QQQ')
            )
            
            # Create minimal config
            env_config = {
                'ALPACA_API_KEY': api_key,
                'ALPACA_SECRET_KEY': secret_key,
                'ALPACA_PAPER_TRADING': str(paper_trading).lower(),
                'ALPACA_DATA_FEED': 'iex',
                'WATCHLIST_SYMBOLS': watchlist,
                'REDIS_HOST': 'localhost',
                'REDIS_PORT': '6379',
                'REDIS_DB': '0',
                'API_HOST': '0.0.0.0',
                'API_PORT': '8000',
                'LOG_LEVEL': 'INFO'
            }
            
            installer.create_env_file(env_config)
            console.print("✅ Environment configuration updated")
        
        if reset_cli:
            console.print("\n[bold yellow]CLI Configuration[/bold yellow]")
            
            output_format = click.prompt(
                "Default output format",
                type=click.Choice(['table', 'json', 'csv', 'yaml']),
                default=config.output_format
            )
            
            api_url = Prompt.ask("API URL", default=config.api_url)
            ws_url = Prompt.ask("WebSocket URL", default=config.ws_url)
            
            # Update CLI config
            config.output_format = output_format
            config.api_url = api_url
            config.ws_url = ws_url
            config.save_config()
            
            console.print("✅ CLI configuration updated")
        
        console.print(Panel.fit(
            "[bold green]Configuration Updated![/bold green]\n\n"
            "Run 'wagehood install status' to verify the changes.",
            title="✅ Success"
        ))


# Export the command group
install = install_commands