"""
Administrative Commands Module

This module provides comprehensive administrative commands for the Wagehood CLI,
including system control, backup/restore operations, maintenance tasks, and troubleshooting.
"""

import os
import json
import subprocess
import signal
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation, log_api_call
from ..config import CLIConfig, ConfigManager


class AdminAPIClient:
    """Client for administrative API operations."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize administrative API client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_api_headers())
        self.logger = CLILogger("admin_client")
    
    def restart_system(self) -> Dict[str, Any]:
        """Restart the system."""
        url = self.config.get_api_url("admin/restart")
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def shutdown_system(self) -> Dict[str, Any]:
        """Shutdown the system."""
        url = self.config.get_api_url("admin/shutdown")
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        url = self.config.get_api_url("admin/info")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def clear_cache(self, cache_type: str = "all") -> Dict[str, Any]:
        """Clear system cache."""
        url = self.config.get_api_url("admin/cache/clear")
        data = {"cache_type": cache_type}
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_logs(self, component: Optional[str] = None,
                 level: Optional[str] = None,
                 limit: int = 100) -> Dict[str, Any]:
        """Get system logs."""
        url = self.config.get_api_url("admin/logs")
        params = {"limit": limit}
        
        if component:
            params["component"] = component
        if level:
            params["level"] = level
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, params=params, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def run_maintenance(self, task: str) -> Dict[str, Any]:
        """Run maintenance task."""
        url = self.config.get_api_url("admin/maintenance")
        data = {"task": task}
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=60)  # Longer timeout for maintenance
            response.raise_for_status()
            return response.json()
    
    def backup_data(self, backup_type: str = "full") -> Dict[str, Any]:
        """Create data backup."""
        url = self.config.get_api_url("admin/backup")
        data = {"backup_type": backup_type}
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=300)  # 5 minute timeout
            response.raise_for_status()
            return response.json()
    
    def restore_data(self, backup_id: str) -> Dict[str, Any]:
        """Restore data from backup."""
        url = self.config.get_api_url("admin/restore")
        data = {"backup_id": backup_id}
        
        with log_api_call(self.logger, "POST", url):
            response = self.session.post(url, json=data, timeout=300)  # 5 minute timeout
            response.raise_for_status()
            return response.json()
    
    def list_backups(self) -> Dict[str, Any]:
        """List available backups."""
        url = self.config.get_api_url("admin/backups")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()


class SystemController:
    """Local system control operations."""
    
    def __init__(self, config: CLIConfig, formatter: OutputFormatter):
        """
        Initialize system controller.
        
        Args:
            config: CLI configuration
            formatter: Output formatter
        """
        self.config = config
        self.formatter = formatter
        self.logger = CLILogger("system_controller")
    
    def start_api_server(self, port: int = 8000, background: bool = False) -> None:
        """Start the API server locally."""
        script_path = Path(__file__).parent.parent.parent.parent / "run_api.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"API script not found: {script_path}")
        
        cmd = ["python", str(script_path), "--port", str(port)]
        
        if background:
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Save PID for later management
            pid_file = Path.home() / ".wagehood" / "api.pid"
            pid_file.parent.mkdir(exist_ok=True)
            pid_file.write_text(str(process.pid))
            
            self.formatter.print_success(f"API server started in background (PID: {process.pid})")
            self.formatter.print_info(f"Server running on port {port}")
        else:
            # Start in foreground
            self.formatter.print_info(f"Starting API server on port {port}...")
            try:
                subprocess.run(cmd, check=True)
            except KeyboardInterrupt:
                self.formatter.print_info("API server stopped")
    
    def stop_api_server(self) -> None:
        """Stop the locally running API server."""
        pid_file = Path.home() / ".wagehood" / "api.pid"
        
        if not pid_file.exists():
            self.formatter.print_warning("No PID file found. Server may not be running.")
            return
        
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to stop
            time.sleep(2)
            
            # Check if process is still running
            try:
                os.kill(pid, 0)  # This doesn't kill, just checks if process exists
                # If we get here, process is still running, force kill
                os.kill(pid, signal.SIGKILL)
                self.formatter.print_warning("Process force killed")
            except ProcessLookupError:
                # Process has stopped
                pass
            
            pid_file.unlink()
            self.formatter.print_success("API server stopped")
            
        except (ValueError, ProcessLookupError) as e:
            self.formatter.print_error(f"Error stopping server: {e}")
            # Clean up stale PID file
            if pid_file.exists():
                pid_file.unlink()
    
    def start_realtime_processor(self, background: bool = False) -> None:
        """Start the real-time processor locally."""
        # Try to use the global command first, fallback to direct script
        script_path = Path(__file__).parent.parent.parent.parent / "run_realtime.py"
        
        try:
            # Check if wagehood global command is available
            subprocess.run(["which", "wagehood"], check=True, capture_output=True)
            cmd = ["wagehood", "admin", "run-realtime"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to direct script execution
            if not script_path.exists():
                raise FileNotFoundError(f"Real-time script not found: {script_path}")
            cmd = ["python", str(script_path)]
        
        if background:
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Save PID for later management
            pid_file = Path.home() / ".wagehood" / "realtime.pid"
            pid_file.parent.mkdir(exist_ok=True)
            pid_file.write_text(str(process.pid))
            
            self.formatter.print_success(f"Real-time processor started in background (PID: {process.pid})")
        else:
            # Start in foreground
            self.formatter.print_info("Starting real-time processor...")
            try:
                subprocess.run(cmd, check=True)
            except KeyboardInterrupt:
                self.formatter.print_info("Real-time processor stopped")
    
    def stop_realtime_processor(self) -> None:
        """Stop the locally running real-time processor."""
        pid_file = Path.home() / ".wagehood" / "realtime.pid"
        
        if not pid_file.exists():
            self.formatter.print_warning("No PID file found. Processor may not be running.")
            return
        
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to stop
            time.sleep(2)
            
            # Check if process is still running
            try:
                os.kill(pid, 0)
                # Force kill if still running
                os.kill(pid, signal.SIGKILL)
                self.formatter.print_warning("Process force killed")
            except ProcessLookupError:
                pass
            
            pid_file.unlink()
            self.formatter.print_success("Real-time processor stopped")
            
        except (ValueError, ProcessLookupError) as e:
            self.formatter.print_error(f"Error stopping processor: {e}")
            if pid_file.exists():
                pid_file.unlink()
    
    def get_running_processes(self) -> Dict[str, Any]:
        """Get information about running processes."""
        processes = {}
        
        # Check API server
        api_pid_file = Path.home() / ".wagehood" / "api.pid"
        if api_pid_file.exists():
            try:
                pid = int(api_pid_file.read_text().strip())
                os.kill(pid, 0)  # Check if process exists
                processes["api_server"] = {"pid": pid, "status": "running"}
            except (ValueError, ProcessLookupError):
                processes["api_server"] = {"pid": None, "status": "stopped"}
                # Clean up stale PID file
                api_pid_file.unlink()
        else:
            processes["api_server"] = {"pid": None, "status": "stopped"}
        
        # Check real-time processor
        rt_pid_file = Path.home() / ".wagehood" / "realtime.pid"
        if rt_pid_file.exists():
            try:
                pid = int(rt_pid_file.read_text().strip())
                os.kill(pid, 0)
                processes["realtime_processor"] = {"pid": pid, "status": "running"}
            except (ValueError, ProcessLookupError):
                processes["realtime_processor"] = {"pid": None, "status": "stopped"}
                rt_pid_file.unlink()
        else:
            processes["realtime_processor"] = {"pid": None, "status": "stopped"}
        
        return processes


# Click command group
@click.group(name='admin')
@click.pass_context
def admin_commands(ctx):
    """Administrative commands for system management."""
    pass


@admin_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def info(ctx, output_format):
    """
    Get detailed system information.
    
    Examples:
        wagehood admin info
        wagehood admin info --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        # Try to get remote system info first
        try:
            client = AdminAPIClient(config)
            with formatter.create_progress_bar("Fetching system info...") as progress:
                task = progress.add_task("Loading...", total=None)
                data = client.get_system_info()
                progress.update(task, completed=1)
            
            formatter.print_data(data, "Remote System Information")
        
        except requests.exceptions.RequestException:
            formatter.print_warning("Could not connect to remote system, showing local info")
        
        # Get local process information
        controller = SystemController(config, formatter)
        local_processes = controller.get_running_processes()
        
        formatter.print_data(local_processes, "Local Process Information")
        
    except Exception as e:
        formatter.print_error(f"Error getting system information: {e}")
        ctx.exit(1)


@admin_commands.group()
@click.pass_context
def service(ctx):
    """Service management commands."""
    pass


@service.command()
@click.option('--port', '-p', type=int, default=8000,
              help='Port to run API server on')
@click.option('--background', '-b', is_flag=True,
              help='Run in background')
@click.pass_context
def start_api(ctx, port, background):
    """
    Start the API server locally.
    
    Examples:
        wagehood admin service start-api
        wagehood admin service start-api --port 8001 --background
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        controller = SystemController(config, formatter)
        controller.start_api_server(port=port, background=background)
        
    except FileNotFoundError as e:
        formatter.print_error(f"API server script not found: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to start API server: {e}")
        ctx.exit(1)


@service.command()
@click.pass_context
def stop_api(ctx):
    """
    Stop the locally running API server.
    
    Examples:
        wagehood admin service stop-api
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        controller = SystemController(config, formatter)
        controller.stop_api_server()
        
    except Exception as e:
        formatter.print_error(f"Failed to stop API server: {e}")
        ctx.exit(1)


@service.command()
@click.option('--background', '-b', is_flag=True,
              help='Run in background')
@click.pass_context
def start_realtime(ctx, background):
    """
    Start the real-time processor locally.
    
    Examples:
        wagehood admin service start-realtime
        wagehood admin service start-realtime --background
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        controller = SystemController(config, formatter)
        controller.start_realtime_processor(background=background)
        
    except FileNotFoundError as e:
        formatter.print_error(f"Real-time processor script not found: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to start real-time processor: {e}")
        ctx.exit(1)


@service.command()
@click.pass_context
def stop_realtime(ctx):
    """
    Stop the locally running real-time processor.
    
    Examples:
        wagehood admin service stop-realtime
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        controller = SystemController(config, formatter)
        controller.stop_realtime_processor()
        
    except Exception as e:
        formatter.print_error(f"Failed to stop real-time processor: {e}")
        ctx.exit(1)


@service.command()
@click.pass_context
def status(ctx):
    """
    Show service status.
    
    Examples:
        wagehood admin service status
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        controller = SystemController(config, formatter)
        processes = controller.get_running_processes()
        
        formatter.print_data(processes, "Service Status")
        
        # Show summary
        running_count = len([p for p in processes.values() if p["status"] == "running"])
        total_count = len(processes)
        
        if running_count == total_count:
            formatter.print_success(f"All services running ({running_count}/{total_count})")
        elif running_count > 0:
            formatter.print_warning(f"Some services running ({running_count}/{total_count})")
        else:
            formatter.print_error("No services running")
        
    except Exception as e:
        formatter.print_error(f"Failed to get service status: {e}")
        ctx.exit(1)


@admin_commands.group()
@click.pass_context
def cache(ctx):
    """Cache management commands."""
    pass


@cache.command()
@click.option('--type', 'cache_type', default='all',
              type=click.Choice(['all', 'data', 'config', 'results'], case_sensitive=False),
              help='Type of cache to clear')
@click.option('--confirm/--no-confirm', default=True,
              help='Confirm before clearing cache')
@click.pass_context
def clear(ctx, cache_type, confirm):
    """
    Clear system cache.
    
    Examples:
        wagehood admin cache clear
        wagehood admin cache clear --type data
        wagehood admin cache clear --no-confirm
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if confirm and not Confirm.ask(f"Clear {cache_type} cache?"):
        formatter.print_info("Cache clear cancelled")
        return
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Clearing cache...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.clear_cache(cache_type)
            progress.update(task, completed=1)
        
        formatter.print_success(f"Cache cleared successfully")
        formatter.print_data(data, "Cache Clear Result")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to clear cache: {e}")
        ctx.exit(1)


@admin_commands.group()
@click.pass_context
def logs(ctx):
    """Log management commands."""
    pass


@logs.command()
@click.option('--component', '-c', help='Filter by component')
@click.option('--level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
              help='Filter by log level')
@click.option('--limit', type=int, default=100,
              help='Maximum number of log entries')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def show(ctx, component, level, limit, output_format):
    """
    Show system logs.
    
    Examples:
        wagehood admin logs show
        wagehood admin logs show --component ingestion --level ERROR
        wagehood admin logs show --limit 50 --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Fetching logs...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_logs(component=component, level=level, limit=limit)
            progress.update(task, completed=1)
        
        formatter.print_data(data, "System Logs")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to fetch logs: {e}")
        ctx.exit(1)


@admin_commands.group()
@click.pass_context
def backup(ctx):
    """Backup and restore commands."""
    pass


@backup.command()
@click.option('--type', 'backup_type', default='full',
              type=click.Choice(['full', 'config', 'data'], case_sensitive=False),
              help='Type of backup to create')
@click.pass_context
def create(ctx, backup_type):
    """
    Create system backup.
    
    Examples:
        wagehood admin backup create
        wagehood admin backup create --type config
        wagehood admin backup create --type data
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Creating backup...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.backup_data(backup_type)
            progress.update(task, completed=1)
        
        formatter.print_success("Backup created successfully")
        formatter.print_data(data, "Backup Information")
        
        # Show helpful information
        backup_id = data.get('backup_id')
        if backup_id:
            formatter.print_info(f"Backup ID: {backup_id}")
            formatter.print_info(f"Restore with: wagehood admin backup restore {backup_id}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to create backup: {e}")
        ctx.exit(1)


@backup.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def list(ctx, output_format):
    """
    List available backups.
    
    Examples:
        wagehood admin backup list
        wagehood admin backup list --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Fetching backups...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.list_backups()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "Available Backups")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to list backups: {e}")
        ctx.exit(1)


@backup.command()
@click.argument('backup_id')
@click.option('--confirm/--no-confirm', default=True,
              help='Confirm before restoring')
@click.pass_context
def restore(ctx, backup_id, confirm):
    """
    Restore from backup.
    
    BACKUP_ID: ID of the backup to restore
    
    Examples:
        wagehood admin backup restore abc123
        wagehood admin backup restore abc123 --no-confirm
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if confirm and not Confirm.ask(f"Restore from backup {backup_id}? This will overwrite current data."):
        formatter.print_info("Restore cancelled")
        return
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Restoring from backup...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.restore_data(backup_id)
            progress.update(task, completed=1)
        
        formatter.print_success("Restore completed successfully")
        formatter.print_data(data, "Restore Information")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to restore backup: {e}")
        ctx.exit(1)


@admin_commands.group()
@click.pass_context
def maintenance(ctx):
    """Maintenance task commands."""
    pass


@maintenance.command()
@click.option('--task', '-t', default='cleanup',
              type=click.Choice(['cleanup', 'optimize', 'vacuum', 'reindex'], case_sensitive=False),
              help='Type of maintenance task')
@click.pass_context
def run(ctx, task):
    """
    Run maintenance tasks.
    
    Examples:
        wagehood admin maintenance run
        wagehood admin maintenance run --task optimize
        wagehood admin maintenance run --task reindex
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar(f"Running {task} maintenance...") as progress:
            task_id = progress.add_task("Processing...", total=None)
            data = client.run_maintenance(task)
            progress.update(task_id, completed=1)
        
        formatter.print_success(f"Maintenance task '{task}' completed successfully")
        formatter.print_data(data, "Maintenance Result")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to run maintenance: {e}")
        ctx.exit(1)


@admin_commands.command()
@click.option('--confirm/--no-confirm', default=True,
              help='Confirm before restarting')
@click.pass_context
def restart(ctx, confirm):
    """
    Restart the system.
    
    Examples:
        wagehood admin restart
        wagehood admin restart --no-confirm
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if confirm and not Confirm.ask("Restart the system?"):
        formatter.print_info("Restart cancelled")
        return
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Restarting system...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.restart_system()
            progress.update(task, completed=1)
        
        formatter.print_success("System restart initiated")
        formatter.print_data(data, "Restart Information")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to restart system: {e}")
        ctx.exit(1)


@admin_commands.command()
@click.option('--confirm/--no-confirm', default=True,
              help='Confirm before shutting down')
@click.pass_context
def shutdown(ctx, confirm):
    """
    Shutdown the system.
    
    Examples:
        wagehood admin shutdown
        wagehood admin shutdown --no-confirm
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if confirm and not Confirm.ask("Shutdown the system?"):
        formatter.print_info("Shutdown cancelled")
        return
    
    try:
        client = AdminAPIClient(config)
        
        with formatter.create_progress_bar("Shutting down system...") as progress:
            task = progress.add_task("Processing...", total=None)
            data = client.shutdown_system()
            progress.update(task, completed=1)
        
        formatter.print_success("System shutdown initiated")
        formatter.print_data(data, "Shutdown Information")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Failed to shutdown system: {e}")
        ctx.exit(1)


@admin_commands.command()
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO',
              help='Logging level')
@click.pass_context
def run_realtime(ctx, log_level):
    """
    Run the real-time data processor directly.
    
    This command directly executes the real-time processor script, primarily
    used by service managers and for testing purposes.
    
    Examples:
        wagehood admin run-realtime
        wagehood admin run-realtime --log-level DEBUG
    """
    import sys
    import os
    from pathlib import Path
    
    try:
        # Get the path to run_realtime.py
        script_path = Path(__file__).parent.parent.parent.parent / "run_realtime.py"
        
        if not script_path.exists():
            click.echo(f"Error: Real-time script not found: {script_path}", err=True)
            ctx.exit(1)
        
        # Set log level environment variable if provided
        if log_level:
            os.environ['LOG_LEVEL'] = log_level.upper()
        
        # Execute the real-time processor
        import subprocess
        cmd = [sys.executable, str(script_path)]
        
        # Add log level argument if the script supports it
        if '--log-level' in sys.argv:
            cmd.extend(['--log-level', log_level.upper()])
        
        # Run the script directly
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        click.echo("\nReal-time processor stopped by user", err=True)
        ctx.exit(0)
    except subprocess.CalledProcessError as e:
        click.echo(f"Real-time processor failed with exit code {e.returncode}", err=True)
        ctx.exit(e.returncode)
    except Exception as e:
        click.echo(f"Error running real-time processor: {e}", err=True)
        ctx.exit(1)


# Export the command group
admin = admin_commands