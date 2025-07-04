"""
Monitoring Commands Module

This module provides comprehensive monitoring and system health commands for the Wagehood CLI,
including health checks, performance metrics, alerts, and real-time system monitoring.
"""

import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.text import Text

from ..utils.output import OutputFormatter, format_timestamp, format_duration, format_file_size
from ..utils.logging import CLILogger, log_operation, log_api_call
from ..config import CLIConfig


class MonitorAPIClient:
    """Client for monitoring API operations."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize monitoring API client.
        
        Args:
            config: CLI configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.get_api_headers())
        self.logger = CLILogger("monitor_client")
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health information."""
        url = self.config.get_api_url("realtime/monitor/health")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed system health information."""
        url = self.config.get_api_url("health/detailed")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        url = self.config.get_api_url("realtime/monitor/stats")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_alerts(self, limit: int = 50, offset: int = 0,
                   alert_type: Optional[str] = None,
                   component: Optional[str] = None,
                   acknowledged: Optional[bool] = None) -> Dict[str, Any]:
        """Get system alerts."""
        url = self.config.get_api_url("realtime/monitor/alerts")
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if alert_type:
            params['alert_type'] = alert_type
        if component:
            params['component'] = component
        if acknowledged is not None:
            params['acknowledged'] = acknowledged
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, params=params, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()
    
    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health check."""
        url = self.config.get_api_url("health")
        
        with log_api_call(self.logger, "GET", url):
            response = self.session.get(url, timeout=self.config.api_timeout)
            response.raise_for_status()
            return response.json()


class SystemMonitor:
    """Real-time system monitor."""
    
    def __init__(self, config: CLIConfig, formatter: OutputFormatter):
        """
        Initialize system monitor.
        
        Args:
            config: CLI configuration
            formatter: Output formatter
        """
        self.config = config
        self.formatter = formatter
        self.client = MonitorAPIClient(config)
        self.running = False
    
    def start_monitoring(self, refresh_interval: int = 5, duration: Optional[int] = None):
        """
        Start real-time monitoring.
        
        Args:
            refresh_interval: Refresh interval in seconds
            duration: Optional duration in seconds
        """
        self.running = True
        start_time = datetime.now()
        
        try:
            with Live(console=self.formatter.console, refresh_per_second=1/refresh_interval) as live:
                while self.running:
                    try:
                        # Get current system data
                        health_data = self.client.get_health()
                        stats_data = self.client.get_stats()
                        
                        # Create monitoring display
                        display = self._create_monitor_display(health_data, stats_data)
                        live.update(display)
                        
                        # Check duration limit
                        if duration and (datetime.now() - start_time).seconds >= duration:
                            break
                        
                        time.sleep(refresh_interval)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        self.formatter.print_error(f"Monitoring error: {e}")
                        time.sleep(refresh_interval)
                        
        except KeyboardInterrupt:
            self.formatter.print_info("Monitoring stopped by user")
        finally:
            self.running = False
    
    def _create_monitor_display(self, health_data: Dict[str, Any], 
                               stats_data: Dict[str, Any]) -> Layout:
        """Create monitoring display layout."""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header with timestamp and status
        status = health_data.get('status', 'unknown')
        status_color = self._get_status_color(status)
        header_text = f"[bold]Wagehood System Monitor[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        header_text += f" - Status: [{status_color}]{status.upper()}[/{status_color}]"
        
        layout["header"].update(Panel(header_text, border_style="cyan"))
        
        # Main content
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Left side - Health and Components
        left_content = self._create_health_display(health_data)
        layout["left"].update(left_content)
        
        # Right side - Statistics
        right_content = self._create_stats_display(stats_data)
        layout["right"].update(right_content)
        
        # Footer with controls
        footer_text = "[dim]Press Ctrl+C to stop monitoring[/dim]"
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return layout
    
    def _create_health_display(self, health_data: Dict[str, Any]) -> Panel:
        """Create health status display."""
        table = Table(title="System Health", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Overall status
        status = health_data.get('status', 'unknown')
        status_color = self._get_status_color(status)
        table.add_row("Overall", f"[{status_color}]{status.upper()}[/{status_color}]", "")
        
        # Component statuses
        components = health_data.get('components', {})
        for component, status in components.items():
            status_color = self._get_status_color(status)
            table.add_row(
                component.title(),
                f"[{status_color}]{status.upper()}[/{status_color}]",
                ""
            )
        
        # Uptime
        uptime_seconds = health_data.get('uptime_seconds')
        if uptime_seconds:
            uptime_str = format_duration(uptime_seconds)
            table.add_row("Uptime", uptime_str, "")
        
        return Panel(table, border_style="green")
    
    def _create_stats_display(self, stats_data: Dict[str, Any]) -> Panel:
        """Create statistics display."""
        table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit")
        
        # Performance metrics
        performance = stats_data.get('performance', [])
        for metric in performance:
            component = metric.get('component', 'unknown')
            events_processed = metric.get('events_processed', 0)
            events_per_second = metric.get('events_per_second', 0.0)
            errors = metric.get('errors', 0)
            memory_usage = metric.get('memory_usage_mb', 0.0)
            
            table.add_row(f"{component} - Events", str(events_processed), "total")
            table.add_row(f"{component} - Rate", f"{events_per_second:.2f}", "events/sec")
            table.add_row(f"{component} - Errors", str(errors), "count")
            table.add_row(f"{component} - Memory", f"{memory_usage:.1f}", "MB")
        
        # System metrics
        uptime = stats_data.get('uptime_seconds', 0)
        if uptime:
            table.add_row("System Uptime", format_duration(uptime), "")
        
        running = stats_data.get('running', False)
        table.add_row("System Running", "Yes" if running else "No", "")
        
        return Panel(table, border_style="blue")
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status."""
        status = status.lower()
        if status == 'healthy':
            return 'green'
        elif status == 'degraded':
            return 'yellow'
        elif status == 'unhealthy':
            return 'red'
        else:
            return 'dim'


# Click command group
@click.group(name='monitor')
@click.pass_context
def monitor_commands(ctx):
    """System monitoring and health check commands."""
    pass


@monitor_commands.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed health information')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def health(ctx, detailed, output_format):
    """
    Check system health status.
    
    Examples:
        wagehood_cli.py monitor health
        wagehood_cli.py monitor health --detailed
        wagehood_cli.py monitor health --format json
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = MonitorAPIClient(config)
        
        with formatter.create_progress_bar("Checking system health...") as progress:
            task = progress.add_task("Loading...", total=None)
            
            if detailed:
                data = client.get_detailed_health()
                title = "Detailed System Health"
            else:
                data = client.get_health()
                title = "System Health"
            
            progress.update(task, completed=1)
        
        formatter.print_data(data, title)
        
        # Show status summary
        status = data.get('status', 'unknown')
        if status == 'healthy':
            formatter.print_success("System is healthy!")
        elif status == 'degraded':
            formatter.print_warning("System is degraded")
        elif status == 'unhealthy':
            formatter.print_error("System is unhealthy")
        else:
            formatter.print_info(f"System status: {status}")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@monitor_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def stats(ctx, output_format):
    """
    Get system performance statistics.
    
    Examples:
        wagehood_cli.py monitor stats
        wagehood_cli.py monitor stats --format table
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = MonitorAPIClient(config)
        
        with formatter.create_progress_bar("Fetching system statistics...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_stats()
            progress.update(task, completed=1)
        
        formatter.print_data(data, "System Statistics")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@monitor_commands.command()
@click.option('--limit', '-l', type=int, default=50,
              help='Maximum number of alerts to return')
@click.option('--offset', type=int, default=0,
              help='Number of alerts to skip')
@click.option('--type', 'alert_type',
              type=click.Choice(['error', 'warning', 'info'], case_sensitive=False),
              help='Filter by alert type')
@click.option('--component', help='Filter by component name')
@click.option('--acknowledged/--unacknowledged', default=None,
              help='Filter by acknowledgment status')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def alerts(ctx, limit, offset, alert_type, component, acknowledged, output_format):
    """
    Get system alerts and notifications.
    
    Examples:
        wagehood_cli.py monitor alerts
        wagehood_cli.py monitor alerts --type error --limit 10
        wagehood_cli.py monitor alerts --component ingestion --unacknowledged
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = MonitorAPIClient(config)
        
        with formatter.create_progress_bar("Fetching alerts...") as progress:
            task = progress.add_task("Loading...", total=None)
            data = client.get_alerts(
                limit=limit, 
                offset=offset,
                alert_type=alert_type,
                component=component,
                acknowledged=acknowledged
            )
            progress.update(task, completed=1)
        
        formatter.print_data(data, "System Alerts")
        
        # Show alert summary
        total_count = data.get('total_count', 0)
        unacknowledged_count = data.get('unacknowledged_count', 0)
        
        formatter.print_info(f"Total alerts: {total_count}")
        if unacknowledged_count > 0:
            formatter.print_warning(f"Unacknowledged alerts: {unacknowledged_count}")
        else:
            formatter.print_success("No unacknowledged alerts")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@monitor_commands.command()
@click.option('--interval', '-i', type=int, default=5,
              help='Refresh interval in seconds')
@click.option('--duration', '-d', type=int,
              help='Duration in seconds to monitor')
@click.pass_context
def live(ctx, interval, duration):
    """
    Start real-time system monitoring.
    
    Examples:
        wagehood_cli.py monitor live
        wagehood_cli.py monitor live --interval 10
        wagehood_cli.py monitor live --duration 300
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    formatter.print_info("Starting real-time system monitoring...")
    formatter.print_info("Press Ctrl+C to stop")
    
    try:
        monitor = SystemMonitor(config, formatter)
        monitor.start_monitoring(refresh_interval=interval, duration=duration)
        
    except KeyboardInterrupt:
        formatter.print_info("Monitoring stopped by user")
    except Exception as e:
        formatter.print_error(f"Monitoring error: {e}")
        ctx.exit(1)


@monitor_commands.command()
@click.option('--output-file', '-o', type=click.Path(),
              help='Save report to file')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'csv', 'yaml', 'html'], case_sensitive=False),
              help='Report format')
@click.pass_context
def report(ctx, output_file, output_format):
    """
    Generate comprehensive system health report.
    
    Examples:
        wagehood_cli.py monitor report
        wagehood_cli.py monitor report --output-file health_report.json
        wagehood_cli.py monitor report --format html --output-file report.html
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        client = MonitorAPIClient(config)
        
        with formatter.create_progress_bar("Generating health report...") as progress:
            # Collect all monitoring data
            health_task = progress.add_task("Health data...", total=None)
            health_data = client.get_detailed_health()
            progress.update(health_task, completed=1)
            
            stats_task = progress.add_task("Statistics...", total=None)
            stats_data = client.get_stats()
            progress.update(stats_task, completed=1)
            
            alerts_task = progress.add_task("Alerts...", total=None)
            alerts_data = client.get_alerts(limit=100)
            progress.update(alerts_task, completed=1)
        
        # Compile comprehensive report
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'health': health_data,
            'statistics': stats_data,
            'alerts': alerts_data,
            'summary': {
                'overall_status': health_data.get('status', 'unknown'),
                'uptime_seconds': health_data.get('uptime_seconds', 0),
                'total_alerts': alerts_data.get('total_count', 0),
                'unacknowledged_alerts': alerts_data.get('unacknowledged_count', 0),
                'components_healthy': len([
                    c for c, s in health_data.get('components', {}).items() 
                    if s == 'healthy'
                ]),
                'components_degraded': len([
                    c for c, s in health_data.get('components', {}).items() 
                    if s == 'degraded'
                ]),
                'components_unhealthy': len([
                    c for c, s in health_data.get('components', {}).items() 
                    if s == 'unhealthy'
                ])
            }
        }
        
        if output_file:
            # Save to file
            if output_format == 'html':
                _save_html_report(report_data, output_file)
            else:
                formatter.export_to_file(report_data, output_file, output_format or 'json')
        else:
            # Display on console
            formatter.print_data(report_data, "System Health Report")
        
        formatter.print_success("Health report generated successfully")
        
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"API request failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


@monitor_commands.command()
@click.option('--timeout', type=int, default=30,
              help='Ping timeout in seconds')
@click.pass_context
def ping(ctx, timeout):
    """
    Ping the system to check basic connectivity.
    
    Examples:
        wagehood_cli.py monitor ping
        wagehood_cli.py monitor ping --timeout 60
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    
    try:
        client = MonitorAPIClient(config)
        
        start_time = datetime.now()
        
        with formatter.create_progress_bar("Pinging system...") as progress:
            task = progress.add_task("Connecting...", total=None)
            data = client.get_basic_health()
            progress.update(task, completed=1)
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
        
        formatter.print_success(f"System is reachable!")
        formatter.print_info(f"Response time: {response_time:.2f} ms")
        formatter.print_info(f"Server version: {data.get('version', 'unknown')}")
        formatter.print_info(f"Server time: {data.get('timestamp', 'unknown')}")
        
    except requests.exceptions.Timeout:
        formatter.print_error(f"Ping timeout after {timeout} seconds")
        ctx.exit(1)
    except requests.exceptions.RequestException as e:
        formatter.print_error(f"Ping failed: {e}")
        ctx.exit(1)
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        ctx.exit(1)


def _save_html_report(report_data: Dict[str, Any], filename: str) -> None:
    """Save report as HTML file."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wagehood System Health Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .status-healthy {{ color: green; font-weight: bold; }}
            .status-degraded {{ color: orange; font-weight: bold; }}
            .status-unhealthy {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Wagehood System Health Report</h1>
            <p>Generated: {report_data['generated_at']}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Overall Status: <span class="status-{report_data['summary']['overall_status']}">{report_data['summary']['overall_status'].upper()}</span></p>
            <p>Uptime: {format_duration(report_data['summary']['uptime_seconds'])}</p>
            <p>Total Alerts: {report_data['summary']['total_alerts']}</p>
            <p>Unacknowledged Alerts: {report_data['summary']['unacknowledged_alerts']}</p>
        </div>
        
        <div class="section">
            <h2>Component Health</h2>
            <table>
                <tr><th>Component</th><th>Status</th></tr>
    """
    
    for component, status in report_data['health'].get('components', {}).items():
        html_content += f'<tr><td>{component}</td><td class="status-{status}">{status.upper()}</td></tr>'
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Recent Alerts</h2>
            <table>
                <tr><th>Type</th><th>Component</th><th>Message</th><th>Timestamp</th></tr>
    """
    
    for alert in report_data['alerts'].get('alerts', [])[:10]:  # Show last 10 alerts
        html_content += f"""
        <tr>
            <td>{alert.get('type', '')}</td>
            <td>{alert.get('component', '')}</td>
            <td>{alert.get('message', '')}</td>
            <td>{alert.get('timestamp', '')}</td>
        </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html_content)