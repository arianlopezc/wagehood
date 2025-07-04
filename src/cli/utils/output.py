"""
Output Formatting Utilities

This module provides comprehensive output formatting capabilities for the Wagehood CLI,
supporting multiple formats including JSON, tables, CSV, and YAML with rich formatting.
"""

import json
import csv
import yaml
import sys
from io import StringIO
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from decimal import Decimal

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
import pandas as pd


class OutputFormatter:
    """Comprehensive output formatter for CLI data display."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize output formatter.
        
        Args:
            console: Rich console instance
        """
        self.console = console or Console()
        self.format = "table"
        self.use_color = True
        self.pager = True
        self.max_rows = 100
        
    def set_format(self, format: str) -> None:
        """Set output format."""
        self.format = format.lower()
    
    def set_color(self, use_color: bool) -> None:
        """Set color usage."""
        self.use_color = use_color
    
    def set_pager(self, use_pager: bool) -> None:
        """Set pager usage."""
        self.pager = use_pager
    
    def set_max_rows(self, max_rows: int) -> None:
        """Set maximum rows to display."""
        self.max_rows = max_rows
    
    def print_data(self, data: Any, title: Optional[str] = None, 
                   format: Optional[str] = None) -> None:
        """
        Print data in the specified format.
        
        Args:
            data: Data to print
            title: Optional title for the output
            format: Output format override
        """
        output_format = format or self.format
        
        if output_format == "json":
            self._print_json(data, title)
        elif output_format == "table":
            self._print_table(data, title)
        elif output_format == "csv":
            self._print_csv(data, title)
        elif output_format == "yaml":
            self._print_yaml(data, title)
        else:
            self._print_json(data, title)
    
    def _print_json(self, data: Any, title: Optional[str] = None) -> None:
        """Print data as JSON."""
        if title:
            self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        
        json_str = json.dumps(data, indent=2, default=self._json_serializer)
        
        if self.use_color:
            json_obj = JSON(json_str)
            self.console.print(json_obj)
        else:
            self.console.print(json_str)
    
    def _print_table(self, data: Any, title: Optional[str] = None) -> None:
        """Print data as a table."""
        if isinstance(data, dict):
            self._print_dict_table(data, title)
        elif isinstance(data, list):
            self._print_list_table(data, title)
        else:
            # Fallback to JSON for other types
            self._print_json(data, title)
    
    def _print_dict_table(self, data: Dict, title: Optional[str] = None) -> None:
        """Print dictionary as a table."""
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2, default=self._json_serializer)
                # Truncate long values
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
            else:
                value_str = str(value)
            
            table.add_row(str(key), value_str)
        
        self.console.print(table)
    
    def _print_list_table(self, data: List, title: Optional[str] = None) -> None:
        """Print list as a table."""
        if not data:
            self.console.print(f"[yellow]No data to display[/yellow]")
            return
        
        # Check if list contains dictionaries
        if isinstance(data[0], dict):
            self._print_dict_list_table(data, title)
        else:
            # Simple list
            table = Table(title=title, show_header=True, header_style="bold cyan")
            table.add_column("Index", style="cyan")
            table.add_column("Value", style="green")
            
            for i, item in enumerate(data):
                table.add_row(str(i), str(item))
            
            self.console.print(table)
    
    def _print_dict_list_table(self, data: List[Dict], title: Optional[str] = None) -> None:
        """Print list of dictionaries as a table."""
        if not data:
            self.console.print(f"[yellow]No data to display[/yellow]")
            return
        
        # Get all unique keys from all dictionaries
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        all_keys = sorted(all_keys)
        
        # Create table
        table = Table(title=title, show_header=True, header_style="bold cyan")
        
        for key in all_keys:
            table.add_column(str(key), style="green")
        
        # Add rows
        for item in data:
            if isinstance(item, dict):
                row_data = []
                for key in all_keys:
                    value = item.get(key, "")
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, default=self._json_serializer)
                        # Truncate long values
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                    else:
                        value_str = str(value)
                    row_data.append(value_str)
                
                table.add_row(*row_data)
        
        self.console.print(table)
    
    def _print_csv(self, data: Any, title: Optional[str] = None) -> None:
        """Print data as CSV."""
        if title:
            self.console.print(f"# {title}")
        
        output = StringIO()
        
        if isinstance(data, dict):
            writer = csv.writer(output)
            writer.writerow(["Key", "Value"])
            for key, value in data.items():
                writer.writerow([key, self._serialize_value(value)])
        
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries
            fieldnames = set()
            for item in data:
                if isinstance(item, dict):
                    fieldnames.update(item.keys())
            
            fieldnames = sorted(fieldnames)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                if isinstance(item, dict):
                    serialized_item = {}
                    for key in fieldnames:
                        value = item.get(key, "")
                        serialized_item[key] = self._serialize_value(value)
                    writer.writerow(serialized_item)
        
        elif isinstance(data, list):
            # Simple list
            writer = csv.writer(output)
            writer.writerow(["Index", "Value"])
            for i, item in enumerate(data):
                writer.writerow([i, self._serialize_value(item)])
        
        else:
            # Other types
            writer = csv.writer(output)
            writer.writerow(["Value"])
            writer.writerow([self._serialize_value(data)])
        
        self.console.print(output.getvalue())
    
    def _print_yaml(self, data: Any, title: Optional[str] = None) -> None:
        """Print data as YAML."""
        if title:
            self.console.print(f"# {title}")
        
        yaml_str = yaml.dump(data, default_flow_style=False, indent=2)
        
        if self.use_color:
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
            self.console.print(syntax)
        else:
            self.console.print(yaml_str)
    
    def print_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Print error message."""
        error_text = f"[bold red]Error:[/bold red] {message}"
        if exception:
            error_text += f"\n[dim red]{str(exception)}[/dim red]"
        
        self.console.print(error_text)
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"[bold green]Success:[/bold green] {message}")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[bold blue]Info:[/bold blue] {message}")
    
    def print_status(self, message: str) -> None:
        """Print status message."""
        self.console.print(f"[dim]{message}[/dim]")
    
    def print_header(self, title: str) -> None:
        """Print formatted header."""
        panel = Panel(
            Align.center(f"[bold cyan]{title}[/bold cyan]"),
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def print_separator(self) -> None:
        """Print separator line."""
        self.console.print("─" * self.console.width)
    
    def create_progress_bar(self, description: str = "Processing...") -> Progress:
        """Create a progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        )
    
    def print_tree(self, data: Dict, title: Optional[str] = None) -> None:
        """Print hierarchical data as a tree."""
        tree = Tree(title or "Data", style="bold cyan")
        self._add_tree_nodes(tree, data)
        self.console.print(tree)
    
    def _add_tree_nodes(self, tree: Tree, data: Any) -> None:
        """Recursively add nodes to tree."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    subtree = tree.add(f"[cyan]{key}[/cyan]")
                    self._add_tree_nodes(subtree, value)
                else:
                    tree.add(f"[cyan]{key}[/cyan]: [green]{value}[/green]")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    subtree = tree.add(f"[cyan][{i}][/cyan]")
                    self._add_tree_nodes(subtree, item)
                else:
                    tree.add(f"[cyan][{i}][/cyan]: [green]{item}[/green]")
        else:
            tree.add(f"[green]{data}[/green]")
    
    def print_columns(self, data: List[Any], title: Optional[str] = None) -> None:
        """Print data in columns."""
        if title:
            self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        
        columns = Columns(
            [str(item) for item in data],
            equal=True,
            expand=True
        )
        self.console.print(columns)
    
    def print_statistics(self, data: Dict[str, Any]) -> None:
        """Print statistics in a formatted way."""
        stats_table = Table(title="Statistics", show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        for key, value in data.items():
            stats_table.add_row(key, str(value))
        
        self.console.print(stats_table)
    
    def print_key_value_pairs(self, data: Dict[str, Any], title: Optional[str] = None) -> None:
        """Print key-value pairs in a formatted table."""
        table = Table(title=title, show_header=False, box=None)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            table.add_row(f"{key}:", str(value))
        
        self.console.print(table)
    
    def _json_serializer(self, obj: Any) -> str:
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for CSV output."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=self._json_serializer)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Decimal):
            return str(float(value))
        else:
            return str(value)
    
    def export_to_file(self, data: Any, filename: str, format: str = "json") -> None:
        """
        Export data to file in specified format.
        
        Args:
            data: Data to export
            filename: Output filename
            format: Export format (json, csv, yaml)
        """
        try:
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            
            elif format.lower() == "csv":
                with open(filename, 'w', newline='') as f:
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        fieldnames = set()
                        for item in data:
                            if isinstance(item, dict):
                                fieldnames.update(item.keys())
                        
                        fieldnames = sorted(fieldnames)
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for item in data:
                            if isinstance(item, dict):
                                serialized_item = {}
                                for key in fieldnames:
                                    value = item.get(key, "")
                                    serialized_item[key] = self._serialize_value(value)
                                writer.writerow(serialized_item)
                    else:
                        # Simple data
                        writer = csv.writer(f)
                        if isinstance(data, dict):
                            writer.writerow(["Key", "Value"])
                            for key, value in data.items():
                                writer.writerow([key, self._serialize_value(value)])
                        elif isinstance(data, list):
                            writer.writerow(["Index", "Value"])
                            for i, item in enumerate(data):
                                writer.writerow([i, self._serialize_value(item)])
                        else:
                            writer.writerow(["Value"])
                            writer.writerow([self._serialize_value(data)])
            
            elif format.lower() == "yaml":
                with open(filename, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.print_success(f"Data exported to {filename}")
            
        except Exception as e:
            self.print_error(f"Failed to export data to {filename}", e)
    
    def print_live_data(self, data_generator, title: Optional[str] = None):
        """Print live updating data."""
        with Live(console=self.console, refresh_per_second=4) as live:
            for data in data_generator:
                if title:
                    live.update(Panel(self._format_live_data(data), title=title))
                else:
                    live.update(self._format_live_data(data))
    
    def _format_live_data(self, data: Any) -> str:
        """Format data for live display."""
        if isinstance(data, dict):
            formatted = ""
            for key, value in data.items():
                formatted += f"[cyan]{key}[/cyan]: [green]{value}[/green]\n"
            return formatted
        else:
            return str(data)


class DataFrameFormatter:
    """Formatter specifically for pandas DataFrames."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def print_dataframe(self, df: pd.DataFrame, title: Optional[str] = None, 
                       max_rows: int = 100) -> None:
        """Print pandas DataFrame as a rich table."""
        if title:
            self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        
        # Limit rows if necessary
        if len(df) > max_rows:
            df_display = df.head(max_rows)
            self.console.print(f"[yellow]Showing first {max_rows} of {len(df)} rows[/yellow]")
        else:
            df_display = df
        
        # Create table
        table = Table(show_header=True, header_style="bold cyan")
        
        # Add columns
        for col in df_display.columns:
            table.add_column(str(col), style="green")
        
        # Add rows
        for _, row in df_display.iterrows():
            table.add_row(*[str(value) for value in row])
        
        self.console.print(table)
    
    def print_dataframe_info(self, df: pd.DataFrame) -> None:
        """Print DataFrame information."""
        info_data = {
            "Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "Columns": ", ".join(df.columns.tolist()),
            "Data Types": df.dtypes.value_counts().to_dict()
        }
        
        formatter = OutputFormatter(self.console)
        formatter.print_key_value_pairs(info_data, "DataFrame Info")
    
    def print_dataframe_summary(self, df: pd.DataFrame) -> None:
        """Print DataFrame summary statistics."""
        summary = df.describe()
        self.print_dataframe(summary, "Summary Statistics")


# Utility functions for quick formatting
def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """Format timestamp for display."""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """Format number for display."""
    if isinstance(number, float):
        return f"{number:.{precision}f}"
    return str(number)


def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage for display."""
    return f"{value:.{precision}f}%"


def format_currency(value: float, currency: str = "$", precision: int = 2) -> str:
    """Format currency for display."""
    return f"{currency}{value:.{precision}f}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"