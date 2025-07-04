"""
Bash Completion Support for Wagehood CLI

This module provides comprehensive bash completion functionality for the Wagehood CLI,
including command completion, option completion, and dynamic value completion.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import click


def install_completion(shell: str = "bash") -> None:
    """
    Install shell completion for the CLI.
    
    Args:
        shell: Shell type (bash, zsh, fish)
    """
    script_name = "wagehood_cli.py"
    
    if shell == "bash":
        install_bash_completion(script_name)
    elif shell == "zsh":
        install_zsh_completion(script_name)
    elif shell == "fish":
        install_fish_completion(script_name)
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def install_bash_completion(script_name: str) -> None:
    """Install bash completion."""
    completion_script = generate_bash_completion(script_name)
    
    # Try different completion directories
    completion_dirs = [
        Path.home() / ".bash_completion.d",
        Path("/usr/local/etc/bash_completion.d"),
        Path("/etc/bash_completion.d")
    ]
    
    for comp_dir in completion_dirs:
        if comp_dir.exists() and comp_dir.is_dir():
            comp_file = comp_dir / f"{script_name}_completion.bash"
            try:
                comp_file.write_text(completion_script)
                print(f"Bash completion installed to: {comp_file}")
                print("Restart your shell or run: source ~/.bashrc")
                return
            except PermissionError:
                continue
    
    # Fallback: install to user's bash completion directory
    user_comp_dir = Path.home() / ".bash_completion.d"
    user_comp_dir.mkdir(exist_ok=True)
    comp_file = user_comp_dir / f"{script_name}_completion.bash"
    comp_file.write_text(completion_script)
    
    # Add to .bashrc if not already there
    bashrc = Path.home() / ".bashrc"
    bashrc_content = bashrc.read_text() if bashrc.exists() else ""
    
    completion_line = f"source {comp_file}"
    if completion_line not in bashrc_content:
        with open(bashrc, "a") as f:
            f.write(f"\n# Wagehood CLI completion\n{completion_line}\n")
    
    print(f"Bash completion installed to: {comp_file}")
    print("Restart your shell or run: source ~/.bashrc")


def generate_bash_completion(script_name: str) -> str:
    """Generate bash completion script."""
    return f'''#!/bin/bash

# Bash completion for {script_name}

_{script_name.replace(".", "_")}_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Main commands
    local commands="data config monitor admin interactive help config-wizard"
    
    # Data subcommands
    local data_commands="latest indicators signals stream historical export"
    
    # Config subcommands
    local config_commands="show set reset validate export import-config watchlist indicators strategies system summary validate-remote"
    
    # Monitor subcommands
    local monitor_commands="health stats alerts live report ping"
    
    # Admin subcommands
    local admin_commands="info service cache logs backup maintenance restart shutdown"
    
    # Service subcommands
    local service_commands="start-api stop-api start-realtime stop-realtime status"
    
    # Cache subcommands
    local cache_commands="clear"
    
    # Logs subcommands
    local logs_commands="show"
    
    # Backup subcommands
    local backup_commands="create list restore"
    
    # Maintenance subcommands
    local maintenance_commands="run"
    
    # Watchlist subcommands
    local watchlist_commands="show add remove"
    
    # Export subcommands
    local export_commands="create status download"
    
    # Global options
    local global_opts="--config --api-url --ws-url --output-format --no-color --verbose --quiet --version --help"
    
    # Format options
    local format_opts="json table csv yaml"
    
    # Alert type options
    local alert_types="error warning info"
    
    # Cache type options
    local cache_types="all data config results"
    
    # Backup type options
    local backup_types="full config data"
    
    # Maintenance task options
    local maintenance_tasks="cleanup optimize vacuum reindex"
    
    # Common symbols (can be expanded)
    local symbols="AAPL SPY QQQ TSLA MSFT GOOGL AMZN META NVDA"
    
    case "${{COMP_CWORD}}" in
        1)
            COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
            ;;
        2)
            case "$prev" in
                data)
                    COMPREPLY=($(compgen -W "$data_commands" -- "$cur"))
                    ;;
                config)
                    COMPREPLY=($(compgen -W "$config_commands" -- "$cur"))
                    ;;
                monitor)
                    COMPREPLY=($(compgen -W "$monitor_commands" -- "$cur"))
                    ;;
                admin)
                    COMPREPLY=($(compgen -W "$admin_commands" -- "$cur"))
                    ;;
                --output-format|-f)
                    COMPREPLY=($(compgen -W "$format_opts" -- "$cur"))
                    ;;
                --config|-c)
                    COMPREPLY=($(compgen -f -- "$cur"))
                    ;;
                *)
                    COMPREPLY=($(compgen -W "$global_opts" -- "$cur"))
                    ;;
            esac
            ;;
        3)
            case "${{COMP_WORDS[1]}}" in
                data)
                    case "$prev" in
                        latest|indicators|signals|historical)
                            COMPREPLY=($(compgen -W "$symbols" -- "$cur"))
                            ;;
                        stream)
                            COMPREPLY=($(compgen -W "$symbols" -- "$cur"))
                            ;;
                        export)
                            COMPREPLY=($(compgen -W "$export_commands" -- "$cur"))
                            ;;
                    esac
                    ;;
                config)
                    case "$prev" in
                        watchlist)
                            COMPREPLY=($(compgen -W "$watchlist_commands" -- "$cur"))
                            ;;
                        indicators|strategies|system)
                            COMPREPLY=($(compgen -W "show update" -- "$cur"))
                            ;;
                    esac
                    ;;
                monitor)
                    case "$prev" in
                        alerts)
                            COMPREPLY=($(compgen -W "--type --component --limit --acknowledged --unacknowledged" -- "$cur"))
                            ;;
                    esac
                    ;;
                admin)
                    case "$prev" in
                        service)
                            COMPREPLY=($(compgen -W "$service_commands" -- "$cur"))
                            ;;
                        cache)
                            COMPREPLY=($(compgen -W "$cache_commands" -- "$cur"))
                            ;;
                        logs)
                            COMPREPLY=($(compgen -W "$logs_commands" -- "$cur"))
                            ;;
                        backup)
                            COMPREPLY=($(compgen -W "$backup_commands" -- "$cur"))
                            ;;
                        maintenance)
                            COMPREPLY=($(compgen -W "$maintenance_commands" -- "$cur"))
                            ;;
                    esac
                    ;;
            esac
            ;;
        *)
            # Handle options and values
            case "$prev" in
                --output-format|-f)
                    COMPREPLY=($(compgen -W "$format_opts" -- "$cur"))
                    ;;
                --type)
                    # Context-dependent type completion
                    if [[ "${{COMP_WORDS[@]}}" == *"alerts"* ]]; then
                        COMPREPLY=($(compgen -W "$alert_types" -- "$cur"))
                    elif [[ "${{COMP_WORDS[@]}}" == *"cache"* ]]; then
                        COMPREPLY=($(compgen -W "$cache_types" -- "$cur"))
                    elif [[ "${{COMP_WORDS[@]}}" == *"backup"* ]]; then
                        COMPREPLY=($(compgen -W "$backup_types" -- "$cur"))
                    fi
                    ;;
                --task|-t)
                    if [[ "${{COMP_WORDS[@]}}" == *"maintenance"* ]]; then
                        COMPREPLY=($(compgen -W "$maintenance_tasks" -- "$cur"))
                    fi
                    ;;
                --config|-c|--output-file|-o)
                    COMPREPLY=($(compgen -f -- "$cur"))
                    ;;
                *)
                    # Default to global options
                    COMPREPLY=($(compgen -W "$global_opts" -- "$cur"))
                    ;;
            esac
            ;;
    esac
}}

complete -F _{script_name.replace(".", "_")}_completion {script_name}
'''


def install_zsh_completion(script_name: str) -> None:
    """Install zsh completion."""
    completion_script = generate_zsh_completion(script_name)
    
    # Try different completion directories for zsh
    completion_dirs = [
        Path(os.environ.get("ZSH_CUSTOM", Path.home() / ".oh-my-zsh/custom")) / "completions",
        Path("/usr/local/share/zsh/site-functions"),
        Path.home() / ".zsh" / "completions"
    ]
    
    for comp_dir in completion_dirs:
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_file = comp_dir / f"_{script_name}"
        try:
            comp_file.write_text(completion_script)
            print(f"Zsh completion installed to: {comp_file}")
            print("Restart your shell or run: exec zsh")
            return
        except PermissionError:
            continue
    
    print("Failed to install zsh completion")


def generate_zsh_completion(script_name: str) -> str:
    """Generate zsh completion script."""
    return f'''#compdef {script_name}

# Zsh completion for {script_name}

_wagehood_cli() {{
    local context state line
    typeset -A opt_args
    
    _arguments -C \\
        '(-c --config)'{'{-c,--config}'}'[Configuration file]:file:_files' \\
        '--api-url[API base URL]' \\
        '--ws-url[WebSocket base URL]' \\
        '(-f --output-format)'{'{-f,--output-format}'}'[Output format]:(json table csv yaml)' \\
        '--no-color[Disable colored output]' \\
        '(-v --verbose)'{'{-v,--verbose}'}'[Increase verbosity]' \\
        '(-q --quiet)'{'{-q,--quiet}'}'[Suppress output]' \\
        '--version[Show version]' \\
        '--help[Show help]' \\
        '1: :_wagehood_commands' \\
        '*:: :->args'
    
    case $state in
        args)
            case $words[1] in
                data)
                    _wagehood_data_commands
                    ;;
                config)
                    _wagehood_config_commands
                    ;;
                monitor)
                    _wagehood_monitor_commands
                    ;;
                admin)
                    _wagehood_admin_commands
                    ;;
            esac
            ;;
    esac
}}

_wagehood_commands() {{
    local commands=(
        'data:Data-related commands'
        'config:Configuration management'
        'monitor:System monitoring'
        'admin:Administrative commands'
        'interactive:Interactive mode'
        'help:Show help'
        'config-wizard:Configuration wizard'
    )
    _describe 'commands' commands
}}

_wagehood_data_commands() {{
    local commands=(
        'latest:Get latest data'
        'indicators:Get indicators'
        'signals:Get trading signals'
        'stream:Stream real-time data'
        'historical:Get historical data'
        'export:Data export commands'
    )
    _describe 'data commands' commands
}}

_wagehood_config_commands() {{
    local commands=(
        'show:Show configuration'
        'set:Set configuration values'
        'reset:Reset to defaults'
        'validate:Validate configuration'
        'export:Export configuration'
        'import-config:Import configuration'
        'watchlist:Watchlist management'
        'indicators:Indicator configuration'
        'strategies:Strategy configuration'
        'system:System configuration'
        'summary:Configuration summary'
        'validate-remote:Validate remote config'
    )
    _describe 'config commands' commands
}}

_wagehood_monitor_commands() {{
    local commands=(
        'health:Check system health'
        'stats:Get system statistics'
        'alerts:Get system alerts'
        'live:Real-time monitoring'
        'report:Generate health report'
        'ping:Ping system'
    )
    _describe 'monitor commands' commands
}}

_wagehood_admin_commands() {{
    local commands=(
        'info:System information'
        'service:Service management'
        'cache:Cache management'
        'logs:Log management'
        'backup:Backup operations'
        'maintenance:Maintenance tasks'
        'restart:Restart system'
        'shutdown:Shutdown system'
    )
    _describe 'admin commands' commands
}}

_wagehood_cli "$@"
'''


def install_fish_completion(script_name: str) -> None:
    """Install fish completion."""
    completion_script = generate_fish_completion(script_name)
    
    # Fish completion directory
    comp_dir = Path.home() / ".config" / "fish" / "completions"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    comp_file = comp_dir / f"{script_name}.fish"
    comp_file.write_text(completion_script)
    
    print(f"Fish completion installed to: {comp_file}")
    print("Restart your shell or run: exec fish")


def generate_fish_completion(script_name: str) -> str:
    """Generate fish completion script."""
    return f'''# Fish completion for {script_name}

# Main commands
complete -c {script_name} -n "__fish_use_subcommand" -a "data" -d "Data-related commands"
complete -c {script_name} -n "__fish_use_subcommand" -a "config" -d "Configuration management"
complete -c {script_name} -n "__fish_use_subcommand" -a "monitor" -d "System monitoring"
complete -c {script_name} -n "__fish_use_subcommand" -a "admin" -d "Administrative commands"
complete -c {script_name} -n "__fish_use_subcommand" -a "interactive" -d "Interactive mode"
complete -c {script_name} -n "__fish_use_subcommand" -a "help" -d "Show help"
complete -c {script_name} -n "__fish_use_subcommand" -a "config-wizard" -d "Configuration wizard"

# Global options
complete -c {script_name} -s c -l config -d "Configuration file" -r
complete -c {script_name} -l api-url -d "API base URL" -r
complete -c {script_name} -l ws-url -d "WebSocket base URL" -r
complete -c {script_name} -s f -l output-format -d "Output format" -xa "json table csv yaml"
complete -c {script_name} -l no-color -d "Disable colored output"
complete -c {script_name} -s v -l verbose -d "Increase verbosity"
complete -c {script_name} -s q -l quiet -d "Suppress output"
complete -c {script_name} -l version -d "Show version"
complete -c {script_name} -l help -d "Show help"

# Data subcommands
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "latest" -d "Get latest data"
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "indicators" -d "Get indicators"
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "signals" -d "Get trading signals"
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "stream" -d "Stream real-time data"
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "historical" -d "Get historical data"
complete -c {script_name} -n "__fish_seen_subcommand_from data" -a "export" -d "Data export commands"

# Config subcommands
complete -c {script_name} -n "__fish_seen_subcommand_from config" -a "show" -d "Show configuration"
complete -c {script_name} -n "__fish_seen_subcommand_from config" -a "set" -d "Set configuration values"
complete -c {script_name} -n "__fish_seen_subcommand_from config" -a "reset" -d "Reset to defaults"
complete -c {script_name} -n "__fish_seen_subcommand_from config" -a "validate" -d "Validate configuration"
complete -c {script_name} -n "__fish_seen_subcommand_from config" -a "watchlist" -d "Watchlist management"

# Monitor subcommands
complete -c {script_name} -n "__fish_seen_subcommand_from monitor" -a "health" -d "Check system health"
complete -c {script_name} -n "__fish_seen_subcommand_from monitor" -a "stats" -d "Get system statistics"
complete -c {script_name} -n "__fish_seen_subcommand_from monitor" -a "alerts" -d "Get system alerts"
complete -c {script_name} -n "__fish_seen_subcommand_from monitor" -a "live" -d "Real-time monitoring"

# Admin subcommands
complete -c {script_name} -n "__fish_seen_subcommand_from admin" -a "info" -d "System information"
complete -c {script_name} -n "__fish_seen_subcommand_from admin" -a "service" -d "Service management"
complete -c {script_name} -n "__fish_seen_subcommand_from admin" -a "cache" -d "Cache management"
complete -c {script_name} -n "__fish_seen_subcommand_from admin" -a "backup" -d "Backup operations"

# Common symbols
set -l symbols AAPL SPY QQQ TSLA MSFT GOOGL AMZN META NVDA

# Symbol completion for data commands
complete -c {script_name} -n "__fish_seen_subcommand_from data; and __fish_seen_subcommand_from latest indicators signals historical" -xa "$symbols"
'''


def uninstall_completion(shell: str = "bash") -> None:
    """
    Uninstall shell completion.
    
    Args:
        shell: Shell type (bash, zsh, fish)
    """
    script_name = "wagehood_cli.py"
    
    if shell == "bash":
        uninstall_bash_completion(script_name)
    elif shell == "zsh":
        uninstall_zsh_completion(script_name)
    elif shell == "fish":
        uninstall_fish_completion(script_name)
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def uninstall_bash_completion(script_name: str) -> None:
    """Uninstall bash completion."""
    completion_files = [
        Path.home() / ".bash_completion.d" / f"{script_name}_completion.bash",
        Path("/usr/local/etc/bash_completion.d") / f"{script_name}_completion.bash",
        Path("/etc/bash_completion.d") / f"{script_name}_completion.bash"
    ]
    
    removed = False
    for comp_file in completion_files:
        if comp_file.exists():
            try:
                comp_file.unlink()
                print(f"Removed: {comp_file}")
                removed = True
            except PermissionError:
                print(f"Permission denied: {comp_file}")
    
    if not removed:
        print("No bash completion files found")


def uninstall_zsh_completion(script_name: str) -> None:
    """Uninstall zsh completion."""
    completion_files = [
        Path(os.environ.get("ZSH_CUSTOM", Path.home() / ".oh-my-zsh/custom")) / "completions" / f"_{script_name}",
        Path("/usr/local/share/zsh/site-functions") / f"_{script_name}",
        Path.home() / ".zsh" / "completions" / f"_{script_name}"
    ]
    
    removed = False
    for comp_file in completion_files:
        if comp_file.exists():
            try:
                comp_file.unlink()
                print(f"Removed: {comp_file}")
                removed = True
            except PermissionError:
                print(f"Permission denied: {comp_file}")
    
    if not removed:
        print("No zsh completion files found")


def uninstall_fish_completion(script_name: str) -> None:
    """Uninstall fish completion."""
    comp_file = Path.home() / ".config" / "fish" / "completions" / f"{script_name}.fish"
    
    if comp_file.exists():
        comp_file.unlink()
        print(f"Removed: {comp_file}")
    else:
        print("No fish completion file found")


def generate_completion_script(shell: str, script_name: str) -> str:
    """
    Generate completion script for a specific shell.
    
    Args:
        shell: Shell type (bash, zsh, fish)
        script_name: Name of the CLI script
        
    Returns:
        Completion script content
    """
    if shell == "bash":
        return generate_bash_completion(script_name)
    elif shell == "zsh":
        return generate_zsh_completion(script_name)
    elif shell == "fish":
        return generate_fish_completion(script_name)
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def detect_shell() -> str:
    """
    Detect the current shell.
    
    Returns:
        Shell name (bash, zsh, fish, or unknown)
    """
    shell = os.environ.get("SHELL", "")
    
    if "bash" in shell:
        return "bash"
    elif "zsh" in shell:
        return "zsh"
    elif "fish" in shell:
        return "fish"
    else:
        return "unknown"


def is_completion_installed(shell: str, script_name: str) -> bool:
    """
    Check if completion is already installed for a shell.
    
    Args:
        shell: Shell type
        script_name: CLI script name
        
    Returns:
        True if completion is installed
    """
    if shell == "bash":
        completion_files = [
            Path.home() / ".bash_completion.d" / f"{script_name}_completion.bash",
            Path("/usr/local/etc/bash_completion.d") / f"{script_name}_completion.bash",
            Path("/etc/bash_completion.d") / f"{script_name}_completion.bash"
        ]
        return any(f.exists() for f in completion_files)
    
    elif shell == "zsh":
        completion_files = [
            Path(os.environ.get("ZSH_CUSTOM", Path.home() / ".oh-my-zsh/custom")) / "completions" / f"_{script_name}",
            Path("/usr/local/share/zsh/site-functions") / f"_{script_name}",
            Path.home() / ".zsh" / "completions" / f"_{script_name}"
        ]
        return any(f.exists() for f in completion_files)
    
    elif shell == "fish":
        comp_file = Path.home() / ".config" / "fish" / "completions" / f"{script_name}.fish"
        return comp_file.exists()
    
    return False


def get_completion_status() -> Dict[str, Any]:
    """
    Get completion installation status for all shells.
    
    Returns:
        Dictionary with completion status for each shell
    """
    script_name = "wagehood_cli.py"
    current_shell = detect_shell()
    
    status = {
        "current_shell": current_shell,
        "shells": {}
    }
    
    for shell in ["bash", "zsh", "fish"]:
        status["shells"][shell] = {
            "installed": is_completion_installed(shell, script_name),
            "supported": True
        }
    
    return status