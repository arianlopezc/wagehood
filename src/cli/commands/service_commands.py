"""
Service Management Commands Module

This module provides comprehensive auto-start service management commands 
for the Wagehood CLI, including cross-platform service installation, 
configuration, and management.
"""

import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
try:
    import pwd
    import grp
except ImportError:
    # Windows doesn't have pwd/grp modules
    pwd = None
    grp = None

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation
from ..config import CLIConfig


class ServiceManagerBase:
    """Base class for platform-specific service managers."""
    
    def __init__(self, console: Console):
        """Initialize service manager."""
        self.console = console
        self.logger = CLILogger("service_manager")
        self.project_root = Path.cwd()
        self.service_name = "wagehood"
        self.service_description = "Wagehood Real-time Trading System"
        
    def get_current_user(self) -> str:
        """Get current user name."""
        return os.getenv('USER') or os.getenv('USERNAME') or 'wagehood'
    
    def get_python_executable(self) -> str:
        """Get Python executable path."""
        return sys.executable
    
    def get_service_script_path(self) -> Path:
        """Get path to the service startup script."""
        return self.project_root / "run_realtime.py"
    
    def get_global_command(self) -> List[str]:
        """Get global wagehood command for service execution."""
        return ["wagehood", "admin", "run-realtime"]
    
    def validate_permissions(self) -> Tuple[bool, str]:
        """Validate permissions for service installation."""
        raise NotImplementedError("Subclasses must implement validate_permissions")
    
    def install_service(self) -> Tuple[bool, str]:
        """Install auto-start service."""
        raise NotImplementedError("Subclasses must implement install_service")
    
    def uninstall_service(self) -> Tuple[bool, str]:
        """Uninstall auto-start service."""
        raise NotImplementedError("Subclasses must implement uninstall_service")
    
    def enable_service(self) -> Tuple[bool, str]:
        """Enable auto-start service."""
        raise NotImplementedError("Subclasses must implement enable_service")
    
    def disable_service(self) -> Tuple[bool, str]:
        """Disable auto-start service."""
        raise NotImplementedError("Subclasses must implement disable_service")
    
    def get_service_status(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Get service status and details."""
        raise NotImplementedError("Subclasses must implement get_service_status")
    
    def start_service(self) -> Tuple[bool, str]:
        """Start the service."""
        raise NotImplementedError("Subclasses must implement start_service")
    
    def stop_service(self) -> Tuple[bool, str]:
        """Stop the service."""
        raise NotImplementedError("Subclasses must implement stop_service")
    
    def restart_service(self) -> Tuple[bool, str]:
        """Restart the service."""
        success, msg = self.stop_service()
        if success:
            time.sleep(2)
            return self.start_service()
        return success, msg


class LinuxSystemdManager(ServiceManagerBase):
    """SystemD service manager for Linux."""
    
    def __init__(self, console: Console):
        super().__init__(console)
        self.service_file = f"/etc/systemd/system/{self.service_name}.service"
        self.user_service_dir = Path.home() / ".config/systemd/user"
        self.user_service_file = self.user_service_dir / f"{self.service_name}.service"
        
    def validate_permissions(self) -> Tuple[bool, str]:
        """Check if we have permissions to install system service."""
        if os.geteuid() == 0:
            return True, "Running as root - can install system service"
        
        # Check if user service directory exists or can be created
        try:
            self.user_service_dir.mkdir(parents=True, exist_ok=True)
            return True, "Can install user service"
        except PermissionError:
            return False, "Cannot create user service directory"
    
    def _create_service_content(self, user_service: bool = False) -> str:
        """Create systemd service file content."""
        user = self.get_current_user()
        python_path = self.get_python_executable()
        script_path = self.get_service_script_path()
        working_dir = self.project_root
        global_cmd = self.get_global_command()
        
        env_file = working_dir / ".env"
        
        # Try to use global command first, fallback to script
        try:
            # Check if wagehood command is available
            subprocess.run(["which", "wagehood"], check=True, capture_output=True)
            exec_start = f"{' '.join(global_cmd)} --log-level INFO"
            exec_comment = "# Using global wagehood command"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to direct script execution
            exec_start = f"{python_path} {script_path} --log-level INFO"
            exec_comment = "# Using direct script execution (fallback)"
        
        content = f"""[Unit]
Description={self.service_description}
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
Environment=PATH={os.environ.get('PATH', '')}
Environment=PYTHONPATH={working_dir}/src
EnvironmentFile=-{env_file}
{exec_comment}
ExecStart={exec_start}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier={self.service_name}

# Resource limits
LimitNOFILE=65535
MemoryMax=2G

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={working_dir}

[Install]
WantedBy={'default.target' if user_service else 'multi-user.target'}
"""
        return content
    
    def install_service(self) -> Tuple[bool, str]:
        """Install systemd service."""
        try:
            is_root = os.geteuid() == 0
            use_user_service = not is_root
            
            if use_user_service:
                service_file = self.user_service_file
                self.user_service_dir.mkdir(parents=True, exist_ok=True)
            else:
                service_file = self.service_file
            
            # Create service file content
            content = self._create_service_content(use_user_service)
            
            # Write service file
            with open(service_file, 'w') as f:
                f.write(content)
            
            # Set permissions
            os.chmod(service_file, 0o644)
            
            # Reload systemd
            if use_user_service:
                subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            else:
                subprocess.run(['systemctl', 'daemon-reload'], check=True)
            
            service_type = "user" if use_user_service else "system"
            return True, f"Successfully installed {service_type} service: {service_file}"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to reload systemd: {e}"
        except Exception as e:
            return False, f"Failed to install service: {e}"
    
    def uninstall_service(self) -> Tuple[bool, str]:
        """Uninstall systemd service."""
        try:
            is_root = os.geteuid() == 0
            use_user_service = not is_root
            
            # Stop and disable service first
            self.disable_service()
            self.stop_service()
            
            if use_user_service:
                service_file = self.user_service_file
                if service_file.exists():
                    service_file.unlink()
                subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            else:
                service_file = Path(self.service_file)
                if service_file.exists():
                    service_file.unlink()
                subprocess.run(['systemctl', 'daemon-reload'], check=True)
            
            return True, "Successfully uninstalled service"
            
        except Exception as e:
            return False, f"Failed to uninstall service: {e}"
    
    def enable_service(self) -> Tuple[bool, str]:
        """Enable systemd service."""
        try:
            is_root = os.geteuid() == 0
            
            if is_root:
                subprocess.run(['systemctl', 'enable', self.service_name], check=True)
            else:
                subprocess.run(['systemctl', '--user', 'enable', self.service_name], check=True)
            
            return True, "Service enabled for auto-start"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to enable service: {e}"
    
    def disable_service(self) -> Tuple[bool, str]:
        """Disable systemd service."""
        try:
            is_root = os.geteuid() == 0
            
            if is_root:
                subprocess.run(['systemctl', 'disable', self.service_name], check=True)
            else:
                subprocess.run(['systemctl', '--user', 'disable', self.service_name], check=True)
            
            return True, "Service disabled from auto-start"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to disable service: {e}"
    
    def start_service(self) -> Tuple[bool, str]:
        """Start systemd service."""
        try:
            is_root = os.geteuid() == 0
            
            if is_root:
                subprocess.run(['systemctl', 'start', self.service_name], check=True)
            else:
                subprocess.run(['systemctl', '--user', 'start', self.service_name], check=True)
            
            return True, "Service started successfully"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to start service: {e}"
    
    def stop_service(self) -> Tuple[bool, str]:
        """Stop systemd service."""
        try:
            is_root = os.geteuid() == 0
            
            if is_root:
                subprocess.run(['systemctl', 'stop', self.service_name], check=True)
            else:
                subprocess.run(['systemctl', '--user', 'stop', self.service_name], check=True)
            
            return True, "Service stopped successfully"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to stop service: {e}"
    
    def get_service_status(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Get systemd service status."""
        try:
            is_root = os.geteuid() == 0
            details = {}
            
            # Check if service file exists
            if is_root:
                service_exists = Path(self.service_file).exists()
                service_file = self.service_file
            else:
                service_exists = self.user_service_file.exists()
                service_file = str(self.user_service_file)
            
            if not service_exists:
                return False, "Service not installed", {"service_file": service_file}
            
            # Get service status
            cmd = ['systemctl', '--user' if not is_root else '', 'is-active', self.service_name]
            cmd = [x for x in cmd if x]  # Remove empty strings
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            is_active = result.returncode == 0 and result.stdout.strip() == 'active'
            
            # Get enabled status
            cmd = ['systemctl', '--user' if not is_root else '', 'is-enabled', self.service_name]
            cmd = [x for x in cmd if x]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            is_enabled = result.returncode == 0 and result.stdout.strip() == 'enabled'
            
            details = {
                "service_file": service_file,
                "active": is_active,
                "enabled": is_enabled,
                "service_type": "user" if not is_root else "system"
            }
            
            status_msg = f"Service {'active' if is_active else 'inactive'}, {'enabled' if is_enabled else 'disabled'}"
            return True, status_msg, details
            
        except Exception as e:
            return False, f"Failed to get service status: {e}", {}


class MacOSLaunchdManager(ServiceManagerBase):
    """Launchd service manager for macOS."""
    
    def __init__(self, console: Console):
        super().__init__(console)
        self.plist_name = f"com.wagehood.{self.service_name}"
        self.user_agents_dir = Path.home() / "Library/LaunchAgents"
        self.plist_file = self.user_agents_dir / f"{self.plist_name}.plist"
        
    def validate_permissions(self) -> Tuple[bool, str]:
        """Check permissions for launchd service."""
        try:
            self.user_agents_dir.mkdir(parents=True, exist_ok=True)
            return True, "Can install user launch agent"
        except PermissionError:
            return False, "Cannot create LaunchAgents directory"
    
    def _create_plist_content(self) -> str:
        """Create launchd plist content."""
        user = self.get_current_user()
        python_path = self.get_python_executable()
        script_path = self.get_service_script_path()
        working_dir = self.project_root
        global_cmd = self.get_global_command()
        
        # Load environment variables from .env file
        env_file = working_dir / ".env"
        env_vars = {}
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        env_vars['PATH'] = os.environ.get('PATH', '')
        env_vars['PYTHONPATH'] = str(working_dir / "src")
        
        env_dict_xml = ""
        for key, value in env_vars.items():
            env_dict_xml += f"""        <key>{key}</key>
        <string>{value}</string>
"""
        
        # Try to use global command first, fallback to script
        try:
            # Check if wagehood command is available
            subprocess.run(["which", "wagehood"], check=True, capture_output=True)
            program_args = [
                f"        <string>{global_cmd[0]}</string>",
                f"        <string>{global_cmd[1]}</string>", 
                f"        <string>{global_cmd[2]}</string>",
                "        <string>--log-level</string>",
                "        <string>INFO</string>"
            ]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to direct script execution
            program_args = [
                f"        <string>{python_path}</string>",
                f"        <string>{script_path}</string>",
                "        <string>--log-level</string>",
                "        <string>INFO</string>"
            ]
        
        program_args_xml = "\n".join(program_args)
        
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.plist_name}</string>
    
    <key>ProgramArguments</key>
    <array>
{program_args_xml}
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
{env_dict_xml}    </dict>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    
    <key>ThrottleInterval</key>
    <integer>10</integer>
    
    <key>StandardOutPath</key>
    <string>{working_dir}/logs/wagehood.out.log</string>
    
    <key>StandardErrorPath</key>
    <string>{working_dir}/logs/wagehood.err.log</string>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>Nice</key>
    <integer>0</integer>
</dict>
</plist>
"""
        return content
    
    def install_service(self) -> Tuple[bool, str]:
        """Install launchd plist."""
        try:
            # Create LaunchAgents directory
            self.user_agents_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logs directory
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create plist content
            content = self._create_plist_content()
            
            # Write plist file
            with open(self.plist_file, 'w') as f:
                f.write(content)
            
            # Set permissions
            os.chmod(self.plist_file, 0o644)
            
            return True, f"Successfully installed launch agent: {self.plist_file}"
            
        except Exception as e:
            return False, f"Failed to install launch agent: {e}"
    
    def uninstall_service(self) -> Tuple[bool, str]:
        """Uninstall launchd plist."""
        try:
            # Stop and disable first
            self.disable_service()
            self.stop_service()
            
            # Remove plist file
            if self.plist_file.exists():
                self.plist_file.unlink()
            
            return True, "Successfully uninstalled launch agent"
            
        except Exception as e:
            return False, f"Failed to uninstall launch agent: {e}"
    
    def enable_service(self) -> Tuple[bool, str]:
        """Enable launchd service."""
        try:
            subprocess.run(['launchctl', 'load', str(self.plist_file)], check=True)
            return True, "Launch agent enabled for auto-start"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to enable launch agent: {e}"
    
    def disable_service(self) -> Tuple[bool, str]:
        """Disable launchd service."""
        try:
            subprocess.run(['launchctl', 'unload', str(self.plist_file)], check=False)
            return True, "Launch agent disabled from auto-start"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to disable launch agent: {e}"
    
    def start_service(self) -> Tuple[bool, str]:
        """Start launchd service."""
        try:
            subprocess.run(['launchctl', 'start', self.plist_name], check=True)
            return True, "Launch agent started successfully"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to start launch agent: {e}"
    
    def stop_service(self) -> Tuple[bool, str]:
        """Stop launchd service."""
        try:
            subprocess.run(['launchctl', 'stop', self.plist_name], check=True)
            return True, "Launch agent stopped successfully"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed to stop launch agent: {e}"
    
    def get_service_status(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Get launchd service status."""
        try:
            if not self.plist_file.exists():
                return False, "Launch agent not installed", {"plist_file": str(self.plist_file)}
            
            # Check if service is loaded
            result = subprocess.run(['launchctl', 'list'], capture_output=True, text=True)
            is_loaded = self.plist_name in result.stdout
            
            details = {
                "plist_file": str(self.plist_file),
                "loaded": is_loaded,
                "service_type": "user"
            }
            
            status_msg = f"Launch agent {'loaded' if is_loaded else 'not loaded'}"
            return True, status_msg, details
            
        except Exception as e:
            return False, f"Failed to get launch agent status: {e}", {}


class WindowsServiceManager(ServiceManagerBase):
    """Windows service manager."""
    
    def __init__(self, console: Console):
        super().__init__(console)
        self.startup_folder = Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"
        self.startup_script = self.startup_folder / f"{self.service_name}_startup.bat"
        
    def validate_permissions(self) -> Tuple[bool, str]:
        """Check permissions for Windows startup."""
        try:
            self.startup_folder.mkdir(parents=True, exist_ok=True)
            return True, "Can install startup script"
        except PermissionError:
            return False, "Cannot access startup folder"
    
    def _create_startup_script(self) -> str:
        """Create Windows startup script."""
        python_path = self.get_python_executable()
        script_path = self.get_service_script_path()
        working_dir = self.project_root
        global_cmd = self.get_global_command()
        
        # Try to use global command first, fallback to script
        try:
            # Check if wagehood command is available
            subprocess.run(["where", "wagehood"], check=True, capture_output=True, shell=True)
            exec_command = f"{' '.join(global_cmd)} --log-level INFO"
            comment = "REM Using global wagehood command"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to direct script execution
            exec_command = f'"{python_path}" "{script_path}" --log-level INFO'
            comment = "REM Using direct script execution (fallback)"
        
        content = f"""@echo off
{comment}
cd /d "{working_dir}"
{exec_command}
pause
"""
        return content
    
    def install_service(self) -> Tuple[bool, str]:
        """Install Windows startup script."""
        try:
            # Create startup folder
            self.startup_folder.mkdir(parents=True, exist_ok=True)
            
            # Create startup script
            content = self._create_startup_script()
            
            # Write script file
            with open(self.startup_script, 'w') as f:
                f.write(content)
            
            return True, f"Successfully installed startup script: {self.startup_script}"
            
        except Exception as e:
            return False, f"Failed to install startup script: {e}"
    
    def uninstall_service(self) -> Tuple[bool, str]:
        """Uninstall Windows startup script."""
        try:
            if self.startup_script.exists():
                self.startup_script.unlink()
            
            return True, "Successfully uninstalled startup script"
            
        except Exception as e:
            return False, f"Failed to uninstall startup script: {e}"
    
    def enable_service(self) -> Tuple[bool, str]:
        """Enable is same as install for Windows."""
        return self.install_service()
    
    def disable_service(self) -> Tuple[bool, str]:
        """Disable is same as uninstall for Windows."""
        return self.uninstall_service()
    
    def start_service(self) -> Tuple[bool, str]:
        """Start service manually on Windows."""
        try:
            global_cmd = self.get_global_command()
            python_path = self.get_python_executable()
            script_path = self.get_service_script_path()
            
            # Try global command first, fallback to script
            try:
                subprocess.run(["where", "wagehood"], check=True, capture_output=True, shell=True)
                subprocess.Popen(global_cmd, cwd=self.project_root)
                return True, "Service started manually using global command"
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.Popen([python_path, str(script_path)], cwd=self.project_root)
                return True, "Service started manually using script"
            
        except Exception as e:
            return False, f"Failed to start service: {e}"
    
    def stop_service(self) -> Tuple[bool, str]:
        """Stop service on Windows."""
        try:
            # Try to kill Python processes running the script
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], check=False)
            return True, "Attempted to stop service"
            
        except Exception as e:
            return False, f"Failed to stop service: {e}"
    
    def get_service_status(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Get Windows service status."""
        try:
            script_exists = self.startup_script.exists()
            
            details = {
                "startup_script": str(self.startup_script),
                "installed": script_exists,
                "service_type": "startup_script"
            }
            
            status_msg = f"Startup script {'installed' if script_exists else 'not installed'}"
            return True, status_msg, details
            
        except Exception as e:
            return False, f"Failed to get service status: {e}", {}


def get_service_manager(console: Console) -> ServiceManagerBase:
    """Get platform-appropriate service manager."""
    system = platform.system().lower()
    
    if system == 'linux':
        return LinuxSystemdManager(console)
    elif system == 'darwin':
        return MacOSLaunchdManager(console)
    elif system == 'windows':
        return WindowsServiceManager(console)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# Click command group
@click.group(name='service')
def service_commands():
    """Service management commands for auto-start configuration."""
    pass


@service_commands.command()
@click.option('--force', is_flag=True, help='Force reinstallation if service exists')
@click.pass_context
def install(ctx, force):
    """
    Install auto-start service for the Wagehood system.
    
    This command creates platform-specific service configurations:
    - Linux: systemd service files
    - macOS: launchd plist files
    - Windows: startup scripts
    
    Examples:
        wagehood service install
        wagehood service install --force
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Installation"):
        
        console.print(Panel.fit(
            f"[bold green]Installing Auto-Start Service[/bold green]\n"
            f"Platform: {platform.system()}",
            title="🔧 Service Installer"
        ))
        
        # Check permissions
        has_perms, perm_msg = manager.validate_permissions()
        if not has_perms:
            formatter.print_error(f"Permission error: {perm_msg}")
            return
        
        console.print(f"✅ {perm_msg}")
        
        # Check if service already exists
        service_exists, status_msg, details = manager.get_service_status()
        if service_exists and not force:
            console.print(f"⚠️  Service already installed: {status_msg}")
            if not Confirm.ask("Reinstall service?"):
                return
        
        # Install service
        with console.status("[spinner]Installing service..."):
            success, msg = manager.install_service()
        
        if success:
            console.print(f"✅ {msg}")
            
            # Enable by default
            enable_success, enable_msg = manager.enable_service()
            if enable_success:
                console.print(f"✅ {enable_msg}")
            else:
                console.print(f"⚠️  {enable_msg}")
            
            console.print(Panel.fit(
                "[bold green]Service Installation Complete![/bold green]\n\n"
                "The Wagehood system will now start automatically when your computer boots.\n\n"
                "Next steps:\n"
                "• Check status: [cyan]wagehood service status[/cyan]\n"
                "• Start now: [cyan]wagehood service start[/cyan]\n"
                "• View logs: Check system logs for service output",
                title="✅ Success"
            ))
        else:
            formatter.print_error(f"Installation failed: {msg}")


@service_commands.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def uninstall(ctx, confirm):
    """
    Uninstall auto-start service.
    
    Examples:
        wagehood service uninstall
        wagehood service uninstall --confirm
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Uninstallation"):
        
        if not confirm:
            if not Confirm.ask("Are you sure you want to uninstall the auto-start service?"):
                formatter.print_info("Uninstallation cancelled")
                return
        
        console.print(Panel.fit(
            "[bold yellow]Uninstalling Auto-Start Service[/bold yellow]",
            title="🗑️  Service Uninstaller"
        ))
        
        # Uninstall service
        with console.status("[spinner]Uninstalling service..."):
            success, msg = manager.uninstall_service()
        
        if success:
            console.print(f"✅ {msg}")
            console.print("The Wagehood system will no longer start automatically.")
        else:
            formatter.print_error(f"Uninstallation failed: {msg}")


@service_commands.command()
@click.pass_context
def enable(ctx):
    """
    Enable auto-start service.
    
    Examples:
        wagehood service enable
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Enable"):
        
        with console.status("[spinner]Enabling service..."):
            success, msg = manager.enable_service()
        
        if success:
            console.print(f"✅ {msg}")
        else:
            formatter.print_error(f"Enable failed: {msg}")


@service_commands.command()
@click.pass_context
def disable(ctx):
    """
    Disable auto-start service.
    
    Examples:
        wagehood service disable
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Disable"):
        
        with console.status("[spinner]Disabling service..."):
            success, msg = manager.disable_service()
        
        if success:
            console.print(f"✅ {msg}")
        else:
            formatter.print_error(f"Disable failed: {msg}")


@service_commands.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['json', 'table', 'yaml'], case_sensitive=False),
              help='Override output format')
@click.pass_context
def status(ctx, output_format):
    """
    Check auto-start service status.
    
    Examples:
        wagehood service status
        wagehood service status --format json
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    if output_format:
        formatter.set_format(output_format)
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Status Check"):
        
        console.print(Panel.fit(
            "[bold blue]Auto-Start Service Status[/bold blue]",
            title="📊 Status Check"
        ))
        
        # Get service status
        service_exists, status_msg, details = manager.get_service_status()
        
        # Create status table
        status_table = Table(show_header=True, header_style="bold magenta")
        status_table.add_column("Property", style="dim")
        status_table.add_column("Value", justify="left")
        
        status_table.add_row("Platform", platform.system())
        status_table.add_row("Service Type", details.get("service_type", "unknown"))
        
        if service_exists:
            status_table.add_row("Status", Text("✅ Installed", style="green"))
            status_table.add_row("Details", status_msg)
            
            # Add platform-specific details
            if "service_file" in details:
                status_table.add_row("Service File", details["service_file"])
            if "plist_file" in details:
                status_table.add_row("Plist File", details["plist_file"])
            if "startup_script" in details:
                status_table.add_row("Startup Script", details["startup_script"])
            
            if "active" in details:
                active_text = "✅ Active" if details["active"] else "❌ Inactive"
                status_table.add_row("Active", Text(active_text, style="green" if details["active"] else "red"))
            
            if "enabled" in details:
                enabled_text = "✅ Enabled" if details["enabled"] else "❌ Disabled"
                status_table.add_row("Auto-start", Text(enabled_text, style="green" if details["enabled"] else "red"))
            
            if "loaded" in details:
                loaded_text = "✅ Loaded" if details["loaded"] else "❌ Not Loaded"
                status_table.add_row("Loaded", Text(loaded_text, style="green" if details["loaded"] else "red"))
            
        else:
            status_table.add_row("Status", Text("❌ Not Installed", style="red"))
            status_table.add_row("Details", status_msg)
        
        console.print(status_table)
        
        # Show available commands
        if service_exists:
            console.print("\n[bold]Available Commands:[/bold]")
            console.print("• [cyan]wagehood service start[/cyan] - Start the service")
            console.print("• [cyan]wagehood service stop[/cyan] - Stop the service")
            console.print("• [cyan]wagehood service restart[/cyan] - Restart the service")
            console.print("• [cyan]wagehood service disable[/cyan] - Disable auto-start")
            console.print("• [cyan]wagehood service uninstall[/cyan] - Remove service")
        else:
            console.print("\n[bold]Next Steps:[/bold]")
            console.print("• [cyan]wagehood service install[/cyan] - Install auto-start service")


@service_commands.command()
@click.pass_context
def start(ctx):
    """
    Start the auto-start service.
    
    Examples:
        wagehood service start
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Start"):
        
        with console.status("[spinner]Starting service..."):
            success, msg = manager.start_service()
        
        if success:
            console.print(f"✅ {msg}")
        else:
            formatter.print_error(f"Start failed: {msg}")


@service_commands.command()
@click.pass_context
def stop(ctx):
    """
    Stop the auto-start service.
    
    Examples:
        wagehood service stop
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Stop"):
        
        with console.status("[spinner]Stopping service..."):
            success, msg = manager.stop_service()
        
        if success:
            console.print(f"✅ {msg}")
        else:
            formatter.print_error(f"Stop failed: {msg}")


@service_commands.command()
@click.pass_context
def restart(ctx):
    """
    Restart the auto-start service.
    
    Examples:
        wagehood service restart
    """
    console = ctx.obj['console']
    formatter = ctx.obj['formatter']
    
    try:
        manager = get_service_manager(console)
    except RuntimeError as e:
        formatter.print_error(str(e))
        return
    
    with log_operation(manager.logger, "Service Restart"):
        
        console.print("🔄 Restarting service...")
        
        # Stop first
        with console.status("[spinner]Stopping service..."):
            stop_success, stop_msg = manager.stop_service()
        
        if stop_success:
            console.print(f"✅ {stop_msg}")
        else:
            console.print(f"⚠️  {stop_msg}")
        
        # Wait a moment
        time.sleep(2)
        
        # Start again
        with console.status("[spinner]Starting service..."):
            start_success, start_msg = manager.start_service()
        
        if start_success:
            console.print(f"✅ {start_msg}")
        else:
            formatter.print_error(f"Start failed: {start_msg}")


# Export the command group
service = service_commands