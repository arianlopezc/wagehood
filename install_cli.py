#!/usr/bin/env python3
"""Install script for Wagehood CLI."""

import os
import sys
import subprocess
from pathlib import Path
import platform


def create_cli_wrapper():
    """Create CLI wrapper script."""
    project_root = Path(__file__).parent.absolute()
    
    wrapper_content = f"""#!/usr/bin/env python3
# Wagehood CLI wrapper
import sys
sys.path.insert(0, '{project_root}')

from src.cli.main import main

if __name__ == '__main__':
    main()
"""
    return wrapper_content


def check_and_install_dependencies():
    """Check and install all required dependencies."""
    print("\n📦 Checking dependencies...")
    
    # Read requirements from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("⚠️  requirements.txt not found, installing essential dependencies...")
        required_packages = [
            "click>=8.1.0",
            "alpaca-trade-api>=3.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "python-dotenv>=1.0.0"
        ]
    else:
        with open(requirements_file, 'r') as f:
            required_packages = [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#')
            ]
    
    # Check which packages need to be installed
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        # Extract package name from requirement string
        package_name = package.split('>=')[0].split('==')[0].split('[')[0]
        import_name = package_name.replace('-', '_')
        
        # Special cases for import names
        import_map = {
            'alpaca_trade_api': 'alpaca_trade_api',
            'python_dotenv': 'dotenv',
            'pytest_asyncio': 'pytest_asyncio',
            'ta_lib': 'talib'
        }
        
        if import_name in import_map:
            import_name = import_map[import_name]
        
        try:
            __import__(import_name)
            installed_packages.append(package_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"   - {package}")
        
        try:
            # Check if we're on macOS with Homebrew Python
            system = platform.system()
            pip_args = []
            
            if system == "Darwin":  # macOS
                # Use --break-system-packages for Homebrew Python on macOS
                pip_args = ["--break-system-packages", "--user"]
            else:
                # Use --user for other systems
                pip_args = ["--user"]
            
            # Update pip first
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"] + pip_args,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                # If pip upgrade fails, continue anyway
                pass
            
            # Install missing packages
            install_cmd = [sys.executable, "-m", "pip", "install"] + pip_args + missing_packages
            
            result = subprocess.run(
                install_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ All dependencies installed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install some dependencies")
            if e.stderr:
                print(f"Error: {e.stderr}")
            print("\nYou may need to install these manually:")
            for package in missing_packages:
                print(f"   pip install --user {package}")
            
            # Continue anyway for packages that did install
            print("\n⚠️  Continuing with installation...")
    else:
        print("✅ All Python dependencies are already installed!")
    
    # Special check for TA-Lib
    try:
        import talib
        print("✅ TA-Lib is installed")
    except ImportError:
        print("\n⚠️  TA-Lib is not installed.")
        print("TA-Lib requires system libraries. Please install:")
        system = platform.system()
        if system == "Darwin":  # macOS
            print("   brew install ta-lib")
            print("   pip install ta-lib")
        elif system == "Linux":
            print("   sudo apt-get install ta-lib")
            print("   pip install ta-lib")
        print("\nNote: Some strategies require TA-Lib to function properly.")


def install_cli():
    """Install Wagehood CLI globally."""
    print("🚀 Wagehood CLI Installer")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        sys.exit(1)
    
    # Check and install all dependencies
    check_and_install_dependencies()
    
    # Determine install location based on OS
    system = platform.system()
    if system == "Darwin":  # macOS
        bin_dir = Path.home() / ".local" / "bin"
    elif system == "Linux":
        bin_dir = Path.home() / ".local" / "bin"
    else:
        print(f"❌ Unsupported system: {system}")
        sys.exit(1)
    
    # Create bin directory if it doesn't exist
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CLI wrapper
    cli_path = bin_dir / "wagehood"
    print(f"\n📝 Creating CLI wrapper at {cli_path}")
    
    wrapper_content = create_cli_wrapper()
    cli_path.write_text(wrapper_content)
    cli_path.chmod(0o755)
    
    # Check if bin directory is in PATH and add it automatically
    if str(bin_dir) not in os.environ.get("PATH", ""):
        print(f"\n🔧 Adding {bin_dir} to your PATH...")
        
        shell = os.environ.get("SHELL", "").split("/")[-1]
        path_added = False
        
        if shell == "zsh":
            zshrc = Path.home() / ".zshrc"
            # Check if PATH export already exists (check multiple patterns)
            if zshrc.exists():
                with open(zshrc, 'r') as f:
                    content = f.read()
                    # Check for various PATH patterns
                    path_patterns = [
                        f'PATH=\\$PATH:{bin_dir}',
                        f'PATH=$PATH:{bin_dir}',
                        f'PATH="$PATH:{bin_dir}"',
                        'PATH="$PATH:$HOME/.local/bin"',
                        'PATH=$PATH:$HOME/.local/bin',
                        '.local/bin'  # Generic check for .local/bin
                    ]
                    if any(pattern in content for pattern in path_patterns):
                        print("   PATH already configured in ~/.zshrc")
                        path_added = True
            
            if not path_added:
                with open(zshrc, 'a') as f:
                    f.write(f'\n# Added by Wagehood installer\nexport PATH="$PATH:{bin_dir}"\n')
                print("   ✅ Added PATH to ~/.zshrc")
                print("\n   📋 To use 'wagehood' command:")
                print(f'   • In this terminal: source ~/.zshrc')
                print(f'   • In new terminals: wagehood will work automatically')
                path_added = True
                
        elif shell == "bash":
            bashrc = Path.home() / ".bashrc"
            # Check if PATH export already exists (check multiple patterns)
            if bashrc.exists():
                with open(bashrc, 'r') as f:
                    content = f.read()
                    # Check for various PATH patterns
                    path_patterns = [
                        f'PATH=\\$PATH:{bin_dir}',
                        f'PATH=$PATH:{bin_dir}',
                        f'PATH="$PATH:{bin_dir}"',
                        'PATH="$PATH:$HOME/.local/bin"',
                        'PATH=$PATH:$HOME/.local/bin',
                        '.local/bin'  # Generic check for .local/bin
                    ]
                    if any(pattern in content for pattern in path_patterns):
                        print("   PATH already configured in ~/.bashrc")
                        path_added = True
            
            if not path_added:
                with open(bashrc, 'a') as f:
                    f.write(f'\n# Added by Wagehood installer\nexport PATH="$PATH:{bin_dir}"\n')
                print("   ✅ Added PATH to ~/.bashrc")
                print("\n   📋 To use 'wagehood' command:")
                print(f'   • In this terminal: source ~/.bashrc')
                print(f'   • In new terminals: wagehood will work automatically')
                path_added = True
        else:
            print(f"   ⚠️  Unknown shell: {shell}")
            print(f'   Manually add to PATH: export PATH=$PATH:{bin_dir}')
    else:
        print(f"\n✅ {bin_dir} is already in PATH")
    
    print("\n✅ CLI installed successfully!")
    print("\n🔧 Now configure the CLI:")
    print("   wagehood configure")
    
    # Check if we're in an interactive terminal
    if sys.stdin.isatty():
        # Ask if user wants to configure now
        configure_now = input("\nWould you like to configure now? (y/n): ").lower().strip()
        if configure_now == 'y':
            # Import and run configuration
            sys.path.insert(0, str(Path(__file__).parent))
            from src.cli.main import cli
            from src.cli.config import ConfigManager
            
            config_manager = ConfigManager()
            
            print("\n🔧 Wagehood Configuration")
            print("=" * 40)
            
            # Get API credentials
            api_key = input("Enter your Alpaca API key: ").strip()
            secret_key = input("Enter your Alpaca secret key: ").strip()
            
            # Get symbols
            symbols_input = input("Enter supported symbols (comma-separated) [AAPL,GOOGL,MSFT,AMZN,TSLA]: ").strip()
            if not symbols_input:
                symbols_input = "AAPL,GOOGL,MSFT,AMZN,TSLA"
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            # Save configuration
            config_manager.update({
                'api_key': api_key,
                'secret_key': secret_key,
                'symbols': symbols
            })
            
            # DO NOT modify .env file - it should be manually maintained
            # setup_env_file(api_key, secret_key, symbols)  # REMOVED: Don't modify .env
            
            # Save environment variables to shell configs from existing .env file
            setup_env_vars_in_shell()
            
            print(f"\n✅ Configuration saved!")
            print(f"📊 Tracking {len(symbols)} symbols: {', '.join(symbols[:5])}" + 
                  (" ..." if len(symbols) > 5 else ""))
            
            # Check Discord webhook configuration from .env file
            print("\n🔔 Discord Notification Configuration Check")
            print("Checking Discord webhook URLs from .env file...")
            
            # Read existing Discord webhooks from environment (already loaded from .env)
            webhook_checks = [
                ('DISCORD_WEBHOOK_INFRA', 'Infrastructure/service notifications'),
                ('DISCORD_WEBHOOK_MACD_RSI', 'MACD-RSI signals (1d timeframe)'),
                ('DISCORD_WEBHOOK_SUPPORT_RESISTANCE', 'Support/Resistance signals (1d timeframe)'),
                ('DISCORD_WEBHOOK_RSI_TREND', 'RSI Trend signals (1h timeframe)'),
                ('DISCORD_WEBHOOK_BOLLINGER', 'Bollinger Band signals (1h timeframe)'),
                ('DISCORD_WEBHOOK_EOD_SUMMARY', 'End-of-Day Summary (daily 5PM ET)')
            ]
            
            configured_webhooks = 0
            for env_var, description in webhook_checks:
                if os.getenv(env_var):
                    print(f"  ✅ {description}: Configured")
                    configured_webhooks += 1
                else:
                    print(f"  ❌ {description}: Not configured")
            
            if configured_webhooks == 0:
                print("\n⚠️  No Discord webhooks configured in .env file")
                print("   To configure Discord notifications, manually edit the .env file")
            else:
                print(f"\n✅ {configured_webhooks} Discord webhooks configured")
            
            # Ask which services to set up
            print("\n🚀 Service Configuration")
            print("Available services:")
            print("  1. Trigger analysis cron jobs (every 10 seconds)")
            print("  2. Notification workers (Discord notifications)")
            print("  3. Daily summary schedule (5PM ET cron job)")
            print("  4. All services (recommended)")
            
            service_choice = input("Which services would you like to set up? (1/2/3/4): ").strip()
            
            # Set up summary schedule if requested
            if service_choice in ["3", "4"]:
                if os.getenv('DISCORD_WEBHOOK_EOD_SUMMARY'):
                    setup_summary_schedule = input("\nSet up daily summary schedule (5PM ET)? (y/n): ").lower().strip()
                    if setup_summary_schedule == 'y':
                        setup_summary_cron()
                else:
                    print("⚠️  Cannot set up summary schedule without EOD_SUMMARY Discord webhook")
            
            # Set up cron jobs if requested
            setup_cron_now = input("\nSet up trigger cron jobs now? (y/n): ").lower().strip()
            if setup_cron_now == 'y':
                print("\n🚀 Setting up cron jobs...")
                setup_cron_jobs_immediately(service_choice)
            
            # Ask about auto-start (only for notification workers)
            if service_choice in ["2", "4"]:
                setup_autostart = input("\nSet up notification workers to start automatically on boot? (y/n): ").lower().strip()
                if setup_autostart == 'y':
                    setup_macos_autostart(service_choice)
    
    print("\n🎉 Installation complete!")
    print("\n⚠️  IMPORTANT: Discord webhook URLs must be configured manually in the .env file")
    print("   Edit .env and add your Discord webhook URLs for notifications to work")
    print("\nAvailable commands:")
    print("  📊 Analysis:")
    print("    wagehood submit            - Run backtesting analysis")
    print("    wagehood summary generate  - Generate today's signal summary")
    print("    wagehood test              - Run integration tests")
    print("  🔧 Configuration:")
    print("    wagehood configure         - Configure API credentials")
    print("    wagehood symbols list      - List tracked symbols")
    print("    wagehood symbols add/remove - Manage symbols")
    print("  📅 Scheduling:")
    print("    wagehood schedule setup    - Set up daily 5PM ET summary")
    print("    wagehood schedule status   - Check summary schedule")
    print("    wagehood schedule remove   - Remove summary schedule")
    print("  ⚡ Trigger Analysis:")
    print("    wagehood triggers test-1h  - Test 1-hour analysis trigger manually")
    print("    wagehood triggers test-1d  - Test 1-day analysis trigger manually") 
    print("    wagehood triggers status   - Check trigger system status")
    print("  🔔 Notifications:")
    print("    wagehood notifications config  - Show Discord configuration")
    print("    wagehood notifications status  - Check notification service status")
    print("    wagehood notifications test    - Test Discord connectivity")


def setup_cron_jobs_immediately(service_choice: str = "4"):
    """Set up cron jobs immediately after configuration."""
    project_root = Path(__file__).parent.absolute()
    
    # Create .wagehood directory if it doesn't exist
    (Path.home() / '.wagehood').mkdir(exist_ok=True)
    
    services_setup = []
    
    # Set up trigger cron jobs
    if service_choice in ["1", "4"]:
        cron_setup_script = project_root / "setup_cron_jobs.py"
        
        if cron_setup_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(cron_setup_script), "setup"],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    services_setup.append("Trigger analysis cron jobs")
                    print("✅ Trigger analysis cron jobs configured successfully")
                else:
                    print(f"❌ Failed to set up trigger cron jobs: {result.stderr}")
            except Exception as e:
                print(f"❌ Failed to set up trigger cron jobs: {str(e)}")
        else:
            print(f"❌ Cron setup script not found at {cron_setup_script}")
    
    # Start notification workers
    if service_choice in ["2", "4"]:
        notification_worker_script = project_root / "start_notification_workers.py"
        notification_pid_file = Path.home() / '.wagehood' / 'notification_workers.pid'
        
        if notification_worker_script.exists() and not notification_pid_file.exists():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(notification_worker_script)],
                    cwd=project_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                notification_pid_file.write_text(str(process.pid))
                services_setup.append(f"Notification workers (PID: {process.pid})")
            except Exception as e:
                print(f"❌ Failed to start notification workers: {str(e)}")
        elif notification_pid_file.exists():
            print("⚠️  Notification workers may already be running.")
    
    if services_setup:
        print(f"✅ Configured: {', '.join(services_setup)}")
    else:
        print("⚠️  No services were configured.")


def setup_summary_cron():
    """Set up the daily summary cron job."""
    print("\n📅 Setting up daily summary schedule...")
    
    try:
        project_root = Path(__file__).parent.absolute()
        scheduled_script = project_root / "run_scheduled_summary.py"
        
        if not scheduled_script.exists():
            print(f"❌ Scheduled script not found at {scheduled_script}")
            return False
        
        # Create cron job for 5pm ET daily
        cron_command = f"0 17 * * * cd {project_root} && {sys.executable} {scheduled_script}"
        
        # Check if cron job already exists
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        if 'run_scheduled_summary.py' in current_crontab:
            print("⚠️  A scheduled summary job already exists - updating...")
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
            print("✅ Daily summary schedule configured successfully!")
            print("📊 The summary will run daily at 5:00 PM ET and send results to Discord.")
            return True
        else:
            print("❌ Failed to set up cron job.")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up summary schedule: {str(e)}")
        return False



def setup_macos_autostart(service_choice: str = "4"):
    """Set up auto-start on macOS using launchd (only for notification workers)."""
    project_root = Path(__file__).parent.absolute()
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(exist_ok=True)
    
    print(f"\n🔧 Setting up notification workers auto-start...")
    services_created = []
    
    # Create notification workers auto-start
    if service_choice in ["2", "4"]:
        notification_plist_file = plist_dir / "com.wagehood.notifications.plist"
        notification_plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wagehood.notifications</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{project_root}/start_notification_workers.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_root}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.wagehood/notifications.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.wagehood/notifications_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>"""
        
        notification_plist_file.write_text(notification_plist_content)
        
        try:
            subprocess.run(["launchctl", "load", str(notification_plist_file)], check=True)
            services_created.append("Notification workers")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not load notification workers auto-start service")
    
    if services_created:
        print(f"✅ {', '.join(services_created)} will start automatically on boot")
        print("ℹ️  Note: Trigger analysis is managed by cron jobs (already configured)")
    else:
        print("⚠️  No auto-start services were configured")
        print("⚠️  You can start notification workers manually with: wagehood notifications start")


def setup_env_file(api_key: str, secret_key: str, symbols: list):
    """DEPRECATED: DO NOT USE - .env file should be manually maintained.
    
    This function would overwrite the .env file which can lose Discord webhook
    configurations and other settings. The .env file should only be edited manually.
    """
    # This function is intentionally left empty to prevent accidental usage
    raise NotImplementedError("The .env file should not be modified by the installer. Edit it manually.")


def update_env_file_with_webhooks(discord_webhooks: dict):
    """DEPRECATED: DO NOT USE - .env file should be manually maintained.
    
    This function would modify the .env file which can overwrite existing
    Discord webhook configurations. The .env file should only be edited manually.
    """
    # This function is intentionally left empty to prevent accidental usage
    raise NotImplementedError("The .env file should not be modified by the installer. Edit it manually.")


def setup_env_vars_in_shell():
    """Set up environment variables in shell configuration files.
    
    This function reads from the existing .env file and copies the
    environment variables to the shell configuration. It does NOT
    modify the .env file.
    """
    print("🔧 Setting up environment variables in shell...")
    
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Please ensure .env file exists with your configuration")
        return
    
    # Read all environment variables from .env file
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    
    if not env_vars:
        print("⚠️  No environment variables found in .env file")
        return
    
    shell = os.environ.get("SHELL", "").split("/")[-1]
    
    # Prepare environment variable exports
    env_exports = [f'export {key}="{value}"' for key, value in env_vars.items() if value]
    
    if shell == "zsh":
        config_file = Path.home() / ".zshrc"
    elif shell == "bash":
        config_file = Path.home() / ".bashrc"
    else:
        print(f"⚠️  Unknown shell: {shell}")
        print("Manually add environment variables from .env file to your shell config")
        return
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Remove old Wagehood environment variables
        lines = content.split('\n')
        new_lines = []
        skip_wagehood_section = False
        
        for line in lines:
            if line.strip() == "# Wagehood environment variables (added by installer)":
                skip_wagehood_section = True
                continue
            elif skip_wagehood_section and (line.strip() == "" or line.startswith("# End Wagehood")):
                skip_wagehood_section = False
                continue
            elif not skip_wagehood_section:
                new_lines.append(line)
        
        # Add new environment variables
        content = '\n'.join(new_lines)
        content += '\n\n# Wagehood environment variables (added by installer)\n'
        for export in env_exports:
            content += export + '\n'
        content += '# End Wagehood environment variables\n'
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated environment variables in {config_file}")
        print(f"💡 Run 'source {config_file}' to apply changes in current terminal")
    else:
        print(f"❌ Shell config file {config_file} not found")


def sync_env_to_shell():
    """Sync .env file to shell configurations."""
    setup_env_vars_in_shell()


if __name__ == "__main__":
    # Check if sync-env argument is passed
    if len(sys.argv) > 1 and sys.argv[1] == "sync-env":
        sync_env_to_shell()
    else:
        install_cli()