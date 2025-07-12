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
            # Check if PATH export already exists
            if zshrc.exists():
                with open(zshrc, 'r') as f:
                    content = f.read()
                    if f'PATH=\\$PATH:{bin_dir}' in content or f'PATH=$PATH:{bin_dir}' in content:
                        print("   PATH already configured in ~/.zshrc")
                        path_added = True
            
            if not path_added:
                with open(zshrc, 'a') as f:
                    f.write(f'\n# Added by Wagehood installer\nexport PATH="$PATH:{bin_dir}"\n')
                print("   ✅ Added PATH to ~/.zshrc")
                print("\n   ⚠️  To use 'wagehood' in this terminal session, run:")
                print(f'   source ~/.zshrc')
                path_added = True
                
        elif shell == "bash":
            bashrc = Path.home() / ".bashrc"
            # Check if PATH export already exists
            if bashrc.exists():
                with open(bashrc, 'r') as f:
                    content = f.read()
                    if f'PATH=\\$PATH:{bin_dir}' in content or f'PATH=$PATH:{bin_dir}' in content:
                        print("   PATH already configured in ~/.bashrc")
                        path_added = True
            
            if not path_added:
                with open(bashrc, 'a') as f:
                    f.write(f'\n# Added by Wagehood installer\nexport PATH="$PATH:{bin_dir}"\n')
                print("   ✅ Added PATH to ~/.bashrc")
                print("\n   ⚠️  To use 'wagehood' in this terminal session, run:")
                print(f'   source ~/.bashrc')
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
            
            print(f"\n✅ Configuration saved!")
            print(f"📊 Tracking {len(symbols)} symbols: {', '.join(symbols[:5])}" + 
                  (" ..." if len(symbols) > 5 else ""))
            
            # Start workers immediately
            start_workers_now = input("\nStart workers now? (y/n): ").lower().strip()
            if start_workers_now == 'y':
                print("\n🚀 Starting workers...")
                start_workers_immediately()
            
            # Ask about auto-start
            setup_autostart = input("\nSet up workers to start automatically on boot? (y/n): ").lower().strip()
            if setup_autostart == 'y':
                setup_macos_autostart()
    
    print("\n🎉 Installation complete!")
    print("\nAvailable commands:")
    print("  wagehood configure         - Configure API credentials")
    print("  wagehood submit            - Submit analysis job")
    print("  wagehood jobs              - List all jobs")
    print("  wagehood symbols list      - List tracked symbols")
    print("  wagehood symbols add/remove - Manage symbols")
    print("  wagehood test              - Run integration tests")
    print("  wagehood workers start     - Start workers")
    print("  wagehood workers stop      - Stop workers")
    print("  wagehood workers status    - Check worker status")


def start_workers_immediately():
    """Start workers immediately after configuration."""
    project_root = Path(__file__).parent.absolute()
    worker_script = project_root / "start_workers.py"
    
    if not worker_script.exists():
        print(f"❌ Worker script not found at {worker_script}")
        return
    
    # Check if workers are already running
    pid_file = Path.home() / '.wagehood' / 'workers.pid'
    if pid_file.exists():
        print("⚠️  Workers may already be running.")
        return
    
    try:
        # Create .wagehood directory if it doesn't exist
        (Path.home() / '.wagehood').mkdir(exist_ok=True)
        
        # Start worker process in background
        process = subprocess.Popen(
            [sys.executable, str(worker_script)],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Save PID
        pid_file.write_text(str(process.pid))
        
        print(f"✅ Workers started successfully (PID: {process.pid})")
        
    except Exception as e:
        print(f"❌ Failed to start workers: {str(e)}")


def setup_macos_autostart():
    """Set up auto-start on macOS using launchd."""
    project_root = Path(__file__).parent.absolute()
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(exist_ok=True)
    
    plist_file = plist_dir / "com.wagehood.workers.plist"
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wagehood.workers</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{project_root}/start_workers.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_root}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.wagehood/workers.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.wagehood/workers_error.log</string>
</dict>
</plist>"""
    
    print(f"\n🔧 Setting up auto-start...")
    plist_file.write_text(plist_content)
    
    # Load the launch agent
    try:
        subprocess.run(["launchctl", "load", str(plist_file)], check=True)
        print("✅ Workers will start automatically on boot")
    except subprocess.CalledProcessError:
        print("⚠️  Failed to set up auto-start. You can start workers manually with: wagehood workers start")


if __name__ == "__main__":
    install_cli()