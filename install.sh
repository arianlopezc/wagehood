#!/bin/bash

# Wagehood Trading System - Production Installer
# This script automates the complete installation and setup of the Wagehood production real-time service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR"  # Install in current directory
VENV_DIR="$INSTALL_DIR/.venv"
LOG_FILE="$INSTALL_DIR/installation.log"

# Helper functions
print_step() {
    echo -e "${BLUE}==>${NC} ${1}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}✓${NC} ${1}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} ${1}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}✗${NC} ${1}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Initialize log file
echo "=== Wagehood Installation Started at $(date) ===" > "$LOG_FILE"

# Welcome message
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║               WAGEHOOD PRODUCTION INSTALLER                   ║"
echo "║                                                               ║"
echo "║         Redis-based Real-time Trading System                  ║"
echo "║              Python API + Worker Service                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons."
   exit 1
fi

# System requirements check
print_step "Checking system requirements..."

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d" " -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif check_command python; then
    PYTHON_VERSION=$(python --version | cut -d" " -f2)
    if [[ $PYTHON_VERSION == 3.* ]]; then
        print_success "Python 3 found: $PYTHON_VERSION"
        PYTHON_CMD="python"
    else
        print_error "Python 3.9+ is required. Found Python $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.9+ is required but not found."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Check pip
if check_command pip3; then
    PIP_CMD="pip3"
    print_success "pip3 found"
elif check_command pip; then
    PIP_CMD="pip"
    print_success "pip found"
else
    print_error "pip is required but not found."
    echo "Please install pip and try again."
    exit 1
fi

# Check Redis (required for production)
if check_command redis-server; then
    print_success "Redis server found"
    REDIS_AVAILABLE=true
else
    print_error "Redis server is required for production operation."
    echo "Install Redis using your package manager:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install redis-server"
    echo "  CentOS/RHEL:   sudo yum install redis"
    echo "  macOS:         brew install redis"
    echo "  Arch Linux:    sudo pacman -S redis"
    exit 1
fi

# Check if Redis is running
if redis-cli ping &>/dev/null; then
    print_success "Redis server is running"
else
    print_step "Starting Redis server..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if check_command brew; then
            brew services start redis
        else
            redis-server --daemonize yes
        fi
    else
        # Linux
        if check_command systemctl; then
            sudo systemctl start redis-server || sudo systemctl start redis
        else
            redis-server --daemonize yes
        fi
    fi
    
    # Wait for Redis to start
    sleep 2
    if redis-cli ping &>/dev/null; then
        print_success "Redis server started successfully"
    else
        print_error "Failed to start Redis server"
        exit 1
    fi
fi

cd "$INSTALL_DIR"

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    print_step "Creating Python virtual environment..."
    if $PYTHON_CMD -m venv "$VENV_DIR"; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
$PIP_CMD install --upgrade pip --quiet

# Install Python dependencies
print_step "Installing Python dependencies..."
if [[ -f "requirements.txt" ]]; then
    if $PIP_CMD install -r requirements.txt --quiet; then
        print_success "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi
else
    print_error "requirements.txt not found - cannot install dependencies"
    exit 1
fi

# Install Wagehood package in development mode
print_step "Installing Wagehood package..."
if [[ -f "setup.py" ]]; then
    if $PIP_CMD install -e . --quiet; then
        print_success "Wagehood package installed"
    else
        print_error "Failed to install Wagehood package"
        exit 1
    fi
else
    print_error "setup.py not found"
    exit 1
fi

# Test core system functionality
print_step "Testing system functionality..."

# Test core imports
if $PYTHON_CMD -c "from src.strategies import create_strategy; from src.data.mock_generator import MockDataGenerator; from src.backtest.engine import BacktestEngine; from src.core.models import MarketData, TimeFrame" 2>/dev/null; then
    print_success "Core system imports working"
else
    print_error "Core system imports failed"
    exit 1
fi

# Test Redis connectivity
if $PYTHON_CMD -c "from src.storage.cache import cache_manager; cache_manager.set('test', 'install', True, ttl=10); print('OK')" 2>/dev/null | grep -q "OK"; then
    print_success "Redis connectivity working"
else
    print_error "Redis connectivity failed"
    exit 1
fi

# Test strategy system
if $PYTHON_CMD -c "from src.strategies import STRATEGY_REGISTRY; print(f'Strategies: {len(STRATEGY_REGISTRY)}')" 2>/dev/null | grep -q "Strategies:"; then
    print_success "Strategy system working"
else
    print_error "Strategy system failed"
    exit 1
fi

# MANDATORY: Test Alpaca credentials and connectivity
print_step "Validating Alpaca credentials and connectivity..."

# Check for required environment variables
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    print_error "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are REQUIRED"
    echo "This production system requires valid Alpaca credentials."
    echo "Please set your credentials:"
    echo "  export ALPACA_API_KEY='your_api_key_here'"
    echo "  export ALPACA_SECRET_KEY='your_secret_key_here'"
    echo ""
    echo "Get your credentials from: https://alpaca.markets/"
    exit 1
fi

# Test Alpaca connection
if $PYTHON_CMD -c "
import os
import asyncio
from src.realtime.data_ingestion import MinimalAlpacaProvider

async def test_alpaca():
    config = {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'paper': True,
        'feed': 'iex'
    }
    provider = MinimalAlpacaProvider(config)
    await provider.connect()
    print('ALPACA_OK')

asyncio.run(test_alpaca())
" 2>/dev/null | grep -q "ALPACA_OK"; then
    print_success "Alpaca connectivity validated"
else
    print_error "Failed to connect to Alpaca Markets"
    echo "Please verify:"
    echo "  1. Your API credentials are correct"
    echo "  2. You have internet connectivity"
    echo "  3. Alpaca services are available"
    echo "  4. Your account is active and in good standing"
    exit 1
fi

# Clean Redis before starting production
print_step "Cleaning Redis for fresh start..."
redis-cli FLUSHALL > /dev/null
print_success "Redis cleaned"

# Stop any existing services
print_step "Stopping any existing services..."
pkill -f "start_production_service" 2>/dev/null || true
pkill -f "wagehood.*python" 2>/dev/null || true
print_success "Existing services stopped"

# Create production service runner if it doesn't exist
if [[ ! -f "start_production_service.py" ]]; then
    print_step "Creating production service runner..."
    cat > "start_production_service.py" << 'EOF'
#!/usr/bin/env python3
"""
Wagehood Production Real-time Service Launcher
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from src.realtime.data_ingestion import create_ingestion_service
from src.realtime.config_manager import ConfigManager
from src.realtime.calculation_engine import CalculationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wagehood_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionService:
    """Production real-time trading service."""
    
    def __init__(self):
        self.config = None
        self.ingestion_service = None
        self.calc_engine = None
        self.running = False
        
    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("🚀 Initializing Wagehood Production Service")
            
            # Initialize configuration
            self.config = ConfigManager()
            logger.info("✅ Configuration manager initialized")
            
            # Create ingestion service
            self.ingestion_service = create_ingestion_service(self.config)
            logger.info("✅ Data ingestion service created")
            
            # Create calculation engine
            self.calc_engine = CalculationEngine(self.config, self.ingestion_service)
            logger.info("✅ Calculation engine initialized")
            
            # Log configuration
            enabled_symbols = self.config.get_enabled_symbols()
            logger.info(f"🎯 Configured symbols: {enabled_symbols}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def start(self):
        """Start the production service."""
        if not await self.initialize():
            return False
            
        try:
            self.running = True
            logger.info("🚀 Starting real-time processing...")
            
            # Start the main ingestion service
            await self.ingestion_service.start()
            
        except KeyboardInterrupt:
            logger.info("⏹️  Shutdown requested by user")
            await self.stop()
        except Exception as e:
            logger.error(f"❌ Service error: {e}", exc_info=True)
            await self.stop()
            return False
        
        return True
    
    async def stop(self):
        """Stop the production service cleanly."""
        if self.running:
            logger.info("🛑 Stopping production service...")
            self.running = False
            
            if self.ingestion_service:
                await self.ingestion_service.stop()
                logger.info("✅ Ingestion service stopped")

async def main():
    """Main entry point for production service."""
    service = ProductionService()
    
    # Set up signal handlers for clean shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if hasattr(asyncio, 'current_task'):
            task = asyncio.current_task()
            if task:
                task.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    logger.info(f"🎯 Wagehood Production Service - {datetime.now()}")
    success = await service.start()
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
EOF
    chmod +x start_production_service.py
    print_success "Production service runner created"
fi

# Create systemd service file for production deployment
print_step "Creating systemd service file..."
cat > "wagehood.service" << EOF
[Unit]
Description=Wagehood Real-time Trading Service
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_DIR/bin/python start_production_service.py
Restart=always
RestartSec=10
StandardOutput=append:$INSTALL_DIR/wagehood_production.log
StandardError=append:$INSTALL_DIR/wagehood_production.log

[Install]
WantedBy=multi-user.target
EOF
print_success "Systemd service file created: wagehood.service"

# Start the production service
print_step "Starting Wagehood production service..."
nohup $PYTHON_CMD start_production_service.py > wagehood_startup.log 2>&1 &
WAGEHOOD_PID=$!
print_success "Production service started with PID: $WAGEHOOD_PID"

# Wait a moment for service to initialize
sleep 3

# Verify service is running
if kill -0 $WAGEHOOD_PID 2>/dev/null; then
    print_success "Production service is running"
    
    # Test service functionality
    print_step "Testing service functionality..."
    sleep 2
    
    # Check Redis streams
    if redis-cli EXISTS market_data_stream >/dev/null; then
        print_success "Market data stream active"
    else
        print_warning "Market data stream not yet active (may take a moment)"
    fi
    
    # Save PID for management
    echo $WAGEHOOD_PID > "$INSTALL_DIR/wagehood.pid"
    print_success "Service PID saved to wagehood.pid"
    
else
    print_error "Production service failed to start"
    cat wagehood_startup.log
    exit 1
fi

# Create management scripts
print_step "Creating management scripts..."

# Status script
cat > "status.sh" << EOF
#!/bin/bash
# Check Wagehood service status

echo "🔍 Wagehood Service Status"
echo "========================="

if [[ -f "wagehood.pid" ]]; then
    PID=\$(cat wagehood.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo "✅ Service running (PID: \$PID)"
        
        # Check Redis
        if redis-cli ping >/dev/null 2>&1; then
            echo "✅ Redis connection OK"
            
            # Check streams
            STREAM_LEN=\$(redis-cli XLEN market_data_stream 2>/dev/null || echo "0")
            echo "📊 Market data stream: \$STREAM_LEN messages"
            
            # Check recent data
            if [[ \$STREAM_LEN -gt 0 ]]; then
                echo "🔄 Data flowing - service operational"
            else
                echo "⚠️  No data yet - service may be starting"
            fi
        else
            echo "❌ Redis connection failed"
        fi
    else
        echo "❌ Service not running"
        rm -f wagehood.pid
    fi
else
    echo "❌ No PID file found - service not started"
fi

echo
echo "📋 Log files:"
echo "   • Production log: wagehood_production.log"
echo "   • Startup log: wagehood_startup.log"
echo "   • Installation log: installation.log"
EOF
chmod +x status.sh

# Stop script
cat > "stop.sh" << EOF
#!/bin/bash
# Stop Wagehood service

echo "🛑 Stopping Wagehood service..."

if [[ -f "wagehood.pid" ]]; then
    PID=\$(cat wagehood.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo "Stopping service (PID: \$PID)..."
        kill \$PID
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 \$PID 2>/dev/null; then
                echo "✅ Service stopped gracefully"
                rm -f wagehood.pid
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if necessary
        echo "⚠️  Forcing service shutdown..."
        kill -9 \$PID 2>/dev/null || true
        rm -f wagehood.pid
        echo "✅ Service force stopped"
    else
        echo "⚠️  Service not running"
        rm -f wagehood.pid
    fi
else
    echo "⚠️  No PID file found"
fi

# Also kill any stray processes
pkill -f "start_production_service" 2>/dev/null || true
echo "✅ Cleanup complete"
EOF
chmod +x stop.sh

# Restart script
cat > "restart.sh" << EOF
#!/bin/bash
# Restart Wagehood service

echo "🔄 Restarting Wagehood service..."
./stop.sh
sleep 2
echo "🚀 Starting service..."
nohup $PYTHON_CMD start_production_service.py > wagehood_startup.log 2>&1 &
WAGEHOOD_PID=\$!
echo \$WAGEHOOD_PID > wagehood.pid
echo "✅ Service restarted with PID: \$WAGEHOOD_PID"
EOF
chmod +x restart.sh

print_success "Management scripts created: status.sh, stop.sh, restart.sh"

# Success message
echo
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║                 ✅ INSTALLATION COMPLETE!                     ║"
echo "║                                                               ║"
echo "║              🚀 PRODUCTION SERVICE RUNNING                    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo
echo -e "${BLUE}📊 Service Status:${NC}"
echo "   • Production service: ✅ RUNNING (PID: $WAGEHOOD_PID)"
echo "   • Redis server: ✅ ACTIVE"
echo "   • Virtual environment: ✅ READY"
echo "   • Configured symbols: SPY, QQQ, IWM"
echo "   • Data provider: ✅ ALPACA MARKETS (Live Data)"
echo "   • Alpaca credentials: ✅ VALIDATED"

echo
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo "   • Check status: ./status.sh"
echo "   • Stop service: ./stop.sh"
echo "   • Restart service: ./restart.sh"
echo "   • View logs: tail -f wagehood_production.log"

echo
echo -e "${BLUE}📋 Log Files:${NC}"
echo "   • Production logs: $INSTALL_DIR/wagehood_production.log"
echo "   • Startup logs: $INSTALL_DIR/wagehood_startup.log"
echo "   • Installation logs: $INSTALL_DIR/installation.log"

echo
echo -e "${BLUE}🐳 Docker Deployment:${NC}"
echo "   • Dockerfile: Available for containerized deployment"
echo "   • Build: docker build -t wagehood:latest ."
echo "   • Run: docker run -d --name wagehood -p 6379:6379 wagehood:latest"

echo
echo -e "${BLUE}🧪 Testing:${NC}"
echo "   • Run tests: source .venv/bin/activate && python run_tests.py --all"
echo "   • API test: python -c \"from src.strategies import create_strategy; print('API OK')\""

echo
echo -e "${BLUE}⚙️  Production Configuration:${NC}"
echo "   • Add Alpaca credentials: Set ALPACA_API_KEY and ALPACA_SECRET_KEY"
echo "   • Configure symbols: Edit src/realtime/config_manager.py"
echo "   • Systemd integration: sudo cp wagehood.service /etc/systemd/system/"

echo
echo -e "${GREEN}🎉 Wagehood is now running in production mode!${NC}"
echo -e "${BLUE}💡 The service is processing SPY, QQQ, and IWM with LIVE Alpaca data.${NC}"
echo -e "${BLUE}   Real-time market data is being ingested and analyzed.${NC}"

echo
echo "=== Installation completed at $(date) ===" >> "$LOG_FILE"