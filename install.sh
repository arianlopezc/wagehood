#!/bin/bash

# Wagehood Trading System - Automated Installer
# This script automates the complete installation and setup of the Wagehood worker + CLI architecture

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/wagehood"
VENV_DIR="$INSTALL_DIR/venv"

# Helper functions
print_step() {
    echo -e "${BLUE}==>${NC} ${1}"
}

print_success() {
    echo -e "${GREEN}✓${NC} ${1}"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} ${1}"
}

print_error() {
    echo -e "${RED}✗${NC} ${1}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Welcome message
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║               WAGEHOOD TRADING SYSTEM INSTALLER               ║"
echo "║                                                               ║"
echo "║         Real-time trading analysis platform                  ║"
echo "║              Worker + CLI Architecture                        ║"
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
        print_error "Python 3.8+ is required. Found Python $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.8+ is required but not found."
    echo "Please install Python 3.8+ and try again."
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

# Check git
if check_command git; then
    print_success "Git found"
else
    print_error "Git is required but not found."
    echo "Please install Git and try again."
    exit 1
fi

# Check if Redis is available (optional)
if check_command redis-server; then
    print_success "Redis server found"
    REDIS_AVAILABLE=true
else
    print_warning "Redis server not found. It will be needed for real-time features."
    REDIS_AVAILABLE=false
fi

# Handle installation directory
print_step "Preparing installation directory..."

# Check if we're already in the source directory
if [[ "$SCRIPT_DIR" == "$INSTALL_DIR" ]]; then
    print_success "Installing from source directory: $INSTALL_DIR"
    SKIP_COPY=true
else
    SKIP_COPY=false
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Do you want to remove it and continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
            print_success "Removed existing installation"
        else
            print_error "Installation cancelled"
            exit 1
        fi
    fi

    mkdir -p "$INSTALL_DIR"
    print_success "Created installation directory: $INSTALL_DIR"

    # Copy source code to installation directory
    print_step "Copying Wagehood source code..."
    if cp -r "$SCRIPT_DIR/"* "$INSTALL_DIR/"; then
        print_success "Source code copied successfully"
    else
        print_error "Failed to copy source code"
        exit 1
    fi
fi

cd "$INSTALL_DIR"

# Create virtual environment
print_step "Creating Python virtual environment..."
if $PYTHON_CMD -m venv "$VENV_DIR"; then
    print_success "Virtual environment created"
else
    print_error "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install Python dependencies
print_step "Installing Python dependencies..."
if [[ -f "requirements.txt" ]]; then
    if $PIP_CMD install -r requirements.txt; then
        print_success "Python dependencies installed successfully"
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi
else
    print_error "requirements.txt not found - cannot install dependencies"
    exit 1
fi

# Install Wagehood package in development mode (optional)
print_step "Installing Wagehood package in development mode..."
if [[ -f "setup.py" ]]; then
    if $PIP_CMD install -e .; then
        print_success "Wagehood package installed in development mode"
    else
        print_warning "Failed to install Wagehood package in development mode"
        print_warning "This is optional - the CLI tools will still work directly"
    fi
else
    print_warning "setup.py not found - skipping package installation"
fi

# Test CLI tools
print_step "Testing CLI tools..."

# Test market analysis CLI
print_step "Testing market analysis CLI..."
if python3 "$INSTALL_DIR/market_analysis_cli.py" --help > /dev/null 2>&1; then
    print_success "Market analysis CLI is working correctly"
    MARKET_ANALYSIS_CLI_WORKING=true
else
    print_warning "Market analysis CLI test failed - may need Redis connection"
    MARKET_ANALYSIS_CLI_WORKING=false
fi

# Test market watch CLI
print_step "Testing market watch CLI..."
if python3 "$INSTALL_DIR/market_watch.py" --help > /dev/null 2>&1; then
    print_success "Market watch CLI is working correctly"
    MARKET_WATCH_CLI_WORKING=true
else
    print_warning "Market watch CLI test failed"
    MARKET_WATCH_CLI_WORKING=false
fi

# Test Python import of core modules
print_step "Testing core module imports..."
if python3 -c "import sys; sys.path.insert(0, '$INSTALL_DIR/src'); from src.core.models import Signal, Trade, MarketData" 2>/dev/null; then
    print_success "Core modules import successfully"
    CORE_MODULES_WORKING=true
else
    print_warning "Core modules import failed - some features may not work"
    CORE_MODULES_WORKING=false
fi

# Redis installation guidance
if [[ "$REDIS_AVAILABLE" == false ]]; then
    print_step "Redis installation guidance..."
    echo
    echo "Redis is required for real-time trading features."
    echo "Install Redis using your package manager:"
    echo
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install redis-server"
    echo "  CentOS/RHEL:   sudo yum install redis"
    echo "  macOS:         brew install redis"
    echo "  Arch Linux:    sudo pacman -S redis"
    echo
    echo "After installing Redis, start it with:"
    echo "  sudo systemctl start redis-server  # Linux"
    echo "  brew services start redis          # macOS"
    echo
fi

# Success message and next steps
echo
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║                    ✅ INSTALLATION COMPLETE!                  ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo
echo -e "${BLUE}📍 Installation Details:${NC}"
echo "   • Installation directory: $INSTALL_DIR"
echo "   • Virtual environment: $VENV_DIR"
echo "   • Python dependencies: Installed from requirements.txt"
echo "   • CLI Tools: market_analysis_cli.py, market_watch.py"
echo "   • Architecture: Worker + CLI (Redis-based)"

echo
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo
echo "1. ${YELLOW}Activate the virtual environment:${NC}"
echo "   source $VENV_DIR/bin/activate"
echo
echo "2. ${YELLOW}Navigate to the installation directory:${NC}"
echo "   cd $INSTALL_DIR"
echo
echo "3. ${YELLOW}Start Redis server (required for real-time features):${NC}"
if [[ "$REDIS_AVAILABLE" == true ]]; then
    echo "   redis-server  # Redis is already installed"
else
    echo "   # Install Redis first (see guidance below)"
fi
echo
echo "4. ${YELLOW}Run the market analysis CLI:${NC}"
echo "   python3 market_analysis_cli.py --help"
echo "   python3 market_analysis_cli.py signals"
echo
echo "5. ${YELLOW}Run the market watch tool:${NC}"
echo "   python3 market_watch.py"
echo
echo "6. ${YELLOW}Start the real-time processor:${NC}"
echo "   python3 run_realtime.py"
echo
echo "7. ${YELLOW}Run tests to verify everything works:${NC}"
echo "   python3 run_tests.py"
echo

if [[ "$REDIS_AVAILABLE" == false ]]; then
    echo -e "${YELLOW}⚠️  Don't forget to install and start Redis before running the setup!${NC}"
    echo
fi

echo -e "${BLUE}📖 Documentation:${NC}"
echo "   • CLI Usage Guide: $INSTALL_DIR/CLI_USAGE.md"
echo "   • Market Analysis CLI: python3 market_analysis_cli.py --help"
echo "   • Market Watch Tool: python3 market_watch.py --help"
echo "   • Real-time Processor: python3 run_realtime.py --help"
echo "   • Read the README: $INSTALL_DIR/README.md"

echo
echo -e "${GREEN}Happy Trading! 📈${NC}"
echo
echo -e "${BLUE}💡 Installation Summary:${NC}"
echo "   The Wagehood trading system uses a worker + CLI architecture."
echo "   CLI tools connect to Redis-based workers for real-time analysis."
if [[ "$MARKET_ANALYSIS_CLI_WORKING" == true && "$MARKET_WATCH_CLI_WORKING" == true ]]; then
    echo "   ✅ Both CLI tools are ready to use!"
elif [[ "$MARKET_ANALYSIS_CLI_WORKING" == true || "$MARKET_WATCH_CLI_WORKING" == true ]]; then
    echo "   ⚠️  Some CLI tools may need Redis connection to work fully."
else
    echo "   ⚠️  CLI tools may need dependencies or Redis connection."
fi
echo
echo "   🔧 To verify installation later, run:"
echo "      source $VENV_DIR/bin/activate && cd $INSTALL_DIR"
echo "      python3 run_tests.py"

# Create activation script for development work
cat > "$INSTALL_DIR/activate_wagehood.sh" << EOF
#!/bin/bash
# Wagehood Environment Activation Script
echo "Activating Wagehood development environment..."
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
echo "Development environment activated."
echo ""
echo "Available CLI tools:"
echo "  • Market Analysis: python3 market_analysis_cli.py --help"
echo "  • Market Watch: python3 market_watch.py"
echo "  • Real-time Processor: python3 run_realtime.py"
echo "  • Test Suite: python3 run_tests.py"
echo ""
echo "Architecture: Worker + CLI (Redis-based)"
echo "Make sure Redis is running for real-time features!"
EOF

chmod +x "$INSTALL_DIR/activate_wagehood.sh"
print_success "Created development activation script: $INSTALL_DIR/activate_wagehood.sh"

# Create installation verification script
cat > "$INSTALL_DIR/verify_install.sh" << EOF
#!/bin/bash
# Script to verify Wagehood installation

echo "🔍 Verifying Wagehood installation..."
echo

# Check virtual environment
if [[ -d "$VENV_DIR" ]]; then
    echo "✅ Virtual environment found: $VENV_DIR"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check CLI tools
echo "Testing CLI tools..."
if python3 market_analysis_cli.py --help > /dev/null 2>&1; then
    echo "✅ Market Analysis CLI is working"
else
    echo "❌ Market Analysis CLI failed"
fi

if python3 market_watch.py --help > /dev/null 2>&1; then
    echo "✅ Market Watch CLI is working"
else
    echo "❌ Market Watch CLI failed"
fi

# Check core modules
echo "Testing core modules..."
if python3 -c "import sys; sys.path.insert(0, 'src'); from src.core.models import Signal, Trade, MarketData" 2>/dev/null; then
    echo "✅ Core modules import successfully"
else
    echo "❌ Core modules import failed"
fi

# Check Redis connection (optional)
echo "Testing Redis connection..."
if python3 -c "import redis; redis.Redis(host='localhost', port=6379).ping()" 2>/dev/null; then
    echo "✅ Redis connection successful"
else
    echo "⚠️  Redis connection failed - real-time features may not work"
fi

echo
echo "🎉 Installation verification complete!"
echo "Use './activate_wagehood.sh' to activate the environment."
EOF

chmod +x "$INSTALL_DIR/verify_install.sh"
print_success "Created verification script: $INSTALL_DIR/verify_install.sh"

# Create uninstall script
cat > "$INSTALL_DIR/uninstall.sh" << EOF
#!/bin/bash
# Wagehood Uninstall Script

echo "🗑️  Uninstalling Wagehood..."

# Remove virtual environment
if [[ -d "$VENV_DIR" ]]; then
    echo "Removing virtual environment..."
    rm -rf "$VENV_DIR"
    echo "✅ Virtual environment removed"
fi

# Optional: Remove the pip package if installed
if source "$VENV_DIR/bin/activate" 2>/dev/null; then
    if command -v pip3 > /dev/null 2>&1; then
        pip3 uninstall wagehood -y 2>/dev/null || echo "No pip package to remove"
    elif command -v pip > /dev/null 2>&1; then
        pip uninstall wagehood -y 2>/dev/null || echo "No pip package to remove"
    fi
fi

echo "✅ Wagehood components removed"
echo "📁 Installation directory remains at: $(dirname "$0")"
echo "   You can manually remove it if desired."
echo "   rm -rf $INSTALL_DIR"
EOF

chmod +x "$INSTALL_DIR/uninstall.sh"
print_success "Created uninstall script: $INSTALL_DIR/uninstall.sh"