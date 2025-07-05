#!/bin/bash

# Wagehood Trading System - Automated Installer
# This script automates the complete installation and setup of the Wagehood CLI

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
echo "║   Professional-grade real-time trading analysis platform     ║"
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

# Install Wagehood package globally
print_step "Installing Wagehood package..."
if [[ -f "setup.py" ]]; then
    # Install in development mode for easier debugging and updates
    if $PIP_CMD install -e .; then
        print_success "Wagehood package installed in development mode"
    else
        print_error "Failed to install Wagehood package"
        exit 1
    fi
else
    print_error "setup.py not found - cannot install package"
    exit 1
fi

# Configure shell profiles for PATH access
print_step "Configuring shell profiles for global access..."

# Get the pip install location for user packages
PYTHON_USER_BASE=$($PYTHON_CMD -m site --user-base)
USER_BIN_DIR="$PYTHON_USER_BASE/bin"

# Add user bin directory to PATH for various shells
configure_shell_path() {
    local shell_config="$1"
    local shell_name="$2"
    
    if [[ -f "$shell_config" ]]; then
        if ! grep -q "$USER_BIN_DIR" "$shell_config" 2>/dev/null; then
            echo "# Wagehood CLI PATH configuration" >> "$shell_config"
            echo "export PATH=\"$USER_BIN_DIR:\$PATH\"" >> "$shell_config"
            print_success "Added PATH configuration to $shell_name ($shell_config)"
        else
            print_success "PATH already configured in $shell_name"
        fi
    fi
}

# Configure for bash
configure_shell_path "$HOME/.bashrc" "bash"
configure_shell_path "$HOME/.bash_profile" "bash profile"

# Configure for zsh
configure_shell_path "$HOME/.zshrc" "zsh"

# Configure for fish
if [[ -d "$HOME/.config/fish" ]]; then
    FISH_CONFIG="$HOME/.config/fish/config.fish"
    mkdir -p "$HOME/.config/fish"
    if [[ ! -f "$FISH_CONFIG" ]] || ! grep -q "$USER_BIN_DIR" "$FISH_CONFIG" 2>/dev/null; then
        echo "# Wagehood CLI PATH configuration" >> "$FISH_CONFIG"
        echo "set -gx PATH \"$USER_BIN_DIR\" \$PATH" >> "$FISH_CONFIG"
        print_success "Added PATH configuration to fish shell"
    else
        print_success "PATH already configured in fish shell"
    fi
fi

print_success "Shell profiles configured for global CLI access"

# Test CLI installation
print_step "Testing CLI installation..."

# First, test the local CLI to ensure it works
if "$INSTALL_DIR/wagehood_cli.py" --version > /dev/null 2>&1; then
    print_success "Local CLI is working correctly"
else
    print_error "Local CLI test failed"
    exit 1
fi

# Test global wagehood command
print_step "Testing global wagehood command..."

# Update PATH for current session
export PATH="$USER_BIN_DIR:$PATH"

# Test if wagehood command is available globally
if command -v wagehood > /dev/null 2>&1; then
    # Test the global command
    if wagehood --version > /dev/null 2>&1; then
        print_success "Global 'wagehood' command is working correctly"
        GLOBAL_CMD_WORKING=true
    else
        print_warning "Global 'wagehood' command found but not working properly"
        GLOBAL_CMD_WORKING=false
    fi
else
    print_warning "Global 'wagehood' command not immediately available"
    print_warning "This may require restarting your terminal or running 'source ~/.bashrc'"
    GLOBAL_CMD_WORKING=false
fi

# Additional verification: test command from different directory
print_step "Testing command from different directory..."
ORIGINAL_DIR=$(pwd)
cd /tmp
if command -v wagehood > /dev/null 2>&1 && wagehood --version > /dev/null 2>&1; then
    print_success "Global command works from any directory"
    GLOBAL_CMD_WORKING=true
else
    print_warning "Global command may not work from all directories yet"
fi
cd "$ORIGINAL_DIR"

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
echo "   • Python package: wagehood (installed globally)"
echo "   • Global command: wagehood"
echo "   • Command location: $USER_BIN_DIR/wagehood"

echo
if [[ "$GLOBAL_CMD_WORKING" == true ]]; then
    echo -e "${BLUE}🚀 Next Steps (using global command):${NC}"
    echo
    echo "1. ${YELLOW}Run the interactive setup:${NC}"
    echo "   wagehood install setup"
    echo
    echo "2. ${YELLOW}Check system status:${NC}"
    echo "   wagehood install status"
    echo
    echo "3. ${YELLOW}Install auto-start service (optional):${NC}"
    echo "   wagehood service install"
    echo
    echo "4. ${YELLOW}Start trading system:${NC}"
    echo "   wagehood install start"
    echo
    echo "5. ${YELLOW}Get help with any command:${NC}"
    echo "   wagehood --help"
    echo "   wagehood <command> --help"
else
    echo -e "${BLUE}🚀 Next Steps (restart terminal first):${NC}"
    echo
    echo "1. ${YELLOW}Restart your terminal or run:${NC}"
    echo "   source ~/.bashrc  # for bash"
    echo "   source ~/.zshrc   # for zsh"
    echo
    echo "2. ${YELLOW}Verify global command works:${NC}"
    echo "   wagehood --version"
    echo
    echo "3. ${YELLOW}Run the interactive setup:${NC}"
    echo "   wagehood install setup"
    echo
    echo "4. ${YELLOW}Check system status:${NC}"
    echo "   wagehood install status"
    echo
    echo "5. ${YELLOW}Alternative - activate environment manually:${NC}"
    echo "   source $VENV_DIR/bin/activate"
    echo "   cd $INSTALL_DIR"
    echo "   ./wagehood_cli.py install setup"
fi
echo

if [[ "$REDIS_AVAILABLE" == false ]]; then
    echo -e "${YELLOW}⚠️  Don't forget to install and start Redis before running the setup!${NC}"
    echo
fi

echo -e "${BLUE}📖 Documentation:${NC}"
if [[ "$GLOBAL_CMD_WORKING" == true ]]; then
    echo "   • View all commands: wagehood --help"
    echo "   • Get help with any command: wagehood <command> --help"
else
    echo "   • View all commands: wagehood --help (after restarting terminal)"
    echo "   • Alternative: ./wagehood_cli.py --help (from $INSTALL_DIR)"
fi
echo "   • Read the README: $INSTALL_DIR/README.md"

echo
echo -e "${GREEN}Happy Trading! 📈${NC}"
echo
echo -e "${BLUE}💡 Installation Summary:${NC}"
echo "   The Wagehood CLI has been installed using proper Python packaging."
echo "   You can now use 'wagehood' command from anywhere on your system."
if [[ "$GLOBAL_CMD_WORKING" == true ]]; then
    echo "   ✅ Global command is ready to use immediately!"
else
    echo "   ⏳ Global command will be ready after restarting your terminal."
fi
echo
echo "   🔧 To verify installation later, run:"
echo "      $INSTALL_DIR/verify_global_install.sh"

# Create activation script for development work
cat > "$INSTALL_DIR/activate_wagehood.sh" << 'EOF'
#!/bin/bash
# Wagehood Environment Activation Script for Development
echo "Activating Wagehood development environment..."
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
echo "Development environment activated."
echo ""
echo "Available commands:"
echo "  • Global command: wagehood <command>"
echo "  • Local development: ./wagehood_cli.py <command>"
echo "  • Help: wagehood --help or ./wagehood_cli.py --help"
echo ""
echo "Note: The 'wagehood' command is globally available even outside this environment."
EOF

chmod +x "$INSTALL_DIR/activate_wagehood.sh"
print_success "Created development activation script: $INSTALL_DIR/activate_wagehood.sh"

# Create a global command verification script
cat > "$INSTALL_DIR/verify_global_install.sh" << 'EOF'
#!/bin/bash
# Script to verify global Wagehood installation

echo "🔍 Verifying Wagehood global installation..."
echo

# Check if command exists
if command -v wagehood > /dev/null 2>&1; then
    echo "✅ 'wagehood' command found in PATH"
    echo "   Location: $(which wagehood)"
    
    # Test the command
    if wagehood --version > /dev/null 2>&1; then
        echo "✅ Command is working correctly"
        echo "   Version: $(wagehood --version 2>/dev/null || echo 'Could not get version')"
    else
        echo "❌ Command found but not working properly"
        exit 1
    fi
    
    # Test from different directory
    echo
    echo "Testing from different directory..."
    ORIGINAL_DIR=$(pwd)
    cd /tmp
    if wagehood --version > /dev/null 2>&1; then
        echo "✅ Command works from any directory"
    else
        echo "❌ Command doesn't work from all directories"
    fi
    cd "$ORIGINAL_DIR"
    
else
    echo "❌ 'wagehood' command not found in PATH"
    echo
    echo "Try the following:"
    echo "1. Restart your terminal"
    echo "2. Or run: source ~/.bashrc (bash) or source ~/.zshrc (zsh)"
    echo "3. Or check your PATH contains the user bin directory"
    exit 1
fi

echo
echo "🎉 Global installation verification complete!"
EOF

chmod +x "$INSTALL_DIR/verify_global_install.sh"
print_success "Created verification script: $INSTALL_DIR/verify_global_install.sh"

# Create uninstall script
cat > "$INSTALL_DIR/uninstall.sh" << 'EOF'
#!/bin/bash
# Wagehood Uninstall Script

echo "🗑️  Uninstalling Wagehood..."

# Remove the pip package
if command -v pip3 > /dev/null 2>&1; then
    pip3 uninstall wagehood -y
elif command -v pip > /dev/null 2>&1; then
    pip uninstall wagehood -y
fi

echo "✅ Wagehood package uninstalled"
echo "📁 Installation directory remains at: $(dirname "$0")"
echo "   You can manually remove it if desired."
echo "🔧 Shell configurations remain in .bashrc/.zshrc for PATH"
echo "   You can manually remove them if desired."
EOF

chmod +x "$INSTALL_DIR/uninstall.sh"
print_success "Created uninstall script: $INSTALL_DIR/uninstall.sh"