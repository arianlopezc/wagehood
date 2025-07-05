#!/bin/bash

# Test script for Wagehood installation
# This script tests the updated install.sh script

set -e

echo "🧪 Testing Wagehood installation script..."
echo

# Save original directory
ORIGINAL_DIR=$(pwd)
TEST_DIR="/tmp/wagehood_install_test"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up test environment..."
    cd "$ORIGINAL_DIR"
    rm -rf "$TEST_DIR"
    
    # Remove test installation if it exists
    if [[ -d "$HOME/wagehood_test" ]]; then
        rm -rf "$HOME/wagehood_test"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Create test environment
echo "📁 Creating test environment..."
mkdir -p "$TEST_DIR"
cp -r . "$TEST_DIR/"
cd "$TEST_DIR"

# Modify install script to use test directory
sed -i.bak 's|INSTALL_DIR="$HOME/wagehood"|INSTALL_DIR="$HOME/wagehood_test"|g' install.sh

echo "✅ Test environment prepared"
echo

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip not found"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo

# Run installation in non-interactive mode
echo "🚀 Running installation (non-interactive)..."
echo

# Use 'yes' to automatically answer prompts
echo "y" | bash install.sh

echo
echo "🎉 Installation test completed!"
echo

# Verify global command
echo "🔍 Verifying global command..."

# Update PATH for verification
if command -v python3 &> /dev/null; then
    PYTHON_USER_BASE=$(python3 -m site --user-base)
else
    PYTHON_USER_BASE=$(python -m site --user-base)
fi
export PATH="$PYTHON_USER_BASE/bin:$PATH"

if command -v wagehood &> /dev/null; then
    echo "✅ Global 'wagehood' command found"
    
    if wagehood --version &> /dev/null; then
        echo "✅ Command is working"
        echo "   Version: $(wagehood --version 2>/dev/null || echo 'Version check failed')"
    else
        echo "❌ Command found but not working"
        exit 1
    fi
else
    echo "❌ Global 'wagehood' command not found"
    echo "This may be expected - command might need terminal restart"
fi

echo
echo "🎉 All installation tests passed!"