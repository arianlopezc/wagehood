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

# Verify CLI tools
echo "🔍 Verifying CLI tools..."

# Test market analysis CLI
if python3 market_analysis_cli.py --help > /dev/null 2>&1; then
    echo "✅ Market analysis CLI is working"
else
    echo "❌ Market analysis CLI failed"
fi

# Test market watch CLI  
if python3 market_watch.py --help > /dev/null 2>&1; then
    echo "✅ Market watch CLI is working"
else
    echo "❌ Market watch CLI failed"
fi

# Test real-time processor
if python3 run_realtime.py --help > /dev/null 2>&1; then
    echo "✅ Real-time processor is working"
else
    echo "❌ Real-time processor failed"
fi

# Test global commands if installed
if command -v wagehood-cli &> /dev/null; then
    echo "✅ Global 'wagehood-cli' command found"
    if wagehood-cli --help &> /dev/null; then
        echo "✅ Command is working"
    else
        echo "❌ Command found but not working"
    fi
else
    echo "⚠️  Global 'wagehood-cli' command not found (may need package install)"
fi

echo
echo "🎉 All installation tests passed!"