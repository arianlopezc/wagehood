#!/usr/bin/env python3
"""
Test runner for the trading system.

This script runs the complete test suite with coverage reporting,
performance benchmarks, and detailed analysis.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_command(cmd: List[str], cwd: str = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def install_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "pytest-benchmark>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "httpx>=0.24.0",
        "memory-profiler>=0.60.0",
        "psutil>=5.9.0"
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        exit_code, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", dep])
        if exit_code != 0:
            print(f"  ❌ Failed to install {dep}: {stderr}")
            return False
        else:
            print(f"  ✅ Installed {dep}")
    
    return True


def run_unit_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ]
    
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            f"--cov-fail-under={args.min_coverage}"
        ])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "Unit Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def run_integration_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run integration tests."""
    print("🔗 Running integration tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short"
    ]
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "Integration Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def run_performance_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run performance tests."""
    print("⚡ Running performance tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "performance",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "Performance Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def run_slow_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run slow tests."""
    print("🐌 Running slow tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "slow",
        "--tb=short"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "Slow Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def run_memory_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run memory usage tests."""
    print("🧠 Running memory tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "memory",
        "--tb=short"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "Memory Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def run_api_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run API tests."""
    print("🌐 Running API tests...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_api.py",
        "-v",
        "--tb=short"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd)
    end_time = time.time()
    
    return {
        "name": "API Tests",
        "exit_code": exit_code,
        "duration": end_time - start_time,
        "stdout": stdout,
        "stderr": stderr
    }


def generate_test_report(results: List[Dict[str, Any]], args: argparse.Namespace):
    """Generate test report."""
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    total_duration = sum(r["duration"] for r in results)
    passed_tests = sum(1 for r in results if r["exit_code"] == 0)
    failed_tests = len(results) - passed_tests
    
    print(f"Total test suites: {len(results)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print()
    
    for result in results:
        status = "✅ PASSED" if result["exit_code"] == 0 else "❌ FAILED"
        print(f"{result['name']:<20} {status:<10} ({result['duration']:.2f}s)")
    
    print()
    
    # Detailed results
    if args.verbose:
        print("📋 DETAILED RESULTS")
        print("-" * 80)
        
        for result in results:
            print(f"\n{result['name']}:")
            print(f"Exit code: {result['exit_code']}")
            print(f"Duration: {result['duration']:.2f}s")
            
            if result['stdout']:
                print("STDOUT:")
                print(result['stdout'][:1000] + "..." if len(result['stdout']) > 1000 else result['stdout'])
            
            if result['stderr']:
                print("STDERR:")
                print(result['stderr'][:500] + "..." if len(result['stderr']) > 500 else result['stderr'])
    
    # Coverage report
    if args.coverage:
        print("\n📈 COVERAGE REPORT")
        print("-" * 80)
        if os.path.exists("htmlcov/index.html"):
            print("HTML coverage report generated: htmlcov/index.html")
        print("Run 'python -m http.server 8000' in project root and visit http://127.0.0.1:8000/htmlcov/")
    
    # Performance summary
    performance_results = [r for r in results if "Performance" in r["name"]]
    if performance_results:
        print("\n⚡ PERFORMANCE SUMMARY")
        print("-" * 80)
        for result in performance_results:
            if "benchmark" in result['stdout'].lower():
                print("Benchmark results available in stdout above")
    
    # Overall status
    print("\n" + "="*80)
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"💥 {failed_tests} TEST SUITE(S) FAILED!")
        return 1


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Trading System Test Runner")
    
    # Test selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--slow", action="store_true", help="Run slow tests")
    parser.add_argument("--memory", action="store_true", help="Run memory tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Test options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--min-coverage", type=int, default=90, help="Minimum coverage percentage")
    parser.add_argument("--markers", type=str, help="Pytest markers to run")
    
    args = parser.parse_args()
    
    # If no specific tests selected, run unit and integration by default
    if not any([args.unit, args.integration, args.performance, args.slow, args.memory, args.api, args.all]):
        args.unit = True
        args.integration = True
    
    if args.all:
        args.unit = True
        args.integration = True
        args.performance = True
        args.api = True
    
    print("🚀 Trading System Test Runner")
    print("="*50)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return 1
    
    # Check if pytest is available
    exit_code, _, _ = run_command([sys.executable, "-c", "import pytest"])
    if exit_code != 0:
        print("❌ pytest not found. Install with: pip install pytest")
        return 1
    
    # Run tests
    results = []
    
    if args.unit:
        results.append(run_unit_tests(args))
    
    if args.integration:
        results.append(run_integration_tests(args))
    
    if args.performance:
        results.append(run_performance_tests(args))
    
    if args.slow:
        results.append(run_slow_tests(args))
    
    if args.memory:
        results.append(run_memory_tests(args))
    
    if args.api:
        results.append(run_api_tests(args))
    
    # Generate report
    return generate_test_report(results, args)


if __name__ == "__main__":
    sys.exit(main())