#!/usr/bin/env python3
"""
Test Runner Script for Wagehood Project

This script executes all tests in the codebase and provides a comprehensive test report.
It handles different test categories and reports results with detailed statistics.
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    """
    Run a command with timeout and capture output.
    
    Args:
        cmd: Command to run as list of strings
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def discover_test_files() -> List[Path]:
    """
    Discover all test files in the project.
    
    Returns:
        List of test file paths
    """
    test_files = []
    project_root = Path(__file__).parent
    
    # Find all test files
    for test_dir in ["tests", "src"]:
        test_path = project_root / test_dir
        if test_path.exists():
            for test_file in test_path.rglob("test_*.py"):
                test_files.append(test_file.relative_to(project_root))
    
    return sorted(test_files)


def categorize_tests(test_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Categorize tests by type.
    
    Args:
        test_files: List of test file paths
        
    Returns:
        Dictionary of test categories
    """
    categories = {
        "unit": [],
        "integration": [],
        "validation": [],
        "other": []
    }
    
    for test_file in test_files:
        test_name = test_file.name.lower()
        
        if "integration" in test_name or "e2e" in test_name:
            categories["integration"].append(test_file)
        elif "validation" in test_name or "math" in test_name or "talib" in test_name:
            categories["validation"].append(test_file)
        elif test_name.startswith("test_") and not any(keyword in test_name for keyword in ["integration", "validation", "math", "talib"]):
            categories["unit"].append(test_file)
        else:
            categories["other"].append(test_file)
    
    return categories


def run_test_category(category: str, test_files: List[Path], verbose: bool = False) -> Dict[str, any]:
    """
    Run tests in a specific category.
    
    Args:
        category: Test category name
        test_files: List of test files in category
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary with test results
    """
    if not test_files:
        return {
            "category": category,
            "total_files": 0,
            "passed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "execution_time": 0,
            "details": []
        }
    
    print(f"\n{'='*60}")
    print(f"Running {category.upper()} tests ({len(test_files)} files)")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = {
        "category": category,
        "total_files": len(test_files),
        "passed_files": 0,
        "failed_files": 0,
        "skipped_files": 0,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "execution_time": 0,
        "details": []
    }
    
    for test_file in test_files:
        print(f"\nRunning: {test_file}")
        
        # Build pytest command
        cmd = ["python3", "-m", "pytest", str(test_file)]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short"])
        
        # Set timeout based on test type
        timeout = 300  # Default 5 minutes
        if "e2e" in test_file.name.lower():
            timeout = 600  # 10 minutes for e2e tests
        elif "integration" in test_file.name.lower():
            timeout = 480  # 8 minutes for integration tests
        
        # Run the test
        return_code, stdout, stderr = run_command(cmd, timeout=timeout)
        
        # Parse results
        file_result = {
            "file": str(test_file),
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if return_code == 0:
            results["passed_files"] += 1
            print(f"  ✓ PASSED")
        else:
            results["failed_files"] += 1
            print(f"  ✗ FAILED (exit code: {return_code})")
            if stderr:
                print(f"    Error: {stderr.strip()}")
        
        # Parse test counts from output
        if "passed" in stdout or "failed" in stdout or "error" in stdout:
            # Try to extract test statistics
            lines = stdout.split('\n')
            for line in lines:
                if "passed" in line or "failed" in line or "error" in line:
                    # Simple parsing - could be improved
                    if "passed" in line and "failed" not in line:
                        try:
                            passed = int(line.split()[0])
                            results["passed_tests"] += passed
                            results["total_tests"] += passed
                        except:
                            pass
                    elif "failed" in line:
                        try:
                            # Handle various formats
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "failed" in part and i > 0:
                                    failed = int(parts[i-1])
                                    results["failed_tests"] += failed
                                    results["total_tests"] += failed
                                    break
                        except:
                            pass
        
        results["details"].append(file_result)
    
    results["execution_time"] = time.time() - start_time
    
    print(f"\n{category.upper()} Results:")
    print(f"  Files: {results['passed_files']}/{results['total_files']} passed")
    print(f"  Tests: {results['passed_tests']} passed, {results['failed_tests']} failed")
    print(f"  Time: {results['execution_time']:.2f}s")
    
    return results


def run_all_tests(verbose: bool = False) -> Dict[str, any]:
    """
    Run all tests in the codebase.
    
    Args:
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary with comprehensive test results
    """
    print("Wagehood Test Runner")
    print("===================")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Discover tests
    test_files = discover_test_files()
    print(f"\nDiscovered {len(test_files)} test files")
    
    if not test_files:
        print("No test files found!")
        return {"error": "No test files found"}
    
    # Categorize tests
    categories = categorize_tests(test_files)
    
    # Print test summary
    print("\nTest Categories:")
    for category, files in categories.items():
        if files:
            print(f"  {category}: {len(files)} files")
            if verbose:
                for file in files:
                    print(f"    - {file}")
    
    # Run tests by category
    overall_start = time.time()
    all_results = {}
    
    for category in ["unit", "validation", "integration", "other"]:
        if categories[category]:
            results = run_test_category(category, categories[category], verbose)
            all_results[category] = results
    
    # Calculate overall statistics
    total_execution_time = time.time() - overall_start
    total_files = sum(len(files) for files in categories.values())
    total_passed_files = sum(r.get("passed_files", 0) for r in all_results.values())
    total_failed_files = sum(r.get("failed_files", 0) for r in all_results.values())
    total_tests = sum(r.get("total_tests", 0) for r in all_results.values())
    total_passed_tests = sum(r.get("passed_tests", 0) for r in all_results.values())
    total_failed_tests = sum(r.get("failed_tests", 0) for r in all_results.values())
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_execution_time:.2f}s")
    print(f"")
    print(f"Files:  {total_passed_files}/{total_files} passed ({total_failed_files} failed)")
    print(f"Tests:  {total_passed_tests} passed, {total_failed_tests} failed")
    
    if total_failed_files == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n❌ {total_failed_files} test files failed")
        
        # Show failed files
        print(f"\nFailed test files:")
        for category, results in all_results.items():
            for detail in results.get("details", []):
                if detail["return_code"] != 0:
                    print(f"  - {detail['file']}")
    
    # Calculate pass rate
    if total_files > 0:
        pass_rate = (total_passed_files / total_files) * 100
        print(f"\nPass rate: {pass_rate:.1f}%")
    
    # Store overall results
    all_results["summary"] = {
        "total_files": total_files,
        "passed_files": total_passed_files,
        "failed_files": total_failed_files,
        "total_tests": total_tests,
        "passed_tests": total_passed_tests,
        "failed_tests": total_failed_tests,
        "execution_time": total_execution_time,
        "pass_rate": pass_rate if total_files > 0 else 0
    }
    
    return all_results


def main():
    """Main function to run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all tests in the Wagehood project")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--category", choices=["unit", "integration", "validation", "other"], 
                       help="Run only tests in specific category")
    
    args = parser.parse_args()
    
    if args.category:
        # Run specific category
        test_files = discover_test_files()
        categories = categorize_tests(test_files)
        if categories[args.category]:
            run_test_category(args.category, categories[args.category], args.verbose)
        else:
            print(f"No tests found in category: {args.category}")
    else:
        # Run all tests
        results = run_all_tests(args.verbose)
        
        # Exit with error code if any tests failed
        if results.get("summary", {}).get("failed_files", 0) > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()