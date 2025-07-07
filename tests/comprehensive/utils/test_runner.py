"""
Test runner for orchestrating comprehensive test execution.

This module provides the main test runner that coordinates the execution
of all test suites in the correct order with proper dependency management.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
import redis
from pathlib import Path

from .report_generator import ReportGenerator, TestResult
from .performance_monitor import PerformanceMonitor


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestSuite:
    """Represents a test suite with its configuration."""
    name: str
    path: str
    dependencies: List[str]
    parallel: bool = True
    timeout: int = 300  # 5 minutes default
    environment: str = "unit"
    markers: List[str] = None
    
    def __post_init__(self):
        if self.markers is None:
            self.markers = []


class TestRunner:
    """
    Main test runner that orchestrates comprehensive test execution.
    
    Handles test suite scheduling, dependency management, parallel execution,
    and result collection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        self.report_generator = ReportGenerator()
        self.performance_monitor = PerformanceMonitor()
        self.redis_client = None
        self.test_suites = self._initialize_test_suites()
        self.results: Dict[str, TestResult] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for test execution."""
        logger = logging.getLogger("comprehensive_test_runner")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path(__file__).parent.parent / "logs" / "test_execution.log"
        log_file.parent.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_test_suites(self) -> List[TestSuite]:
        """Initialize all test suites with their configurations."""
        return [
            TestSuite(
                name="mathematical",
                path="tests/comprehensive/mathematical",
                dependencies=[],
                parallel=True,
                environment="unit",
                markers=["mathematical", "unit"]
            ),
            TestSuite(
                name="integration",
                path="tests/comprehensive/integration",
                dependencies=["mathematical"],
                parallel=False,  # Limited by API rate limits
                environment="integration",
                markers=["integration", "redis", "alpaca"]
            ),
            TestSuite(
                name="workers",
                path="tests/comprehensive/workers",
                dependencies=["mathematical", "integration"],
                parallel=True,
                environment="integration",
                markers=["workers", "redis"]
            ),
            TestSuite(
                name="e2e",
                path="tests/comprehensive/e2e",
                dependencies=["mathematical", "integration", "workers"],
                parallel=False,  # Sequential due to shared state
                environment="e2e",
                markers=["e2e", "redis", "alpaca", "slow"]
            ),
            TestSuite(
                name="performance",
                path="tests/comprehensive/performance",
                dependencies=["mathematical", "integration", "workers"],
                parallel=False,  # Isolated execution
                environment="performance",
                markers=["performance", "stress", "memory", "slow"]
            )
        ]
    
    def setup_environment(self, environment: str) -> bool:
        """Set up the test environment."""
        try:
            self.logger.info(f"Setting up environment: {environment}")
            
            # Set up Redis if required
            if self.config.get('redis', False):
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=1,  # Use test database
                    decode_responses=True
                )
                # Test Redis connection
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            
            # Set up other environment requirements
            if environment == "integration":
                self._setup_integration_environment()
            elif environment == "e2e":
                self._setup_e2e_environment()
            elif environment == "performance":
                self._setup_performance_environment()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up environment {environment}: {e}")
            return False
    
    def _setup_integration_environment(self):
        """Set up integration test environment."""
        # Mock external APIs
        # Set up test data
        pass
    
    def _setup_e2e_environment(self):
        """Set up end-to-end test environment."""
        # Set up live connections
        # Initialize test portfolio
        pass
    
    def _setup_performance_environment(self):
        """Set up performance test environment."""
        # Start performance monitoring
        self.performance_monitor.start()
        # Set up test data generators
        pass
    
    def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite."""
        self.logger.info(f"Starting test suite: {suite.name}")
        start_time = time.time()
        
        try:
            # Set up environment
            if not self.setup_environment(suite.environment):
                return TestResult(
                    name=suite.name,
                    status=TestStatus.ERROR,
                    duration=0,
                    error="Failed to set up environment"
                )
            
            # Build pytest command
            pytest_args = [
                suite.path,
                "-v",
                "--tb=short",
                f"--timeout={suite.timeout}",
                "--junit-xml=tests/comprehensive/reports/junit.xml"
            ]
            
            # Add markers
            if suite.markers:
                marker_expr = " and ".join(suite.markers)
                pytest_args.extend(["-m", marker_expr])
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            duration = time.time() - start_time
            
            if exit_code == 0:
                status = TestStatus.PASSED
                self.logger.info(f"Test suite {suite.name} passed in {duration:.2f}s")
            else:
                status = TestStatus.FAILED
                self.logger.error(f"Test suite {suite.name} failed in {duration:.2f}s")
            
            return TestResult(
                name=suite.name,
                status=status,
                duration=duration,
                exit_code=exit_code
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test suite {suite.name} error: {e}")
            return TestResult(
                name=suite.name,
                status=TestStatus.ERROR,
                duration=duration,
                error=str(e)
            )
    
    def check_dependencies(self, suite: TestSuite) -> bool:
        """Check if all dependencies for a test suite have passed."""
        for dep in suite.dependencies:
            if dep not in self.results:
                return False
            if self.results[dep].status not in [TestStatus.PASSED]:
                return False
        return True
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all test suites in the correct order."""
        self.logger.info("Starting comprehensive test execution")
        
        completed_suites = set()
        remaining_suites = self.test_suites.copy()
        
        while remaining_suites:
            # Find suites that can be run (dependencies met)
            ready_suites = [
                suite for suite in remaining_suites
                if self.check_dependencies(suite)
            ]
            
            if not ready_suites:
                # No suites ready - check for circular dependencies
                self.logger.error("No suites ready to run - possible circular dependency")
                break
            
            # Separate parallel and sequential suites
            parallel_suites = [s for s in ready_suites if s.parallel]
            sequential_suites = [s for s in ready_suites if not s.parallel]
            
            # Run parallel suites
            if parallel_suites:
                self._run_parallel_suites(parallel_suites)
            
            # Run sequential suites
            for suite in sequential_suites:
                result = self.run_test_suite(suite)
                self.results[suite.name] = result
                completed_suites.add(suite.name)
                remaining_suites.remove(suite)
        
        # Generate comprehensive report
        self._generate_final_report()
        
        return self.results
    
    def _run_parallel_suites(self, suites: List[TestSuite]):
        """Run multiple test suites in parallel."""
        with ThreadPoolExecutor(max_workers=len(suites)) as executor:
            future_to_suite = {
                executor.submit(self.run_test_suite, suite): suite
                for suite in suites
            }
            
            for future in as_completed(future_to_suite):
                suite = future_to_suite[future]
                try:
                    result = future.result()
                    self.results[suite.name] = result
                except Exception as e:
                    self.logger.error(f"Suite {suite.name} raised exception: {e}")
                    self.results[suite.name] = TestResult(
                        name=suite.name,
                        status=TestStatus.ERROR,
                        duration=0,
                        error=str(e)
                    )
    
    def _generate_final_report(self):
        """Generate the final comprehensive test report."""
        self.logger.info("Generating comprehensive test report")
        
        # Generate HTML report
        self.report_generator.generate_comprehensive_report(
            self.results,
            output_path="tests/comprehensive/reports/comprehensive_report.html"
        )
        
        # Generate performance report if performance tests were run
        if "performance" in self.results:
            self.performance_monitor.generate_report(
                output_path="tests/comprehensive/reports/performance_report.html"
            )
        
        # Log summary
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r.status == TestStatus.PASSED)
        failed_suites = sum(1 for r in self.results.values() if r.status == TestStatus.FAILED)
        
        self.logger.info(f"Test execution completed: {passed_suites}/{total_suites} suites passed")
        
        if failed_suites > 0:
            self.logger.error(f"{failed_suites} test suites failed")
    
    def cleanup(self):
        """Clean up resources after test execution."""
        if self.redis_client:
            self.redis_client.flushdb()  # Clear test database
        
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop()


# CLI interface for running tests
def main():
    """Main entry point for the comprehensive test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--suite", help="Run specific test suite")
    parser.add_argument("--environment", default="unit", help="Test environment")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--report", help="Output report path")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.suite:
        # Run specific suite
        suite = next((s for s in runner.test_suites if s.name == args.suite), None)
        if suite:
            result = runner.run_test_suite(suite)
            print(f"Suite {suite.name}: {result.status.value}")
        else:
            print(f"Suite {args.suite} not found")
    else:
        # Run all tests
        results = runner.run_all_tests()
        
        # Print summary
        for name, result in results.items():
            print(f"{name}: {result.status.value} ({result.duration:.2f}s)")
    
    runner.cleanup()


if __name__ == "__main__":
    main()