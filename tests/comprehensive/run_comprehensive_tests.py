#!/usr/bin/env python3
"""
Comprehensive Test Runner for Wagehood Trading System

This script orchestrates the execution of the complete test suite,
including mathematical validation, integration tests, worker process tests,
end-to-end system tests, and performance benchmarking.

Usage:
    python run_comprehensive_tests.py [options]

Options:
    --suite SUITE_NAME      Run specific test suite (mathematical, integration, workers, e2e, performance)
    --environment ENV       Test environment (unit, integration, e2e, performance)
    --parallel              Enable parallel test execution where possible
    --report-dir DIR        Directory for test reports (default: tests/comprehensive/reports)
    --log-level LEVEL       Logging level (DEBUG, INFO, WARNING, ERROR)
    --timeout SECONDS       Test timeout in seconds (default: 1800)
    --coverage              Enable coverage reporting
    --performance           Enable performance monitoring
    --validate-math         Run mathematical validation tests only
    --quick                 Run quick test subset
    --ci                    Run in CI mode (optimized for CI/CD)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.comprehensive.utils.test_runner import TestRunner, TestSuite, TestStatus
from tests.comprehensive.utils.report_generator import ReportGenerator
from tests.comprehensive.utils.performance_monitor import PerformanceMonitor
from tests.comprehensive.utils.data_validator import DataValidator


class ComprehensiveTestRunner:
    """
    Main comprehensive test runner that orchestrates all test execution.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = self._setup_logging()
        self.report_dir = Path(args.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.test_runner = TestRunner()
        self.report_generator = ReportGenerator()
        self.performance_monitor = PerformanceMonitor()
        self.data_validator = DataValidator()

        # Test configuration
        self.test_suites = self._configure_test_suites()
        self.results = {}

        self.logger.info(f"Comprehensive test runner initialized")
        self.logger.info(f"Report directory: {self.report_dir}")
        self.logger.info(f"Test environment: {args.environment}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("comprehensive_tests")
        logger.setLevel(getattr(logging, self.args.log_level.upper()))

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.report_dir / "comprehensive_tests.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _configure_test_suites(self) -> List[TestSuite]:
        """Configure test suites based on command line arguments."""
        base_suites = [
            TestSuite(
                name="mathematical",
                path="tests/comprehensive/mathematical",
                dependencies=[],
                parallel=True,
                timeout=self.args.timeout,
                environment=self.args.environment,
                markers=["mathematical", "unit"],
            ),
            TestSuite(
                name="integration",
                path="tests/comprehensive/integration",
                dependencies=["mathematical"] if not self.args.quick else [],
                parallel=False,
                timeout=self.args.timeout,
                environment=self.args.environment,
                markers=["integration", "redis", "alpaca"],
            ),
            TestSuite(
                name="workers",
                path="tests/comprehensive/workers",
                dependencies=(
                    ["mathematical", "integration"] if not self.args.quick else []
                ),
                parallel=True,
                timeout=self.args.timeout,
                environment=self.args.environment,
                markers=["workers", "redis"],
            ),
            TestSuite(
                name="e2e",
                path="tests/comprehensive/e2e",
                dependencies=(
                    ["mathematical", "integration", "workers"]
                    if not self.args.quick
                    else []
                ),
                parallel=False,
                timeout=self.args.timeout,
                environment=self.args.environment,
                markers=["e2e", "redis", "alpaca", "slow"],
            ),
            TestSuite(
                name="performance",
                path="tests/comprehensive/performance",
                dependencies=["mathematical"] if not self.args.quick else [],
                parallel=False,
                timeout=self.args.timeout,
                environment=self.args.environment,
                markers=["performance", "stress", "memory", "slow"],
            ),
        ]

        # Filter suites based on arguments
        if self.args.suite:
            base_suites = [s for s in base_suites if s.name == self.args.suite]

        if self.args.validate_math:
            base_suites = [s for s in base_suites if s.name == "mathematical"]

        if self.args.quick:
            # In quick mode, remove dependencies and reduce timeouts
            for suite in base_suites:
                suite.dependencies = []
                suite.timeout = min(suite.timeout, 300)  # 5 minutes max

        if self.args.ci:
            # CI mode optimizations
            for suite in base_suites:
                suite.timeout = min(suite.timeout, 900)  # 15 minutes max
                if suite.name == "performance":
                    suite.markers.append("not stress")  # Skip stress tests in CI

        return base_suites

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all configured test suites."""
        self.logger.info("Starting comprehensive test execution")
        start_time = time.time()

        # Start performance monitoring if enabled
        if self.args.performance:
            self.performance_monitor.start()

        try:
            # Run test suites
            if self.args.suite:
                # Run specific suite
                suite = next(
                    (s for s in self.test_suites if s.name == self.args.suite), None
                )
                if suite:
                    self.results[suite.name] = self.test_runner.run_test_suite(suite)
                else:
                    self.logger.error(f"Suite {self.args.suite} not found")
                    return {"error": f"Suite {self.args.suite} not found"}
            else:
                # Run all suites
                self.results = self.test_runner.run_all_tests()

            # Generate reports
            self._generate_reports()

            # Calculate summary
            summary = self._calculate_summary()

            total_time = time.time() - start_time
            self.logger.info(
                f"Comprehensive test execution completed in {total_time:.2f}s"
            )

            return {
                "summary": summary,
                "results": self.results,
                "execution_time": total_time,
                "reports": self._get_report_paths(),
            }

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return {"error": str(e)}

        finally:
            # Stop performance monitoring
            if self.args.performance and hasattr(self, "performance_monitor"):
                self.performance_monitor.stop()

    def _generate_reports(self):
        """Generate all test reports."""
        self.logger.info("Generating test reports")

        # Generate comprehensive HTML report
        html_report = self.report_generator.generate_comprehensive_report(
            self.results, output_path=str(self.report_dir / "comprehensive_report.html")
        )
        self.logger.info(f"Generated HTML report: {html_report}")

        # Generate JSON report for CI/CD
        json_report = self.report_generator.generate_json_report(
            self.results, output_path=str(self.report_dir / "test_results.json")
        )
        self.logger.info(f"Generated JSON report: {json_report}")

        # Generate summary report
        summary_report = self.report_generator.generate_summary_report(
            self.results, output_path=str(self.report_dir / "test_summary.txt")
        )
        self.logger.info(f"Generated summary report: {summary_report}")

        # Generate performance report if monitoring was enabled
        if self.args.performance:
            performance_report = self.performance_monitor.generate_report(
                output_path=str(self.report_dir / "performance_report.html")
            )
            self.logger.info(f"Generated performance report: {performance_report}")

            # Export performance metrics
            performance_metrics = self.performance_monitor.export_metrics(
                output_path=str(self.report_dir / "performance_metrics.json")
            )
            self.logger.info(f"Exported performance metrics: {performance_metrics}")

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate test execution summary."""
        total_suites = len(self.results)
        passed_suites = sum(
            1 for r in self.results.values() if r.status == TestStatus.PASSED
        )
        failed_suites = sum(
            1 for r in self.results.values() if r.status == TestStatus.FAILED
        )
        error_suites = sum(
            1 for r in self.results.values() if r.status == TestStatus.ERROR
        )

        success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        total_duration = sum(r.duration for r in self.results.values())

        return {
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "error_suites": error_suites,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "environment": self.args.environment,
            "parallel_enabled": self.args.parallel,
            "performance_monitoring": self.args.performance,
        }

    def _get_report_paths(self) -> Dict[str, str]:
        """Get paths to generated reports."""
        return {
            "html_report": str(self.report_dir / "comprehensive_report.html"),
            "json_report": str(self.report_dir / "test_results.json"),
            "summary_report": str(self.report_dir / "test_summary.txt"),
            "performance_report": str(self.report_dir / "performance_report.html"),
            "performance_metrics": str(self.report_dir / "performance_metrics.json"),
            "log_file": str(self.report_dir / "comprehensive_tests.log"),
        }

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up test resources")

        if hasattr(self, "test_runner"):
            self.test_runner.cleanup()

        if hasattr(self, "performance_monitor"):
            self.performance_monitor.stop()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Wagehood Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Test selection
    parser.add_argument(
        "--suite",
        choices=["mathematical", "integration", "workers", "e2e", "performance"],
        help="Run specific test suite",
    )

    parser.add_argument(
        "--environment",
        choices=["unit", "integration", "e2e", "performance"],
        default="unit",
        help="Test environment (default: unit)",
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel test execution where possible",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Test timeout in seconds (default: 1800)",
    )

    # Reporting options
    parser.add_argument(
        "--report-dir",
        default="tests/comprehensive/reports",
        help="Directory for test reports (default: tests/comprehensive/reports)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Feature flags
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )

    parser.add_argument(
        "--performance", action="store_true", help="Enable performance monitoring"
    )

    parser.add_argument(
        "--validate-math",
        action="store_true",
        help="Run mathematical validation tests only",
    )

    parser.add_argument("--quick", action="store_true", help="Run quick test subset")

    parser.add_argument(
        "--ci", action="store_true", help="Run in CI mode (optimized for CI/CD)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create test runner
    test_runner = ComprehensiveTestRunner(args)

    try:
        # Run tests
        results = test_runner.run_all_tests()

        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print("\\n" + "=" * 60)
            print("COMPREHENSIVE TEST EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Total Suites: {summary['total_suites']}")
            print(f"Passed: {summary['passed_suites']}")
            print(f"Failed: {summary['failed_suites']}")
            print(f"Errors: {summary['error_suites']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Duration: {summary['total_duration']:.2f}s")
            print(f"Environment: {summary['environment']}")
            print("=" * 60)

            # Print individual results
            if "results" in results:
                print("\\nSUITE RESULTS:")
                for suite_name, result in results["results"].items():
                    status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
                    print(
                        f"{status_symbol} {suite_name}: {result.status.value} ({result.duration:.2f}s)"
                    )

            # Print report paths
            if "reports" in results:
                print("\\nREPORTS GENERATED:")
                for report_name, path in results["reports"].items():
                    if Path(path).exists():
                        print(f"  {report_name}: {path}")

        # Exit with appropriate code
        if "error" in results:
            print(f"\\nERROR: {results['error']}")
            sys.exit(1)
        elif "summary" in results and results["summary"]["failed_suites"] > 0:
            print("\\nSome tests failed!")
            sys.exit(1)
        else:
            print("\\nAll tests passed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        test_runner.cleanup()


if __name__ == "__main__":
    main()
