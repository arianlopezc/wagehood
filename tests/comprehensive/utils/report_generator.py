"""
Report generator for comprehensive test results.

This module provides functionality to generate detailed HTML reports
from test execution results, including visualizations and metrics.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import jinja2


class TestStatus(Enum):
    """Test execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Represents the result of a test execution."""

    name: str
    status: TestStatus
    duration: float
    exit_code: int = 0
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ReportGenerator:
    """
    Generates comprehensive test reports in various formats.

    Supports HTML reports with visualizations, JSON reports for
    programmatic access, and summary reports for CI/CD integration.
    """

    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir)
        )
        self._ensure_templates()

    def _ensure_templates(self):
        """Ensure required templates exist."""
        self._create_comprehensive_template()
        self._create_mathematical_template()
        self._create_integration_template()
        self._create_performance_template()

    def _create_comprehensive_template(self):
        """Create the comprehensive report template."""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wagehood Comprehensive Test Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .summary-card.passed {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        }
        .summary-card.failed {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        }
        .summary-card.error {
            background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        }
        .test-suite {
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .test-suite-header {
            background-color: #f8f9fa;
            padding: 15px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-suite-header.passed {
            background-color: #d4edda;
            color: #155724;
        }
        .test-suite-header.failed {
            background-color: #f8d7da;
            color: #721c24;
        }
        .test-suite-header.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .test-suite-content {
            padding: 15px;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-passed { background-color: #28a745; color: white; }
        .status-failed { background-color: #dc3545; color: white; }
        .status-error { background-color: #6c757d; color: white; }
        .status-skipped { background-color: #ffc107; color: black; }
        .error-details {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .timestamp {
            color: #6c757d;
            font-size: 14px;
        }
        .duration {
            color: #6c757d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wagehood Comprehensive Test Report</h1>
            <p class="timestamp">Generated: {{ timestamp }}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Suites</h3>
                <h2>{{ summary.total_suites }}</h2>
            </div>
            <div class="summary-card passed">
                <h3>Passed</h3>
                <h2>{{ summary.passed_suites }}</h2>
            </div>
            <div class="summary-card failed">
                <h3>Failed</h3>
                <h2>{{ summary.failed_suites }}</h2>
            </div>
            <div class="summary-card error">
                <h3>Errors</h3>
                <h2>{{ summary.error_suites }}</h2>
            </div>
        </div>
        
        <div class="test-results">
            {% for result in results %}
            <div class="test-suite">
                <div class="test-suite-header {{ result.status.value }}">
                    <span>{{ result.name|title }} Test Suite</span>
                    <div>
                        <span class="status-badge status-{{ result.status.value }}">
                            {{ result.status.value }}
                        </span>
                        <span class="duration">{{ "%.2f"|format(result.duration) }}s</span>
                    </div>
                </div>
                <div class="test-suite-content">
                    <p><strong>Executed:</strong> {{ result.timestamp }}</p>
                    {% if result.error %}
                    <div class="error-details">{{ result.error }}</div>
                    {% endif %}
                    {% if result.details %}
                    <div class="test-details">
                        <h4>Details:</h4>
                        <pre>{{ result.details | tojson(indent=2) }}</pre>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """

        template_path = self.template_dir / "comprehensive_report.html"
        template_path.write_text(template_content)

    def _create_mathematical_template(self):
        """Create mathematical validation report template."""
        # Create a specialized template for mathematical validation results
        pass

    def _create_integration_template(self):
        """Create integration test report template."""
        # Create a specialized template for integration test results
        pass

    def _create_performance_template(self):
        """Create performance test report template."""
        # Create a specialized template for performance test results
        pass

    def generate_comprehensive_report(
        self,
        results: Dict[str, TestResult],
        output_path: str = "comprehensive_report.html",
    ) -> str:
        """
        Generate a comprehensive HTML report from test results.

        Args:
            results: Dictionary of test results keyed by suite name
            output_path: Path to save the report

        Returns:
            Path to the generated report
        """
        # Calculate summary statistics
        summary = self._calculate_summary(results)

        # Prepare template context
        context = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "results": list(results.values()),
        }

        # Generate report
        template = self.jinja_env.get_template("comprehensive_report.html")
        html_content = template.render(**context)

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)

        return str(output_file)

    def generate_mathematical_report(
        self, results: Dict[str, Any], output_path: str = "mathematical_validation.html"
    ) -> str:
        """Generate specialized mathematical validation report."""
        # Implementation for mathematical validation report
        pass

    def generate_integration_report(
        self, results: Dict[str, Any], output_path: str = "integration_results.html"
    ) -> str:
        """Generate specialized integration test report."""
        # Implementation for integration test report
        pass

    def generate_performance_report(
        self, results: Dict[str, Any], output_path: str = "performance_report.html"
    ) -> str:
        """Generate specialized performance test report."""
        # Implementation for performance test report
        pass

    def generate_json_report(
        self, results: Dict[str, TestResult], output_path: str = "test_results.json"
    ) -> str:
        """
        Generate a JSON report for programmatic access.

        Args:
            results: Dictionary of test results
            output_path: Path to save the JSON report

        Returns:
            Path to the generated JSON report
        """
        # Convert results to JSON-serializable format
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._calculate_summary(results),
            "results": {name: asdict(result) for name, result in results.items()},
        }

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2, default=str)

        return str(output_file)

    def generate_summary_report(
        self, results: Dict[str, TestResult], output_path: str = "test_summary.txt"
    ) -> str:
        """
        Generate a text summary report for CI/CD integration.

        Args:
            results: Dictionary of test results
            output_path: Path to save the summary report

        Returns:
            Path to the generated summary report
        """
        summary = self._calculate_summary(results)

        # Generate text report
        report_lines = [
            "WAGEHOOD COMPREHENSIVE TEST SUMMARY",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Suites: {summary['total_suites']}",
            f"Passed: {summary['passed_suites']}",
            f"Failed: {summary['failed_suites']}",
            f"Errors: {summary['error_suites']}",
            f"Success Rate: {summary['success_rate']:.1f}%",
            "",
            "SUITE RESULTS:",
            "-" * 20,
        ]

        for name, result in results.items():
            status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
            report_lines.append(
                f"{status_symbol} {name}: {result.status.value} ({result.duration:.2f}s)"
            )

        report_content = "\n".join(report_lines)

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_content)

        return str(output_file)

    def _calculate_summary(self, results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        total_suites = len(results)
        passed_suites = sum(
            1 for r in results.values() if r.status == TestStatus.PASSED
        )
        failed_suites = sum(
            1 for r in results.values() if r.status == TestStatus.FAILED
        )
        error_suites = sum(1 for r in results.values() if r.status == TestStatus.ERROR)
        skipped_suites = sum(
            1 for r in results.values() if r.status == TestStatus.SKIPPED
        )

        success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        total_duration = sum(r.duration for r in results.values())

        return {
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "error_suites": error_suites,
            "skipped_suites": skipped_suites,
            "success_rate": success_rate,
            "total_duration": total_duration,
        }
