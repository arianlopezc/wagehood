"""
Data validation utilities for comprehensive testing.

This module provides functionality to validate trading data, calculations,
and system outputs against expected results and mathematical properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path


class ValidationSeverity(Enum):
    """Severity levels for validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Container for validation results."""

    name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    tolerance: Optional[float] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ValidationSuite:
    """Container for multiple validation results."""

    name: str
    results: List[ValidationResult] = field(default_factory=list)
    passed: Optional[bool] = None

    def add_result(self, result: ValidationResult):
        """Add a validation result to the suite."""
        self.results.append(result)
        self._update_status()

    def _update_status(self):
        """Update the overall passed status based on results."""
        if not self.results:
            self.passed = None
        else:
            self.passed = all(r.passed for r in self.results)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        by_severity = {}
        for severity in ValidationSeverity:
            by_severity[severity.value] = len(
                [r for r in self.results if r.severity == severity]
            )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "by_severity": by_severity,
        }


class DataValidator:
    """
    Comprehensive data validator for trading system validation.

    Provides validation for mathematical calculations, data integrity,
    trading logic, and system outputs.
    """

    def __init__(self, tolerance: float = 1e-8):
        """
        Initialize the data validator.

        Args:
            tolerance: Default tolerance for numerical comparisons
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        self.validation_suites: Dict[str, ValidationSuite] = {}

    def create_suite(self, name: str) -> ValidationSuite:
        """
        Create a new validation suite.

        Args:
            name: Name of the validation suite

        Returns:
            The created validation suite
        """
        suite = ValidationSuite(name=name)
        self.validation_suites[name] = suite
        return suite

    def validate_numerical_equality(
        self,
        expected: Union[float, np.ndarray, pd.Series],
        actual: Union[float, np.ndarray, pd.Series],
        name: str,
        tolerance: Optional[float] = None,
        suite_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate numerical equality with tolerance.

        Args:
            expected: Expected value(s)
            actual: Actual value(s)
            name: Name of the validation
            tolerance: Tolerance for comparison (uses default if None)
            suite_name: Name of suite to add result to

        Returns:
            ValidationResult
        """
        if tolerance is None:
            tolerance = self.tolerance

        try:
            # Handle different data types
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                passed = abs(expected - actual) <= tolerance
                details = {"difference": abs(expected - actual), "tolerance": tolerance}
            elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
                passed = np.allclose(expected, actual, atol=tolerance)
                details = {
                    "max_difference": np.max(np.abs(expected - actual)),
                    "tolerance": tolerance,
                    "shape_match": expected.shape == actual.shape,
                }
            elif isinstance(expected, pd.Series) and isinstance(actual, pd.Series):
                passed = np.allclose(expected.values, actual.values, atol=tolerance)
                details = {
                    "max_difference": np.max(np.abs(expected.values - actual.values)),
                    "tolerance": tolerance,
                    "length_match": len(expected) == len(actual),
                }
            else:
                passed = False
                details = {
                    "error": f"Unsupported data types: {type(expected)}, {type(actual)}"
                }

            severity = (
                ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO
            )
            message = f"Numerical equality check {'passed' if passed else 'failed'}"

            result = ValidationResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Exception during validation: {str(e)}",
                details={"exception": str(e)},
                expected=expected,
                actual=actual,
                tolerance=tolerance,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

    def validate_data_integrity(
        self,
        data: pd.DataFrame,
        name: str,
        required_columns: Optional[List[str]] = None,
        suite_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate data integrity for trading data.

        Args:
            data: DataFrame to validate
            name: Name of the validation
            required_columns: List of required column names
            suite_name: Name of suite to add result to

        Returns:
            ValidationResult
        """
        issues = []
        details = {}

        try:
            # Check if data is empty
            if data.empty:
                issues.append("Data is empty")

            # Check for required columns
            if required_columns:
                missing_cols = set(required_columns) - set(data.columns)
                if missing_cols:
                    issues.append(f"Missing required columns: {missing_cols}")
                details["missing_columns"] = list(missing_cols)

            # Check for null values
            null_counts = data.isnull().sum()
            if null_counts.any():
                issues.append("Data contains null values")
                details["null_counts"] = null_counts[null_counts > 0].to_dict()

            # Check for duplicate indices
            if data.index.duplicated().any():
                issues.append("Data contains duplicate indices")
                details["duplicate_indices"] = data.index.duplicated().sum()

            # Check for infinite values in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if np.isinf(data[col]).any():
                    issues.append(f"Column '{col}' contains infinite values")
                    details[f"{col}_infinite_count"] = np.isinf(data[col]).sum()

            # Check data types
            details["data_types"] = data.dtypes.to_dict()
            details["shape"] = data.shape

            passed = len(issues) == 0
            severity = (
                ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO
            )
            message = f"Data integrity check {'passed' if passed else 'failed'}"

            if issues:
                message += f": {'; '.join(issues)}"

            result = ValidationResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Exception during data integrity validation: {str(e)}",
                details={"exception": str(e)},
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

    def validate_indicator_properties(
        self,
        indicator_values: pd.Series,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        should_be_bounded: bool = False,
        suite_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate mathematical properties of technical indicators.

        Args:
            indicator_values: Series containing indicator values
            name: Name of the validation
            min_value: Expected minimum value
            max_value: Expected maximum value
            should_be_bounded: Whether the indicator should be bounded
            suite_name: Name of suite to add result to

        Returns:
            ValidationResult
        """
        issues = []
        details = {}

        try:
            # Basic statistics
            details["count"] = len(indicator_values)
            details["min"] = indicator_values.min()
            details["max"] = indicator_values.max()
            details["mean"] = indicator_values.mean()
            details["std"] = indicator_values.std()
            details["null_count"] = indicator_values.isnull().sum()

            # Check bounds
            if min_value is not None and indicator_values.min() < min_value:
                issues.append(
                    f"Minimum value {indicator_values.min()} below expected {min_value}"
                )

            if max_value is not None and indicator_values.max() > max_value:
                issues.append(
                    f"Maximum value {indicator_values.max()} above expected {max_value}"
                )

            # Check for bounded indicators (like RSI, Stochastic)
            if should_be_bounded:
                if min_value is not None and max_value is not None:
                    out_of_bounds = (
                        (indicator_values < min_value) | (indicator_values > max_value)
                    ).sum()
                    if out_of_bounds > 0:
                        issues.append(
                            f"{out_of_bounds} values out of bounds [{min_value}, {max_value}]"
                        )
                        details["out_of_bounds_count"] = out_of_bounds

            # Check for reasonable values (no extreme outliers)
            if len(indicator_values) > 0:
                q99 = indicator_values.quantile(0.99)
                q01 = indicator_values.quantile(0.01)
                details["q99"] = q99
                details["q01"] = q01

                # Check for extreme outliers (values beyond 3 standard deviations)
                mean_val = indicator_values.mean()
                std_val = indicator_values.std()
                if std_val > 0:
                    outliers = np.abs(indicator_values - mean_val) > 3 * std_val
                    outlier_count = outliers.sum()
                    if (
                        outlier_count > len(indicator_values) * 0.05
                    ):  # More than 5% outliers
                        issues.append(f"High number of outliers: {outlier_count}")
                        details["outlier_count"] = outlier_count

            passed = len(issues) == 0
            severity = (
                ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO
            )
            message = (
                f"Indicator properties validation {'passed' if passed else 'failed'}"
            )

            if issues:
                message += f": {'; '.join(issues)}"

            result = ValidationResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Exception during indicator validation: {str(e)}",
                details={"exception": str(e)},
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

    def validate_trading_signals(
        self,
        signals: pd.Series,
        name: str,
        expected_signal_count: Optional[int] = None,
        suite_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate trading signals.

        Args:
            signals: Series containing trading signals (1, 0, -1)
            name: Name of the validation
            expected_signal_count: Expected number of signals
            suite_name: Name of suite to add result to

        Returns:
            ValidationResult
        """
        issues = []
        details = {}

        try:
            # Signal statistics
            signal_counts = signals.value_counts()
            details["signal_counts"] = signal_counts.to_dict()
            details["total_signals"] = len(signals[signals != 0])
            details["buy_signals"] = (signals == 1).sum()
            details["sell_signals"] = (signals == -1).sum()
            details["no_signal"] = (signals == 0).sum()

            # Check for valid signal values
            valid_signals = signals.isin([-1, 0, 1])
            if not valid_signals.all():
                invalid_count = (~valid_signals).sum()
                issues.append(
                    f"{invalid_count} invalid signal values (not in [-1, 0, 1])"
                )
                details["invalid_signals"] = invalid_count

            # Check expected signal count
            if expected_signal_count is not None:
                actual_count = details["total_signals"]
                if (
                    abs(actual_count - expected_signal_count)
                    > 0.1 * expected_signal_count
                ):
                    issues.append(
                        f"Signal count {actual_count} differs from expected {expected_signal_count}"
                    )

            # Check for signal balance (not too many consecutive signals)
            if len(signals) > 0:
                # Check for long runs of the same signal
                max_consecutive = 0
                current_run = 1
                for i in range(1, len(signals)):
                    if signals.iloc[i] == signals.iloc[i - 1] and signals.iloc[i] != 0:
                        current_run += 1
                        max_consecutive = max(max_consecutive, current_run)
                    else:
                        current_run = 1

                details["max_consecutive_signals"] = max_consecutive
                if max_consecutive > len(signals) * 0.1:  # More than 10% consecutive
                    issues.append(f"Long run of consecutive signals: {max_consecutive}")

            passed = len(issues) == 0
            severity = (
                ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO
            )
            message = f"Trading signals validation {'passed' if passed else 'failed'}"

            if issues:
                message += f": {'; '.join(issues)}"

            result = ValidationResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Exception during signal validation: {str(e)}",
                details={"exception": str(e)},
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

    def validate_portfolio_calculations(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        name: str,
        initial_capital: float = 100000,
        suite_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate portfolio calculations and performance metrics.

        Args:
            portfolio_values: Series of portfolio values over time
            returns: Series of portfolio returns
            name: Name of the validation
            initial_capital: Initial portfolio capital
            suite_name: Name of suite to add result to

        Returns:
            ValidationResult
        """
        issues = []
        details = {}

        try:
            # Basic portfolio statistics
            details["initial_value"] = (
                portfolio_values.iloc[0] if len(portfolio_values) > 0 else 0
            )
            details["final_value"] = (
                portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0
            )
            details["total_return"] = (
                details["final_value"] / details["initial_value"] - 1
            ) * 100
            details["return_count"] = len(returns)
            details["return_mean"] = returns.mean()
            details["return_std"] = returns.std()

            # Check initial capital
            if len(portfolio_values) > 0:
                actual_initial = portfolio_values.iloc[0]
                if abs(actual_initial - initial_capital) > 0.01:
                    issues.append(
                        f"Initial capital {actual_initial} differs from expected {initial_capital}"
                    )

            # Check for negative portfolio values
            negative_values = (portfolio_values < 0).sum()
            if negative_values > 0:
                issues.append(f"Portfolio has {negative_values} negative values")
                details["negative_values"] = negative_values

            # Check return consistency
            if len(portfolio_values) > 1 and len(returns) > 0:
                # Calculate returns from portfolio values
                calculated_returns = portfolio_values.pct_change().dropna()
                if len(calculated_returns) == len(returns):
                    return_diff = np.abs(calculated_returns.values - returns.values)
                    max_diff = np.max(return_diff)
                    if max_diff > 0.001:  # 0.1% tolerance
                        issues.append(
                            f"Return calculation inconsistency: max diff {max_diff:.4f}"
                        )
                        details["max_return_diff"] = max_diff

            # Check for extreme returns (potential calculation errors)
            if len(returns) > 0:
                extreme_returns = np.abs(returns) > 0.5  # 50% single-period return
                extreme_count = extreme_returns.sum()
                if extreme_count > 0:
                    issues.append(f"Found {extreme_count} extreme returns (>50%)")
                    details["extreme_returns"] = extreme_count

            passed = len(issues) == 0
            severity = (
                ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO
            )
            message = (
                f"Portfolio calculations validation {'passed' if passed else 'failed'}"
            )

            if issues:
                message += f": {'; '.join(issues)}"

            result = ValidationResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message,
                details=details,
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

        except Exception as e:
            result = ValidationResult(
                name=name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Exception during portfolio validation: {str(e)}",
                details={"exception": str(e)},
            )

            if suite_name and suite_name in self.validation_suites:
                self.validation_suites[suite_name].add_result(result)

            return result

    def export_results(self, output_path: str = "validation_results.json") -> str:
        """
        Export all validation results to a JSON file.

        Args:
            output_path: Path to save the results

        Returns:
            Path to the exported file
        """
        export_data = {}

        for suite_name, suite in self.validation_suites.items():
            export_data[suite_name] = {
                "name": suite.name,
                "passed": suite.passed,
                "summary": suite.get_summary(),
                "results": [
                    {
                        "name": result.name,
                        "passed": result.passed,
                        "severity": result.severity.value,
                        "message": result.message,
                        "details": result.details,
                        "tolerance": result.tolerance,
                    }
                    for result in suite.results
                ],
            }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return str(output_file)
