"""
Comprehensive Mathematical Test Runner

This module provides a comprehensive test runner for all mathematical validation tests
including indicators, strategies, edge cases, and incremental calculations.
"""

import pytest
import sys
import os
import time
from typing import Dict, List, Any
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MathematicalTestRunner:
    """Comprehensive test runner for mathematical validation"""
    
    def __init__(self):
        self.test_modules = [
            'test_moving_averages',
            'test_momentum', 
            'test_volatility',
            'test_moving_average_crossover',
            'test_macd_rsi',
            'test_edge_cases',
            'test_incremental_validation'
        ]
        self.results = {}
    
    def run_indicator_tests(self):
        """Run all indicator tests"""
        logger.info("Running technical indicator tests...")
        
        indicator_modules = [
            'indicators/test_moving_averages.py',
            'indicators/test_momentum.py',
            'indicators/test_volatility.py'
        ]
        
        results = {}
        for module in indicator_modules:
            logger.info(f"Running {module}...")
            start_time = time.time()
            
            exit_code = pytest.main([
                os.path.join(os.path.dirname(__file__), module),
                '-v',
                '--tb=short'
            ])
            
            duration = time.time() - start_time
            results[module] = {
                'exit_code': exit_code,
                'duration': duration,
                'status': 'PASSED' if exit_code == 0 else 'FAILED'
            }
            
            logger.info(f"{module}: {results[module]['status']} ({duration:.2f}s)")
        
        return results
    
    def run_strategy_tests(self):
        """Run all strategy tests"""
        logger.info("Running trading strategy tests...")
        
        strategy_modules = [
            'strategies/test_moving_average_crossover.py',
            'strategies/test_macd_rsi.py'
        ]
        
        results = {}
        for module in strategy_modules:
            logger.info(f"Running {module}...")
            start_time = time.time()
            
            exit_code = pytest.main([
                os.path.join(os.path.dirname(__file__), module),
                '-v',
                '--tb=short'
            ])
            
            duration = time.time() - start_time
            results[module] = {
                'exit_code': exit_code,
                'duration': duration,
                'status': 'PASSED' if exit_code == 0 else 'FAILED'
            }
            
            logger.info(f"{module}: {results[module]['status']} ({duration:.2f}s)")
        
        return results
    
    def run_edge_case_tests(self):
        """Run edge case tests"""
        logger.info("Running edge case tests...")
        
        start_time = time.time()
        exit_code = pytest.main([
            os.path.join(os.path.dirname(__file__), 'test_edge_cases.py'),
            '-v',
            '--tb=short'
        ])
        
        duration = time.time() - start_time
        result = {
            'exit_code': exit_code,
            'duration': duration,
            'status': 'PASSED' if exit_code == 0 else 'FAILED'
        }
        
        logger.info(f"Edge case tests: {result['status']} ({duration:.2f}s)")
        return result
    
    def run_incremental_validation_tests(self):
        """Run incremental validation tests"""
        logger.info("Running incremental validation tests...")
        
        start_time = time.time()
        exit_code = pytest.main([
            os.path.join(os.path.dirname(__file__), 'test_incremental_validation.py'),
            '-v',
            '--tb=short'
        ])
        
        duration = time.time() - start_time
        result = {
            'exit_code': exit_code,
            'duration': duration,
            'status': 'PASSED' if exit_code == 0 else 'FAILED'
        }
        
        logger.info(f"Incremental validation tests: {result['status']} ({duration:.2f}s)")
        return result
    
    def run_all_tests(self):
        """Run all mathematical validation tests"""
        logger.info("Starting comprehensive mathematical test suite...")
        
        total_start_time = time.time()
        
        # Run all test categories
        self.results['indicators'] = self.run_indicator_tests()
        self.results['strategies'] = self.run_strategy_tests()
        self.results['edge_cases'] = self.run_edge_case_tests()
        self.results['incremental_validation'] = self.run_incremental_validation_tests()
        
        total_duration = time.time() - total_start_time
        
        # Generate summary report
        self.generate_summary_report(total_duration)
        
        return self.results
    
    def generate_summary_report(self, total_duration: float):
        """Generate a comprehensive summary report"""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE MATHEMATICAL TEST SUITE SUMMARY")
        logger.info("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        # Indicator tests summary
        logger.info("\nTECHNICAL INDICATORS:")
        logger.info("-" * 40)
        for module, result in self.results['indicators'].items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            logger.info(f"{status_symbol} {module}: {result['status']} ({result['duration']:.2f}s)")
            total_tests += 1
            if result['status'] == 'PASSED':
                passed_tests += 1
            else:
                failed_tests += 1
        
        # Strategy tests summary
        logger.info("\nTRADING STRATEGIES:")
        logger.info("-" * 40)
        for module, result in self.results['strategies'].items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            logger.info(f"{status_symbol} {module}: {result['status']} ({result['duration']:.2f}s)")
            total_tests += 1
            if result['status'] == 'PASSED':
                passed_tests += 1
            else:
                failed_tests += 1
        
        # Edge case tests summary
        logger.info("\nEDGE CASE VALIDATION:")
        logger.info("-" * 40)
        result = self.results['edge_cases']
        status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
        logger.info(f"{status_symbol} Edge Cases: {result['status']} ({result['duration']:.2f}s)")
        total_tests += 1
        if result['status'] == 'PASSED':
            passed_tests += 1
        else:
            failed_tests += 1
        
        # Incremental validation summary
        logger.info("\nINCREMENTAL VALIDATION:")
        logger.info("-" * 40)
        result = self.results['incremental_validation']
        status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
        logger.info(f"{status_symbol} Incremental Validation: {result['status']} ({result['duration']:.2f}s)")
        total_tests += 1
        if result['status'] == 'PASSED':
            passed_tests += 1
        else:
            failed_tests += 1
        
        # Overall summary
        logger.info("\nOVERALL SUMMARY:")
        logger.info("=" * 40)
        logger.info(f"Total Test Categories: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        
        if failed_tests == 0:
            logger.info("\n🎉 ALL MATHEMATICAL VALIDATION TESTS PASSED!")
            logger.info("The trading system has been mathematically validated.")
        else:
            logger.warning(f"\n⚠️  {failed_tests} test categories failed.")
            logger.warning("Please review the failed tests before deploying.")
        
        logger.info("=" * 80)
    
    def run_specific_category(self, category: str):
        """Run tests for a specific category"""
        if category == 'indicators':
            return self.run_indicator_tests()
        elif category == 'strategies':
            return self.run_strategy_tests()
        elif category == 'edge_cases':
            return self.run_edge_case_tests()
        elif category == 'incremental':
            return self.run_incremental_validation_tests()
        else:
            logger.error(f"Unknown category: {category}")
            return None
    
    def run_quick_validation(self):
        """Run a quick validation subset of tests"""
        logger.info("Running quick mathematical validation...")
        
        # Run a subset of key tests
        quick_tests = [
            'indicators/test_moving_averages.py::TestSimpleMovingAverage::test_sma_basic_calculation',
            'indicators/test_momentum.py::TestRelativeStrengthIndex::test_rsi_basic_calculation',
            'indicators/test_volatility.py::TestBollingerBands::test_bollinger_bands_basic_calculation',
            'strategies/test_moving_average_crossover.py::TestGoldenCrossSignalGeneration::test_basic_golden_cross_detection',
            'test_incremental_validation.py::TestMovingAverageIncremental::test_sma_incremental_vs_batch'
        ]
        
        start_time = time.time()
        results = {}
        
        for test in quick_tests:
            test_path = os.path.join(os.path.dirname(__file__), test)
            exit_code = pytest.main([test_path, '-v'])
            results[test] = 'PASSED' if exit_code == 0 else 'FAILED'
        
        duration = time.time() - start_time
        
        passed = sum(1 for status in results.values() if status == 'PASSED')
        total = len(results)
        
        logger.info(f"\nQuick validation completed: {passed}/{total} passed ({duration:.2f}s)")
        
        if passed == total:
            logger.info("✓ Quick validation PASSED - Core mathematical functions are working correctly")
        else:
            logger.warning("✗ Quick validation FAILED - Check core mathematical implementations")
        
        return results


def main():
    """Main entry point for running mathematical validation tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive mathematical validation tests')
    parser.add_argument('--category', choices=['indicators', 'strategies', 'edge_cases', 'incremental'],
                      help='Run tests for specific category only')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick validation subset')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = MathematicalTestRunner()
    
    if args.quick:
        results = runner.run_quick_validation()
    elif args.category:
        results = runner.run_specific_category(args.category)
    else:
        results = runner.run_all_tests()
    
    # Exit with non-zero code if any tests failed
    if isinstance(results, dict):
        if args.quick:
            exit_code = 0 if all(status == 'PASSED' for status in results.values()) else 1
        else:
            # Check all results for failures
            exit_code = 0
            for category_results in results.values():
                if isinstance(category_results, dict):
                    for result in category_results.values():
                        if result.get('status') == 'FAILED':
                            exit_code = 1
                            break
                elif category_results.get('status') == 'FAILED':
                    exit_code = 1
                    break
    else:
        exit_code = 0 if results and results.get('status') == 'PASSED' else 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()