#!/usr/bin/env python3
"""
Test Orchestration and Management System

Coordinates execution of all test suites with dependency management,
parallel execution, comprehensive logging, and detailed reporting.
"""

import asyncio
import logging
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

# Test framework imports
from .performance_monitor import PerformanceMonitor
from .data_manager import TestDataManager
from .logger import TestLogger

logger = logging.getLogger(__name__)


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    path: str
    dependencies: List[str]
    timeout: int
    parallel: bool
    critical: bool
    description: str
    tags: List[str]


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    start_time: datetime
    end_time: datetime
    exit_code: int
    stdout: str
    stderr: str
    metrics: Dict[str, Any]
    error_details: Optional[str] = None


class TestOrchestrator:
    """Orchestrates comprehensive test suite execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize test orchestrator."""
        self.config_path = config_path or self._get_default_config_path()
        self.test_suites = {}
        self.execution_order = []
        self.results = {}
        self.performance_monitor = PerformanceMonitor()
        self.data_manager = TestDataManager()
        self.test_logger = TestLogger()
        
        # Execution control
        self.max_parallel_jobs = multiprocessing.cpu_count()
        self.global_timeout = 7200  # 2 hours
        self.continue_on_failure = True
        
        self._load_test_configuration()
        
    def _get_default_config_path(self) -> str:
        """Get default test configuration path."""
        return os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'test_suites.json'
        )
    
    def _load_test_configuration(self):
        """Load test suite configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            for suite_config in config.get('suites', []):
                suite = TestSuite(**suite_config)
                self.test_suites[suite.name] = suite
                
            logger.info(f"Loaded {len(self.test_suites)} test suites")
            
        except FileNotFoundError:
            logger.warning(f"Test configuration not found: {self.config_path}")
            self._create_default_configuration()
        except Exception as e:
            logger.error(f"Failed to load test configuration: {e}")
            raise
    
    def _create_default_configuration(self):
        """Create default test configuration."""
        default_suites = [
            {
                "name": "mathematical",
                "path": "mathematical/",
                "dependencies": [],
                "timeout": 300,
                "parallel": True,
                "critical": True,
                "description": "Mathematical validation tests for strategies and indicators",
                "tags": ["math", "validation", "critical"]
            },
            {
                "name": "integration_alpaca",
                "path": "integration/test_alpaca_api_integration.py",
                "dependencies": [],
                "timeout": 600,
                "parallel": False,
                "critical": True,
                "description": "Alpaca API integration tests",
                "tags": ["integration", "alpaca", "api"]
            },
            {
                "name": "integration_streaming",
                "path": "integration/test_realtime_streaming.py",
                "dependencies": ["integration_alpaca"],
                "timeout": 900,
                "parallel": False,
                "critical": True,
                "description": "Real-time streaming integration tests",
                "tags": ["integration", "streaming", "realtime"]
            },
            {
                "name": "workers",
                "path": "workers/",
                "dependencies": ["mathematical"],
                "timeout": 1200,
                "parallel": True,
                "critical": True,
                "description": "Worker process validation tests",
                "tags": ["workers", "processes", "critical"]
            },
            {
                "name": "e2e",
                "path": "e2e/",
                "dependencies": ["mathematical", "integration_alpaca", "workers"],
                "timeout": 1800,
                "parallel": False,
                "critical": True,
                "description": "End-to-end system integration tests",
                "tags": ["e2e", "integration", "critical"]
            },
            {
                "name": "performance",
                "path": "performance/",
                "dependencies": ["mathematical", "workers"],
                "timeout": 2400,
                "parallel": True,
                "critical": False,
                "description": "Performance and stress tests",
                "tags": ["performance", "stress", "benchmark"]
            }
        ]
        
        for suite_config in default_suites:
            suite = TestSuite(**suite_config)
            self.test_suites[suite.name] = suite
            
        # Save default configuration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({"suites": default_suites}, f, indent=2)
            
        logger.info(f"Created default test configuration: {self.config_path}")
    
    def calculate_execution_order(self, selected_suites: Optional[List[str]] = None) -> List[str]:
        """Calculate optimal execution order based on dependencies."""
        if selected_suites is None:
            selected_suites = list(self.test_suites.keys())
        
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(suite_name: str):
            if suite_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {suite_name}")
            if suite_name in visited:
                return
                
            temp_visited.add(suite_name)
            
            # Visit dependencies first
            suite = self.test_suites[suite_name]
            for dep in suite.dependencies:
                if dep in selected_suites:
                    visit(dep)
            
            temp_visited.remove(suite_name)
            visited.add(suite_name)
            execution_order.append(suite_name)
        
        # Process all selected suites
        for suite_name in selected_suites:
            if suite_name not in visited:
                visit(suite_name)
        
        self.execution_order = execution_order
        logger.info(f"Execution order: {' → '.join(execution_order)}")
        return execution_order
    
    async def run_comprehensive_tests(
        self,
        suites: Optional[List[str]] = None,
        parallel_mode: str = "auto",  # auto, sequential, parallel
        fail_fast: bool = False,
        dry_run: bool = False
    ) -> Dict[str, TestResult]:
        """Run comprehensive test suite."""
        start_time = datetime.now()
        logger.info("🚀 Starting comprehensive test execution")
        
        # Setup test environment
        await self._setup_test_environment()
        
        # Calculate execution order
        execution_order = self.calculate_execution_order(suites)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            if parallel_mode == "sequential" or not self._can_run_parallel():
                results = await self._run_sequential(execution_order, fail_fast, dry_run)
            elif parallel_mode == "parallel":
                results = await self._run_parallel(execution_order, fail_fast, dry_run)
            else:  # auto
                results = await self._run_adaptive(execution_order, fail_fast, dry_run)
            
            self.results = results
            
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
            
            # Generate comprehensive report
            await self._generate_final_report(start_time)
            
            # Cleanup test environment
            await self._cleanup_test_environment()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summary
        total_tests = len(results)
        passed_tests = len([r for r in results.values() if r.status == "passed"])
        failed_tests = len([r for r in results.values() if r.status == "failed"])
        
        logger.info(f"✅ Test execution completed in {duration:.1f}s")
        logger.info(f"📊 Results: {passed_tests}/{total_tests} passed, {failed_tests} failed")
        
        return results
    
    async def _run_sequential(
        self, 
        execution_order: List[str], 
        fail_fast: bool, 
        dry_run: bool
    ) -> Dict[str, TestResult]:
        """Run tests sequentially."""
        results = {}
        
        for suite_name in execution_order:
            logger.info(f"🧪 Running test suite: {suite_name}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would execute {suite_name}")
                continue
            
            result = await self._execute_test_suite(suite_name)
            results[suite_name] = result
            
            # Check for failures
            if result.status == "failed" and fail_fast:
                logger.error(f"❌ Test suite {suite_name} failed, stopping execution")
                break
                
            # Check critical suite failures
            suite = self.test_suites[suite_name]
            if result.status == "failed" and suite.critical:
                logger.error(f"💥 Critical test suite {suite_name} failed")
                if not self.continue_on_failure:
                    break
        
        return results
    
    async def _run_parallel(
        self, 
        execution_order: List[str], 
        fail_fast: bool, 
        dry_run: bool
    ) -> Dict[str, TestResult]:
        """Run tests in parallel where possible."""
        results = {}
        executed = set()
        
        while len(executed) < len(execution_order):
            # Find suites ready to run (dependencies satisfied)
            ready_suites = []
            for suite_name in execution_order:
                if suite_name in executed:
                    continue
                    
                suite = self.test_suites[suite_name]
                deps_satisfied = all(dep in executed for dep in suite.dependencies)
                
                if deps_satisfied:
                    ready_suites.append(suite_name)
            
            if not ready_suites:
                logger.error("No ready suites found - possible circular dependency")
                break
            
            # Determine parallel groups
            parallel_suites = [s for s in ready_suites if self.test_suites[s].parallel]
            sequential_suites = [s for s in ready_suites if not self.test_suites[s].parallel]
            
            # Run parallel suites concurrently
            if parallel_suites:
                logger.info(f"🔄 Running parallel suites: {', '.join(parallel_suites)}")
                
                if dry_run:
                    for suite_name in parallel_suites:
                        logger.info(f"[DRY RUN] Would execute {suite_name} (parallel)")
                        executed.add(suite_name)
                else:
                    tasks = [self._execute_test_suite(suite_name) for suite_name in parallel_suites]
                    parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for suite_name, result in zip(parallel_suites, parallel_results):
                        if isinstance(result, Exception):
                            logger.error(f"❌ Exception in {suite_name}: {result}")
                            result = TestResult(
                                suite_name=suite_name,
                                status="error",
                                duration=0,
                                start_time=datetime.now(),
                                end_time=datetime.now(),
                                exit_code=-1,
                                stdout="",
                                stderr=str(result),
                                metrics={},
                                error_details=str(result)
                            )
                        
                        results[suite_name] = result
                        executed.add(suite_name)
                        
                        # Check for critical failures
                        if result.status == "failed" and fail_fast:
                            logger.error(f"❌ Critical failure in {suite_name}, stopping")
                            return results
            
            # Run sequential suites one by one
            for suite_name in sequential_suites:
                logger.info(f"🧪 Running sequential suite: {suite_name}")
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would execute {suite_name} (sequential)")
                    executed.add(suite_name)
                else:
                    result = await self._execute_test_suite(suite_name)
                    results[suite_name] = result
                    executed.add(suite_name)
                    
                    # Check for failures
                    if result.status == "failed" and fail_fast:
                        logger.error(f"❌ Test suite {suite_name} failed, stopping")
                        return results
        
        return results
    
    async def _run_adaptive(
        self, 
        execution_order: List[str], 
        fail_fast: bool, 
        dry_run: bool
    ) -> Dict[str, TestResult]:
        """Run tests with adaptive strategy based on system resources."""
        # Check system resources
        cpu_count = multiprocessing.cpu_count()
        available_memory = self.performance_monitor.get_available_memory()
        
        # Decide strategy based on resources
        if cpu_count >= 4 and available_memory > 4096:  # 4GB
            logger.info(f"🚀 Using parallel execution (CPU: {cpu_count}, Memory: {available_memory}MB)")
            return await self._run_parallel(execution_order, fail_fast, dry_run)
        else:
            logger.info(f"🐌 Using sequential execution (CPU: {cpu_count}, Memory: {available_memory}MB)")
            return await self._run_sequential(execution_order, fail_fast, dry_run)
    
    async def _execute_test_suite(self, suite_name: str) -> TestResult:
        """Execute a single test suite."""
        suite = self.test_suites[suite_name]
        start_time = datetime.now()
        
        logger.info(f"▶️  Executing {suite_name}: {suite.description}")
        
        # Prepare command
        test_path = os.path.join(
            os.path.dirname(self.config_path), 
            '..', 
            suite.path
        )
        
        if os.path.isdir(test_path):
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
        else:
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
        
        # Add test markers
        if suite.tags:
            cmd.extend(["-m", " and ".join(suite.tags)])
        
        # Execute test
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(self.config_path)
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=suite.timeout
                )
                exit_code = process.returncode
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ Test suite {suite_name} timed out after {suite.timeout}s")
                process.kill()
                await process.wait()
                
                return TestResult(
                    suite_name=suite_name,
                    status="failed",
                    duration=suite.timeout,
                    start_time=start_time,
                    end_time=datetime.now(),
                    exit_code=-9,
                    stdout="",
                    stderr=f"Test timed out after {suite.timeout} seconds",
                    metrics={},
                    error_details="Timeout"
                )
        
        except Exception as e:
            logger.error(f"💥 Failed to execute {suite_name}: {e}")
            return TestResult(
                suite_name=suite_name,
                status="error",
                duration=0,
                start_time=start_time,
                end_time=datetime.now(),
                exit_code=-1,
                stdout="",
                stderr=str(e),
                metrics={},
                error_details=str(e)
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Determine status
        if exit_code == 0:
            status = "passed"
            logger.info(f"✅ {suite_name} passed in {duration:.1f}s")
        else:
            status = "failed"
            logger.error(f"❌ {suite_name} failed in {duration:.1f}s (exit code: {exit_code})")
        
        # Collect metrics
        metrics = {
            "exit_code": exit_code,
            "duration": duration,
            "memory_peak": self.performance_monitor.get_peak_memory(),
            "cpu_avg": self.performance_monitor.get_average_cpu()
        }
        
        return TestResult(
            suite_name=suite_name,
            status=status,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            exit_code=exit_code,
            stdout=stdout.decode('utf-8') if stdout else "",
            stderr=stderr.decode('utf-8') if stderr else "",
            metrics=metrics,
            error_details=stderr.decode('utf-8') if stderr and exit_code != 0 else None
        )
    
    async def _setup_test_environment(self):
        """Setup test environment."""
        logger.info("🛠️  Setting up test environment...")
        
        # Initialize data manager
        await self.data_manager.setup()
        
        # Initialize test logger
        self.test_logger.setup()
        
        # Clear previous test artifacts
        await self._cleanup_previous_runs()
        
        logger.info("✅ Test environment ready")
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        # Cleanup data manager
        await self.data_manager.cleanup()
        
        # Archive logs
        self.test_logger.archive_logs()
        
        logger.info("✅ Test environment cleaned up")
    
    async def _cleanup_previous_runs(self):
        """Cleanup artifacts from previous test runs."""
        # Remove old test data
        test_data_dir = os.path.join(os.path.dirname(self.config_path), '..', 'data', 'temp')
        if os.path.exists(test_data_dir):
            import shutil
            shutil.rmtree(test_data_dir)
            os.makedirs(test_data_dir, exist_ok=True)
        
        # Clear Redis test namespaces
        try:
            from src.storage.cache import cache_manager
            cache_manager.clear_namespace("test")
        except Exception as e:
            logger.debug(f"Failed to clear Redis test data: {e}")
    
    async def _generate_final_report(self, start_time: datetime):
        """Generate comprehensive final report."""
        logger.info("📊 Generating comprehensive test report...")
        
        # Collect all metrics
        performance_metrics = self.performance_monitor.get_comprehensive_metrics()
        
        # Generate report
        report_data = {
            "execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": (datetime.now() - start_time).total_seconds(),
                "execution_order": self.execution_order
            },
            "test_results": {name: asdict(result) for name, result in self.results.items()},
            "performance_metrics": performance_metrics,
            "system_info": {
                "cpu_count": multiprocessing.cpu_count(),
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Save reports
        reports_dir = os.path.join(os.path.dirname(self.config_path), '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report_path = os.path.join(reports_dir, f"comprehensive_test_report_{timestamp}.json")
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # HTML report (simplified version without jinja2 dependency)
        html_report_path = os.path.join(reports_dir, f"comprehensive_test_report_{timestamp}.html")
        self._generate_simple_html_report(report_data, html_report_path)
        
        logger.info(f"📄 Reports generated:")
        logger.info(f"   JSON: {json_report_path}")
        logger.info(f"   HTML: {html_report_path}")
    
    def _generate_simple_html_report(self, report_data: Dict[str, Any], output_path: str):
        """Generate a simple HTML report without jinja2 dependency."""
        execution_summary = report_data.get('execution_summary', {})
        test_results = report_data.get('test_results', {})
        
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results.values() if r.get('status') == 'passed'])
        failed_tests = len([r for r in test_results.values() if r.get('status') == 'failed'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .test-result {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .error {{ background-color: #fff3cd; color: #856404; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Test Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <div>Total Tests</div>
                <div class="metric-value">{total_tests}</div>
            </div>
            <div class="metric-card">
                <div>Success Rate</div>
                <div class="metric-value">{success_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <div>Passed</div>
                <div class="metric-value">{passed_tests}</div>
            </div>
            <div class="metric-card">
                <div>Failed</div>
                <div class="metric-value">{failed_tests}</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
"""
        
        for suite_name, result in test_results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            status_class = 'passed' if status == 'passed' else 'failed' if status == 'failed' else 'error'
            
            html_content += f"""
            <div class="test-result {status_class}">
                <strong>{suite_name}</strong> - {status.upper()} ({duration:.2f}s)
            </div>
"""
        
        html_content += """
        </div>
    </div>
</body>
</html>
        """
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
    
    def _can_run_parallel(self) -> bool:
        """Check if parallel execution is feasible."""
        return multiprocessing.cpu_count() >= 2
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        if not self.results:
            return {"status": "not_executed"}
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r.status == "passed"])
        failed_tests = len([r for r in self.results.values() if r.status == "failed"])
        error_tests = len([r for r in self.results.values() if r.status == "error"])
        
        total_duration = sum(r.duration for r in self.results.values())
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "status": "passed" if failed_tests == 0 and error_tests == 0 else "failed"
        }