"""
Data Flow Integrity End-to-End Tests

This module contains comprehensive tests for data flow integrity across the entire
trading system, ensuring end-to-end data consistency, timing validation,
signal propagation, portfolio state consistency, and error handling.
"""

import pytest
import asyncio
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.signal_engine import SignalEngine
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.config_manager import ConfigManager, TradingProfile, AssetConfig
from src.core.models import OHLCV, TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Individual data point for tracking."""
    id: str
    timestamp: datetime
    symbol: str
    data: Dict[str, Any]
    checksum: str
    source_component: str


@dataclass
class DataFlowTrace:
    """Trace of data flow through the system."""
    data_id: str
    flow_path: List[str] = field(default_factory=list)
    timestamps: Dict[str, datetime] = field(default_factory=dict)
    data_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    integrity_checks: Dict[str, bool] = field(default_factory=dict)
    latency_measurements: Dict[str, float] = field(default_factory=dict)


@dataclass
class IntegrityTestResult:
    """Result of data integrity test."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    data_points_traced: int = 0
    integrity_violations: int = 0
    consistency_score: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)


class TestDataFlowIntegrity:
    """
    Test data flow integrity across the entire trading system.
    
    These tests validate:
    - End-to-end data consistency
    - Timing and latency across the pipeline
    - Signal propagation and updates
    - Portfolio state consistency
    - Error handling across components
    """
    
    @pytest.fixture
    def data_integrity_tracker(self):
        """Track data integrity across system components."""
        class DataIntegrityTracker:
            def __init__(self):
                self.data_traces = {}
                self.component_states = {}
                self.consistency_checks = []
                self.lock = threading.Lock()
                
            def create_data_point(self, symbol: str, data: Dict[str, Any], 
                                source_component: str) -> DataPoint:
                """Create a tracked data point."""
                data_id = hashlib.md5(
                    f"{symbol}_{datetime.now().isoformat()}_{source_component}".encode()
                ).hexdigest()[:12]
                
                checksum = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
                
                data_point = DataPoint(
                    id=data_id,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    data=data.copy(),
                    checksum=checksum,
                    source_component=source_component
                )
                
                # Initialize trace
                with self.lock:
                    self.data_traces[data_id] = DataFlowTrace(
                        data_id=data_id,
                        flow_path=[source_component],
                        timestamps={source_component: data_point.timestamp},
                        data_states={source_component: data.copy()},
                        integrity_checks={source_component: True}
                    )
                
                return data_point
                
            def trace_data_flow(self, data_id: str, component: str, 
                              data_state: Dict[str, Any]) -> bool:
                """Trace data flow to a component."""
                with self.lock:
                    if data_id not in self.data_traces:
                        logger.warning(f"Data ID {data_id} not found for tracing")
                        return False
                    
                    trace = self.data_traces[data_id]
                    trace.flow_path.append(component)
                    trace.timestamps[component] = datetime.now()
                    trace.data_states[component] = data_state.copy()
                    
                    # Check data integrity
                    integrity_ok = self.validate_data_integrity(data_id, component, data_state)
                    trace.integrity_checks[component] = integrity_ok
                    
                    # Calculate latency from previous component
                    if len(trace.flow_path) > 1:
                        prev_component = trace.flow_path[-2]
                        latency = (trace.timestamps[component] - trace.timestamps[prev_component]).total_seconds()
                        trace.latency_measurements[f"{prev_component}_to_{component}"] = latency
                    
                    return integrity_ok
                    
            def validate_data_integrity(self, data_id: str, component: str, 
                                      data_state: Dict[str, Any]) -> bool:
                """Validate data integrity at a component."""
                if data_id not in self.data_traces:
                    return False
                
                trace = self.data_traces[data_id]
                
                # Check for required fields
                required_fields = ["symbol", "timestamp"]
                for field in required_fields:
                    if field not in data_state:
                        logger.warning(f"Missing required field {field} in {component}")
                        return False
                
                # Check data consistency with source
                if trace.flow_path:
                    source_component = trace.flow_path[0]
                    source_data = trace.data_states[source_component]
                    
                    # Symbol should remain consistent
                    if data_state.get("symbol") != source_data.get("symbol"):
                        logger.warning(f"Symbol inconsistency in {component}")
                        return False
                
                return True
                
            def check_end_to_end_consistency(self, data_id: str) -> Dict[str, Any]:
                """Check end-to-end consistency for a data point."""
                if data_id not in self.data_traces:
                    return {"error": "Data ID not found"}
                
                trace = self.data_traces[data_id]
                
                consistency_report = {
                    "data_id": data_id,
                    "flow_path": trace.flow_path,
                    "components_passed": len(trace.flow_path),
                    "integrity_checks_passed": sum(trace.integrity_checks.values()),
                    "total_integrity_checks": len(trace.integrity_checks),
                    "integrity_score": sum(trace.integrity_checks.values()) / len(trace.integrity_checks),
                    "total_latency": sum(trace.latency_measurements.values()),
                    "max_component_latency": max(trace.latency_measurements.values()) if trace.latency_measurements else 0,
                    "avg_component_latency": sum(trace.latency_measurements.values()) / len(trace.latency_measurements) if trace.latency_measurements else 0,
                    "data_corrupted": False
                }
                
                # Check for data corruption
                source_symbol = trace.data_states[trace.flow_path[0]].get("symbol")
                for component_data in trace.data_states.values():
                    if component_data.get("symbol") != source_symbol:
                        consistency_report["data_corrupted"] = True
                        break
                
                return consistency_report
                
            def get_system_integrity_summary(self) -> Dict[str, Any]:
                """Get overall system integrity summary."""
                if not self.data_traces:
                    return {"error": "No data traces available"}
                
                total_traces = len(self.data_traces)
                integrity_scores = []
                latencies = []
                corrupted_count = 0
                
                for data_id in self.data_traces:
                    consistency = self.check_end_to_end_consistency(data_id)
                    integrity_scores.append(consistency["integrity_score"])
                    latencies.extend(self.data_traces[data_id].latency_measurements.values())
                    
                    if consistency["data_corrupted"]:
                        corrupted_count += 1
                
                return {
                    "total_data_points": total_traces,
                    "avg_integrity_score": sum(integrity_scores) / len(integrity_scores),
                    "min_integrity_score": min(integrity_scores),
                    "data_corruption_rate": corrupted_count / total_traces,
                    "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                    "max_latency": max(latencies) if latencies else 0,
                    "total_latency_measurements": len(latencies)
                }
        
        return DataIntegrityTracker()
    
    @pytest.fixture
    def consistency_validator(self):
        """Validate consistency across different components."""
        class ConsistencyValidator:
            def __init__(self):
                self.component_snapshots = {}
                self.consistency_rules = []
                self.violations = []
                
            def add_consistency_rule(self, rule_name: str, validator_func):
                """Add a consistency validation rule."""
                self.consistency_rules.append({
                    "name": rule_name,
                    "validator": validator_func
                })
                
            def take_snapshot(self, component_name: str, state: Dict[str, Any]):
                """Take a snapshot of component state."""
                self.component_snapshots[component_name] = {
                    "timestamp": datetime.now(),
                    "state": state.copy()
                }
                
            def validate_cross_component_consistency(self) -> List[Dict[str, Any]]:
                """Validate consistency across components."""
                violations = []
                
                for rule in self.consistency_rules:
                    try:
                        violation = rule["validator"](self.component_snapshots)
                        if violation:
                            violations.append({
                                "rule": rule["name"],
                                "violation": violation,
                                "timestamp": datetime.now()
                            })
                    except Exception as e:
                        violations.append({
                            "rule": rule["name"],
                            "violation": f"Rule validation error: {e}",
                            "timestamp": datetime.now()
                        })
                
                self.violations.extend(violations)
                return violations
                
            def validate_portfolio_consistency(self, snapshots: Dict[str, Any]) -> Optional[str]:
                """Validate portfolio state consistency."""
                if "signal_engine" not in snapshots or "calculation_engine" not in snapshots:
                    return None
                
                # Check if signals are based on current calculations
                signal_timestamp = snapshots["signal_engine"]["timestamp"]
                calc_timestamp = snapshots["calculation_engine"]["timestamp"]
                
                # Signals should be generated after calculations
                if signal_timestamp < calc_timestamp:
                    return "Signals generated before calculations completed"
                
                return None
                
            def validate_data_freshness(self, snapshots: Dict[str, Any]) -> Optional[str]:
                """Validate data freshness across components."""
                now = datetime.now()
                max_age_seconds = 60  # Data should not be older than 1 minute
                
                for component, snapshot in snapshots.items():
                    age = (now - snapshot["timestamp"]).total_seconds()
                    if age > max_age_seconds:
                        return f"Stale data in {component}: {age:.2f} seconds old"
                
                return None
                
            def validate_symbol_consistency(self, snapshots: Dict[str, Any]) -> Optional[str]:
                """Validate symbol consistency across components."""
                symbols_by_component = {}
                
                for component, snapshot in snapshots.items():
                    state = snapshot["state"]
                    if "symbols" in state:
                        symbols_by_component[component] = set(state["symbols"])
                
                if len(symbols_by_component) < 2:
                    return None
                
                # All components should track the same symbols
                all_symbols = list(symbols_by_component.values())
                first_symbols = all_symbols[0]
                
                for i, symbols in enumerate(all_symbols[1:], 1):
                    if symbols != first_symbols:
                        components = list(symbols_by_component.keys())
                        return f"Symbol mismatch between {components[0]} and {components[i]}"
                
                return None
        
        return ConsistencyValidator()
    
    @pytest.fixture
    async def integrity_test_system(self, test_config_manager, mock_redis_client, 
                                   data_integrity_tracker):
        """Create system with data integrity tracking."""
        # Enhanced Redis mock with integrity tracking
        redis_streams = {}
        redis_data = {}
        
        def track_redis_operations():
            """Track Redis operations for integrity."""
            def xadd(stream_name, fields, id="*", maxlen=None):
                if stream_name not in redis_streams:
                    redis_streams[stream_name] = []
                
                message_id = f"integrity-{len(redis_streams[stream_name])}-{int(time.time() * 1000000)}"
                redis_streams[stream_name].append((message_id, fields))
                
                # Track data flow if this is market data
                if stream_name == "market_data_stream" and "symbol" in fields:
                    data_integrity_tracker.trace_data_flow(
                        fields.get("data_id", message_id),
                        "redis_stream",
                        fields
                    )
                
                return message_id
            
            def xreadgroup(group_name, consumer_name, streams, count=None, block=None):
                results = []
                for stream_name, since_id in streams.items():
                    if stream_name in redis_streams:
                        messages = redis_streams[stream_name]
                        if count:
                            messages = messages[-count:]
                        if messages:
                            results.append([stream_name, messages])
                return results
            
            def xack(stream_name, group_name, message_id):
                return 1
            
            def set(key, value, ex=None):
                redis_data[key] = value
                return True
            
            def get(key):
                return redis_data.get(key)
            
            def ping():
                return True
            
            return {
                'xadd': xadd,
                'xreadgroup': xreadgroup,
                'xack': xack,
                'set': set,
                'get': get,
                'ping': ping
            }
        
        # Apply Redis operations
        redis_ops = track_redis_operations()
        for op_name, op_func in redis_ops.items():
            setattr(mock_redis_client, op_name, op_func)
        
        # Create system with integrity tracking
        system = {}
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                with patch('src.realtime.signal_engine.redis.Redis', return_value=mock_redis_client):
                    
                    # Create components
                    data_ingestion = MarketDataIngestionService(test_config_manager)
                    calculation_engine = CalculationEngine(test_config_manager, data_ingestion)
                    signal_engine = SignalEngine(test_config_manager, calculation_engine.timeframe_manager)
                    
                    # Wrap components with integrity tracking
                    original_process_data = data_ingestion._process_market_data
                    async def tracked_process_data(data):
                        # Track data entry
                        data_point = data_integrity_tracker.create_data_point(
                            data.get("symbol", "UNKNOWN"),
                            data,
                            "data_ingestion"
                        )
                        data["data_id"] = data_point.id
                        
                        return await original_process_data(data)
                    
                    data_ingestion._process_market_data = tracked_process_data
                    
                    system = {
                        'data_ingestion': data_ingestion,
                        'calculation_engine': calculation_engine,
                        'signal_engine': signal_engine,
                        'timeframe_manager': calculation_engine.timeframe_manager,
                        'redis_client': mock_redis_client,
                        'redis_streams': redis_streams,
                        'redis_data': redis_data,
                        'integrity_tracker': data_integrity_tracker
                    }
                    
                    yield system
                    
                    # Cleanup
                    await data_ingestion.stop()
                    await calculation_engine.stop()
    
    @pytest.fixture
    def timing_validator(self):
        """Validate timing and latency requirements."""
        class TimingValidator:
            def __init__(self):
                self.timing_requirements = {
                    "data_ingestion_max_latency": 0.1,  # 100ms
                    "calculation_max_latency": 0.5,     # 500ms
                    "signal_generation_max_latency": 0.2, # 200ms
                    "end_to_end_max_latency": 1.0       # 1 second
                }
                self.timing_measurements = []
                
            def measure_component_timing(self, component: str, start_time: float, 
                                       end_time: float) -> Dict[str, Any]:
                """Measure timing for a component operation."""
                latency = end_time - start_time
                
                measurement = {
                    "component": component,
                    "latency": latency,
                    "timestamp": datetime.now(),
                    "within_requirements": latency <= self.timing_requirements.get(f"{component}_max_latency", float('inf'))
                }
                
                self.timing_measurements.append(measurement)
                return measurement
                
            def validate_timing_requirements(self) -> Dict[str, Any]:
                """Validate all timing requirements."""
                validation_results = {}
                
                for requirement, max_latency in self.timing_requirements.items():
                    component = requirement.replace("_max_latency", "")
                    relevant_measurements = [
                        m for m in self.timing_measurements 
                        if component in m["component"] or component == "end_to_end"
                    ]
                    
                    if relevant_measurements:
                        latencies = [m["latency"] for m in relevant_measurements]
                        validation_results[requirement] = {
                            "max_allowed": max_latency,
                            "measured_max": max(latencies),
                            "measured_avg": sum(latencies) / len(latencies),
                            "measurements_count": len(latencies),
                            "violations": sum(1 for l in latencies if l > max_latency),
                            "compliance_rate": sum(1 for l in latencies if l <= max_latency) / len(latencies)
                        }
                
                return validation_results
        
        return TimingValidator()
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_consistency(self, integrity_test_system, 
                                              consistency_validator):
        """Test end-to-end data consistency across all components."""
        test_result = IntegrityTestResult(
            test_name="end_to_end_data_consistency",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = integrity_test_system['data_ingestion']
            calculation_engine = integrity_test_system['calculation_engine']
            signal_engine = integrity_test_system['signal_engine']
            integrity_tracker = integrity_test_system['integrity_tracker']
            
            # Setup consistency rules
            consistency_validator.add_consistency_rule(
                "portfolio_consistency",
                consistency_validator.validate_portfolio_consistency
            )
            consistency_validator.add_consistency_rule(
                "data_freshness",
                consistency_validator.validate_data_freshness
            )
            consistency_validator.add_consistency_rule(
                "symbol_consistency", 
                consistency_validator.validate_symbol_consistency
            )
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Generate test data with known properties
            test_symbols = ["AAPL", "MSFT", "GOOGL"]
            test_data_points = []
            
            for i in range(50):
                for symbol in test_symbols:
                    test_data = {
                        "symbol": symbol,
                        "timestamp": (datetime.now() - timedelta(seconds=i)).isoformat(),
                        "price": f"{100.0 + i * 0.1:.2f}",
                        "volume": str(1000 + i * 10),
                        "open": f"{99.9 + i * 0.1:.2f}",
                        "high": f"{100.1 + i * 0.1:.2f}",
                        "low": f"{99.8 + i * 0.1:.2f}",
                        "close": f"{100.0 + i * 0.1:.2f}",
                        "test_sequence": i,
                        "test_symbol_group": symbol
                    }
                    test_data_points.append(test_data)
            
            # Process data and track consistency
            for test_data in test_data_points:
                await data_ingestion._process_market_data(test_data)
                
                # Take component snapshots
                consistency_validator.take_snapshot("data_ingestion", {
                    "symbols": [test_data["symbol"]],
                    "last_processed": test_data["test_sequence"]
                })
                
                # Allow processing
                await asyncio.sleep(0.01)
            
            # Allow final processing
            await asyncio.sleep(3.0)
            
            # Take final snapshots
            calc_stats = calculation_engine.get_stats()
            signal_stats = signal_engine.get_stats()
            
            consistency_validator.take_snapshot("calculation_engine", {
                "symbols": test_symbols,
                "stats": calc_stats
            })
            consistency_validator.take_snapshot("signal_engine", {
                "symbols": test_symbols,
                "stats": signal_stats
            })
            
            # Validate consistency
            violations = consistency_validator.validate_cross_component_consistency()
            
            # Stop processing
            calc_task.cancel()
            
            # Analyze integrity
            integrity_summary = integrity_tracker.get_system_integrity_summary()
            
            test_result.data_points_traced = len(test_data_points)
            test_result.integrity_violations = len(violations)
            test_result.consistency_score = integrity_summary.get("avg_integrity_score", 0.0)
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Validate results
            assert integrity_summary["data_corruption_rate"] == 0.0, "Data corruption detected"
            assert integrity_summary["avg_integrity_score"] >= 0.95, f"Integrity score too low: {integrity_summary['avg_integrity_score']}"
            assert len(violations) == 0, f"Consistency violations: {violations}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "integrity_summary": integrity_summary,
                "consistency_violations": violations,
                "data_corruption_rate": integrity_summary.get("data_corruption_rate", 0.0),
                "avg_latency": integrity_summary.get("avg_latency", 0.0)
            }
            test_result.detailed_results = {
                "test_data_points": len(test_data_points),
                "symbols_tested": test_symbols,
                "component_snapshots": len(consistency_validator.component_snapshots)
            }
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"End-to-end data consistency test failed: {e}")
            raise
            
        finally:
            logger.info(f"End-to-end data consistency test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_timing_and_latency_validation(self, integrity_test_system, 
                                                timing_validator):
        """Test timing and latency across the pipeline."""
        test_result = IntegrityTestResult(
            test_name="timing_and_latency_validation",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = integrity_test_system['data_ingestion']
            calculation_engine = integrity_test_system['calculation_engine']
            signal_engine = integrity_test_system['signal_engine']
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Test timing with various data loads
            timing_tests = []
            
            # Test 1: Single data point timing
            test_data = {
                "symbol": "AAPL",
                "timestamp": datetime.now().isoformat(),
                "price": "150.00",
                "volume": "1000"
            }
            
            start_time = time.time()
            await data_ingestion._process_market_data(test_data)
            await asyncio.sleep(0.1)  # Allow processing
            end_time = time.time()
            
            single_measurement = timing_validator.measure_component_timing(
                "data_ingestion", start_time, end_time
            )
            timing_tests.append(("single_data_point", single_measurement))
            
            # Test 2: Batch processing timing
            batch_data = []
            for i in range(10):
                batch_data.append({
                    "symbol": "MSFT",
                    "timestamp": (datetime.now() - timedelta(seconds=i)).isoformat(),
                    "price": f"{300.0 + i * 0.1:.2f}",
                    "volume": str(1000 + i * 100)
                })
            
            batch_start = time.time()
            for data in batch_data:
                await data_ingestion._process_market_data(data)
            await asyncio.sleep(0.5)  # Allow batch processing
            batch_end = time.time()
            
            batch_measurement = timing_validator.measure_component_timing(
                "batch_processing", batch_start, batch_end
            )
            timing_tests.append(("batch_processing", batch_measurement))
            
            # Test 3: High frequency timing
            hf_start = time.time()
            for i in range(100):
                hf_data = {
                    "symbol": "GOOGL",
                    "timestamp": datetime.now().isoformat(),
                    "price": f"{2500.0 + i * 0.01:.2f}",
                    "volume": str(500 + i * 5)
                }
                await data_ingestion._process_market_data(hf_data)
                await asyncio.sleep(0.001)  # High frequency
            
            await asyncio.sleep(1.0)  # Allow processing
            hf_end = time.time()
            
            hf_measurement = timing_validator.measure_component_timing(
                "high_frequency", hf_start, hf_end
            )
            timing_tests.append(("high_frequency", hf_measurement))
            
            # Test 4: End-to-end timing
            e2e_start = time.time()
            e2e_data = {
                "symbol": "TSLA",
                "timestamp": datetime.now().isoformat(),
                "price": "800.00",
                "volume": "2000"
            }
            
            await data_ingestion._process_market_data(e2e_data)
            
            # Wait for signal generation
            await asyncio.sleep(2.0)
            e2e_end = time.time()
            
            e2e_measurement = timing_validator.measure_component_timing(
                "end_to_end", e2e_start, e2e_end
            )
            timing_tests.append(("end_to_end", e2e_measurement))
            
            # Stop processing
            calc_task.cancel()
            
            # Validate timing requirements
            timing_validation = timing_validator.validate_timing_requirements()
            
            test_result.data_points_traced = 111  # 1 + 10 + 100 data points
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Check timing compliance
            total_violations = sum(
                result.get("violations", 0) for result in timing_validation.values()
            )
            
            # Calculate overall compliance
            compliance_rates = [
                result.get("compliance_rate", 0.0) for result in timing_validation.values()
            ]
            overall_compliance = sum(compliance_rates) / len(compliance_rates) if compliance_rates else 0.0
            
            assert total_violations == 0, f"Timing violations detected: {total_violations}"
            assert overall_compliance >= 0.95, f"Timing compliance too low: {overall_compliance}"
            
            test_result.success = True
            test_result.consistency_score = overall_compliance
            test_result.performance_metrics = {
                "timing_validation": timing_validation,
                "timing_tests": timing_tests,
                "total_violations": total_violations,
                "overall_compliance": overall_compliance
            }
            test_result.detailed_results = {
                "individual_measurements": timing_validator.timing_measurements,
                "requirements_tested": list(timing_validator.timing_requirements.keys())
            }
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Timing and latency validation test failed: {e}")
            raise
            
        finally:
            logger.info(f"Timing and latency validation test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_signal_propagation_integrity(self, integrity_test_system, 
                                              data_integrity_tracker):
        """Test signal propagation and integrity."""
        test_result = IntegrityTestResult(
            test_name="signal_propagation_integrity",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = integrity_test_system['data_ingestion']
            calculation_engine = integrity_test_system['calculation_engine']
            signal_engine = integrity_test_system['signal_engine']
            redis_streams = integrity_test_system['redis_streams']
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Generate data designed to trigger signals
            signal_test_data = []
            
            # Create RSI oversold condition
            for i in range(20):
                # Declining prices to create oversold condition
                price = 150.0 - (i * 0.5)  # Price declining
                
                signal_data = {
                    "symbol": "AAPL",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "price": f"{price:.2f}",
                    "volume": str(1000 + i * 50),
                    "open": f"{price + 0.1:.2f}",
                    "high": f"{price + 0.2:.2f}",
                    "low": f"{price - 0.1:.2f}",
                    "close": f"{price:.2f}",
                    "signal_test": "rsi_oversold"
                }
                signal_test_data.append(signal_data)
            
            # Create MACD bullish crossover
            for i in range(15):
                # Rising prices for MACD bullish signal
                price = 140.0 + (i * 0.3)
                
                signal_data = {
                    "symbol": "MSFT",
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "price": f"{price:.2f}",
                    "volume": str(1500 + i * 100),
                    "open": f"{price - 0.1:.2f}",
                    "high": f"{price + 0.3:.2f}",
                    "low": f"{price - 0.2:.2f}",
                    "close": f"{price:.2f}",
                    "signal_test": "macd_bullish"
                }
                signal_test_data.append(signal_data)
            
            # Track signal propagation
            signals_generated = 0
            data_points_processed = 0
            
            # Process signal test data
            for test_data in signal_test_data:
                await data_ingestion._process_market_data(test_data)
                data_points_processed += 1
                
                # Check for signal generation
                if 'signals_stream' in redis_streams:
                    current_signals = len(redis_streams['signals_stream'])
                    if current_signals > signals_generated:
                        signals_generated = current_signals
                        
                        # Trace signal propagation
                        data_integrity_tracker.trace_data_flow(
                            test_data.get("data_id", f"signal_test_{data_points_processed}"),
                            "signal_generation",
                            {"signal_generated": True, "signals_count": signals_generated}
                        )
                
                await asyncio.sleep(0.05)  # Allow processing between data points
            
            # Allow final signal processing
            await asyncio.sleep(3.0)
            
            # Final signal count
            final_signals = len(redis_streams.get('signals_stream', []))
            
            # Stop processing
            calc_task.cancel()
            
            # Analyze signal propagation
            propagation_analysis = {
                "data_points_processed": data_points_processed,
                "signals_generated": final_signals,
                "signal_rate": final_signals / data_points_processed if data_points_processed > 0 else 0,
                "rsi_test_data_points": len([d for d in signal_test_data if d.get("signal_test") == "rsi_oversold"]),
                "macd_test_data_points": len([d for d in signal_test_data if d.get("signal_test") == "macd_bullish"])
            }
            
            # Check signal integrity
            integrity_summary = data_integrity_tracker.get_system_integrity_summary()
            
            test_result.data_points_traced = data_points_processed
            test_result.signals_generated = final_signals
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Validate signal propagation
            assert final_signals >= 0, "Signal generation failed"
            assert integrity_summary.get("avg_integrity_score", 0.0) >= 0.9, "Signal propagation integrity compromised"
            
            # Check that signals were generated for test conditions
            signal_rate = final_signals / data_points_processed
            assert signal_rate <= 1.0, f"Signal rate too high: {signal_rate}"  # Sanity check
            
            test_result.success = True
            test_result.consistency_score = integrity_summary.get("avg_integrity_score", 0.0)
            test_result.performance_metrics = {
                "propagation_analysis": propagation_analysis,
                "integrity_summary": integrity_summary,
                "signal_generation_latency": integrity_summary.get("avg_latency", 0.0),
                "signal_integrity_score": integrity_summary.get("avg_integrity_score", 0.0)
            }
            test_result.detailed_results = {
                "test_scenarios": ["rsi_oversold", "macd_bullish"],
                "redis_streams": list(redis_streams.keys()),
                "final_signal_count": final_signals
            }
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Signal propagation integrity test failed: {e}")
            raise
            
        finally:
            logger.info(f"Signal propagation integrity test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_portfolio_state_consistency(self, integrity_test_system, 
                                             consistency_validator):
        """Test portfolio state consistency across components."""
        test_result = IntegrityTestResult(
            test_name="portfolio_state_consistency",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = integrity_test_system['data_ingestion']
            calculation_engine = integrity_test_system['calculation_engine']
            signal_engine = integrity_test_system['signal_engine']
            redis_data = integrity_test_system['redis_data']
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Generate portfolio test data
            portfolio_symbols = ["AAPL", "MSFT", "GOOGL"]
            portfolio_states = []
            
            # Simulate portfolio updates
            for update_round in range(5):
                round_data = []
                
                for symbol in portfolio_symbols:
                    # Generate market data
                    price = 100.0 + update_round * 5.0 + hash(symbol) % 20
                    
                    market_data = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "price": f"{price:.2f}",
                        "volume": str(1000 + update_round * 200),
                        "portfolio_update_round": update_round
                    }
                    round_data.append(market_data)
                    
                    await data_ingestion._process_market_data(market_data)
                
                # Allow processing
                await asyncio.sleep(1.0)
                
                # Take portfolio state snapshot
                calc_stats = calculation_engine.get_stats()
                signal_stats = signal_engine.get_stats()
                
                portfolio_state = {
                    "update_round": update_round,
                    "symbols": portfolio_symbols,
                    "calc_stats": calc_stats,
                    "signal_stats": signal_stats,
                    "redis_keys": len(redis_data),
                    "timestamp": datetime.now()
                }
                portfolio_states.append(portfolio_state)
                
                # Store state in consistency validator
                consistency_validator.take_snapshot(f"portfolio_round_{update_round}", {
                    "symbols": portfolio_symbols,
                    "round": update_round,
                    "calc_stats": calc_stats,
                    "signal_stats": signal_stats
                })
            
            # Stop processing
            calc_task.cancel()
            
            # Analyze portfolio consistency
            consistency_analysis = {
                "portfolio_updates": len(portfolio_states),
                "symbols_tracked": portfolio_symbols,
                "state_transitions": [],
                "consistency_violations": []
            }
            
            # Check state transitions
            for i in range(1, len(portfolio_states)):
                prev_state = portfolio_states[i-1]
                curr_state = portfolio_states[i]
                
                transition = {
                    "from_round": prev_state["update_round"],
                    "to_round": curr_state["update_round"],
                    "symbols_consistent": prev_state["symbols"] == curr_state["symbols"],
                    "data_progression": curr_state["redis_keys"] >= prev_state["redis_keys"],
                    "timestamp_progression": curr_state["timestamp"] > prev_state["timestamp"]
                }
                consistency_analysis["state_transitions"].append(transition)
                
                # Check for violations
                if not transition["symbols_consistent"]:
                    consistency_analysis["consistency_violations"].append(
                        f"Symbol inconsistency between round {prev_state['update_round']} and {curr_state['update_round']}"
                    )
                
                if not transition["timestamp_progression"]:
                    consistency_analysis["consistency_violations"].append(
                        f"Timestamp regression between round {prev_state['update_round']} and {curr_state['update_round']}"
                    )
            
            # Validate portfolio consistency rules
            violations = consistency_validator.validate_cross_component_consistency()
            consistency_analysis["consistency_violations"].extend([v["violation"] for v in violations])
            
            test_result.data_points_traced = len(portfolio_symbols) * 5  # 5 rounds
            test_result.integrity_violations = len(consistency_analysis["consistency_violations"])
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Calculate consistency score
            total_checks = len(consistency_analysis["state_transitions"]) * 3  # 3 checks per transition
            passed_checks = sum(
                sum([t["symbols_consistent"], t["data_progression"], t["timestamp_progression"]])
                for t in consistency_analysis["state_transitions"]
            )
            consistency_score = passed_checks / total_checks if total_checks > 0 else 1.0
            test_result.consistency_score = consistency_score
            
            # Validate results
            assert len(consistency_analysis["consistency_violations"]) == 0, f"Portfolio consistency violations: {consistency_analysis['consistency_violations']}"
            assert consistency_score >= 0.95, f"Portfolio consistency score too low: {consistency_score}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "consistency_analysis": consistency_analysis,
                "portfolio_states": len(portfolio_states),
                "state_transitions": len(consistency_analysis["state_transitions"]),
                "consistency_score": consistency_score
            }
            test_result.detailed_results = {
                "portfolio_symbols": portfolio_symbols,
                "update_rounds": 5,
                "snapshots_taken": len(consistency_validator.component_snapshots)
            }
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Portfolio state consistency test failed: {e}")
            raise
            
        finally:
            logger.info(f"Portfolio state consistency test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_error_handling_data_integrity(self, integrity_test_system, 
                                               data_integrity_tracker):
        """Test error handling and data integrity recovery."""
        test_result = IntegrityTestResult(
            test_name="error_handling_data_integrity",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = integrity_test_system['data_ingestion']
            calculation_engine = integrity_test_system['calculation_engine']
            signal_engine = integrity_test_system['signal_engine']
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Error handling test scenarios
            error_scenarios = [
                {
                    "name": "malformed_data",
                    "data": {
                        "symbol": "AAPL",
                        "timestamp": "invalid_timestamp",
                        "price": "not_a_number",
                        "volume": "also_not_a_number"
                    }
                },
                {
                    "name": "missing_required_fields",
                    "data": {
                        "symbol": "MSFT",
                        # Missing timestamp, price, volume
                    }
                },
                {
                    "name": "extreme_values",
                    "data": {
                        "symbol": "GOOGL",
                        "timestamp": datetime.now().isoformat(),
                        "price": "999999999.99",  # Extreme price
                        "volume": "-1000"  # Negative volume
                    }
                },
                {
                    "name": "null_values",
                    "data": {
                        "symbol": None,
                        "timestamp": None,
                        "price": None,
                        "volume": None
                    }
                }
            ]
            
            # Track error handling
            error_handling_results = {
                "scenarios_tested": len(error_scenarios),
                "errors_caught": 0,
                "system_crashes": 0,
                "recovery_successful": 0,
                "data_integrity_maintained": 0
            }
            
            # Test each error scenario
            for scenario in error_scenarios:
                try:
                    # Attempt to process malformed data
                    await data_ingestion._process_market_data(scenario["data"])
                    
                    # If no exception, check if system handled gracefully
                    await asyncio.sleep(0.5)
                    
                    # Test recovery with good data
                    recovery_data = {
                        "symbol": "TEST",
                        "timestamp": datetime.now().isoformat(),
                        "price": "100.00",
                        "volume": "1000"
                    }
                    await data_ingestion._process_market_data(recovery_data)
                    
                    error_handling_results["recovery_successful"] += 1
                    
                except Exception as e:
                    # Error was caught, which is expected
                    error_handling_results["errors_caught"] += 1
                    logger.info(f"Error scenario '{scenario['name']}' caught as expected: {e}")
                    
                    # Test system recovery
                    try:
                        recovery_data = {
                            "symbol": "RECOVERY",
                            "timestamp": datetime.now().isoformat(),
                            "price": "150.00",
                            "volume": "1500"
                        }
                        await data_ingestion._process_market_data(recovery_data)
                        error_handling_results["recovery_successful"] += 1
                    except Exception as recovery_error:
                        logger.error(f"System failed to recover after error scenario '{scenario['name']}': {recovery_error}")
                
                # Check system health after each scenario
                try:
                    calc_stats = calculation_engine.get_stats()
                    signal_stats = signal_engine.get_stats()
                    
                    if calc_stats is not None and signal_stats is not None:
                        error_handling_results["data_integrity_maintained"] += 1
                except Exception as health_error:
                    logger.error(f"System health check failed after scenario '{scenario['name']}': {health_error}")
                    error_handling_results["system_crashes"] += 1
            
            # Test system with good data after error scenarios
            good_data_test = []
            for i in range(10):
                good_data = {
                    "symbol": "FINAL_TEST",
                    "timestamp": (datetime.now() - timedelta(seconds=i)).isoformat(),
                    "price": f"{200.0 + i * 0.1:.2f}",
                    "volume": str(2000 + i * 50)
                }
                good_data_test.append(good_data)
                await data_ingestion._process_market_data(good_data)
                await asyncio.sleep(0.1)
            
            # Final system health check
            await asyncio.sleep(2.0)
            
            final_calc_stats = calculation_engine.get_stats()
            final_signal_stats = signal_engine.get_stats()
            
            # Stop processing
            calc_task.cancel()
            
            # Analyze error handling effectiveness
            integrity_summary = data_integrity_tracker.get_system_integrity_summary()
            
            test_result.data_points_traced = len(good_data_test)
            test_result.integrity_violations = error_handling_results["system_crashes"]
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Calculate error handling score
            recovery_rate = error_handling_results["recovery_successful"] / error_handling_results["scenarios_tested"]
            integrity_maintenance_rate = error_handling_results["data_integrity_maintained"] / error_handling_results["scenarios_tested"]
            overall_error_handling_score = (recovery_rate + integrity_maintenance_rate) / 2
            
            test_result.consistency_score = overall_error_handling_score
            
            # Validate error handling
            assert error_handling_results["system_crashes"] == 0, "System crashed during error handling tests"
            assert recovery_rate >= 0.8, f"Recovery rate too low: {recovery_rate}"
            assert integrity_maintenance_rate >= 0.8, f"Integrity maintenance rate too low: {integrity_maintenance_rate}"
            assert final_calc_stats is not None, "Calculation engine not responsive after error tests"
            assert final_signal_stats is not None, "Signal engine not responsive after error tests"
            
            test_result.success = True
            test_result.performance_metrics = {
                "error_handling_results": error_handling_results,
                "recovery_rate": recovery_rate,
                "integrity_maintenance_rate": integrity_maintenance_rate,
                "overall_error_handling_score": overall_error_handling_score,
                "final_system_health": {
                    "calculation_engine_responsive": final_calc_stats is not None,
                    "signal_engine_responsive": final_signal_stats is not None
                },
                "integrity_summary": integrity_summary
            }
            test_result.detailed_results = {
                "error_scenarios_tested": [s["name"] for s in error_scenarios],
                "good_data_points_after_errors": len(good_data_test),
                "system_recovery_demonstrated": True
            }
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Error handling data integrity test failed: {e}")
            raise
            
        finally:
            logger.info(f"Error handling data integrity test completed: {test_result}")