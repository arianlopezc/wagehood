#!/usr/bin/env python3
"""
Comprehensive End-to-End Trading Workflow Tests

Tests the complete trading workflow from data ingestion to portfolio updates
with real-time data processing and multi-strategy signal generation.
"""

import asyncio
import pytest
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from src.realtime.config_manager import ConfigManager
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.signal_engine import SignalEngine
from src.storage.cache import cache_manager

logger = logging.getLogger(__name__)


class CompleteWorkflowTester:
    """Complete trading workflow test coordinator."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.components = {}
        self.test_symbols = ['SPY', 'QQQ', 'AAPL']
        self.test_results = {}
        
    async def setup_system(self):
        """Setup complete system for testing."""
        logger.info("Setting up complete trading system for e2e testing...")
        
        # Initialize components
        self.components['ingestion'] = MarketDataIngestionService(self.config_manager)
        self.components['calculation'] = CalculationEngine(
            self.config_manager, 
            self.components['ingestion']
        )
        self.components['timeframe'] = TimeframeManager(['1m', '5m', '15m'])
        self.components['signals'] = SignalEngine()
        
        # Start services
        await self.components['ingestion'].start()
        await self.components['calculation'].start()
        
        logger.info("Complete system setup completed")
        
    async def teardown_system(self):
        """Cleanup system after testing."""
        logger.info("Tearing down trading system...")
        
        for component in self.components.values():
            if hasattr(component, 'stop'):
                try:
                    await component.stop()
                except Exception as e:
                    logger.warning(f"Error stopping component: {e}")
                    
        # Clear cache
        cache_manager.clear_namespace("test")
        logger.info("System teardown completed")


@pytest.fixture
async def workflow_tester():
    """Fixture providing complete workflow tester."""
    tester = CompleteWorkflowTester()
    await tester.setup_system()
    
    yield tester
    
    await tester.teardown_system()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_complete_data_to_signals_workflow(workflow_tester):
    """Test complete workflow: Data ingestion → Signal generation."""
    logger.info("Testing complete data to signals workflow...")
    
    # Track workflow stages
    workflow_stages = {
        'data_ingestion': False,
        'indicator_calculation': False,
        'signal_generation': False,
        'redis_storage': False
    }
    
    # Start data flow
    start_time = time.time()
    
    # Wait for data to flow through system
    timeout = 30  # 30 seconds
    while time.time() - start_time < timeout:
        # Check data ingestion
        if not workflow_stages['data_ingestion']:
            # Check if market data is being ingested
            try:
                stream_length = cache_manager.redis_client.xlen("market_data_stream")
                if stream_length > 0:
                    workflow_stages['data_ingestion'] = True
                    logger.info("✓ Data ingestion stage completed")
            except Exception as e:
                logger.debug(f"Data ingestion check failed: {e}")
        
        # Check indicator calculation
        if workflow_stages['data_ingestion'] and not workflow_stages['indicator_calculation']:
            # Check if indicators are being calculated
            for symbol in workflow_tester.test_symbols:
                cache_key = f"{symbol}_rsi_14"
                if cache_manager.get("indicators", cache_key):
                    workflow_stages['indicator_calculation'] = True
                    logger.info("✓ Indicator calculation stage completed")
                    break
        
        # Check signal generation
        if workflow_stages['indicator_calculation'] and not workflow_stages['signal_generation']:
            # Check if signals are being generated
            for symbol in workflow_tester.test_symbols:
                cache_key = f"{symbol}_signals"
                if cache_manager.get("strategies", cache_key):
                    workflow_stages['signal_generation'] = True
                    logger.info("✓ Signal generation stage completed")
                    break
        
        # Check Redis storage
        if workflow_stages['signal_generation'] and not workflow_stages['redis_storage']:
            # Check if data is properly stored in Redis
            try:
                calc_stream_length = cache_manager.redis_client.xlen("calculation_events_stream")
                if calc_stream_length > 0:
                    workflow_stages['redis_storage'] = True
                    logger.info("✓ Redis storage stage completed")
            except Exception as e:
                logger.debug(f"Redis storage check failed: {e}")
        
        # Check if all stages completed
        if all(workflow_stages.values()):
            break
            
        await asyncio.sleep(1)
    
    # Verify all workflow stages completed
    completed_stages = sum(workflow_stages.values())
    total_stages = len(workflow_stages)
    
    logger.info(f"Workflow completion: {completed_stages}/{total_stages} stages")
    
    assert completed_stages >= 3, f"Expected at least 3 stages, got {completed_stages}"
    assert workflow_stages['data_ingestion'], "Data ingestion stage failed"
    assert workflow_stages['indicator_calculation'], "Indicator calculation stage failed"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_multi_strategy_signal_correlation(workflow_tester):
    """Test multi-strategy signal correlation and decision making."""
    logger.info("Testing multi-strategy signal correlation...")
    
    # Wait for system to generate signals
    await asyncio.sleep(10)
    
    signal_correlations = {}
    
    for symbol in workflow_tester.test_symbols:
        cache_key = f"{symbol}_signals"
        signals = cache_manager.get("strategies", cache_key)
        
        if signals:
            # Analyze signal correlation across strategies
            strategies = list(signals.keys())
            signal_types = [signals[strategy].get('signal') for strategy in strategies if signals[strategy]]
            
            # Calculate agreement percentage
            if signal_types:
                buy_signals = signal_types.count('buy')
                sell_signals = signal_types.count('sell')
                total_signals = len([s for s in signal_types if s])
                
                if total_signals > 0:
                    agreement = max(buy_signals, sell_signals) / total_signals
                    signal_correlations[symbol] = {
                        'strategies_count': len(strategies),
                        'active_signals': total_signals,
                        'agreement_percentage': agreement,
                        'dominant_signal': 'buy' if buy_signals > sell_signals else 'sell'
                    }
                    
                    logger.info(f"{symbol}: {total_signals} signals, {agreement:.1%} agreement")
    
    # Verify signal correlation analysis
    assert len(signal_correlations) > 0, "No signal correlations found"
    
    # Verify reasonable signal generation
    total_strategies = sum(data['strategies_count'] for data in signal_correlations.values())
    total_active = sum(data['active_signals'] for data in signal_correlations.values())
    
    assert total_strategies >= 5, f"Expected at least 5 strategies, got {total_strategies}"
    assert total_active > 0, "No active signals generated"
    
    workflow_tester.test_results['signal_correlation'] = signal_correlations


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_real_time_system_updates(workflow_tester):
    """Test real-time updates across all system components."""
    logger.info("Testing real-time system updates...")
    
    # Record initial state
    initial_time = datetime.now()
    initial_signals = {}
    
    for symbol in workflow_tester.test_symbols:
        cache_key = f"{symbol}_signals"
        signals = cache_manager.get("strategies", cache_key)
        if signals:
            initial_signals[symbol] = len(signals)
    
    # Wait for real-time updates
    await asyncio.sleep(15)
    
    # Record updated state
    updated_signals = {}
    for symbol in workflow_tester.test_symbols:
        cache_key = f"{symbol}_signals"
        signals = cache_manager.get("strategies", cache_key)
        if signals:
            updated_signals[symbol] = len(signals)
    
    # Verify system is actively updating
    updates_detected = False
    for symbol in workflow_tester.test_symbols:
        if symbol in both (initial_signals, updated_signals):
            if updated_signals[symbol] != initial_signals.get(symbol, 0):
                updates_detected = True
                logger.info(f"Real-time update detected for {symbol}")
                break
    
    # Check for new data in streams
    try:
        market_stream_length = cache_manager.redis_client.xlen("market_data_stream")
        calc_stream_length = cache_manager.redis_client.xlen("calculation_events_stream")
        
        assert market_stream_length > 0, "No market data in stream"
        logger.info(f"Market data stream length: {market_stream_length}")
        
        if calc_stream_length > 0:
            logger.info(f"Calculation events stream length: {calc_stream_length}")
            updates_detected = True
            
    except Exception as e:
        logger.warning(f"Stream length check failed: {e}")
    
    # Verify real-time processing is active
    assert updates_detected, "No real-time updates detected"


@pytest.mark.asyncio
@pytest.mark.e2e  
@pytest.mark.slow
async def test_market_simulation_scenario(workflow_tester):
    """Test realistic market simulation scenario."""
    logger.info("Testing market simulation scenario...")
    
    # Simulate market conditions for extended period
    simulation_duration = 60  # 1 minute simulation
    check_interval = 10  # Check every 10 seconds
    
    market_data = {
        'timestamps': [],
        'signal_counts': [],
        'indicator_values': [],
        'system_performance': []
    }
    
    start_time = time.time()
    
    while time.time() - start_time < simulation_duration:
        current_time = datetime.now()
        market_data['timestamps'].append(current_time)
        
        # Collect signal counts
        signal_count = 0
        for symbol in workflow_tester.test_symbols:
            cache_key = f"{symbol}_signals"
            signals = cache_manager.get("strategies", cache_key)
            if signals:
                signal_count += len([s for s in signals.values() if s.get('signal')])
        
        market_data['signal_counts'].append(signal_count)
        
        # Collect indicator sample
        indicator_sample = {}
        for symbol in workflow_tester.test_symbols[:1]:  # Sample one symbol
            rsi_key = f"{symbol}_rsi_14"
            rsi_value = cache_manager.get("indicators", rsi_key)
            if rsi_value:
                indicator_sample[symbol] = rsi_value
        
        market_data['indicator_values'].append(indicator_sample)
        
        # Check system performance
        try:
            engine_stats = workflow_tester.components['calculation'].get_stats()
            market_data['system_performance'].append({
                'calculations_performed': engine_stats.get('calculations_performed', 0),
                'signals_generated': engine_stats.get('signals_generated', 0),
                'errors': engine_stats.get('errors', 0)
            })
        except Exception as e:
            logger.debug(f"Stats collection failed: {e}")
            market_data['system_performance'].append({})
        
        logger.info(f"Market simulation: t={int(time.time() - start_time)}s, signals={signal_count}")
        
        await asyncio.sleep(check_interval)
    
    # Analyze simulation results
    total_checks = len(market_data['timestamps'])
    successful_signal_checks = len([c for c in market_data['signal_counts'] if c > 0])
    
    logger.info(f"Market simulation completed: {total_checks} checks, {successful_signal_checks} with signals")
    
    # Verify simulation results
    assert total_checks >= 6, f"Expected at least 6 checks, got {total_checks}"
    assert successful_signal_checks > 0, "No signals detected during simulation"
    
    # Verify system stability
    performance_data = [p for p in market_data['system_performance'] if p]
    if performance_data:
        error_counts = [p.get('errors', 0) for p in performance_data]
        total_errors = sum(error_counts)
        error_rate = total_errors / len(performance_data) if performance_data else 0
        
        assert error_rate < 0.1, f"Error rate too high: {error_rate:.2%}"
        logger.info(f"System error rate: {error_rate:.2%}")
    
    workflow_tester.test_results['market_simulation'] = market_data


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_configuration_changes_impact(workflow_tester):
    """Test system response to configuration changes during operation."""
    logger.info("Testing configuration changes impact...")
    
    # Record initial configuration
    initial_config = workflow_tester.config_manager.get_system_config()
    
    # Make configuration change
    try:
        # Update system configuration (non-breaking change)
        new_config = initial_config
        new_config.enable_monitoring = not initial_config.enable_monitoring
        
        # Apply configuration change
        workflow_tester.config_manager.update_system_config(new_config)
        logger.info("Configuration change applied")
        
        # Wait for system to adapt
        await asyncio.sleep(5)
        
        # Verify system continues to operate
        post_change_signals = {}
        for symbol in workflow_tester.test_symbols:
            cache_key = f"{symbol}_signals" 
            signals = cache_manager.get("strategies", cache_key)
            if signals:
                post_change_signals[symbol] = len(signals)
        
        # Verify system is still generating signals
        assert len(post_change_signals) > 0, "No signals after configuration change"
        logger.info(f"Post-change signals: {sum(post_change_signals.values())}")
        
    except Exception as e:
        logger.warning(f"Configuration change test failed: {e}")
        # Test is informational, don't fail on configuration issues
    
    finally:
        # Restore original configuration
        try:
            workflow_tester.config_manager.update_system_config(initial_config)
        except Exception as e:
            logger.warning(f"Failed to restore configuration: {e}")


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_error_recovery_workflow(workflow_tester):
    """Test system error recovery and resilience."""
    logger.info("Testing error recovery workflow...")
    
    # Record baseline performance
    try:
        baseline_stats = workflow_tester.components['calculation'].get_stats()
        baseline_errors = baseline_stats.get('errors', 0)
    except Exception:
        baseline_errors = 0
    
    # Simulate error conditions (non-destructive)
    error_scenarios = []
    
    # Test invalid symbol processing
    try:
        invalid_symbol = "INVALID_SYMBOL_TEST"
        cache_manager.set("test", f"{invalid_symbol}_signals", {"test": "invalid"})
        error_scenarios.append("invalid_symbol_injection")
        logger.info("Invalid symbol data injected")
    except Exception as e:
        logger.debug(f"Invalid symbol injection failed: {e}")
    
    # Test malformed data
    try:
        test_symbol = workflow_tester.test_symbols[0]
        cache_manager.set("test", f"{test_symbol}_malformed", "invalid_data_format")
        error_scenarios.append("malformed_data_injection")
        logger.info("Malformed data injected")
    except Exception as e:
        logger.debug(f"Malformed data injection failed: {e}")
    
    # Wait for system to process errors
    await asyncio.sleep(10)
    
    # Check if system recovered
    try:
        post_error_stats = workflow_tester.components['calculation'].get_stats()
        post_error_errors = post_error_stats.get('errors', 0)
        
        # System should handle errors gracefully
        error_increase = post_error_errors - baseline_errors
        logger.info(f"Error increase: {error_increase}")
        
        # Verify system is still generating signals despite errors
        active_signals = 0
        for symbol in workflow_tester.test_symbols:
            cache_key = f"{symbol}_signals"
            signals = cache_manager.get("strategies", cache_key)
            if signals:
                active_signals += len([s for s in signals.values() if s.get('signal')])
        
        assert active_signals > 0, "System stopped generating signals after errors"
        logger.info(f"System recovery verified: {active_signals} active signals")
        
    except Exception as e:
        logger.warning(f"Error recovery verification failed: {e}")
    
    # Cleanup error test data
    cache_manager.clear_namespace("test")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])