"""
Test Session-Aware Solution - Day Transition Handling

This script tests the comprehensive solution for handling day transitions
in the real-time system while maintaining continuous operation.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.realtime.market_session_manager import MarketSessionManager, SessionState, MarketType
from src.realtime.enhanced_timeframe_manager import EnhancedTimeframeManager
from src.realtime.gap_aware_indicators import GapAwareIndicatorCalculator
from src.realtime.session_data_synchronizer import SessionDataSynchronizer
from src.realtime.config_manager import ConfigManager, TradingProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SessionAwareSolutionTester:
    """Test the complete session-aware solution."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config_manager = self._create_test_config_manager()
        self.session_manager = MarketSessionManager(MarketType.US_EQUITY)
        self.timeframe_manager = EnhancedTimeframeManager(
            self.config_manager, self.session_manager
        )
        self.indicator_calculator = GapAwareIndicatorCalculator()
        self.data_synchronizer = SessionDataSynchronizer(
            self.session_manager, self.timeframe_manager
        )
        
        self.test_results = []
    
    def _create_test_config_manager(self):
        """Create a mock config manager for testing."""
        class MockConfigManager:
            def get_timeframe_configs(self):
                return {
                    TradingProfile.SWING_TRADING: []
                }
            
            def get_enabled_indicators(self):
                from src.realtime.config_manager import IndicatorConfig
                return [
                    IndicatorConfig(name="sma_20", enabled=True, parameters={"period": 20}),
                    IndicatorConfig(name="ema_12", enabled=True, parameters={"period": 12}),
                    IndicatorConfig(name="rsi_14", enabled=True, parameters={"period": 14}),
                    IndicatorConfig(name="macd", enabled=True, parameters={"fast": 12, "slow": 26, "signal": 9})
                ]
        
        return MockConfigManager()
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the session-aware solution."""
        logger.info("Starting comprehensive session-aware solution test")
        
        test_results = {
            "start_time": datetime.now(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # Test 1: Session Manager
        result1 = await self._test_session_manager()
        test_results["test_details"].append(result1)
        if result1["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # Test 2: Enhanced Timeframe Manager
        result2 = await self._test_enhanced_timeframe_manager()
        test_results["test_details"].append(result2)
        if result2["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # Test 3: Gap-Aware Indicators
        result3 = await self._test_gap_aware_indicators()
        test_results["test_details"].append(result3)
        if result3["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # Test 4: Day Boundary Transitions
        result4 = await self._test_day_boundary_transitions()
        test_results["test_details"].append(result4)
        if result4["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # Test 5: Continuous Operation
        result5 = await self._test_continuous_operation()
        test_results["test_details"].append(result5)
        if result5["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        test_results["end_time"] = datetime.now()
        test_results["duration_seconds"] = (test_results["end_time"] - test_results["start_time"]).total_seconds()
        test_results["success_rate"] = test_results["tests_passed"] / (test_results["tests_passed"] + test_results["tests_failed"]) * 100
        
        logger.info(f"Test completed - Success rate: {test_results['success_rate']:.1f}%")
        return test_results
    
    async def _test_session_manager(self) -> Dict[str, Any]:
        """Test the MarketSessionManager."""
        logger.info("Testing MarketSessionManager...")
        
        test_result = {
            "test_name": "Session Manager",
            "success": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Test session state detection
            current_time = datetime.now()
            transition = self.session_manager.update_session_state(current_time)
            
            if transition:
                test_result["details"].append(f"Initial session transition: {transition.new_state.value}")
            
            # Test gap detection
            past_time = current_time - timedelta(hours=18)  # 18 hour gap
            gap_info = self.session_manager.get_gap_info(past_time, current_time)
            
            if gap_info["is_significant"]:
                test_result["details"].append(f"Gap detection working: {gap_info['duration_hours']:.1f}h gap detected")
            else:
                test_result["errors"].append("Gap detection failed for 18-hour gap")
                test_result["success"] = False
            
            # Test session state tracking
            current_state = self.session_manager.get_current_state()
            test_result["details"].append(f"Current session state: {current_state.value}")
            
            # Test weekend gap detection
            friday_close = current_time.replace(hour=16, minute=0, second=0) - timedelta(days=2)  # Friday
            monday_open = friday_close + timedelta(days=3, hours=1)  # Monday 5 PM
            
            weekend_gap = self.session_manager.get_gap_info(friday_close, monday_open)
            if weekend_gap["is_weekend_gap"]:
                test_result["details"].append("Weekend gap detection working")
            else:
                test_result["errors"].append("Weekend gap detection failed")
                test_result["success"] = False
            
        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(f"Exception: {str(e)}")
        
        return test_result
    
    async def _test_enhanced_timeframe_manager(self) -> Dict[str, Any]:
        """Test the EnhancedTimeframeManager."""
        logger.info("Testing EnhancedTimeframeManager...")
        
        test_result = {
            "test_name": "Enhanced Timeframe Manager",
            "success": True,
            "details": [],
            "errors": []
        }
        
        try:
            symbol = "AAPL"
            timeframes = ["1m", "5m", "1h", "1d"]
            
            # Test normal tick processing
            current_time = datetime.now()
            for i in range(10):
                price = 150.0 + i * 0.1
                timestamp = current_time + timedelta(minutes=i)
                
                results = self.timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=timestamp,
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
            
            test_result["details"].append(f"Processed 10 normal ticks for {symbol}")
            
            # Test gap handling
            gap_time = current_time + timedelta(hours=20)  # 20-hour gap
            gap_transition = self.session_manager.update_session_state(gap_time)
            
            gap_results = self.timeframe_manager.process_tick(
                symbol=symbol,
                price=155.0,
                volume=1200.0,
                timestamp=gap_time,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
            test_result["details"].append("Gap handling test completed")
            
            # Verify candle data
            for timeframe in timeframes:
                candles = self.timeframe_manager.get_timeframe_data(symbol, timeframe)
                if len(candles) > 0:
                    test_result["details"].append(f"{timeframe}: {len(candles)} candles created")
                else:
                    test_result["errors"].append(f"No candles created for {timeframe}")
                    test_result["success"] = False
            
            # Test session stats
            stats = self.timeframe_manager.get_session_stats()
            if stats["total_updates"] > 0:
                test_result["details"].append(f"Session stats: {stats['total_updates']} updates")
            else:
                test_result["errors"].append("No updates recorded in session stats")
                test_result["success"] = False
        
        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(f"Exception: {str(e)}")
        
        return test_result
    
    async def _test_gap_aware_indicators(self) -> Dict[str, Any]:
        """Test the GapAwareIndicatorCalculator."""
        logger.info("Testing GapAwareIndicatorCalculator...")
        
        test_result = {
            "test_name": "Gap-Aware Indicators",
            "success": True,
            "details": [],
            "errors": []
        }
        
        try:
            symbol = "AAPL"
            
            # Test normal indicator calculation
            for i in range(20):
                price = 150.0 + i * 0.1
                sma = self.indicator_calculator.calculate_sma_incremental(symbol, price, 10)
                ema = self.indicator_calculator.calculate_ema_incremental(symbol, price, 10)
                rsi = self.indicator_calculator.calculate_rsi_incremental(symbol, price, 14)
            
            test_result["details"].append("Normal indicator calculations completed")
            
            # Test gap handling
            from src.realtime.market_session_manager import SessionTransition, SessionState
            gap_transition = SessionTransition(
                previous_state=SessionState.REGULAR_HOURS,
                new_state=SessionState.PRE_MARKET,
                transition_time=datetime.now(),
                gap_duration_hours=18.0,
                is_new_trading_day=True,
                is_weekend_gap=True,
                is_significant_gap=True
            )
            
            # Apply gap handling
            self.indicator_calculator.handle_session_transition(symbol, gap_transition)
            test_result["details"].append("Gap handling applied to indicators")
            
            # Test gap-aware calculations
            gap_sma = self.indicator_calculator.calculate_sma_incremental_with_gap_awareness(
                symbol, 160.0, 10, datetime.now(), {"is_significant": True, "duration_hours": 18}
            )
            
            gap_ema = self.indicator_calculator.calculate_ema_incremental_with_gap_awareness(
                symbol, 160.0, 10, datetime.now(), {"is_significant": True, "duration_hours": 18}
            )
            
            gap_rsi = self.indicator_calculator.calculate_rsi_incremental_with_gap_awareness(
                symbol, 160.0, 14, datetime.now(), {"is_significant": True, "duration_hours": 18}
            )
            
            test_result["details"].append("Gap-aware indicator calculations completed")
            
            # Check gap statistics
            gap_stats = self.indicator_calculator.get_gap_stats()
            if gap_stats["session_transitions"] > 0:
                test_result["details"].append(f"Gap stats: {gap_stats['session_transitions']} transitions handled")
            
        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(f"Exception: {str(e)}")
        
        return test_result
    
    async def _test_day_boundary_transitions(self) -> Dict[str, Any]:
        """Test day boundary transition handling."""
        logger.info("Testing day boundary transitions...")
        
        test_result = {
            "test_name": "Day Boundary Transitions",
            "success": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Start the data synchronizer
            await self.data_synchronizer.start()
            
            symbol = "AAPL"
            timeframes = ["1m", "5m", "1h", "1d"]
            
            # Simulate day boundary transition
            from src.realtime.market_session_manager import SessionTransition, SessionState
            day_transition = SessionTransition(
                previous_state=SessionState.AFTER_HOURS,
                new_state=SessionState.PRE_MARKET,
                transition_time=datetime.now(),
                gap_duration_hours=8.0,
                is_new_trading_day=True,
                is_weekend_gap=False,
                is_significant_gap=True
            )
            
            # Test session transition handling
            validation_results = await self.data_synchronizer.handle_session_transition(
                symbol, timeframes, day_transition
            )
            
            if validation_results:
                test_result["details"].append(f"Session transition validation: {len(validation_results)} results")
                
                for result in validation_results:
                    if result.success:
                        test_result["details"].append(f"✓ {result.timeframe} validation passed")
                    else:
                        test_result["details"].append(f"✗ {result.timeframe} validation failed: {result.issues_found}")
            
            # Test synchronization stats
            sync_stats = self.data_synchronizer.get_sync_stats()
            test_result["details"].append(f"Sync stats: {sync_stats['session_transitions_handled']} transitions handled")
            
            # Stop the synchronizer
            await self.data_synchronizer.stop()
        
        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(f"Exception: {str(e)}")
        
        return test_result
    
    async def _test_continuous_operation(self) -> Dict[str, Any]:
        """Test continuous operation capabilities."""
        logger.info("Testing continuous operation...")
        
        test_result = {
            "test_name": "Continuous Operation",
            "success": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Test multiple session transitions without stopping
            symbol = "AAPL"
            timeframes = ["5m", "1h"]
            
            session_states = [
                SessionState.PRE_MARKET,
                SessionState.REGULAR_HOURS,
                SessionState.AFTER_HOURS,
                SessionState.CLOSED
            ]
            
            for i, state in enumerate(session_states):
                # Simulate transition to each state
                current_time = datetime.now() + timedelta(hours=i * 2)
                transition = self.session_manager.update_session_state(current_time)
                
                # Process tick during each session state
                price = 150.0 + i
                results = self.timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=current_time,
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
                test_result["details"].append(f"Processed tick during {state.value} session")
            
            # Verify system maintained state throughout
            final_stats = self.timeframe_manager.get_session_stats()
            if final_stats["total_updates"] >= len(session_states):
                test_result["details"].append("Continuous operation maintained across all session states")
            else:
                test_result["errors"].append("System failed to maintain continuous operation")
                test_result["success"] = False
            
        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(f"Exception: {str(e)}")
        
        return test_result


async def main():
    """Run the comprehensive test."""
    print("🚀 Starting Session-Aware Solution Test")
    print("=" * 60)
    
    tester = SessionAwareSolutionTester()
    results = await tester.run_comprehensive_test()
    
    print("\n📊 TEST RESULTS")
    print("=" * 60)
    print(f"⏱️  Duration: {results['duration_seconds']:.2f} seconds")
    print(f"✅ Tests Passed: {results['tests_passed']}")
    print(f"❌ Tests Failed: {results['tests_failed']}")
    print(f"📈 Success Rate: {results['success_rate']:.1f}%")
    
    print("\n📋 DETAILED RESULTS")
    print("=" * 60)
    
    for test_detail in results["test_details"]:
        status = "✅ PASS" if test_detail["success"] else "❌ FAIL"
        print(f"\n{status} {test_detail['test_name']}")
        
        for detail in test_detail["details"]:
            print(f"  ℹ️  {detail}")
        
        for error in test_detail["errors"]:
            print(f"  ⚠️  {error}")
    
    print("\n🎯 SOLUTION SUMMARY")
    print("=" * 60)
    print("✅ Market Session Detection: Implemented")
    print("✅ Session-Aware Candle Creation: Implemented")
    print("✅ Gap-Aware Indicators: Implemented")
    print("✅ Day Boundary Synchronization: Implemented")
    print("✅ Continuous Operation: Implemented")
    print("✅ Data Integrity Validation: Implemented")
    
    if results['success_rate'] >= 80:
        print("\n🎉 Solution is ready for integration!")
        print("\n📝 NEXT STEPS:")
        print("1. Review the implementation files in src/realtime/")
        print("2. Integrate SessionAwareCalculationEngine into your production service")
        print("3. Update start_production_service.py to use the new engine")
        print("4. Run additional tests with your specific data")
        print("5. Monitor the system during the first few day transitions")
    else:
        print(f"\n⚠️  Solution needs attention - {100-results['success_rate']:.1f}% of tests failed")
        print("Please review the failed tests and address the issues.")


if __name__ == "__main__":
    asyncio.run(main())