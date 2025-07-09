"""
Real-time Trading Scenarios End-to-End Tests

This module contains comprehensive tests for real-time trading scenarios including
market open/close, high volatility periods, multi-symbol processing, and
strategy changes during live operation.
"""

import pytest
import asyncio
import time
import threading
import json
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Optional, Tuple, Set
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, field
from enum import Enum
import random
import math

from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.signal_engine import SignalEngine
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.config_manager import ConfigManager, TradingProfile, AssetConfig
from src.core.models import OHLCV, TimeFrame

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    REGULAR_HOURS = "regular_hours"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"


class VolatilityRegime(Enum):
    """Volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketScenario:
    """Market scenario configuration."""
    name: str
    description: str
    session: MarketSession
    volatility: VolatilityRegime
    duration_minutes: int
    symbols: List[str]
    expected_behavior: Dict[str, Any]
    performance_targets: Dict[str, Any]


@dataclass
class RealTimeTestResult:
    """Result of real-time scenario test."""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    market_data_points: int = 0
    signals_generated: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    scenario_validation: Dict[str, Any] = field(default_factory=dict)


class TestRealTimeScenarios:
    """
    Test real-time trading scenarios under various market conditions.
    
    These tests validate:
    - Market open and close scenarios
    - High volatility trading periods
    - Multiple symbol processing simultaneously
    - Strategy changes during live operation
    - System load under realistic trading conditions
    """
    
    @pytest.fixture
    def market_scenarios(self):
        """Define comprehensive market scenarios."""
        return [
            MarketScenario(
                name="market_open_rush",
                description="Market opening with high volume and volatility",
                session=MarketSession.MARKET_OPEN,
                volatility=VolatilityRegime.HIGH,
                duration_minutes=15,
                symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"],
                expected_behavior={
                    "high_volume": True,
                    "price_gaps": True,
                    "increased_signals": True,
                    "volatility_expansion": True
                },
                performance_targets={
                    "max_latency_ms": 100,
                    "min_throughput_msgs_per_sec": 100,
                    "max_memory_increase_mb": 50
                }
            ),
            MarketScenario(
                name="market_close_activity",
                description="Market closing with position adjustments",
                session=MarketSession.MARKET_CLOSE,
                volatility=VolatilityRegime.HIGH,
                duration_minutes=10,
                symbols=["SPY", "QQQ", "IWM", "DIA"],
                expected_behavior={
                    "volume_spike": True,
                    "price_convergence": True,
                    "signal_clustering": True,
                    "liquidity_changes": True
                },
                performance_targets={
                    "max_latency_ms": 50,
                    "min_throughput_msgs_per_sec": 150,
                    "max_memory_increase_mb": 30
                }
            ),
            MarketScenario(
                name="high_volatility_period",
                description="High volatility trading with rapid price movements",
                session=MarketSession.REGULAR_HOURS,
                volatility=VolatilityRegime.EXTREME,
                duration_minutes=20,
                symbols=["TSLA", "NVDA", "AMD", "NFLX"],
                expected_behavior={
                    "rapid_price_changes": True,
                    "frequent_signals": True,
                    "risk_management_active": True,
                    "adaptive_strategies": True
                },
                performance_targets={
                    "max_latency_ms": 25,
                    "min_throughput_msgs_per_sec": 200,
                    "max_memory_increase_mb": 75
                }
            ),
            MarketScenario(
                name="multi_symbol_correlation",
                description="Multiple correlated symbols moving together",
                session=MarketSession.REGULAR_HOURS,
                volatility=VolatilityRegime.NORMAL,
                duration_minutes=30,
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"],
                expected_behavior={
                    "correlated_movements": True,
                    "sector_rotation": True,
                    "cross_asset_signals": True,
                    "portfolio_balancing": True
                },
                performance_targets={
                    "max_latency_ms": 75,
                    "min_throughput_msgs_per_sec": 80,
                    "max_memory_increase_mb": 100
                }
            ),
            MarketScenario(
                name="crypto_24x7_trading",
                description="24/7 cryptocurrency trading simulation",
                session=MarketSession.REGULAR_HOURS,
                volatility=VolatilityRegime.HIGH,
                duration_minutes=25,
                symbols=["BTC-USD", "ETH-USD", "BNB-USD"],
                expected_behavior={
                    "continuous_trading": True,
                    "high_volatility": True,
                    "momentum_signals": True,
                    "trend_following": True
                },
                performance_targets={
                    "max_latency_ms": 30,
                    "min_throughput_msgs_per_sec": 120,
                    "max_memory_increase_mb": 40
                }
            )
        ]
    
    @pytest.fixture
    def realistic_market_simulator(self):
        """Create realistic market data simulator for various scenarios."""
        class RealisticMarketSimulator:
            def __init__(self):
                # Base prices for different assets
                self.base_prices = {
                    "AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3000.0,
                    "META": 200.0, "TSLA": 800.0, "NVDA": 400.0, "NFLX": 350.0,
                    "AMD": 100.0, "SPY": 400.0, "QQQ": 320.0, "IWM": 200.0,
                    "DIA": 350.0, "BTC-USD": 40000.0, "ETH-USD": 2500.0, "BNB-USD": 300.0
                }
                
                # Current prices track
                self.current_prices = self.base_prices.copy()
                
                # Market state
                self.market_state = {
                    "session": MarketSession.REGULAR_HOURS,
                    "volatility": VolatilityRegime.NORMAL,
                    "volume_multiplier": 1.0,
                    "correlation_strength": 0.3
                }
                
                # Price history for correlation
                self.price_history = {symbol: [] for symbol in self.base_prices.keys()}
                
                # Volume patterns
                self.volume_patterns = {
                    MarketSession.PRE_MARKET: 0.3,
                    MarketSession.MARKET_OPEN: 2.5,
                    MarketSession.REGULAR_HOURS: 1.0,
                    MarketSession.MARKET_CLOSE: 2.0,
                    MarketSession.AFTER_HOURS: 0.4
                }
                
                # Volatility multipliers
                self.volatility_multipliers = {
                    VolatilityRegime.LOW: 0.5,
                    VolatilityRegime.NORMAL: 1.0,
                    VolatilityRegime.HIGH: 2.0,
                    VolatilityRegime.EXTREME: 4.0
                }
                
            def set_market_conditions(self, session: MarketSession, volatility: VolatilityRegime):
                """Set market conditions."""
                self.market_state["session"] = session
                self.market_state["volatility"] = volatility
                self.market_state["volume_multiplier"] = self.volume_patterns[session]
                
            def generate_correlated_price_movement(self, symbol: str, market_movement: float = 0.0) -> float:
                """Generate price movement with market correlation."""
                base_volatility = 0.001  # 0.1% base volatility
                
                # Apply volatility regime
                volatility = base_volatility * self.volatility_multipliers[self.market_state["volatility"]]
                
                # Random component
                random_movement = random.gauss(0, volatility)
                
                # Market correlation component
                correlation_strength = self.market_state["correlation_strength"]
                correlated_movement = market_movement * correlation_strength
                
                # Symbol-specific characteristics
                symbol_characteristics = {
                    "TSLA": {"volatility_multiplier": 2.0, "momentum_factor": 1.5},
                    "BTC-USD": {"volatility_multiplier": 3.0, "momentum_factor": 2.0},
                    "ETH-USD": {"volatility_multiplier": 2.5, "momentum_factor": 1.8},
                    "NVDA": {"volatility_multiplier": 1.8, "momentum_factor": 1.3},
                    "SPY": {"volatility_multiplier": 0.5, "momentum_factor": 0.8}
                }
                
                char = symbol_characteristics.get(symbol, {"volatility_multiplier": 1.0, "momentum_factor": 1.0})
                
                # Apply symbol characteristics
                total_movement = (random_movement + correlated_movement) * char["volatility_multiplier"]
                
                return total_movement
                
            def generate_realistic_tick(self, symbol: str, market_movement: float = 0.0) -> Dict[str, Any]:
                """Generate realistic market tick."""
                current_price = self.current_prices[symbol]
                
                # Generate price movement
                price_change = self.generate_correlated_price_movement(symbol, market_movement)
                new_price = current_price * (1 + price_change)
                
                # Ensure price doesn't go negative
                new_price = max(new_price, current_price * 0.8)
                
                # Update current price
                self.current_prices[symbol] = new_price
                
                # Generate OHLC data
                price_range = abs(price_change) * current_price
                open_price = current_price
                high_price = max(current_price, new_price) + price_range * 0.5
                low_price = min(current_price, new_price) - price_range * 0.5
                close_price = new_price
                
                # Generate volume based on market session and volatility
                base_volume = 1000
                volume_multiplier = self.market_state["volume_multiplier"]
                volatility_volume_impact = self.volatility_multipliers[self.market_state["volatility"]]
                
                volume = int(base_volume * volume_multiplier * volatility_volume_impact * (1 + abs(price_change) * 10))
                
                # Store price history
                self.price_history[symbol].append(new_price)
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
                
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "open": f"{open_price:.2f}",
                    "high": f"{high_price:.2f}",
                    "low": f"{low_price:.2f}",
                    "close": f"{close_price:.2f}",
                    "volume": str(volume),
                    "price": f"{close_price:.2f}",
                    "price_change": f"{price_change:.6f}",
                    "price_change_percent": f"{price_change * 100:.4f}%"
                }
                
            def generate_market_scenario(self, scenario: MarketScenario) -> List[Dict[str, Any]]:
                """Generate market data for a specific scenario."""
                self.set_market_conditions(scenario.session, scenario.volatility)
                
                # Calculate ticks per minute based on market session
                base_ticks_per_minute = 4  # 15-second intervals
                session_multiplier = {
                    MarketSession.PRE_MARKET: 0.5,
                    MarketSession.MARKET_OPEN: 3.0,
                    MarketSession.REGULAR_HOURS: 1.0,
                    MarketSession.MARKET_CLOSE: 2.5,
                    MarketSession.AFTER_HOURS: 0.3
                }
                
                ticks_per_minute = int(base_ticks_per_minute * session_multiplier[scenario.session])
                total_ticks = scenario.duration_minutes * ticks_per_minute
                
                scenario_data = []
                market_trend = random.gauss(0, 0.0005)  # Overall market trend
                
                for tick_index in range(total_ticks):
                    # Generate market-wide movement
                    market_movement = random.gauss(market_trend, 0.0002)
                    
                    # Generate ticks for all symbols
                    for symbol in scenario.symbols:
                        tick_data = self.generate_realistic_tick(symbol, market_movement)
                        
                        # Add scenario-specific metadata
                        tick_data["scenario"] = scenario.name
                        tick_data["session"] = scenario.session.value
                        tick_data["volatility_regime"] = scenario.volatility.value
                        tick_data["tick_index"] = tick_index
                        
                        scenario_data.append(tick_data)
                    
                    # Simulate realistic timing
                    time.sleep(0.001)  # Small delay to simulate real-time flow
                
                return scenario_data
                
            def get_market_statistics(self) -> Dict[str, Any]:
                """Get current market statistics."""
                stats = {}
                
                for symbol in self.current_prices.keys():
                    if symbol in self.price_history and self.price_history[symbol]:
                        prices = self.price_history[symbol]
                        price_changes = [
                            (prices[i] - prices[i-1]) / prices[i-1] 
                            for i in range(1, len(prices))
                        ]
                        
                        stats[symbol] = {
                            "current_price": self.current_prices[symbol],
                            "price_change_since_start": (
                                (self.current_prices[symbol] - self.base_prices[symbol]) 
                                / self.base_prices[symbol]
                            ),
                            "volatility": math.sqrt(sum(pc**2 for pc in price_changes) / len(price_changes)) if price_changes else 0,
                            "price_history_length": len(prices),
                            "recent_trend": sum(price_changes[-10:]) / 10 if len(price_changes) >= 10 else 0
                        }
                
                return stats
        
        return RealisticMarketSimulator()
    
    @pytest.fixture
    def strategy_manager(self):
        """Manage strategy changes during testing."""
        class StrategyManager:
            def __init__(self):
                self.available_strategies = [
                    "macd_rsi_strategy",
                    "ma_crossover_strategy", 
                    "rsi_trend_strategy",
                    "bollinger_bands_strategy",
                    "momentum_strategy"
                ]
                self.current_strategies = ["macd_rsi_strategy", "ma_crossover_strategy"]
                self.strategy_changes = []
                
            def change_strategy_mix(self, new_strategies: List[str]):
                """Change the active strategy mix."""
                old_strategies = self.current_strategies.copy()
                self.current_strategies = new_strategies
                
                change_record = {
                    "timestamp": datetime.now(),
                    "old_strategies": old_strategies,
                    "new_strategies": new_strategies,
                    "change_type": "strategy_mix_update"
                }
                self.strategy_changes.append(change_record)
                
                return change_record
                
            def add_strategy(self, strategy: str):
                """Add a strategy to the current mix."""
                if strategy not in self.current_strategies:
                    self.current_strategies.append(strategy)
                    change_record = {
                        "timestamp": datetime.now(),
                        "action": "add_strategy",
                        "strategy": strategy,
                        "current_strategies": self.current_strategies.copy()
                    }
                    self.strategy_changes.append(change_record)
                    return change_record
                    
            def remove_strategy(self, strategy: str):
                """Remove a strategy from the current mix."""
                if strategy in self.current_strategies:
                    self.current_strategies.remove(strategy)
                    change_record = {
                        "timestamp": datetime.now(),
                        "action": "remove_strategy",
                        "strategy": strategy,
                        "current_strategies": self.current_strategies.copy()
                    }
                    self.strategy_changes.append(change_record)
                    return change_record
                    
            def get_strategy_performance(self) -> Dict[str, Any]:
                """Get strategy performance summary."""
                return {
                    "active_strategies": self.current_strategies,
                    "total_changes": len(self.strategy_changes),
                    "change_history": self.strategy_changes,
                    "strategy_diversity": len(self.current_strategies) / len(self.available_strategies)
                }
        
        return StrategyManager()
    
    @pytest.fixture
    async def realtime_system(self, test_config_manager, mock_redis_client):
        """Create real-time system for scenario testing."""
        # Enhanced Redis mock for real-time scenarios
        redis_streams = {}
        redis_data = {}
        redis_metrics = {"messages_processed": 0, "stream_operations": 0}
        
        def enhanced_redis_mock():
            """Enhanced Redis mock with performance tracking."""
            def xadd(stream_name, fields, id="*", maxlen=None):
                nonlocal redis_metrics
                if stream_name not in redis_streams:
                    redis_streams[stream_name] = []
                
                message_id = f"rt-{len(redis_streams[stream_name])}-{int(time.time() * 1000000)}"
                redis_streams[stream_name].append((message_id, fields))
                redis_metrics["messages_processed"] += 1
                redis_metrics["stream_operations"] += 1
                
                # Simulate stream size limits
                if maxlen and len(redis_streams[stream_name]) > maxlen:
                    redis_streams[stream_name] = redis_streams[stream_name][-maxlen:]
                
                return message_id
            
            def xreadgroup(group_name, consumer_name, streams, count=None, block=None):
                nonlocal redis_metrics
                results = []
                
                for stream_name, since_id in streams.items():
                    if stream_name in redis_streams:
                        messages = redis_streams[stream_name]
                        if count:
                            messages = messages[-count:]
                        if messages:
                            results.append([stream_name, messages])
                            redis_metrics["stream_operations"] += 1
                
                return results
            
            def xack(stream_name, group_name, message_id):
                redis_metrics["stream_operations"] += 1
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
        
        # Apply enhanced Redis mock
        redis_ops = enhanced_redis_mock()
        for op_name, op_func in redis_ops.items():
            setattr(mock_redis_client, op_name, op_func)
        
        # Create real-time system
        system = {}
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                with patch('src.realtime.signal_engine.redis.Redis', return_value=mock_redis_client):
                    
                    # Create components
                    data_ingestion = MarketDataIngestionService(test_config_manager)
                    calculation_engine = CalculationEngine(test_config_manager, data_ingestion)
                    signal_engine = SignalEngine(test_config_manager, calculation_engine.timeframe_manager)
                    
                    system = {
                        'data_ingestion': data_ingestion,
                        'calculation_engine': calculation_engine,
                        'signal_engine': signal_engine,
                        'timeframe_manager': calculation_engine.timeframe_manager,
                        'redis_client': mock_redis_client,
                        'redis_streams': redis_streams,
                        'redis_data': redis_data,
                        'redis_metrics': redis_metrics
                    }
                    
                    yield system
                    
                    # Cleanup
                    await data_ingestion.stop()
                    await calculation_engine.stop()
    
    @pytest.mark.asyncio
    async def test_market_open_scenario(self, realtime_system, realistic_market_simulator, 
                                       market_scenarios):
        """Test market opening scenario with high volume and volatility."""
        scenario = next(s for s in market_scenarios if s.name == "market_open_rush")
        
        test_result = RealTimeTestResult(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            signal_engine = realtime_system['signal_engine']
            redis_streams = realtime_system['redis_streams']
            redis_metrics = realtime_system['redis_metrics']
            
            # Generate market open data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start real-time processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Performance tracking
            processing_times = []
            signal_counts = []
            
            # Process market open data
            batch_size = 20  # Process in batches to simulate realistic flow
            for i in range(0, len(market_data), batch_size):
                batch = market_data[i:i + batch_size]
                batch_start = time.time()
                
                # Process batch
                for tick_data in batch:
                    await data_ingestion._process_market_data(tick_data)
                
                # Allow processing
                await asyncio.sleep(0.1)
                
                batch_end = time.time()
                processing_times.append(batch_end - batch_start)
                
                # Track signals
                signal_count = len(redis_streams.get('signals_stream', []))
                signal_counts.append(signal_count)
            
            # Allow final processing
            await asyncio.sleep(2.0)
            
            # Stop processing
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Validate market open behavior
            market_stats = realistic_market_simulator.get_market_statistics()
            
            # Check performance metrics
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            
            # Validate scenario expectations
            scenario_validation = {
                "high_volume_processed": test_result.market_data_points > 100,
                "signals_generated": test_result.signals_generated > 0,
                "processing_performance": avg_processing_time < 1.0,
                "system_responsiveness": max_processing_time < 2.0,
                "market_volatility_detected": any(
                    stats["volatility"] > 0.01 for stats in market_stats.values()
                )
            }
            
            # Validate against scenario targets
            throughput = test_result.market_data_points / test_result.duration_seconds
            latency_ms = avg_processing_time * 1000
            
            assert throughput >= scenario.performance_targets["min_throughput_msgs_per_sec"], \
                f"Throughput too low: {throughput}"
            assert latency_ms <= scenario.performance_targets["max_latency_ms"], \
                f"Latency too high: {latency_ms}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "avg_processing_time": avg_processing_time,
                "max_processing_time": max_processing_time,
                "throughput_msgs_per_sec": throughput,
                "latency_ms": latency_ms,
                "market_statistics": market_stats,
                "redis_metrics": redis_metrics
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Market open scenario test failed: {e}")
            raise
            
        finally:
            logger.info(f"Market open scenario test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_high_volatility_scenario(self, realtime_system, realistic_market_simulator,
                                          market_scenarios):
        """Test high volatility scenario with rapid price movements."""
        scenario = next(s for s in market_scenarios if s.name == "high_volatility_period")
        
        test_result = RealTimeTestResult(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            signal_engine = realtime_system['signal_engine']
            redis_streams = realtime_system['redis_streams']
            
            # Generate high volatility data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Track volatility metrics
            volatility_metrics = {
                "price_changes": [],
                "volume_spikes": [],
                "signal_frequency": []
            }
            
            # Process high volatility data
            for i, tick_data in enumerate(market_data):
                await data_ingestion._process_market_data(tick_data)
                
                # Track volatility indicators
                if "price_change" in tick_data:
                    price_change = float(tick_data["price_change"])
                    volatility_metrics["price_changes"].append(abs(price_change))
                
                volume = int(tick_data["volume"])
                volatility_metrics["volume_spikes"].append(volume)
                
                # Check signal generation frequency
                if i % 50 == 0:  # Check every 50 ticks
                    signal_count = len(redis_streams.get('signals_stream', []))
                    volatility_metrics["signal_frequency"].append(signal_count)
                
                # High frequency processing
                if i % 10 == 0:
                    await asyncio.sleep(0.01)
            
            # Allow processing to complete
            await asyncio.sleep(2.0)
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Analyze volatility response
            avg_price_change = sum(volatility_metrics["price_changes"]) / len(volatility_metrics["price_changes"])
            max_price_change = max(volatility_metrics["price_changes"])
            avg_volume = sum(volatility_metrics["volume_spikes"]) / len(volatility_metrics["volume_spikes"])
            
            # Validate high volatility handling
            scenario_validation = {
                "high_volatility_detected": avg_price_change > 0.005,  # 0.5% average change
                "extreme_moves_handled": max_price_change < 0.1,  # No moves > 10%
                "signals_adapted": test_result.signals_generated > 0,
                "system_stable": True,  # If we complete without crashing
                "volume_response": avg_volume > 1000
            }
            
            # Performance validation
            throughput = test_result.market_data_points / test_result.duration_seconds
            assert throughput >= scenario.performance_targets["min_throughput_msgs_per_sec"], \
                f"Throughput insufficient for high volatility: {throughput}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "avg_price_change": avg_price_change,
                "max_price_change": max_price_change,
                "avg_volume": avg_volume,
                "throughput_msgs_per_sec": throughput,
                "volatility_metrics": volatility_metrics,
                "signals_per_tick": test_result.signals_generated / test_result.market_data_points
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"High volatility scenario test failed: {e}")
            raise
            
        finally:
            logger.info(f"High volatility scenario test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_multi_symbol_processing(self, realtime_system, realistic_market_simulator,
                                         market_scenarios):
        """Test simultaneous processing of multiple symbols."""
        scenario = next(s for s in market_scenarios if s.name == "multi_symbol_correlation")
        
        test_result = RealTimeTestResult(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            redis_streams = realtime_system['redis_streams']
            
            # Generate multi-symbol data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Track per-symbol metrics
            symbol_metrics = {}
            for symbol in scenario.symbols:
                symbol_metrics[symbol] = {
                    "data_points": 0,
                    "signals": 0,
                    "price_changes": []
                }
            
            # Process multi-symbol data
            for tick_data in market_data:
                await data_ingestion._process_market_data(tick_data)
                
                # Track per-symbol metrics
                symbol = tick_data["symbol"]
                if symbol in symbol_metrics:
                    symbol_metrics[symbol]["data_points"] += 1
                    
                    if "price_change" in tick_data:
                        symbol_metrics[symbol]["price_changes"].append(float(tick_data["price_change"]))
                
                # Simulate realistic multi-symbol flow
                await asyncio.sleep(0.001)
            
            # Allow processing to complete
            await asyncio.sleep(3.0)
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Analyze multi-symbol processing
            total_symbols_processed = len([s for s in symbol_metrics.values() if s["data_points"] > 0])
            avg_data_points_per_symbol = sum(s["data_points"] for s in symbol_metrics.values()) / len(symbol_metrics)
            
            # Check correlation processing
            symbol_correlations = {}
            for symbol, metrics in symbol_metrics.items():
                if metrics["price_changes"]:
                    symbol_correlations[symbol] = {
                        "volatility": sum(abs(pc) for pc in metrics["price_changes"]) / len(metrics["price_changes"]),
                        "trend": sum(metrics["price_changes"]) / len(metrics["price_changes"]),
                        "data_points": metrics["data_points"]
                    }
            
            # Validate multi-symbol processing
            scenario_validation = {
                "all_symbols_processed": total_symbols_processed == len(scenario.symbols),
                "balanced_processing": avg_data_points_per_symbol > 0,
                "correlations_detected": len(symbol_correlations) > 0,
                "cross_symbol_signals": test_result.signals_generated > 0,
                "system_scalability": total_symbols_processed >= 5
            }
            
            # Performance validation
            throughput = test_result.market_data_points / test_result.duration_seconds
            assert throughput >= scenario.performance_targets["min_throughput_msgs_per_sec"], \
                f"Multi-symbol throughput too low: {throughput}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "symbols_processed": total_symbols_processed,
                "avg_data_points_per_symbol": avg_data_points_per_symbol,
                "symbol_correlations": symbol_correlations,
                "throughput_msgs_per_sec": throughput,
                "symbol_metrics": symbol_metrics,
                "processing_balance": min(s["data_points"] for s in symbol_metrics.values()) / max(s["data_points"] for s in symbol_metrics.values())
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Multi-symbol processing test failed: {e}")
            raise
            
        finally:
            logger.info(f"Multi-symbol processing test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_strategy_changes_during_operation(self, realtime_system, realistic_market_simulator,
                                                   strategy_manager, market_scenarios):
        """Test strategy changes during live operation."""
        scenario = next(s for s in market_scenarios if s.name == "market_open_rush")
        
        test_result = RealTimeTestResult(
            scenario_name=f"{scenario.name}_with_strategy_changes",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            signal_engine = realtime_system['signal_engine']
            redis_streams = realtime_system['redis_streams']
            
            # Generate market data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Track strategy change effects
            strategy_change_metrics = {
                "changes_applied": 0,
                "signals_before_changes": 0,
                "signals_after_changes": 0,
                "change_timeline": []
            }
            
            # Process data with strategy changes
            data_third = len(market_data) // 3
            
            for i, tick_data in enumerate(market_data):
                await data_ingestion._process_market_data(tick_data)
                
                # Make strategy changes at specific points
                if i == data_third:
                    # First strategy change
                    strategy_change_metrics["signals_before_changes"] = len(redis_streams.get('signals_stream', []))
                    
                    change_record = strategy_manager.add_strategy("momentum_strategy")
                    strategy_change_metrics["changes_applied"] += 1
                    strategy_change_metrics["change_timeline"].append(change_record)
                    
                    logger.info(f"Strategy change 1 applied at tick {i}: {change_record}")
                    
                elif i == data_third * 2:
                    # Second strategy change
                    change_record = strategy_manager.change_strategy_mix([
                        "rsi_trend_strategy", "bollinger_bands_strategy"
                    ])
                    strategy_change_metrics["changes_applied"] += 1
                    strategy_change_metrics["change_timeline"].append(change_record)
                    
                    logger.info(f"Strategy change 2 applied at tick {i}: {change_record}")
                
                # Allow processing
                if i % 100 == 0:
                    await asyncio.sleep(0.1)
            
            # Final processing
            await asyncio.sleep(2.0)
            strategy_change_metrics["signals_after_changes"] = len(redis_streams.get('signals_stream', []))
            
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Analyze strategy change effects
            strategy_performance = strategy_manager.get_strategy_performance()
            
            # Validate strategy changes
            scenario_validation = {
                "strategy_changes_applied": strategy_change_metrics["changes_applied"] == 2,
                "system_continued_processing": test_result.signals_generated > 0,
                "strategy_diversity_maintained": strategy_performance["strategy_diversity"] > 0.2,
                "signals_generated_throughout": strategy_change_metrics["signals_after_changes"] > strategy_change_metrics["signals_before_changes"],
                "no_processing_interruption": True  # If we complete without exceptions
            }
            
            test_result.success = True
            test_result.performance_metrics = {
                "strategy_change_metrics": strategy_change_metrics,
                "strategy_performance": strategy_performance,
                "signals_per_strategy_change": test_result.signals_generated / max(1, strategy_change_metrics["changes_applied"]),
                "adaptation_efficiency": strategy_change_metrics["signals_after_changes"] / max(1, strategy_change_metrics["signals_before_changes"])
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Strategy changes during operation test failed: {e}")
            raise
            
        finally:
            logger.info(f"Strategy changes during operation test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_crypto_24x7_scenario(self, realtime_system, realistic_market_simulator,
                                       market_scenarios):
        """Test 24/7 cryptocurrency trading scenario."""
        scenario = next(s for s in market_scenarios if s.name == "crypto_24x7_trading")
        
        test_result = RealTimeTestResult(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            signal_engine = realtime_system['signal_engine']
            redis_streams = realtime_system['redis_streams']
            
            # Generate crypto market data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Track crypto-specific metrics
            crypto_metrics = {
                "continuous_processing": True,
                "high_volatility_handling": [],
                "momentum_signals": 0,
                "trend_changes": 0
            }
            
            # Process crypto data
            prev_prices = {}
            for tick_data in market_data:
                await data_ingestion._process_market_data(tick_data)
                
                # Track crypto-specific behavior
                symbol = tick_data["symbol"]
                current_price = float(tick_data["price"])
                
                if symbol in prev_prices:
                    price_change = abs(current_price - prev_prices[symbol]) / prev_prices[symbol]
                    crypto_metrics["high_volatility_handling"].append(price_change)
                    
                    # Detect trend changes
                    if price_change > 0.02:  # 2% change
                        crypto_metrics["trend_changes"] += 1
                
                prev_prices[symbol] = current_price
                
                # High-frequency processing for crypto
                await asyncio.sleep(0.0005)
            
            # Allow processing to complete
            await asyncio.sleep(2.0)
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Analyze crypto trading performance
            avg_volatility = sum(crypto_metrics["high_volatility_handling"]) / len(crypto_metrics["high_volatility_handling"])
            max_volatility = max(crypto_metrics["high_volatility_handling"])
            
            # Validate crypto trading
            scenario_validation = {
                "continuous_processing": crypto_metrics["continuous_processing"],
                "high_volatility_handled": avg_volatility > 0.005,  # 0.5% average volatility
                "extreme_volatility_managed": max_volatility < 0.15,  # Max 15% single move
                "momentum_detected": test_result.signals_generated > 0,
                "trend_changes_identified": crypto_metrics["trend_changes"] > 0,
                "crypto_symbols_processed": len(set(d["symbol"] for d in market_data if "BTC" in d["symbol"] or "ETH" in d["symbol"]))
            }
            
            # Performance validation
            throughput = test_result.market_data_points / test_result.duration_seconds
            assert throughput >= scenario.performance_targets["min_throughput_msgs_per_sec"], \
                f"Crypto trading throughput too low: {throughput}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "avg_volatility": avg_volatility,
                "max_volatility": max_volatility,
                "throughput_msgs_per_sec": throughput,
                "crypto_metrics": crypto_metrics,
                "signals_per_trend_change": test_result.signals_generated / max(1, crypto_metrics["trend_changes"]),
                "processing_efficiency": test_result.market_data_points / test_result.duration_seconds
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Crypto 24x7 scenario test failed: {e}")
            raise
            
        finally:
            logger.info(f"Crypto 24x7 scenario test completed: {test_result}")
    
    @pytest.mark.asyncio
    async def test_market_close_scenario(self, realtime_system, realistic_market_simulator,
                                        market_scenarios):
        """Test market close scenario with position adjustments."""
        scenario = next(s for s in market_scenarios if s.name == "market_close_activity")
        
        test_result = RealTimeTestResult(
            scenario_name=scenario.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0,
            success=False
        )
        
        try:
            # Get system components
            data_ingestion = realtime_system['data_ingestion']
            calculation_engine = realtime_system['calculation_engine']
            signal_engine = realtime_system['signal_engine']
            redis_streams = realtime_system['redis_streams']
            
            # Generate market close data
            market_data = realistic_market_simulator.generate_market_scenario(scenario)
            
            # Start processing
            calc_task = asyncio.create_task(calculation_engine.start_processing())
            
            # Track market close metrics
            close_metrics = {
                "volume_spikes": [],
                "price_convergence": [],
                "signal_clustering": [],
                "liquidity_events": 0
            }
            
            # Process market close data
            for i, tick_data in enumerate(market_data):
                await data_ingestion._process_market_data(tick_data)
                
                # Track market close behavior
                volume = int(tick_data["volume"])
                close_metrics["volume_spikes"].append(volume)
                
                # Track signal clustering (signals per time period)
                if i % 20 == 0:  # Check every 20 ticks
                    signal_count = len(redis_streams.get('signals_stream', []))
                    close_metrics["signal_clustering"].append(signal_count)
                
                # Simulate liquidity events
                if volume > 3000:  # High volume threshold
                    close_metrics["liquidity_events"] += 1
                
                # Market close processing frequency
                await asyncio.sleep(0.01)
            
            # Allow processing to complete
            await asyncio.sleep(2.0)
            calc_task.cancel()
            
            # Collect results
            test_result.market_data_points = len(market_data)
            test_result.signals_generated = len(redis_streams.get('signals_stream', []))
            test_result.end_time = datetime.now()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Analyze market close behavior
            avg_volume = sum(close_metrics["volume_spikes"]) / len(close_metrics["volume_spikes"])
            max_volume = max(close_metrics["volume_spikes"])
            
            # Validate market close handling
            scenario_validation = {
                "volume_spike_detected": avg_volume > 1500,  # Higher than normal volume
                "liquidity_events_handled": close_metrics["liquidity_events"] > 0,
                "signal_clustering_observed": len(close_metrics["signal_clustering"]) > 0,
                "market_close_processed": test_result.signals_generated > 0,
                "high_volume_managed": max_volume < 10000  # Reasonable volume cap
            }
            
            # Performance validation
            throughput = test_result.market_data_points / test_result.duration_seconds
            assert throughput >= scenario.performance_targets["min_throughput_msgs_per_sec"], \
                f"Market close throughput too low: {throughput}"
            
            test_result.success = True
            test_result.performance_metrics = {
                "avg_volume": avg_volume,
                "max_volume": max_volume,
                "throughput_msgs_per_sec": throughput,
                "close_metrics": close_metrics,
                "signals_per_liquidity_event": test_result.signals_generated / max(1, close_metrics["liquidity_events"]),
                "volume_volatility": max_volume / avg_volume if avg_volume > 0 else 0
            }
            test_result.scenario_validation = scenario_validation
            
        except Exception as e:
            test_result.error_message = str(e)
            logger.error(f"Market close scenario test failed: {e}")
            raise
            
        finally:
            logger.info(f"Market close scenario test completed: {test_result}")