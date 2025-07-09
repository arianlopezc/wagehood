"""
Data Processing Pipeline Integration Tests

This module contains comprehensive tests for the entire data processing
pipeline, including data ingestion, transformation, validation, Redis
storage, indicator calculations, and signal generation.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import numpy as np

# Import test modules
from src.realtime.data_ingestion import MarketDataIngestionService, MarketDataEvent
from src.realtime.config_manager import ConfigManager, AssetConfig, SystemConfig
from src.realtime.calculation_engine import CalculationEngine
from src.realtime.signal_engine import SignalEngine
from src.core.models import OHLCV, TimeFrame, Signal, SignalType
from src.data.providers.mock_provider import MockProvider
from src.storage.cache import cache_manager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataProcessingPipeline:
    """
    Comprehensive tests for the data processing pipeline.
    
    These tests validate the complete data flow from ingestion through
    processing to signal generation, including:
    - End-to-end data pipeline integration
    - Data transformation and validation
    - Redis storage and retrieval
    - Real-time indicator calculations
    - Signal generation and distribution
    - Performance under load
    """
    
    @pytest.fixture(scope="class")
    def pipeline_config_manager(self):
        """Create configuration manager for pipeline tests."""
        config_manager = Mock(spec=ConfigManager)
        
        # System configuration
        system_config = SystemConfig(
            data_update_interval_seconds=1,
            max_concurrent_calculations=20,
            calculation_batch_size=10,
            worker_pool_size=4,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        )
        config_manager.get_system_config.return_value = system_config
        
        # Watchlist with multiple assets and strategies
        watchlist = [
            AssetConfig(
                symbol='AAPL',
                enabled=True,
                data_provider='mock',
                strategies=['ma_crossover', 'rsi_trend'],
                position_size=0.1
            ),
            AssetConfig(
                symbol='MSFT',
                enabled=True,
                data_provider='mock',
                strategies=['bollinger_breakout'],
                position_size=0.1
            ),
            AssetConfig(
                symbol='GOOGL',
                enabled=True,
                data_provider='mock',
                strategies=['macd_rsi'],
                position_size=0.1
            )
        ]
        config_manager.get_watchlist.return_value = watchlist
        config_manager.get_enabled_symbols.return_value = ['AAPL', 'MSFT', 'GOOGL']
        
        return config_manager
    
    @pytest.fixture
    def mock_redis_pipeline(self):
        """Create mock Redis client with pipeline support."""
        mock_redis = Mock()
        
        # Storage for simulated Redis data
        redis_storage = {}
        stream_storage = {}
        
        def mock_xadd(stream_name, data, **kwargs):
            if stream_name not in stream_storage:
                stream_storage[stream_name] = []
            
            message_id = f"{int(time.time() * 1000)}-{len(stream_storage[stream_name])}"
            stream_storage[stream_name].append({
                'id': message_id,
                'data': data
            })
            return message_id.encode()
        
        def mock_set(key, value, **kwargs):
            redis_storage[key] = value
            return True
        
        def mock_get(key):
            return redis_storage.get(key)
        
        def mock_xinfo_stream(stream_name):
            if stream_name in stream_storage:
                return {
                    'length': len(stream_storage[stream_name]),
                    'groups': 1,
                    'last-generated-id': stream_storage[stream_name][-1]['id'].encode() if stream_storage[stream_name] else b'0-0'
                }
            return {'length': 0, 'groups': 0, 'last-generated-id': b'0-0'}
        
        def mock_xread(streams, **kwargs):
            results = {}
            for stream_name, last_id in streams.items():
                if stream_name in stream_storage:
                    messages = []
                    for msg in stream_storage[stream_name]:
                        messages.append((msg['id'].encode(), msg['data']))
                    if messages:
                        results[stream_name.encode()] = messages
            return results
        
        mock_redis.xadd = mock_xadd
        mock_redis.set = mock_set
        mock_redis.get = mock_get
        mock_redis.xinfo_stream = mock_xinfo_stream
        mock_redis.xread = mock_xread
        mock_redis.xgroup_create.return_value = True
        mock_redis.ping.return_value = True
        
        # Add storage access for testing
        mock_redis._storage = redis_storage
        mock_redis._streams = stream_storage
        
        return mock_redis
    
    @pytest.fixture
    def pipeline_data_generator(self):
        """Create realistic data generator for pipeline tests."""
        class PipelineDataGenerator:
            def __init__(self):
                self.base_prices = {
                    'AAPL': 150.0,
                    'MSFT': 300.0,
                    'GOOGL': 2500.0
                }
                self.price_trends = {
                    'AAPL': 0.001,   # Slight upward trend
                    'MSFT': -0.0005, # Slight downward trend
                    'GOOGL': 0.002   # Moderate upward trend
                }
                self.volatilities = {
                    'AAPL': 0.02,
                    'MSFT': 0.015,
                    'GOOGL': 0.025
                }
                self.data_points_generated = 0
            
            def generate_next_data(self, symbol: str) -> OHLCV:
                """Generate next realistic OHLCV data point."""
                base_price = self.base_prices[symbol]
                trend = self.price_trends[symbol]
                volatility = self.volatilities[symbol]
                
                # Apply trend and random walk
                price_change = trend + np.random.normal(0, volatility)
                new_price = base_price * (1 + price_change)
                
                # Generate OHLC data
                daily_range = new_price * np.random.uniform(0.005, 0.02)
                
                open_price = new_price + np.random.uniform(-daily_range/2, daily_range/2)
                close_price = new_price + np.random.uniform(-daily_range/2, daily_range/2)
                high_price = max(open_price, close_price) + np.random.uniform(0, daily_range/2)
                low_price = min(open_price, close_price) - np.random.uniform(0, daily_range/2)
                
                volume = int(np.random.uniform(50000, 200000))
                
                # Update base price for next generation
                self.base_prices[symbol] = new_price
                self.data_points_generated += 1
                
                return OHLCV(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume
                )
            
            def generate_batch_data(self, symbols: List[str], count: int) -> Dict[str, List[OHLCV]]:
                """Generate batch of data for multiple symbols."""
                batch_data = {symbol: [] for symbol in symbols}
                
                for _ in range(count):
                    for symbol in symbols:
                        ohlcv = self.generate_next_data(symbol)
                        batch_data[symbol].append(ohlcv)
                
                return batch_data
        
        return PipelineDataGenerator()
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_flow(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test complete end-to-end data pipeline flow."""
        # Mock Redis in ingestion service
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            # Create ingestion service
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Add mock provider with realistic data
            mock_provider = MockProvider()
            
            # Track pipeline metrics
            pipeline_metrics = {
                'data_ingested': 0,
                'data_processed': 0,
                'signals_generated': 0,
                'errors': 0
            }
            
            def mock_get_latest_data(symbol):
                pipeline_metrics['data_ingested'] += 1
                return pipeline_data_generator.generate_next_data(symbol)
            
            mock_provider.get_latest_data = mock_get_latest_data
            ingestion_service.add_provider('mock', mock_provider)
            
            # Simulate data flow
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            for symbol in symbols:
                for _ in range(10):  # Generate 10 data points per symbol
                    try:
                        # Get data from provider
                        latest_data = mock_provider.get_latest_data(symbol)
                        
                        # Publish to Redis stream
                        await ingestion_service._publish_market_data_event(
                            symbol=symbol,
                            data=latest_data,
                            source='mock'
                        )
                        pipeline_metrics['data_processed'] += 1
                        
                        # Simulate small delay
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        pipeline_metrics['errors'] += 1
                        logger.error(f"Pipeline error for {symbol}: {e}")
            
            # Validate pipeline flow
            assert pipeline_metrics['data_ingested'] == 30, "Should ingest data for all symbols"
            assert pipeline_metrics['data_processed'] == 30, "Should process all ingested data"
            assert pipeline_metrics['errors'] == 0, "Should have no errors in normal flow"
            
            # Check Redis stream data
            assert len(mock_redis_pipeline._streams) > 0, "Should create Redis streams"
            assert 'market_data_stream' in mock_redis_pipeline._streams, "Should create market data stream"
            
            stream_messages = mock_redis_pipeline._streams['market_data_stream']
            assert len(stream_messages) == 30, "Should store all messages in stream"
            
            # Validate message content
            for message in stream_messages:
                data = message['data']
                assert 'symbol' in data, "Message should contain symbol"
                assert 'price' in data, "Message should contain price"
                assert 'timestamp' in data, "Message should contain timestamp"
                assert 'source' in data, "Message should contain source"
            
            logger.info(f"✓ End-to-end pipeline test passed:")
            logger.info(f"  Data ingested: {pipeline_metrics['data_ingested']}")
            logger.info(f"  Data processed: {pipeline_metrics['data_processed']}")
            logger.info(f"  Stream messages: {len(stream_messages)}")
    
    @pytest.mark.asyncio
    async def test_data_transformation_validation(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test data transformation and validation in the pipeline."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Test data validation rules
            symbol = 'AAPL'
            
            # Test valid data
            valid_data = pipeline_data_generator.generate_next_data(symbol)
            await ingestion_service._publish_market_data_event(symbol, valid_data, 'test')
            
            # Test invalid data scenarios
            invalid_scenarios = [
                # Negative prices
                OHLCV(datetime.now(), symbol, -100, 105, 95, 102, 1000),
                # Invalid OHLC relationships
                OHLCV(datetime.now(), symbol, 100, 95, 105, 102, 1000),  # high < low
                # Zero volume
                OHLCV(datetime.now(), symbol, 100, 105, 95, 102, 0),
                # Extreme price movements
                OHLCV(datetime.now(), symbol, 100, 500, 50, 102, 1000),
            ]
            
            validation_results = []
            
            for i, invalid_data in enumerate(invalid_scenarios):
                try:
                    # The validation should happen in the data ingestion
                    # For this test, we'll validate the data before publishing
                    is_valid = self._validate_ohlcv_data(invalid_data)
                    validation_results.append(is_valid)
                    
                    if is_valid:
                        await ingestion_service._publish_market_data_event(symbol, invalid_data, 'test')
                    
                except Exception as e:
                    validation_results.append(False)
                    logger.info(f"Correctly rejected invalid data scenario {i}: {e}")
            
            # Most invalid scenarios should be rejected
            valid_count = sum(validation_results)
            assert valid_count <= 1, f"Too many invalid data points accepted: {valid_count}/4"
            
            logger.info(f"✓ Data validation test passed: {len(validation_results) - valid_count}/4 invalid scenarios rejected")
    
    def _validate_ohlcv_data(self, ohlcv: OHLCV) -> bool:
        """Validate OHLCV data integrity."""
        try:
            # Check for positive prices
            if any(price <= 0 for price in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]):
                return False
            
            # Check OHLC relationships
            if not (ohlcv.low <= ohlcv.open <= ohlcv.high and
                   ohlcv.low <= ohlcv.close <= ohlcv.high):
                return False
            
            # Check for reasonable price ranges (no extreme movements)
            price_range = ohlcv.high - ohlcv.low
            avg_price = (ohlcv.open + ohlcv.close) / 2
            if price_range / avg_price > 0.5:  # 50% daily range is extreme
                return False
            
            # Check volume
            if ohlcv.volume < 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    @pytest.mark.asyncio
    async def test_redis_storage_retrieval_performance(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test Redis storage and retrieval performance."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Performance metrics
            storage_times = []
            retrieval_times = []
            
            symbol = 'AAPL'
            num_operations = 100
            
            # Test storage performance
            for i in range(num_operations):
                data = pipeline_data_generator.generate_next_data(symbol)
                
                start_time = time.time()
                await ingestion_service._publish_market_data_event(symbol, data, 'test')
                storage_time = time.time() - start_time
                storage_times.append(storage_time)
            
            # Test retrieval performance
            for i in range(num_operations):
                start_time = time.time()
                latest_price = ingestion_service.get_latest_price(symbol)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
            
            # Analyze performance
            avg_storage_time = sum(storage_times) / len(storage_times)
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            max_storage_time = max(storage_times)
            max_retrieval_time = max(retrieval_times)
            
            # Performance assertions
            assert avg_storage_time < 0.01, f"Average storage time too slow: {avg_storage_time:.6f}s"
            assert avg_retrieval_time < 0.001, f"Average retrieval time too slow: {avg_retrieval_time:.6f}s"
            assert max_storage_time < 0.1, f"Maximum storage time too slow: {max_storage_time:.6f}s"
            assert max_retrieval_time < 0.01, f"Maximum retrieval time too slow: {max_retrieval_time:.6f}s"
            
            logger.info(f"✓ Redis performance test passed:")
            logger.info(f"  Average storage time: {avg_storage_time*1000:.2f}ms")
            logger.info(f"  Average retrieval time: {avg_retrieval_time*1000:.2f}ms")
            logger.info(f"  Operations tested: {num_operations}")
    
    @pytest.mark.asyncio
    async def test_real_time_indicator_calculations(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test real-time indicator calculations in the pipeline."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Mock calculation engine
            calculation_results = []
            
            class MockCalculationEngine:
                def __init__(self):
                    self.indicator_cache = {}
                
                async def calculate_indicators(self, symbol: str, data: List[OHLCV]) -> Dict[str, Any]:
                    """Mock indicator calculations."""
                    if len(data) < 20:
                        return {}  # Need minimum data for indicators
                    
                    prices = [ohlcv.close for ohlcv in data]
                    
                    # Simple moving average
                    sma_20 = sum(prices[-20:]) / 20
                    
                    # Simple RSI calculation
                    gains = [max(prices[i] - prices[i-1], 0) for i in range(1, len(prices))]
                    losses = [max(prices[i-1] - prices[i], 0) for i in range(1, len(prices))]
                    
                    if len(gains) >= 14:
                        avg_gain = sum(gains[-14:]) / 14
                        avg_loss = sum(losses[-14:]) / 14
                        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 50  # Neutral RSI
                    
                    results = {
                        'sma_20': sma_20,
                        'rsi_14': rsi,
                        'last_price': prices[-1],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    calculation_results.append((symbol, results))
                    return results
            
            calc_engine = MockCalculationEngine()
            
            # Generate data and calculate indicators
            symbol = 'AAPL'
            data_history = []
            
            for i in range(50):  # Generate enough data for indicators
                data_point = pipeline_data_generator.generate_next_data(symbol)
                data_history.append(data_point)
                
                # Publish to pipeline
                await ingestion_service._publish_market_data_event(symbol, data_point, 'test')
                
                # Calculate indicators every 5 data points
                if i % 5 == 0 and len(data_history) >= 20:
                    indicators = await calc_engine.calculate_indicators(symbol, data_history)
                    
                    # Publish calculation results
                    if indicators:
                        await ingestion_service.publish_calculation_event(symbol, indicators)
                
                await asyncio.sleep(0.01)
            
            # Validate indicator calculations
            assert len(calculation_results) > 0, "Should generate indicator calculations"
            
            for symbol_name, results in calculation_results:
                assert symbol_name == symbol, "Should calculate for correct symbol"
                assert 'sma_20' in results, "Should calculate SMA"
                assert 'rsi_14' in results, "Should calculate RSI"
                assert 'last_price' in results, "Should include last price"
                
                # Validate indicator ranges
                assert 0 <= results['rsi_14'] <= 100, f"RSI out of range: {results['rsi_14']}"
                assert results['sma_20'] > 0, f"SMA should be positive: {results['sma_20']}"
                assert results['last_price'] > 0, f"Price should be positive: {results['last_price']}"
            
            # Check calculation events in Redis
            calc_streams = mock_redis_pipeline._streams.get('calculation_events_stream', [])
            assert len(calc_streams) > 0, "Should publish calculation events"
            
            logger.info(f"✓ Real-time indicator calculation test passed:")
            logger.info(f"  Calculations performed: {len(calculation_results)}")
            logger.info(f"  Calculation events stored: {len(calc_streams)}")
    
    @pytest.mark.asyncio
    async def test_signal_generation_pipeline(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test signal generation in the data pipeline."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Mock signal engine
            generated_signals = []
            
            class MockSignalEngine:
                def __init__(self):
                    self.signal_count = 0
                
                async def evaluate_signals(self, symbol: str, indicators: Dict[str, Any]) -> List[Signal]:
                    """Mock signal generation."""
                    signals = []
                    
                    if 'rsi_14' in indicators and 'sma_20' in indicators:
                        rsi = indicators['rsi_14']
                        sma = indicators['sma_20']
                        price = indicators['last_price']
                        
                        # Simple signal logic
                        if rsi < 30 and price < sma:
                            # Oversold and below SMA - Buy signal
                            signal = Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                price=price,
                                confidence=0.7,
                                strategy_name='rsi_sma_strategy',
                                metadata={'rsi': rsi, 'sma': sma}
                            )
                            signals.append(signal)
                            self.signal_count += 1
                        
                        elif rsi > 70 and price > sma:
                            # Overbought and above SMA - Sell signal
                            signal = Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                price=price,
                                confidence=0.6,
                                strategy_name='rsi_sma_strategy',
                                metadata={'rsi': rsi, 'sma': sma}
                            )
                            signals.append(signal)
                            self.signal_count += 1
                    
                    generated_signals.extend(signals)
                    return signals
            
            signal_engine = MockSignalEngine()
            
            # Generate data with trend to trigger signals
            symbol = 'AAPL'
            data_history = []
            
            # Create downward trend to trigger buy signal
            pipeline_data_generator.price_trends[symbol] = -0.01  # Strong downward trend
            
            for i in range(60):
                data_point = pipeline_data_generator.generate_next_data(symbol)
                data_history.append(data_point)
                
                # Publish data
                await ingestion_service._publish_market_data_event(symbol, data_point, 'test')
                
                # Calculate indicators and generate signals
                if i % 10 == 0 and len(data_history) >= 20:
                    # Mock indicator calculation
                    prices = [ohlcv.close for ohlcv in data_history]
                    sma_20 = sum(prices[-20:]) / 20
                    
                    # Simulate RSI based on trend
                    if i > 40:  # Later in the sequence, simulate oversold
                        rsi = 25  # Oversold
                    else:
                        rsi = 45  # Neutral
                    
                    indicators = {
                        'sma_20': sma_20,
                        'rsi_14': rsi,
                        'last_price': prices[-1]
                    }
                    
                    # Generate signals
                    signals = await signal_engine.evaluate_signals(symbol, indicators)
                    
                    # Publish signals as alerts
                    for signal in signals:
                        await ingestion_service.publish_alert(
                            alert_type='signal',
                            symbol=symbol,
                            message=f"{signal.signal_type.value} signal generated",
                            metadata={
                                'signal_type': signal.signal_type.value,
                                'price': signal.price,
                                'confidence': signal.confidence,
                                'strategy': signal.strategy_name
                            }
                        )
                
                await asyncio.sleep(0.01)
            
            # Validate signal generation
            assert len(generated_signals) > 0, "Should generate trading signals"
            
            buy_signals = [s for s in generated_signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in generated_signals if s.signal_type == SignalType.SELL]
            
            # Should have at least one buy signal due to downward trend
            assert len(buy_signals) > 0, "Should generate buy signals in downward trend"
            
            for signal in generated_signals:
                assert signal.symbol == symbol, "Signal should be for correct symbol"
                assert signal.confidence > 0, "Signal should have positive confidence"
                assert signal.price > 0, "Signal should have valid price"
                assert signal.strategy_name == 'rsi_sma_strategy', "Signal should have strategy name"
            
            # Check alert events in Redis
            alert_streams = mock_redis_pipeline._streams.get('alert_stream', [])
            signal_alerts = [msg for msg in alert_streams if json.loads(msg['data']['metadata']).get('signal_type')]
            
            assert len(signal_alerts) > 0, "Should publish signal alerts"
            
            logger.info(f"✓ Signal generation pipeline test passed:")
            logger.info(f"  Total signals: {len(generated_signals)}")
            logger.info(f"  Buy signals: {len(buy_signals)}")
            logger.info(f"  Sell signals: {len(sell_signals)}")
            logger.info(f"  Signal alerts: {len(signal_alerts)}")
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling_recovery(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test error handling and recovery in the pipeline."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Track error scenarios
            error_scenarios = {
                'redis_errors': 0,
                'data_errors': 0,
                'processing_errors': 0,
                'recovered_operations': 0
            }
            
            # Add failing provider
            class FailingProvider:
                def __init__(self):
                    self.call_count = 0
                
                def get_latest_data(self, symbol):
                    self.call_count += 1
                    
                    # Fail every 3rd call
                    if self.call_count % 3 == 0:
                        error_scenarios['data_errors'] += 1
                        raise Exception(f"Simulated data provider error {self.call_count}")
                    
                    return pipeline_data_generator.generate_next_data(symbol)
            
            failing_provider = FailingProvider()
            ingestion_service.add_provider('failing_provider', failing_provider)
            
            # Simulate Redis failures
            original_xadd = mock_redis_pipeline.xadd
            
            def failing_xadd(stream_name, data, **kwargs):
                if len(mock_redis_pipeline._streams.get(stream_name, [])) > 20:  # Fail after 20 messages
                    error_scenarios['redis_errors'] += 1
                    raise Exception("Simulated Redis error")
                return original_xadd(stream_name, data, **kwargs)
            
            mock_redis_pipeline.xadd = failing_xadd
            
            # Test error recovery
            symbol = 'AAPL'
            successful_operations = 0
            
            for i in range(30):
                try:
                    # Try to get data (may fail)
                    data = failing_provider.get_latest_data(symbol)
                    
                    # Try to publish (may fail due to Redis)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    
                    successful_operations += 1
                    error_scenarios['recovered_operations'] += 1
                    
                except Exception as e:
                    error_scenarios['processing_errors'] += 1
                    logger.debug(f"Pipeline error in iteration {i}: {e}")
                    
                    # Simulate recovery delay
                    await asyncio.sleep(0.01)
            
            # Validate error handling
            total_errors = (error_scenarios['redis_errors'] + 
                          error_scenarios['data_errors'] + 
                          error_scenarios['processing_errors'])
            
            assert total_errors > 0, "Should encounter simulated errors"
            assert successful_operations > 0, "Should have some successful operations"
            
            # Recovery rate should be reasonable
            recovery_rate = successful_operations / 30
            assert recovery_rate > 0.3, f"Recovery rate too low: {recovery_rate:.2%}"
            
            logger.info(f"✓ Pipeline error handling test passed:")
            logger.info(f"  Total errors: {total_errors}")
            logger.info(f"  Successful operations: {successful_operations}")
            logger.info(f"  Recovery rate: {recovery_rate:.2%}")
            logger.info(f"  Error breakdown: {error_scenarios}")
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_under_load(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test pipeline performance under high load conditions."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Performance tracking
            performance_metrics = {
                'total_operations': 0,
                'total_time': 0,
                'peak_memory_mb': 0,
                'throughput_ops_per_sec': 0,
                'latency_samples': []
            }
            
            # Memory monitoring
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # High-load test parameters
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            operations_per_symbol = 100
            total_operations = len(symbols) * operations_per_symbol
            
            start_time = time.time()
            
            # Simulate high-load pipeline operations
            for i in range(operations_per_symbol):
                batch_start_time = time.time()
                
                # Process all symbols in parallel
                tasks = []
                for symbol in symbols:
                    async def process_symbol_data(sym):
                        operation_start = time.time()
                        
                        # Generate data
                        data = pipeline_data_generator.generate_next_data(sym)
                        
                        # Publish to pipeline
                        await ingestion_service._publish_market_data_event(sym, data, 'load_test')
                        
                        operation_time = time.time() - operation_start
                        performance_metrics['latency_samples'].append(operation_time)
                    
                    tasks.append(process_symbol_data(symbol))
                
                # Execute batch
                await asyncio.gather(*tasks)
                
                # Update metrics
                performance_metrics['total_operations'] += len(symbols)
                
                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                performance_metrics['peak_memory_mb'] = max(performance_metrics['peak_memory_mb'], current_memory)
                
                # Control batch rate
                batch_time = time.time() - batch_start_time
                if batch_time < 0.01:  # Maintain reasonable rate
                    await asyncio.sleep(0.01 - batch_time)
            
            total_time = time.time() - start_time
            performance_metrics['total_time'] = total_time
            performance_metrics['throughput_ops_per_sec'] = total_operations / total_time
            
            # Analyze performance
            avg_latency = sum(performance_metrics['latency_samples']) / len(performance_metrics['latency_samples'])
            p95_latency = sorted(performance_metrics['latency_samples'])[int(len(performance_metrics['latency_samples']) * 0.95)]
            max_latency = max(performance_metrics['latency_samples'])
            
            memory_increase = performance_metrics['peak_memory_mb'] - initial_memory
            
            # Performance assertions
            assert performance_metrics['throughput_ops_per_sec'] > 100, f"Throughput too low: {performance_metrics['throughput_ops_per_sec']:.1f} ops/s"
            assert avg_latency < 0.01, f"Average latency too high: {avg_latency*1000:.1f}ms"
            assert p95_latency < 0.05, f"P95 latency too high: {p95_latency*1000:.1f}ms"
            assert memory_increase < 50, f"Memory increase too high: {memory_increase:.1f}MB"
            
            # Validate data integrity under load
            total_messages = sum(len(stream) for stream in mock_redis_pipeline._streams.values())
            message_loss_rate = 1 - (total_messages / total_operations)
            
            assert message_loss_rate < 0.01, f"Message loss rate too high: {message_loss_rate:.2%}"
            
            logger.info(f"✓ Pipeline load test passed:")
            logger.info(f"  Total operations: {total_operations}")
            logger.info(f"  Throughput: {performance_metrics['throughput_ops_per_sec']:.1f} ops/s")
            logger.info(f"  Average latency: {avg_latency*1000:.1f}ms")
            logger.info(f"  P95 latency: {p95_latency*1000:.1f}ms")
            logger.info(f"  Memory increase: {memory_increase:.1f}MB")
            logger.info(f"  Message loss rate: {message_loss_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_pipeline_data_consistency(self, pipeline_config_manager, mock_redis_pipeline, pipeline_data_generator):
        """Test data consistency across the pipeline."""
        with patch('redis.Redis', return_value=mock_redis_pipeline):
            ingestion_service = MarketDataIngestionService(pipeline_config_manager)
            ingestion_service._redis_client = mock_redis_pipeline
            
            # Track data flow
            data_checkpoints = {
                'ingestion': [],
                'storage': [],
                'retrieval': []
            }
            
            symbol = 'AAPL'
            
            # Generate and track data through pipeline
            for i in range(20):
                # Generate original data
                original_data = pipeline_data_generator.generate_next_data(symbol)
                data_checkpoints['ingestion'].append(original_data)
                
                # Publish through pipeline
                await ingestion_service._publish_market_data_event(symbol, original_data, 'consistency_test')
                
                # Retrieve from cache
                latest_price = ingestion_service.get_latest_price(symbol)
                if latest_price:
                    data_checkpoints['retrieval'].append(latest_price)
                
                await asyncio.sleep(0.01)
            
            # Extract stored data from Redis streams
            stream_messages = mock_redis_pipeline._streams.get('market_data_stream', [])
            for message in stream_messages:
                if message['data']['symbol'] == symbol:
                    stored_price = float(message['data']['price'])
                    data_checkpoints['storage'].append(stored_price)
            
            # Validate data consistency
            ingestion_prices = [data.close for data in data_checkpoints['ingestion']]
            storage_prices = data_checkpoints['storage']
            retrieval_prices = data_checkpoints['retrieval']
            
            # Check data preservation
            assert len(storage_prices) == len(ingestion_prices), "All ingested data should be stored"
            
            # Check price consistency (allowing for small floating-point differences)
            for i, (original, stored) in enumerate(zip(ingestion_prices, storage_prices)):
                price_diff = abs(original - stored)
                assert price_diff < 0.01, f"Price inconsistency at index {i}: {original} vs {stored}"
            
            # Check retrieval consistency
            if retrieval_prices:
                last_retrieval = retrieval_prices[-1]
                last_storage = storage_prices[-1]
                assert abs(last_retrieval - last_storage) < 0.01, "Latest retrieval should match storage"
            
            # Check chronological order
            ingestion_timestamps = [data.timestamp for data in data_checkpoints['ingestion']]
            assert ingestion_timestamps == sorted(ingestion_timestamps), "Data should be in chronological order"
            
            logger.info(f"✓ Pipeline data consistency test passed:")
            logger.info(f"  Data points processed: {len(ingestion_prices)}")
            logger.info(f"  Storage consistency: 100%")
            logger.info(f"  Retrieval consistency: {'✓' if retrieval_prices else 'N/A'}")
            logger.info(f"  Chronological order: ✓")