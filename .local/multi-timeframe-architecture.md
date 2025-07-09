# Multi-Strategy Multi-Timeframe Processing Architecture

## Executive Summary

This document outlines the architecture design for enhancing the Wagehood real-time trading system to support multi-strategy multi-timeframe processing. The system will apply all 5 trading strategies to every watchlist symbol across 3 different timeframes (day trading, swing trading, and position trading) while maintaining high performance and backward compatibility.

## Current Architecture Analysis

### Existing Components

1. **ConfigManager** (`config_manager.py`)
   - Manages watchlist symbols and strategies
   - Currently supports single timeframe per symbol
   - Stores configuration in Redis with TTL

2. **CalculationEngine** (`calculation_engine.py`)
   - Processes market data from Redis Streams
   - Calculates indicators and generates signals
   - Uses thread pool for parallel processing
   - Batches messages by symbol for efficiency

3. **MarketDataIngestionService** (`data_ingestion.py`)
   - Fetches data from providers (Mock, Alpaca)
   - Publishes to Redis Streams
   - Uses circuit breakers for reliability

4. **IncrementalIndicatorCalculator** (`incremental_indicators.py`)
   - Maintains state for incremental calculations
   - Reduces computational overhead

5. **Trading Strategies** (`strategies/`)
   - 5 strategies: MACD+RSI, MA Crossover, RSI Trend, Bollinger Breakout, S/R Breakout
   - Each strategy extends TradingStrategy base class

### Current Limitations

1. Single timeframe processing per symbol
2. No differentiation between trading styles (day/swing/position)
3. Limited timeframe aggregation support
4. No multi-timeframe signal correlation

## Proposed Architecture

### Core Design Principles

1. **Separation of Concerns**: Timeframe processing separated from strategy logic
2. **Scalability**: Horizontal scaling through worker distribution
3. **Efficiency**: Shared indicator calculations across strategies
4. **Backward Compatibility**: Existing APIs remain unchanged

### Data Structure Modifications

#### 1. Enhanced AssetConfig

```python
@dataclass
class AssetConfig:
    symbol: str
    enabled: bool
    data_provider: str
    timeframes: Dict[str, TimeframeConfig]  # Changed from List[str]
    priority: int = 1
    asset_type: str = "stock"
    last_updated: Optional[datetime] = None

@dataclass
class TimeframeConfig:
    name: str  # "1s", "1m", "5m", "1h", "1d"
    trading_style: str  # "day", "swing", "position"
    update_frequency_seconds: int
    enabled: bool = True
    buffer_size: int = 1000  # Data points to keep in memory
```

#### 2. Multi-Timeframe Indicator State

```python
@dataclass
class MultiTimeframeIndicatorState:
    """Maintains indicator state across multiple timeframes"""
    symbol: str
    timeframe_states: Dict[str, IndicatorState]  # timeframe -> state
    last_sync_time: datetime
    
@dataclass
class TimeframeAggregator:
    """Aggregates lower timeframe data to higher timeframes"""
    source_timeframe: str
    target_timeframe: str
    aggregation_method: str  # "ohlc", "vwap", "typical"
    buffer: List[MarketDataEvent]
    last_aggregation: datetime
```

#### 3. Enhanced Signal Structure

```python
@dataclass
class MultiTimeframeSignal:
    base_signal: Signal
    timeframe: str
    trading_style: str
    correlated_signals: List[Signal]  # Signals from other timeframes
    composite_confidence: float
    metadata: Dict[str, Any]
```

### Processing Flow Changes

#### 1. Data Ingestion Layer

```python
class EnhancedMarketDataIngestionService(MarketDataIngestionService):
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.timeframe_aggregators = {}
        self.raw_data_buffers = {}  # symbol -> timeframe -> buffer
        
    async def publish_market_data(self, symbol: str, data: OHLCV):
        # Publish to lowest timeframe stream
        await self._publish_to_stream("market_data_1s", data)
        
        # Trigger aggregation for higher timeframes
        await self._aggregate_timeframes(symbol, data)
        
    async def _aggregate_timeframes(self, symbol: str, data: OHLCV):
        # Aggregate to all configured timeframes
        for timeframe in self._get_symbol_timeframes(symbol):
            if self._should_aggregate(symbol, timeframe):
                aggregated_data = await self._aggregate_data(symbol, timeframe)
                await self._publish_to_stream(f"market_data_{timeframe}", aggregated_data)
```

#### 2. Calculation Engine Enhancements

```python
class MultiTimeframeCalculationEngine(CalculationEngine):
    def __init__(self, config_manager: ConfigManager, ingestion_service: MarketDataIngestionService):
        super().__init__(config_manager, ingestion_service)
        self.timeframe_calculators = {}  # timeframe -> calculator
        self.multi_timeframe_states = {}  # symbol -> MTF state
        
    async def _consume_market_data_streams(self):
        """Consume from multiple timeframe streams"""
        streams = self._get_timeframe_streams()
        
        while self._running:
            # Read from all timeframe streams
            messages = await self._read_multi_stream(streams)
            
            # Group by symbol and timeframe
            grouped = self._group_messages_by_symbol_timeframe(messages)
            
            # Process in parallel
            tasks = []
            for (symbol, timeframe), msgs in grouped.items():
                task = self._process_timeframe_batch(symbol, timeframe, msgs)
                tasks.append(task)
                
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _process_timeframe_batch(self, symbol: str, timeframe: str, messages: List):
        """Process messages for specific symbol and timeframe"""
        # Get trading style for this timeframe
        trading_style = self._get_trading_style(symbol, timeframe)
        
        # Calculate indicators for this timeframe
        indicators = await self._calculate_timeframe_indicators(symbol, timeframe, messages)
        
        # Generate signals for ALL strategies
        all_signals = await self._generate_all_strategy_signals(symbol, timeframe, indicators)
        
        # Correlate with other timeframe signals
        correlated_signals = await self._correlate_signals(symbol, all_signals, timeframe)
        
        # Publish results
        await self._publish_calculation_results(symbol, timeframe, correlated_signals)
```

#### 3. Strategy Processing Pipeline

```python
class StrategyProcessor:
    """Processes all strategies for a given symbol and timeframe"""
    
    def __init__(self, strategies: List[TradingStrategy]):
        self.strategies = strategies
        self.shared_indicators = {}  # Cache shared indicator calculations
        
    async def process_all_strategies(self, symbol: str, timeframe: str, 
                                   market_data: MarketData) -> Dict[str, List[Signal]]:
        """Process all strategies efficiently"""
        # Pre-calculate all required indicators
        all_required = self._get_all_required_indicators()
        shared_indicators = await self._calculate_shared_indicators(market_data, all_required)
        
        # Process strategies in parallel
        tasks = []
        for strategy in self.strategies:
            task = self._process_strategy(strategy, market_data, shared_indicators)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return strategy_name -> signals mapping
        return dict(zip([s.name for s in self.strategies], results))
```

### Performance Optimization Strategies

#### 1. Shared Indicator Calculations

```python
class SharedIndicatorCache:
    """Cache for indicators used by multiple strategies"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}  # (symbol, timeframe, indicator) -> (value, timestamp)
        self.ttl = ttl_seconds
        self.calculation_locks = {}  # Prevent duplicate calculations
        
    async def get_or_calculate(self, key: Tuple, calculator: Callable) -> Any:
        """Get from cache or calculate if missing/expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
                
        # Ensure only one calculation per key
        lock_key = str(key)
        if lock_key not in self.calculation_locks:
            self.calculation_locks[lock_key] = asyncio.Lock()
            
        async with self.calculation_locks[lock_key]:
            # Double-check after acquiring lock
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                    
            # Calculate and cache
            value = await calculator()
            self.cache[key] = (value, time.time())
            return value
```

#### 2. Timeframe-Based Worker Distribution

```python
class TimeframeWorkerPool:
    """Distributes work based on timeframe priority"""
    
    def __init__(self, num_workers: int = 4):
        self.workers = {
            "day": [],      # High frequency workers
            "swing": [],    # Medium frequency workers  
            "position": []  # Low frequency workers
        }
        self._distribute_workers(num_workers)
        
    def _distribute_workers(self, total: int):
        """Distribute workers based on timeframe needs"""
        # 50% for day trading, 30% for swing, 20% for position
        self.workers["day"] = [Worker(f"day_{i}") for i in range(total // 2)]
        self.workers["swing"] = [Worker(f"swing_{i}") for i in range(total * 3 // 10)]
        self.workers["position"] = [Worker(f"position_{i}") for i in range(total - total // 2 - total * 3 // 10)]
```

#### 3. Batched Redis Operations

```python
class BatchedRedisPublisher:
    """Batches Redis operations for efficiency"""
    
    def __init__(self, redis_client, batch_size: int = 100, flush_interval: float = 0.1):
        self.redis = redis_client
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()
        self.flush_lock = asyncio.Lock()
        
    async def publish(self, stream: str, data: Dict):
        """Add to buffer and flush if needed"""
        async with self.flush_lock:
            self.buffer.append((stream, data))
            
            if len(self.buffer) >= self.batch_size or \
               time.time() - self.last_flush > self.flush_interval:
                await self._flush()
                
    async def _flush(self):
        """Flush buffer to Redis using pipeline"""
        if not self.buffer:
            return
            
        pipeline = self.redis.pipeline()
        for stream, data in self.buffer:
            pipeline.xadd(stream, data, maxlen=10000)
            
        await pipeline.execute()
        self.buffer.clear()
        self.last_flush = time.time()
```

### Memory Management

#### 1. Sliding Window Buffers

```python
class SlidingWindowBuffer:
    """Memory-efficient buffer for time series data"""
    
    def __init__(self, max_size: int, timeframe: str):
        self.max_size = max_size
        self.timeframe = timeframe
        self.data = deque(maxlen=max_size)
        self.summary_stats = {}  # Keep summary statistics
        
    def add(self, item: MarketDataEvent):
        """Add item and update statistics"""
        self.data.append(item)
        self._update_stats(item)
        
    def get_window(self, size: int) -> List[MarketDataEvent]:
        """Get last N items efficiently"""
        return list(itertools.islice(self.data, max(0, len(self.data) - size), len(self.data)))
```

#### 2. Lazy Loading Indicators

```python
class LazyIndicator:
    """Calculate indicators only when accessed"""
    
    def __init__(self, calculator: Callable, cache_ttl: int = 60):
        self._calculator = calculator
        self._cache = None
        self._cache_time = None
        self._cache_ttl = cache_ttl
        
    @property
    def value(self):
        """Calculate or return cached value"""
        if self._cache is None or time.time() - self._cache_time > self._cache_ttl:
            self._cache = self._calculator()
            self._cache_time = time.time()
        return self._cache
```

### Signal Correlation and Filtering

```python
class SignalCorrelator:
    """Correlates signals across timeframes"""
    
    def correlate_signals(self, symbol: str, signals_by_timeframe: Dict[str, List[Signal]]) -> List[MultiTimeframeSignal]:
        """Correlate signals across timeframes"""
        correlated = []
        
        # Start with highest timeframe signals
        for tf in ["1d", "1h", "5m", "1m", "1s"]:
            if tf not in signals_by_timeframe:
                continue
                
            for signal in signals_by_timeframe[tf]:
                # Find supporting signals in lower timeframes
                supporting = self._find_supporting_signals(signal, signals_by_timeframe, tf)
                
                # Calculate composite confidence
                composite_confidence = self._calculate_composite_confidence(signal, supporting)
                
                correlated.append(MultiTimeframeSignal(
                    base_signal=signal,
                    timeframe=tf,
                    trading_style=self._get_trading_style(tf),
                    correlated_signals=supporting,
                    composite_confidence=composite_confidence,
                    metadata={"correlation_strength": len(supporting) / len(signals_by_timeframe)}
                ))
                
        return correlated
```

## Implementation Plan

### Phase 1: Data Structure Updates (Week 1)
1. Update config models with timeframe support
2. Implement TimeframeAggregator
3. Create multi-timeframe buffer management

### Phase 2: Processing Pipeline (Week 2)
1. Enhance calculation engine for multi-stream consumption
2. Implement shared indicator cache
3. Create strategy processor for parallel execution

### Phase 3: Optimization (Week 3)
1. Implement batched Redis operations
2. Add memory management features
3. Create performance monitoring

### Phase 4: Signal Correlation (Week 4)
1. Implement signal correlator
2. Add composite confidence calculation
3. Create filtering and prioritization logic

### Phase 5: Testing and Tuning (Week 5)
1. Performance benchmarking
2. Memory usage optimization
3. Load testing with full symbol set

## Backward Compatibility

### Maintained APIs
- `ConfigManager` methods remain unchanged
- `CalculationEngine.get_latest_results()` continues to work
- Strategy interfaces unchanged

### Migration Path
1. New features opt-in via configuration
2. Default behavior matches current system
3. Gradual migration tools provided

## Performance Targets

### Processing Capacity
- **Current**: 100 symbols × 1 timeframe × 5 strategies = 500 calculations/second
- **Target**: 100 symbols × 5 timeframes × 5 strategies = 2,500 calculations/second

### Latency Goals
- Day trading signals: < 100ms
- Swing trading signals: < 500ms
- Position trading signals: < 1000ms

### Memory Usage
- Per symbol overhead: < 10MB
- Total system memory: < 4GB for 100 symbols

## Monitoring and Observability

### Key Metrics
1. Calculations per second by timeframe
2. Signal generation latency percentiles
3. Memory usage by component
4. Redis Stream lag
5. Cache hit rates

### Health Checks
```python
class HealthChecker:
    async def check_system_health(self) -> Dict[str, Any]:
        return {
            "calculation_lag": self._check_calculation_lag(),
            "memory_usage": self._check_memory_usage(),
            "redis_streams": self._check_stream_health(),
            "worker_status": self._check_worker_health(),
            "signal_quality": self._check_signal_quality()
        }
```

## Risk Mitigation

### Performance Risks
1. **Risk**: CPU overload from 5x calculation increase
   - **Mitigation**: Shared indicator cache, parallel processing

2. **Risk**: Memory exhaustion from multiple buffers
   - **Mitigation**: Sliding windows, configurable buffer sizes

3. **Risk**: Redis throughput bottleneck
   - **Mitigation**: Batched operations, stream partitioning

### Operational Risks
1. **Risk**: Complex debugging with multiple timeframes
   - **Mitigation**: Enhanced logging, correlation IDs

2. **Risk**: Signal quality degradation
   - **Mitigation**: Composite confidence scoring, validation

## Future Enhancements

1. **Dynamic Timeframe Selection**: Automatically adjust timeframes based on market conditions
2. **ML-Based Signal Fusion**: Use machine learning to optimize signal correlation
3. **Adaptive Buffer Sizing**: Dynamically adjust memory usage based on system load
4. **Cross-Asset Correlation**: Correlate signals across different symbols
5. **Strategy Performance Tracking**: Real-time strategy effectiveness monitoring