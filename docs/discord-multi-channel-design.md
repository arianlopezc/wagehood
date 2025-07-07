# Discord Multi-Channel Notification System - Design Document

## Executive Summary

Design and implementation plan for organizing Discord notifications into separate channels per swing trading strategy, providing better organization and strategy-specific tracking.

## Current System Analysis

### Existing Architecture
- **Single Discord webhook**: All swing trading signals go to one channel
- **Current components**: DiscordNotifier, MessageFormatter, NotificationService, NotificationConfig
- **Rich embed formatting**: Color-coded messages with strategy information
- **Rate limiting**: Configurable notifications per hour
- **Timeframe filtering**: Now properly filters to 1d timeframe only

### Identified Limitations
- All strategies mixed in one channel makes tracking difficult
- No strategy-specific customization (rate limits, confidence thresholds)
- Limited scalability for adding new strategies
- Difficult to analyze performance by individual strategy

## Available Swing Trading Strategies

Based on system analysis, these 5 strategies will each get their own channel:

1. **Moving Average Crossover** (`ma_crossover`)
   - **Description**: Golden Cross/Death Cross signals
   - **Optimal Timeframe**: 1d
   - **Trading Profile**: Position Trading
   - **Academic Win Rate**: Not specifically tested

2. **MACD + RSI Combined** (`macd_rsi`) 
   - **Description**: Dual indicator momentum strategy
   - **Optimal Timeframe**: 1h (but using 1d for notifications)
   - **Trading Profile**: Swing Trading
   - **Academic Win Rate**: 73% (validated)

3. **RSI Trend Following** (`rsi_trend`)
   - **Description**: Trend confirmation and pullback timing
   - **Optimal Timeframe**: 1h (but using 1d for notifications)
   - **Trading Profile**: Swing Trading
   - **Academic Win Rate**: RSI has 81% prediction accuracy

4. **Bollinger Band Breakout** (`bollinger_breakout`)
   - **Description**: Volatility-based breakout strategy
   - **Optimal Timeframe**: 15m (but using 1d for notifications)
   - **Trading Profile**: Day Trading (adapted for swing)
   - **Academic Win Rate**: 78% (validated)

5. **Support/Resistance Breakout** (`sr_breakout`)
   - **Description**: Level-based breakout strategy
   - **Optimal Timeframe**: 1h (but using 1d for notifications)
   - **Trading Profile**: Swing Trading
   - **Confidence Threshold**: 70% (highest due to complexity)

## Multi-Channel Architecture Design

### Channel Organization

```
📈-swing-ma-crossover        # Moving Average Crossover (🟢 Green theme)
📊-swing-macd-rsi           # MACD + RSI Combined (🔵 Blue theme)
📉-swing-rsi-trend          # RSI Trend Following (🟡 Yellow theme)
💥-swing-bollinger-breakout # Bollinger Band Breakout (🟠 Orange theme)
🎯-swing-sr-breakout        # Support/Resistance Breakout (🔴 Red theme)
```

### Strategy-Specific Configurations

Each strategy channel will have customized settings:

| Strategy | Rate Limit/Hour | Min Confidence | Channel Color | Priority |
|----------|-----------------|----------------|---------------|----------|
| MA Crossover | 5 | 65% | Green (65280) | Medium |
| MACD+RSI | 8 | 60% | Blue (3447003) | High |
| RSI Trend | 6 | 60% | Yellow (16776960) | High |
| Bollinger Breakout | 10 | 60% | Orange (16753920) | Medium |
| S/R Breakout | 3 | 70% | Red (16711680) | High |

### Enhanced Message Design

Each strategy will have unique embed formatting:

**MACD+RSI Channel Example:**
```
📊 MACD+RSI SWING SIGNAL
🎯 SPY - SPDR S&P 500 ETF
💰 Price: $623.31 (+1.6%)
📈 Signal: BUY
🎯 Confidence: 87%
⏰ Timeframe: 1d
🔍 MACD: Bullish crossover above signal line
📊 RSI: Oversold recovery from 28.5 to 45.2
📋 Strategy: Dual momentum confirmation
```

## Implementation Architecture

### Core Components

#### 1. `MultiChannelNotificationConfig`
```python
@dataclass
class StrategyChannelConfig:
    strategy_name: str
    webhook_url: str
    enabled: bool = True
    max_notifications_per_hour: int = 5
    min_confidence_threshold: float = 0.6
    channel_color: int = 3447003
    emoji: str = "📊"
    
@dataclass  
class MultiChannelNotificationConfig:
    enabled: bool = True
    strategy_channels: Dict[str, StrategyChannelConfig] = field(default_factory=dict)
    fallback_webhook_url: str = ""
    default_timeframes: List[str] = field(default_factory=lambda: ["1d"])
```

#### 2. `MultiChannelDiscordNotifier`
```python
class MultiChannelDiscordNotifier:
    def __init__(self, config: MultiChannelNotificationConfig):
        self.config = config
        self.strategy_notifiers = {}
        self.strategy_rate_limiters = {}
        
    async def send_strategy_notification(self, strategy: str, signal_data: Dict) -> bool:
        # Route to appropriate strategy channel
        
    def get_strategy_notifier(self, strategy: str) -> DiscordNotifier:
        # Return strategy-specific notifier or fallback
```

#### 3. `StrategyMessageFormatter`
```python
class StrategyMessageFormatter:
    def format_strategy_embed(self, strategy: str, signal_data: Dict) -> Dict:
        # Strategy-specific formatting with custom colors and emojis
        
    def get_strategy_template(self, strategy: str) -> Dict:
        # Return strategy-specific embed template
```

#### 4. `MultiChannelNotificationService`
```python
class MultiChannelNotificationService:
    def __init__(self, config: MultiChannelNotificationConfig):
        self.config = config
        self.notifier = MultiChannelDiscordNotifier(config)
        self.formatter = StrategyMessageFormatter()
        
    async def process_signal_event(self, signal_data: Dict) -> bool:
        strategy = signal_data.get('strategy', 'unknown')
        return await self.notifier.send_strategy_notification(strategy, signal_data)
```

### Configuration Management

#### Environment Variables
```bash
# Global settings
DISCORD_MULTI_CHANNEL_ENABLED=true
DISCORD_FALLBACK_WEBHOOK=https://discord.com/api/webhooks/fallback

# Strategy-specific webhook URLs
DISCORD_WEBHOOK_MA_CROSSOVER=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_MACD_RSI=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_RSI_TREND=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_BOLLINGER_BREAKOUT=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_SR_BREAKOUT=https://discord.com/api/webhooks/...

# Strategy-specific configurations
DISCORD_RATE_LIMIT_MA_CROSSOVER=5
DISCORD_MIN_CONFIDENCE_MA_CROSSOVER=0.65
DISCORD_ENABLED_MA_CROSSOVER=true

DISCORD_RATE_LIMIT_MACD_RSI=8
DISCORD_MIN_CONFIDENCE_MACD_RSI=0.60
DISCORD_ENABLED_MACD_RSI=true

DISCORD_RATE_LIMIT_RSI_TREND=6
DISCORD_MIN_CONFIDENCE_RSI_TREND=0.60
DISCORD_ENABLED_RSI_TREND=true

DISCORD_RATE_LIMIT_BOLLINGER_BREAKOUT=10
DISCORD_MIN_CONFIDENCE_BOLLINGER_BREAKOUT=0.60
DISCORD_ENABLED_BOLLINGER_BREAKOUT=true

DISCORD_RATE_LIMIT_SR_BREAKOUT=3
DISCORD_MIN_CONFIDENCE_SR_BREAKOUT=0.70
DISCORD_ENABLED_SR_BREAKOUT=true
```

## Implementation Plan

### Phase 1: Core Multi-Channel Architecture (Week 1)
**Days 1-2: Configuration Layer**
- Implement `StrategyChannelConfig` and `MultiChannelNotificationConfig`
- Add environment variable loading for strategy-specific settings
- Create configuration validation and error handling

**Days 3-4: Notifier Layer**  
- Implement `MultiChannelDiscordNotifier`
- Add strategy-specific webhook routing
- Implement per-strategy rate limiting

**Days 5-7: Testing and Integration**
- Unit tests for multi-channel configuration
- Integration tests with mock webhooks
- Validate environment variable loading

### Phase 2: Enhanced Formatting and Service (Week 2)
**Days 1-3: Message Formatting**
- Implement `StrategyMessageFormatter`
- Add strategy-specific embed templates
- Create color coding and emoji system

**Days 4-5: Service Integration**
- Implement `MultiChannelNotificationService`
- Integrate with existing notification pipeline
- Add strategy detection and routing logic

**Days 6-7: Testing and Refinement**
- End-to-end testing with real signal data
- Performance testing with multiple strategies
- Error handling and fallback testing

### Phase 3: Discord Server Setup (Week 3)
**Days 1-2: Channel Creation**
- Create 5 strategy-specific channels in Discord server
- Set up proper channel permissions and organization
- Configure channel descriptions and topics

**Days 3-4: Webhook Configuration**
- Generate webhook URLs for each channel
- Update environment variables with real webhook URLs
- Test webhook connectivity for each channel

**Days 5-7: Monitoring and Alerts**
- Set up channel-specific monitoring
- Configure alerts for webhook failures
- Create channel usage analytics

### Phase 4: Deployment and Monitoring (Week 4)
**Days 1-3: Gradual Rollout**
- Deploy to staging environment
- Test with live signal data (using test webhooks)
- Validate strategy routing and filtering

**Days 4-5: Production Deployment**
- Deploy to production with real Discord channels
- Monitor notification delivery and performance
- Collect initial usage metrics

**Days 6-7: Optimization and Documentation**
- Performance optimization based on real usage
- Create user documentation for Discord channels
- Training materials for strategy-specific channels

## Discord Server Setup Requirements

### Required Actions

1. **Create Strategy Channels**
   ```
   #📈-swing-ma-crossover
   #📊-swing-macd-rsi  
   #📉-swing-rsi-trend
   #💥-swing-bollinger-breakout
   #🎯-swing-sr-breakout
   ```

2. **Channel Configuration**
   - Set channel topics describing each strategy
   - Configure appropriate permissions (read-only for general users)
   - Set up channel categories for organization

3. **Webhook Setup**
   - Generate individual webhook URLs for each channel
   - Configure webhook names and avatars for each strategy
   - Test webhook connectivity before production use

4. **Channel Descriptions**
   ```
   📈-swing-ma-crossover: Golden Cross/Death Cross signals using EMA crossovers
   📊-swing-macd-rsi: Dual momentum strategy with 73% academic win rate
   📉-swing-rsi-trend: Trend following with pullback timing (81% RSI accuracy)
   💥-swing-bollinger-breakout: Volatility breakout strategy (78% win rate)
   🎯-swing-sr-breakout: Support/resistance levels (70% min confidence)
   ```

## Benefits of Multi-Channel Design

### Organization Benefits
1. **Clear Strategy Separation**: Users can focus on specific strategies
2. **Reduced Noise**: No mixing of different strategy signals
3. **Easy Performance Tracking**: Per-strategy win/loss tracking
4. **Customized Experience**: Users can mute/unmute specific strategies

### Technical Benefits
1. **Distributed Rate Limiting**: Each channel has its own rate limits
2. **Strategy-Specific Configuration**: Custom confidence thresholds per strategy
3. **Improved Scalability**: Easy to add new strategies without affecting existing ones
4. **Better Error Isolation**: Channel failures don't affect other strategies

### User Experience Benefits
1. **Focused Alerts**: Users receive only relevant strategy notifications
2. **Enhanced Formatting**: Strategy-specific colors and formatting
3. **Clear Identification**: Easy to identify which strategy generated the signal
4. **Historical Tracking**: Better ability to analyze strategy performance over time

## Risk Mitigation

### Configuration Risks
- **Mitigation**: Comprehensive validation and fallback mechanisms
- **Fallback**: Single channel fallback for unknown strategies
- **Testing**: Extensive testing with mock and real webhooks

### Performance Risks  
- **Mitigation**: Optimized rate limiting and connection pooling
- **Monitoring**: Real-time performance monitoring per channel
- **Alerts**: Automated alerts for webhook failures or delays

### User Experience Risks
- **Mitigation**: Clear documentation and gradual rollout
- **Support**: Easy migration path from single to multi-channel
- **Feedback**: User feedback collection and iteration

## Success Metrics

### Technical Metrics
- **Notification Delivery Rate**: >99% successful delivery per channel
- **Response Time**: <2 seconds per notification
- **Error Rate**: <1% webhook failures
- **Rate Limit Compliance**: Zero rate limit violations

### User Engagement Metrics
- **Channel Activity**: Engagement per strategy channel
- **User Preferences**: Which strategies are most followed
- **Signal Performance**: Win/loss rates per strategy channel
- **User Feedback**: Satisfaction with organization and clarity

## Conclusion

The multi-channel Discord notification system provides significant improvements in organization, customization, and user experience while maintaining the robust technical foundation of the existing notification system. The phased implementation approach ensures minimal disruption while delivering enhanced value to users through better strategy-specific organization and tracking capabilities.