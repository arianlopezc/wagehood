"""
Summary Formatter for End-of-Day Discord Messages

Creates rich Discord embeds specifically for daily trading summaries with:
- Clear BUY/SELL signal categorization
- Top opportunities highlighting
- Market overview statistics
- Visual formatting with emojis and colors
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SummaryFormatter:
    """
    Formats end-of-day summary data into rich Discord embeds
    
    Creates visually appealing summaries that clearly separate BUY and SELL signals,
    highlight top opportunities, and provide market overview statistics.
    """
    
    @staticmethod
    def format_summary(result: Any) -> str:
        """
        Format summary result for Discord notification.
        
        Args:
            result: SummaryResult object from run_summary.py
            
        Returns:
            JSON string with Discord webhook content
        """
        import json
        
        # Convert stock signals to the format expected by format_eod_summary
        stock_signal_summaries = []
        for summary in result.signal_summaries:
            if summary.signal_count > 0:
                stock_signal_summaries.append({
                    'symbol': summary.symbol,
                    'signal_count': summary.signal_count,
                    'buy_signals': summary.buy_signals,
                    'sell_signals': summary.sell_signals,
                    'avg_confidence': summary.avg_confidence,
                    'latest_signal': summary.latest_signal,
                    'strategies': summary.strategies,
                    'is_crypto': False
                })
        
        # Convert crypto signals to the same format
        crypto_signal_summaries = []
        if hasattr(result, 'crypto_signal_summaries'):
            for summary in result.crypto_signal_summaries:
                if summary.signal_count > 0:
                    crypto_signal_summaries.append({
                        'symbol': summary.symbol,
                        'signal_count': summary.signal_count,
                        'buy_signals': summary.buy_signals,
                        'sell_signals': summary.sell_signals,
                        'avg_confidence': summary.avg_confidence,
                        'latest_signal': summary.latest_signal,
                        'strategies': summary.strategies,
                        'is_crypto': True
                    })
        
        # Call the enhanced format method with both stock and crypto data
        formatted = SummaryFormatter.format_eod_summary_with_crypto(
            stock_signal_summaries=stock_signal_summaries,
            crypto_signal_summaries=crypto_signal_summaries,
            execution_time=result.generated_at,
            stock_symbols_processed=result.symbols_processed,
            stock_total_signals=result.total_signals,
            crypto_symbols_processed=getattr(result, 'crypto_symbols_processed', 0),
            crypto_total_signals=getattr(result, 'crypto_total_signals', 0)
        )
        
        return json.dumps(formatted)
    
    @staticmethod
    def format_eod_summary_with_crypto(
        stock_signal_summaries: List[Dict[str, Any]],
        crypto_signal_summaries: List[Dict[str, Any]],
        execution_time: datetime,
        stock_symbols_processed: int,
        stock_total_signals: int,
        crypto_symbols_processed: int,
        crypto_total_signals: int
    ) -> Dict[str, Any]:
        """
        Format end-of-day summary with both stock and crypto signals for Discord
        
        Args:
            stock_signal_summaries: List of stock signal summary dictionaries
            crypto_signal_summaries: List of crypto signal summary dictionaries
            execution_time: When the summary was generated
            stock_symbols_processed: Total number of stock symbols analyzed
            stock_total_signals: Total number of stock signals generated
            crypto_symbols_processed: Total number of crypto symbols analyzed
            crypto_total_signals: Total number of crypto signals generated
            
        Returns:
            Dictionary with Discord message content and embeds
        """
        
        # Categorize signals separately for stocks and crypto
        stock_buy_signals, stock_sell_signals = SummaryFormatter._categorize_signals(stock_signal_summaries)
        crypto_buy_signals, crypto_sell_signals = SummaryFormatter._categorize_signals(crypto_signal_summaries)
        
        # Combine all signals for top opportunities
        all_signal_summaries = stock_signal_summaries + crypto_signal_summaries
        
        # Create comprehensive embed with both stock and crypto
        embed = SummaryFormatter._create_comprehensive_embed_with_crypto(
            execution_time, 
            stock_symbols_processed, 
            stock_signal_summaries, 
            stock_total_signals,
            stock_buy_signals, 
            stock_sell_signals,
            crypto_symbols_processed,
            crypto_signal_summaries,
            crypto_total_signals,
            crypto_buy_signals,
            crypto_sell_signals
        )
        
        # Create friendly content summary
        content = SummaryFormatter._create_friendly_content_with_crypto(
            stock_buy_signals, stock_sell_signals, stock_symbols_processed,
            crypto_buy_signals, crypto_sell_signals, crypto_symbols_processed,
            execution_time
        )
        
        return {
            "content": content,
            "embeds": [embed],
            "username": "Wagehood Daily Summary"
        }
    
    @staticmethod
    def format_eod_summary(
        signal_summaries: List[Dict[str, Any]],
        execution_time: datetime,
        total_symbols: int,
        total_signals: int
    ) -> Dict[str, Any]:
        """
        Format end-of-day summary for Discord
        
        Args:
            signal_summaries: List of signal summary dictionaries
            execution_time: When the summary was generated
            total_symbols: Total number of symbols analyzed
            total_signals: Total number of signals generated
            
        Returns:
            Dictionary with Discord message content and embeds
        """
        
        # Extract and categorize signals
        buy_signals, sell_signals = SummaryFormatter._categorize_signals(signal_summaries)
        
        # Create a single comprehensive embed
        embed = SummaryFormatter._create_comprehensive_embed(
            execution_time, total_symbols, len(signal_summaries), total_signals,
            buy_signals, sell_signals
        )
        
        # Create friendly content summary
        content = SummaryFormatter._create_friendly_content(
            buy_signals, sell_signals, total_symbols, execution_time
        )
        
        return {
            "content": content,
            "embeds": [embed],  # Single embed
            "username": "Wagehood Daily Summary"
        }
    
    @staticmethod
    def _categorize_signals(signal_summaries: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Categorize signals into BUY and SELL with metadata"""
        buy_signals = []
        sell_signals = []
        
        for summary in signal_summaries:
            symbol = summary.get('symbol', 'UNKNOWN')
            buy_count = summary.get('buy_signals', 0)
            sell_count = summary.get('sell_signals', 0)
            avg_confidence = summary.get('avg_confidence', 0.0)
            latest_signal = summary.get('latest_signal', {})
            
            # Get strategies that generated signals
            strategies = summary.get('strategies', [])
            
            # Since we only have counts, not individual signals, create simplified entries
            if buy_count > 0:
                buy_signals.append({
                    'symbol': symbol,
                    'confidence': avg_confidence,
                    'strategy': latest_signal.get('strategy', 'unknown') if latest_signal else 'unknown',
                    'strategies': strategies,  # List of all strategies that generated signals
                    'price': latest_signal.get('price', 0) if latest_signal else 0,
                    'signal_name': 'BUY',
                    'timestamp': latest_signal.get('timestamp') if latest_signal else None,
                    'count': buy_count
                })
            
            if sell_count > 0:
                sell_signals.append({
                    'symbol': symbol,
                    'confidence': avg_confidence,
                    'strategy': latest_signal.get('strategy', 'unknown') if latest_signal else 'unknown',
                    'strategies': strategies,  # List of all strategies that generated signals
                    'price': latest_signal.get('price', 0) if latest_signal else 0,
                    'signal_name': 'SELL',
                    'timestamp': latest_signal.get('timestamp') if latest_signal else None,
                    'count': sell_count
                })
        
        # Sort by confidence (highest first)
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        sell_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return buy_signals, sell_signals
    
    @staticmethod
    def _create_main_embed(
        execution_time: datetime,
        total_symbols: int,
        symbols_with_signals: int,
        total_signals: int
    ) -> Dict[str, Any]:
        """Create the main summary embed"""
        
        date_str = execution_time.strftime('%Y-%m-%d')
        time_str = execution_time.strftime('%I:%M %p ET')
        
        embed = {
            "title": f"📊 Daily Trading Summary - {date_str}",
            "description": f"⏰ Generated at {time_str}",
            "color": 0x0099ff,  # Blue
            "fields": [
                {
                    "name": "📈 Total Symbols Analyzed",
                    "value": str(total_symbols),
                    "inline": True
                },
                {
                    "name": "🎯 Symbols with Signals",
                    "value": str(symbols_with_signals),
                    "inline": True
                },
                {
                    "name": "📊 Total Signals",
                    "value": str(total_signals),
                    "inline": True
                }
            ],
            "footer": {
                "text": "Next update: Tomorrow at 5:00 PM ET"
            },
            "timestamp": execution_time.isoformat()
        }
        
        return embed
    
    @staticmethod
    def _create_top_opportunities_embed(
        buy_signals: List[Dict],
        sell_signals: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Create top opportunities embed"""
        
        # Combine and get top 5 signals by confidence
        all_signals = buy_signals + sell_signals
        if not all_signals:
            return None
        
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        top_signals = all_signals[:5]
        
        fields = []
        for i, signal in enumerate(top_signals, 1):
            signal_type = "🟢 BUY" if signal in buy_signals else "🔴 SELL"
            strategy_display = signal['strategy'].replace('_', ' ').title()
            confidence_pct = f"{signal['confidence']:.0%}"
            
            # Include count if available
            count_info = f" ({signal.get('count', 1)} signals)" if signal.get('count', 1) > 1 else ""
            fields.append({
                "name": f"{i}. {signal['symbol']}",
                "value": f"{signal_type} • {strategy_display} • {confidence_pct}{count_info}",
                "inline": False
            })
        
        embed = {
            "title": "🎯 Top Opportunities",
            "description": "Highest confidence signals across all strategies",
            "color": 0xffd700,  # Gold
            "fields": fields
        }
        
        return embed
    
    @staticmethod
    def _create_signals_embed(
        signals: List[Dict],
        title: str,
        color: int,
        emoji: str
    ) -> Dict[str, Any]:
        """Create embed for signal type (BUY or SELL)"""
        
        if not signals:
            return {
                "title": f"{emoji} {title}",
                "description": "No signals detected",
                "color": color
            }
        
        # Group signals by confidence ranges
        strong_signals = [s for s in signals if s['confidence'] >= 0.8]
        moderate_signals = [s for s in signals if 0.6 <= s['confidence'] < 0.8]
        weak_signals = [s for s in signals if s['confidence'] < 0.6]
        
        fields = []
        
        # Strong signals
        if strong_signals:
            symbols_text = ", ".join([s['symbol'] for s in strong_signals[:10]])
            if len(strong_signals) > 10:
                symbols_text += f" +{len(strong_signals) - 10} more"
            
            fields.append({
                "name": f"🔥 Strong Signals ({len(strong_signals)})",
                "value": symbols_text,
                "inline": False
            })
        
        # Moderate signals
        if moderate_signals:
            symbols_text = ", ".join([s['symbol'] for s in moderate_signals[:10]])
            if len(moderate_signals) > 10:
                symbols_text += f" +{len(moderate_signals) - 10} more"
            
            fields.append({
                "name": f"⚡ Moderate Signals ({len(moderate_signals)})",
                "value": symbols_text,
                "inline": False
            })
        
        # Weak signals (only show count if there are many)
        if weak_signals and len(weak_signals) > 5:
            symbols_text = ", ".join([s['symbol'] for s in weak_signals[:5]])
            if len(weak_signals) > 5:
                symbols_text += f" +{len(weak_signals) - 5} more"
            
            fields.append({
                "name": f"💫 Weak Signals ({len(weak_signals)})",
                "value": symbols_text,
                "inline": False
            })
        
        # Add strategy breakdown
        strategy_counts = {}
        for signal in signals:
            strategy = signal['strategy'].replace('_', ' ').title()
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        strategy_text = " • ".join([f"{strategy}: {count}" for strategy, count in strategy_counts.items()])
        
        fields.append({
            "name": "📋 Strategy Breakdown",
            "value": strategy_text,
            "inline": False
        })
        
        embed = {
            "title": f"{emoji} {title} ({len(signals)})",
            "color": color,
            "fields": fields
        }
        
        return embed
    
    @staticmethod
    def _create_comprehensive_embed(
        execution_time: datetime,
        total_symbols: int,
        symbols_with_signals: int,
        total_signals: int,
        buy_signals: List[Dict],
        sell_signals: List[Dict]
    ) -> Dict[str, Any]:
        """Create a single comprehensive embed with all information"""
        
        date_str = execution_time.strftime('%A, %B %d, %Y')
        time_str = execution_time.strftime('%I:%M %p ET')
        
        # Build description based on signals
        if not buy_signals and not sell_signals:
            description = f"I analyzed {total_symbols} symbols today, but didn't find any strong trading signals. The market seems quiet! 😌"
        else:
            description = f"Here's what I found after analyzing {total_symbols} symbols today:"
        
        embed = {
            "title": f"📊 Your Daily Trading Summary",
            "description": description,
            "color": 0x5865F2,  # Discord blurple for a friendly look
            "fields": [],
            "footer": {
                "text": f"⚠️ Not financial advice • Analysis completed at {time_str} • Next update tomorrow at 5PM ET"
            },
            "timestamp": execution_time.isoformat()
        }
        
        # Add market overview
        if total_signals > 0:
            embed["fields"].append({
                "name": "📈 Market Overview",
                "value": f"• **{symbols_with_signals}** symbols showing activity\n• **{total_signals}** total signals detected\n• **{len(buy_signals)}** bullish / **{len(sell_signals)}** bearish",
                "inline": False
            })
        
        # Add top opportunities if any
        all_signals = buy_signals + sell_signals
        if all_signals:
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            top_signals = all_signals[:5]
            
            top_text = []
            for signal in top_signals:
                emoji = "🟢" if signal in buy_signals else "🔴"
                symbol = signal['symbol']
                confidence = f"{signal['confidence']:.0%}"
                
                # Format strategies
                strategies = signal.get('strategies', [])
                if strategies:
                    strategy_names = [s.replace('_', ' ').title() for s in strategies]
                    strategy_text = " & ".join(strategy_names)
                else:
                    # Fallback to single strategy field
                    strategy_text = signal.get('strategy', 'unknown').replace('_', ' ').title()
                
                count = signal.get('count', 1)
                count_text = f" ({count} signals)" if count > 1 else ""
                
                signal_type = "BUY" if signal in buy_signals else "SELL"
                top_text.append(f"{emoji} **{symbol} - {signal_type}** - {strategy_text} - {confidence} confidence{count_text}")
            
            embed["fields"].append({
                "name": "🎯 Top Opportunities",
                "value": "\n".join(top_text),
                "inline": False
            })
        
        # Add signal breakdown if we have many signals
        if len(all_signals) > 5:
            # Group by confidence
            strong = len([s for s in all_signals if s['confidence'] >= 0.8])
            moderate = len([s for s in all_signals if 0.6 <= s['confidence'] < 0.8])
            
            breakdown_text = []
            if strong > 0:
                breakdown_text.append(f"🔥 **{strong}** high confidence (80%+)")
            if moderate > 0:
                breakdown_text.append(f"⚡ **{moderate}** moderate confidence (60-79%)")
            
            if breakdown_text:
                embed["fields"].append({
                    "name": "📊 Signal Strength Distribution",
                    "value": "\n".join(breakdown_text),
                    "inline": False
                })
        
        # Add disclaimer field
        embed["fields"].append({
            "name": "⚠️ Disclaimer",
            "value": "This summary is for informational purposes only and should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.",
            "inline": False
        })
        
        return embed
    
    @staticmethod
    def _create_friendly_content(
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        total_symbols: int,
        execution_time: datetime
    ) -> str:
        """Create a friendly content summary for the message"""
        
        greetings = [
            "Good evening! 🌆",
            "Hey there! 👋",
            "Hello! 🌟",
            "Hi! ✨"
        ]
        
        # Pick a greeting based on the day
        greeting_index = execution_time.day % len(greetings)
        greeting = greetings[greeting_index]
        
        if not buy_signals and not sell_signals:
            return f"{greeting} Today's market scan is complete. No significant signals detected - sometimes no trade is the best trade! 💤"
        
        total_signals = len(buy_signals) + len(sell_signals)
        
        if total_signals == 1:
            return f"{greeting} I found **1 trading signal** for you today! Check it out below 👇"
        else:
            return f"{greeting} I found **{total_signals} trading signals** for you today! Let's see what the market is telling us 👇"
    
    @staticmethod
    def _create_content_summary(
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        total_symbols: int,
        execution_time: datetime
    ) -> str:
        """Create brief content summary for the message"""
        # This method is kept for backward compatibility but redirects to the friendly version
        return SummaryFormatter._create_friendly_content(
            buy_signals, sell_signals, total_symbols, execution_time
        )
    
    @staticmethod
    def format_error_summary(
        execution_time: datetime,
        errors: List[str],
        symbols_attempted: int = 0
    ) -> Dict[str, Any]:
        """Format error summary for Discord when EOD job fails"""
        
        date_str = execution_time.strftime('%Y-%m-%d')
        time_str = execution_time.strftime('%I:%M %p ET')
        
        embed = {
            "title": "🚨 EOD Summary Failed",
            "description": f"Failed to generate daily summary for {date_str}",
            "color": 0xff0000,  # Red
            "fields": [
                {
                    "name": "⏰ Attempted At",
                    "value": time_str,
                    "inline": True
                },
                {
                    "name": "📊 Symbols Attempted",
                    "value": str(symbols_attempted),
                    "inline": True
                },
                {
                    "name": "❌ Error Count",
                    "value": str(len(errors)),
                    "inline": True
                }
            ],
            "timestamp": execution_time.isoformat(),
            "footer": {
                "text": "Will retry tomorrow at 5:00 PM ET"
            }
        }
        
        # Add error details (truncated)
        if errors:
            error_text = "\n".join(errors[:3])  # Show first 3 errors
            if len(errors) > 3:
                error_text += f"\n... and {len(errors) - 3} more errors"
            
            embed["fields"].append({
                "name": "🔍 Error Details",
                "value": f"```{error_text[:1000]}```",  # Discord field limit
                "inline": False
            })
        
        return {
            "content": f"🚨 **EOD Summary Failed** - {date_str}",
            "embeds": [embed],
            "username": "Wagehood EOD Summary"
        }
    
    @staticmethod
    def create_test_summary() -> Dict[str, Any]:
        """Create a test summary for webhook verification"""
        
        test_time = datetime.now()
        
        embed = {
            "title": "🧪 EOD Summary Test",
            "description": "Test message to verify webhook connectivity",
            "color": 0x0099ff,
            "fields": [
                {
                    "name": "📊 Test Status",
                    "value": "✅ Webhook Active",
                    "inline": True
                },
                {
                    "name": "⏰ Test Time",
                    "value": test_time.strftime('%I:%M %p ET'),
                    "inline": True
                }
            ],
            "timestamp": test_time.isoformat(),
            "footer": {
                "text": "Wagehood EOD Summary Test"
            }
        }
        
        return {
            "content": "🧪 **EOD Summary Webhook Test**",
            "embeds": [embed],
            "username": "Wagehood EOD Summary"
        }
    
    @staticmethod
    def _create_comprehensive_embed_with_crypto(
        execution_time: datetime,
        stock_symbols_processed: int,
        stock_signal_summaries: List[Dict[str, Any]],
        stock_total_signals: int,
        stock_buy_signals: List[Dict],
        stock_sell_signals: List[Dict],
        crypto_symbols_processed: int,
        crypto_signal_summaries: List[Dict[str, Any]],
        crypto_total_signals: int,
        crypto_buy_signals: List[Dict],
        crypto_sell_signals: List[Dict]
    ) -> Dict[str, Any]:
        """Create a single comprehensive embed with both stock and crypto information"""
        
        date_str = execution_time.strftime('%A, %B %d, %Y')
        time_str = execution_time.strftime('%I:%M %p ET')
        
        # Combine totals
        total_symbols = stock_symbols_processed + crypto_symbols_processed
        total_signals = stock_total_signals + crypto_total_signals
        
        # Build description based on signals
        if total_signals == 0:
            description = f"I analyzed {total_symbols} symbols today ({stock_symbols_processed} stocks, {crypto_symbols_processed} crypto), but didn't find any strong trading signals. The market seems quiet! 😌"
        else:
            description = f"Here's what I found after analyzing {total_symbols} symbols today ({stock_symbols_processed} stocks, {crypto_symbols_processed} crypto):"
        
        embed = {
            "title": f"📊 Your Daily Trading Summary",
            "description": description,
            "color": 0x5865F2,  # Discord blurple for a friendly look
            "fields": [],
            "footer": {
                "text": f"⚠️ Not financial advice • Analysis completed at {time_str} • Next update tomorrow at 5PM ET"
            },
            "timestamp": execution_time.isoformat()
        }
        
        # Add market overview
        if total_signals > 0:
            # Stock overview
            if stock_total_signals > 0:
                stock_symbols_with_signals = len(stock_signal_summaries)
                embed["fields"].append({
                    "name": "📈 Stock Market Overview",
                    "value": f"• **{stock_symbols_with_signals}** symbols showing activity\n• **{stock_total_signals}** total signals detected\n• **{len(stock_buy_signals)}** bullish / **{len(stock_sell_signals)}** bearish",
                    "inline": False
                })
            
            # Crypto overview
            if crypto_total_signals > 0:
                crypto_symbols_with_signals = len(crypto_signal_summaries)
                embed["fields"].append({
                    "name": "🪙 Crypto Market Overview",
                    "value": f"• **{crypto_symbols_with_signals}** symbols showing activity\n• **{crypto_total_signals}** total signals detected\n• **{len(crypto_buy_signals)}** bullish / **{len(crypto_sell_signals)}** bearish",
                    "inline": False
                })
        
        # Add top opportunities combining both stocks and crypto
        all_signals = []
        for signal in stock_buy_signals + stock_sell_signals:
            signal['market_type'] = '📈'  # Stock emoji
            all_signals.append(signal)
        for signal in crypto_buy_signals + crypto_sell_signals:
            signal['market_type'] = '🪙'  # Crypto emoji
            all_signals.append(signal)
        
        if all_signals:
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            top_signals = all_signals[:5]
            
            top_text = []
            for signal in top_signals:
                emoji = "🟢" if signal['signal_name'] == 'BUY' else "🔴"
                market_emoji = signal['market_type']
                symbol = signal['symbol']
                confidence = f"{signal['confidence']:.0%}"
                
                # Format strategies
                strategies = signal.get('strategies', [])
                if strategies:
                    strategy_names = [s.replace('_', ' ').title() for s in strategies]
                    strategy_text = " & ".join(strategy_names)
                else:
                    # Fallback to single strategy field
                    strategy_text = signal.get('strategy', 'unknown').replace('_', ' ').title()
                
                count = signal.get('count', 1)
                count_text = f" ({count} signals)" if count > 1 else ""
                
                signal_type = signal['signal_name']
                top_text.append(f"{market_emoji} {emoji} **{symbol} - {signal_type}** - {strategy_text} - {confidence} confidence{count_text}")
            
            embed["fields"].append({
                "name": "🎯 Top Opportunities",
                "value": "\n".join(top_text),
                "inline": False
            })
        
        # Add signal breakdown if we have many signals
        if len(all_signals) > 5:
            # Group by confidence
            strong = len([s for s in all_signals if s['confidence'] >= 0.8])
            moderate = len([s for s in all_signals if 0.6 <= s['confidence'] < 0.8])
            
            breakdown_text = []
            if strong > 0:
                breakdown_text.append(f"🔥 **{strong}** high confidence (80%+)")
            if moderate > 0:
                breakdown_text.append(f"⚡ **{moderate}** moderate confidence (60-79%)")
            
            if breakdown_text:
                embed["fields"].append({
                    "name": "📊 Signal Strength Distribution",
                    "value": "\n".join(breakdown_text),
                    "inline": False
                })
        
        # Add disclaimer field
        embed["fields"].append({
            "name": "⚠️ Disclaimer",
            "value": "This summary is for informational purposes only and should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.",
            "inline": False
        })
        
        return embed
    
    @staticmethod
    def _create_friendly_content_with_crypto(
        stock_buy_signals: List[Dict],
        stock_sell_signals: List[Dict],
        stock_symbols_processed: int,
        crypto_buy_signals: List[Dict],
        crypto_sell_signals: List[Dict],
        crypto_symbols_processed: int,
        execution_time: datetime
    ) -> str:
        """Create a friendly content summary for the message including crypto"""
        
        greetings = [
            "Good evening! 🌆",
            "Hey there! 👋",
            "Hello! 🌟",
            "Hi! ✨"
        ]
        
        # Pick a greeting based on the day
        greeting_index = execution_time.day % len(greetings)
        greeting = greetings[greeting_index]
        
        total_signals = len(stock_buy_signals) + len(stock_sell_signals) + len(crypto_buy_signals) + len(crypto_sell_signals)
        
        if total_signals == 0:
            return f"{greeting} Today's market scan is complete. No significant signals detected across stocks or crypto - sometimes no trade is the best trade! 💤"
        
        # Build signal summary text
        signal_parts = []
        
        stock_signals = len(stock_buy_signals) + len(stock_sell_signals)
        if stock_signals > 0:
            signal_parts.append(f"{stock_signals} stock")
        
        crypto_signals = len(crypto_buy_signals) + len(crypto_sell_signals)
        if crypto_signals > 0:
            signal_parts.append(f"{crypto_signals} crypto")
        
        signal_text = " and ".join(signal_parts)
        
        if total_signals == 1:
            return f"{greeting} I found **1 trading signal** for you today! Check it out below 👇"
        else:
            return f"{greeting} I found **{total_signals} trading signals** for you today ({signal_text})! Let's see what the market is telling us 👇"