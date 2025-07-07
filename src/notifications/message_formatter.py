"""
Discord Message Formatter

Formats trading signals and system messages into rich Discord embeds
with proper styling, colors, and structure.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MessageFormatter:
    """
    Formats trading data into Discord-friendly embed messages.
    """
    
    # Discord color constants
    COLORS = {
        'BUY': 65280,      # Green
        'SELL': 16711680,  # Red  
        'HOLD': 3447003,   # Blue
        'WARNING': 16753920,  # Orange
        'SUCCESS': 65280,  # Green
        'INFO': 3447003    # Blue
    }
    
    # Signal emojis
    SIGNAL_EMOJIS = {
        'BUY': '📈',
        'SELL': '📉', 
        'HOLD': '⚖️',
        'UNKNOWN': '❓'
    }
    
    def __init__(self):
        """Initialize message formatter."""
        logger.debug("Message formatter initialized")
    
    def format_signal_embed(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format trading signal into Discord embed.
        
        Args:
            signal_data: Signal information dictionary with keys:
                - symbol: Trading symbol (e.g., 'AAPL')
                - company_name: Company name (optional)
                - signal: Signal type ('BUY', 'SELL', 'HOLD')
                - price: Current price
                - price_change: Price change amount
                - price_change_pct: Price change percentage
                - strategy: Strategy name that generated signal
                - confidence: Signal confidence (0-1)
                - details: Additional signal details
                - timestamp: When signal occurred
                
        Returns:
            Discord embed dictionary
        """
        try:
            # Extract data with defaults
            symbol = signal_data.get('symbol', 'UNKNOWN')
            company_name = signal_data.get('company_name', '')
            signal = signal_data.get('signal', 'UNKNOWN').upper()
            price = signal_data.get('price', 0.0)
            price_change = signal_data.get('price_change', 0.0)
            price_change_pct = signal_data.get('price_change_pct', 0.0)
            strategy = signal_data.get('strategy', 'Unknown Strategy')
            confidence = signal_data.get('confidence', 0.0)
            details = signal_data.get('details', {})
            timestamp = signal_data.get('timestamp')
            
            # Format timestamp
            if timestamp:
                if isinstance(timestamp, str):
                    # Convert string timestamp to datetime if needed
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.utcnow()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()
            
            # Build title with emoji
            emoji = self.SIGNAL_EMOJIS.get(signal, '❓')
            title_symbol = f"{symbol}"
            if company_name:
                title_symbol += f" - {company_name}"
            
            # Format price display
            price_change_sign = '+' if price_change >= 0 else ''
            price_display = f"${price:.2f} ({price_change_sign}{price_change_pct:.1f}%)"
            
            # Format confidence percentage
            confidence_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
            
            # Build signal details
            details_text = self._format_signal_details(details, signal_data)
            
            # Create embed
            embed = {
                "title": "🎯 SWING TRADING SIGNAL",
                "color": self.COLORS.get(signal, self.COLORS['INFO']),
                "fields": [
                    {
                        "name": f"{emoji} {title_symbol}",
                        "value": f"**{signal} SIGNAL**",
                        "inline": False
                    },
                    {
                        "name": "💰 Price",
                        "value": price_display,
                        "inline": True
                    },
                    {
                        "name": "📊 Strategy", 
                        "value": strategy,
                        "inline": True
                    },
                    {
                        "name": "🎯 Confidence",
                        "value": f"{confidence_pct}%",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Paper Trading • Not Financial Advice • Wagehood"
                },
                "timestamp": timestamp.isoformat()
            }
            
            # Add details if available
            if details_text:
                embed["fields"].append({
                    "name": "📋 Signal Details",
                    "value": details_text,
                    "inline": False
                })
            
            return embed
            
        except Exception as e:
            logger.error(f"Error formatting signal embed: {e}")
            return self._create_error_embed(f"Error formatting signal: {str(e)}")
    
    def _format_signal_details(self, details: Dict[str, Any], signal_data: Dict[str, Any]) -> str:
        """
        Format signal details into readable text.
        
        Args:
            details: Signal details dictionary
            signal_data: Full signal data for context
            
        Returns:
            Formatted details string
        """
        try:
            detail_lines = []
            
            # Add specific indicator details if available
            if 'macd' in details:
                macd_value = details['macd']
                detail_lines.append(f"• MACD: {macd_value}")
            
            if 'rsi' in details:
                rsi_value = details['rsi']
                rsi_condition = self._get_rsi_condition(rsi_value)
                detail_lines.append(f"• RSI: {rsi_value:.1f} ({rsi_condition})")
            
            if 'volume_change' in details:
                vol_change = details['volume_change']
                vol_sign = '+' if vol_change >= 0 else ''
                detail_lines.append(f"• Volume: {vol_sign}{vol_change:.1f}% vs average")
            
            if 'bollinger_position' in details:
                bb_pos = details['bollinger_position']
                detail_lines.append(f"• Bollinger: {bb_pos}")
            
            if 'ma_trend' in details:
                trend = details['ma_trend']
                detail_lines.append(f"• Trend: {trend}")
            
            # Add any custom details
            if 'custom_details' in details:
                custom = details['custom_details']
                if isinstance(custom, list):
                    detail_lines.extend([f"• {detail}" for detail in custom])
                elif isinstance(custom, str):
                    detail_lines.append(f"• {custom}")
            
            # If no specific details, add generic info
            if not detail_lines:
                signal = signal_data.get('signal', '').upper()
                if signal == 'BUY':
                    detail_lines.append("• Bullish momentum detected")
                elif signal == 'SELL':
                    detail_lines.append("• Bearish momentum detected")
                elif signal == 'HOLD':
                    detail_lines.append("• Sideways consolidation")
            
            return '\n'.join(detail_lines)
            
        except Exception as e:
            logger.error(f"Error formatting signal details: {e}")
            return "• Signal generated by strategy analysis"
    
    def _get_rsi_condition(self, rsi_value: float) -> str:
        """Get RSI condition description."""
        if rsi_value >= 70:
            return "overbought"
        elif rsi_value <= 30:
            return "oversold"
        elif rsi_value >= 50:
            return "bullish"
        else:
            return "bearish"
    
    def format_system_status_embed(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format system status into Discord embed.
        
        Args:
            status_data: System status information
            
        Returns:
            Discord embed dictionary
        """
        try:
            status = status_data.get('status', 'unknown')
            service_name = status_data.get('service', 'Wagehood')
            
            # Determine color based on status
            if status == 'running':
                color = self.COLORS['SUCCESS']
                emoji = '🚀'
            elif status == 'stopped':
                color = self.COLORS['WARNING'] 
                emoji = '⏹️'
            elif status == 'error':
                color = self.COLORS['SELL']
                emoji = '❌'
            else:
                color = self.COLORS['INFO']
                emoji = '🔍'
            
            embed = {
                "title": f"{emoji} {service_name} System Status",
                "color": color,
                "fields": [
                    {
                        "name": "Status",
                        "value": status.title(),
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Wagehood System Monitor"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add additional fields if available
            if 'symbol_count' in status_data:
                embed["fields"].append({
                    "name": "Symbols Tracking",
                    "value": str(status_data['symbol_count']),
                    "inline": True
                })
            
            if 'data_provider' in status_data:
                embed["fields"].append({
                    "name": "Data Source",
                    "value": status_data['data_provider'],
                    "inline": True
                })
            
            if 'message' in status_data:
                embed["fields"].append({
                    "name": "Details",
                    "value": status_data['message'],
                    "inline": False
                })
            
            return embed
            
        except Exception as e:
            logger.error(f"Error formatting status embed: {e}")
            return self._create_error_embed(f"Error formatting status: {str(e)}")
    
    def _create_error_embed(self, error_message: str) -> Dict[str, Any]:
        """
        Create error embed for fallback cases.
        
        Args:
            error_message: Error description
            
        Returns:
            Error embed dictionary
        """
        return {
            "title": "❌ Notification Error",
            "description": error_message,
            "color": self.COLORS['SELL'],
            "footer": {
                "text": "Wagehood System Error"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def create_test_embed(self) -> Dict[str, Any]:
        """
        Create test embed for webhook verification.
        
        Returns:
            Test embed dictionary
        """
        return {
            "title": "🧪 Wagehood Discord Test",
            "description": "Testing Discord webhook integration",
            "color": self.COLORS['SUCCESS'],
            "fields": [
                {
                    "name": "✅ Status",
                    "value": "Connection successful",
                    "inline": True
                },
                {
                    "name": "🕐 Time",
                    "value": datetime.now().strftime("%H:%M:%S EST"),
                    "inline": True
                },
                {
                    "name": "📡 Source",
                    "value": "Wagehood Trading System",
                    "inline": False
                }
            ],
            "footer": {
                "text": "Discord Integration Test"
            },
            "timestamp": datetime.utcnow().isoformat()
        }