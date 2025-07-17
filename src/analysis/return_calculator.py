#!/usr/bin/env python3
"""
Return Calculator for Backtesting

Calculates percentage returns from trading signals.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ReturnCalculator:
    """Calculate returns from trading signals."""
    
    def __init__(self):
        """Initialize the return calculator."""
        self.trades = []
        self.position = None  # Track current position
        
    def calculate_returns(self, signals: List[Dict[str, Any]], 
                         final_price: Optional[float] = None,
                         final_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate returns from a list of trading signals.
        
        Args:
            signals: List of signal dictionaries with 'signal_type', 'price', 'timestamp'
            final_price: Final price for open positions (if any)
            final_date: Final date for open positions
            
        Returns:
            Dictionary with return statistics
        """
        # Reset state
        self.trades = []
        self.position = None
        
        # Sort signals by timestamp to ensure chronological order
        sorted_signals = sorted(signals, key=lambda x: x['timestamp'])
        
        # Process each signal
        for signal in sorted_signals:
            self._process_signal(signal)
        
        # Handle any open position at the end
        if self.position and final_price and final_date:
            self._close_position(final_price, final_date, 'MARKET_CLOSE')
        
        # Calculate statistics
        return self._calculate_statistics()
    
    def _process_signal(self, signal: Dict[str, Any]) -> None:
        """Process a single signal."""
        signal_type = signal['signal_type']
        price = signal['price']
        timestamp = signal['timestamp']
        
        if signal_type == 'BUY':
            if not self.position:
                # Enter long position
                self.position = {
                    'type': 'long',
                    'entry_signal': 'BUY',
                    'entry_price': price,
                    'entry_date': timestamp,
                    'symbol': signal.get('symbol', 'UNKNOWN')
                }
                logger.debug(f"Entered long position at {price} on {timestamp}")
            else:
                logger.debug(f"Ignoring BUY signal - already in position")
                
        elif signal_type == 'SELL':
            if self.position and self.position['type'] == 'long':
                # Exit long position
                self._close_position(price, timestamp, 'SELL')
            else:
                logger.debug(f"Ignoring SELL signal - not in long position")
    
    def _close_position(self, exit_price: float, exit_date: Any, exit_signal: str) -> None:
        """Close the current position and record the trade."""
        if not self.position:
            return
        
        # Calculate return
        entry_price = self.position['entry_price']
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Record the trade
        trade = {
            'symbol': self.position['symbol'],
            'entry_signal': self.position['entry_signal'],
            'entry_price': entry_price,
            'entry_date': self.position['entry_date'],
            'exit_signal': exit_signal,
            'exit_price': exit_price,
            'exit_date': exit_date,
            'return_pct': return_pct,
            'profit_loss': exit_price - entry_price
        }
        
        self.trades.append(trade)
        logger.debug(f"Closed position: {return_pct:.2f}% return")
        
        # Clear position
        self.position = None
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics from trades."""
        if not self.trades:
            return {
                'total_return_pct': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'trades': []
            }
        
        # Extract returns
        returns = [trade['return_pct'] for trade in self.trades]
        
        # Calculate compound return
        compound_factor = 1.0
        for r in returns:
            compound_factor *= (1 + r / 100)
        total_return_pct = (compound_factor - 1) * 100
        
        # Calculate statistics
        winning_trades = [t for t in self.trades if t['return_pct'] > 0]
        losing_trades = [t for t in self.trades if t['return_pct'] < 0]
        
        # Find best and worst trades
        best_trade = max(self.trades, key=lambda t: t['return_pct'])
        worst_trade = min(self.trades, key=lambda t: t['return_pct'])
        
        # Average return
        avg_return = sum(returns) / len(returns) if returns else 0
        
        # Win rate
        win_rate = (len(winning_trades) / len(self.trades) * 100) if self.trades else 0
        
        return {
            'total_return_pct': round(total_return_pct, 2),
            'num_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'avg_return_per_trade': round(avg_return, 2),
            'best_trade': {
                'return_pct': round(best_trade['return_pct'], 2),
                'entry_date': best_trade['entry_date'],
                'exit_date': best_trade['exit_date']
            } if best_trade else None,
            'worst_trade': {
                'return_pct': round(worst_trade['return_pct'], 2),
                'entry_date': worst_trade['entry_date'],
                'exit_date': worst_trade['exit_date']
            } if worst_trade else None,
            'trades': [self._format_trade(t) for t in self.trades]
        }
    
    def _format_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Format a trade for output."""
        return {
            'entry_date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
            'exit_date': trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date']),
            'entry_price': round(trade['entry_price'], 2),
            'exit_price': round(trade['exit_price'], 2),
            'return_pct': round(trade['return_pct'], 2)
        }