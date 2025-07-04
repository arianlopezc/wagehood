"""
Order execution simulation with realistic market conditions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
import math

from ..core.models import Signal, SignalType, OHLCV, Trade
from .costs import TransactionCostModel, CommissionFreeModel, RealisticCostModel


@dataclass
class OrderResult:
    """Result of order execution"""
    executed: bool
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    reason: str = ""


@dataclass
class Position:
    """Current position in a symbol"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


class OrderExecutor(ABC):
    """Abstract base class for order execution"""
    
    @abstractmethod
    def execute_order(self, signal: Signal, current_bar: OHLCV, 
                     position: Position, available_capital: float) -> OrderResult:
        """
        Execute an order based on signal
        
        Args:
            signal: Trading signal
            current_bar: Current market data
            position: Current position
            available_capital: Available capital for trading
            
        Returns:
            OrderResult with execution details
        """
        pass


class MarketOrderExecutor(OrderExecutor):
    """Market order executor with realistic slippage"""
    
    def __init__(self, 
                 cost_model: TransactionCostModel = None,
                 max_position_size: float = 0.1,
                 min_order_size: float = 1.0,
                 risk_per_trade: float = 0.02):
        """
        Initialize market order executor
        
        Args:
            cost_model: Transaction cost model (default: CommissionFreeModel for commission-free trading)
            max_position_size: Maximum position size as fraction of capital
            min_order_size: Minimum order size
            risk_per_trade: Risk per trade as fraction of capital
        """
        self.cost_model = cost_model or CommissionFreeModel()
        self.max_position_size = max_position_size
        self.min_order_size = min_order_size
        self.risk_per_trade = risk_per_trade
    
    def execute_order(self, signal: Signal, current_bar: OHLCV,
                     position: Position, available_capital: float) -> OrderResult:
        """Execute market order with realistic conditions"""
        
        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(signal, current_bar)
        
        # Calculate order quantity
        order_quantity = self._calculate_order_quantity(
            signal, execution_price, position, available_capital
        )
        
        if abs(order_quantity) < self.min_order_size:
            return OrderResult(
                executed=False,
                quantity=0.0,
                price=execution_price,
                commission=0.0,
                slippage=0.0,
                timestamp=signal.timestamp,
                reason="Order size too small"
            )
        
        # Calculate market conditions for cost model
        market_conditions = self._get_market_conditions(current_bar)
        
        # Calculate transaction costs
        side = 'buy' if order_quantity > 0 else 'sell'
        costs = self.cost_model.calculate_cost(
            order_quantity, execution_price, side, signal.symbol, market_conditions
        )
        
        # Check if we have enough capital
        total_cost = abs(order_quantity) * execution_price + costs.total
        if total_cost > available_capital and order_quantity > 0:
            return OrderResult(
                executed=False,
                quantity=0.0,
                price=execution_price,
                commission=0.0,
                slippage=0.0,
                timestamp=signal.timestamp,
                reason="Insufficient capital"
            )
        
        return OrderResult(
            executed=True,
            quantity=order_quantity,
            price=execution_price,
            commission=costs.commission,
            slippage=costs.slippage,
            timestamp=signal.timestamp,
            reason="Executed successfully"
        )
    
    def _calculate_execution_price(self, signal: Signal, current_bar: OHLCV) -> float:
        """Calculate realistic execution price with slippage"""
        base_price = signal.price
        
        # Add realistic slippage based on signal type and market conditions
        if signal.signal_type == SignalType.BUY:
            # Buy orders typically execute at ask price (higher)
            slippage_factor = 1 + self._calculate_slippage_factor(current_bar, 'buy')
        else:
            # Sell orders typically execute at bid price (lower)
            slippage_factor = 1 - self._calculate_slippage_factor(current_bar, 'sell')
        
        return base_price * slippage_factor
    
    def _calculate_slippage_factor(self, current_bar: OHLCV, side: str) -> float:
        """Calculate slippage factor based on market conditions"""
        # Base slippage (0.01% to 0.1%)
        base_slippage = 0.0001 + random.uniform(0, 0.0009)
        
        # Increase slippage based on volatility
        price_range = (current_bar.high - current_bar.low) / current_bar.close
        volatility_factor = 1 + min(price_range * 10, 2.0)  # Cap at 3x base slippage
        
        # Volume factor (lower volume = higher slippage)
        volume_factor = 1.0
        if current_bar.volume > 0:
            # Normalize volume (assuming average volume is around 1M)
            normalized_volume = current_bar.volume / 1000000
            volume_factor = 1.0 + (1.0 / max(normalized_volume, 0.1))
        
        return base_slippage * volatility_factor * volume_factor
    
    def _calculate_order_quantity(self, signal: Signal, price: float,
                                 position: Position, available_capital: float) -> float:
        """Calculate order quantity based on position sizing rules"""
        
        if signal.signal_type == SignalType.BUY:
            return self._calculate_buy_quantity(signal, price, position, available_capital)
        elif signal.signal_type == SignalType.SELL:
            return self._calculate_sell_quantity(signal, price, position, available_capital)
        elif signal.signal_type == SignalType.CLOSE_LONG:
            return -position.quantity if position.is_long else 0.0
        elif signal.signal_type == SignalType.CLOSE_SHORT:
            return -position.quantity if position.is_short else 0.0
        else:
            return 0.0
    
    def _calculate_buy_quantity(self, signal: Signal, price: float,
                               position: Position, available_capital: float) -> float:
        """Calculate buy order quantity"""
        # Risk-based position sizing
        risk_amount = available_capital * self.risk_per_trade
        
        # Simple position sizing: risk amount divided by price
        base_quantity = risk_amount / price
        
        # Adjust for confidence
        confidence_factor = signal.confidence
        adjusted_quantity = base_quantity * confidence_factor
        
        # Apply position size limits
        max_position_value = available_capital * self.max_position_size
        max_quantity = max_position_value / price
        
        return min(adjusted_quantity, max_quantity)
    
    def _calculate_sell_quantity(self, signal: Signal, price: float,
                                position: Position, available_capital: float) -> float:
        """Calculate sell order quantity"""
        if position.is_long:
            # Sell part or all of long position
            confidence_factor = signal.confidence
            sell_fraction = confidence_factor * 0.5  # Sell up to 50% based on confidence
            return -position.quantity * sell_fraction
        else:
            # Open short position (if allowed)
            return -self._calculate_buy_quantity(signal, price, position, available_capital)
    
    def _get_market_conditions(self, current_bar: OHLCV) -> Dict[str, Any]:
        """Get market conditions for cost calculation"""
        return {
            'volatility_factor': self._calculate_volatility_factor(current_bar),
            'volume_factor': self._calculate_volume_factor(current_bar),
            'avg_volume': current_bar.volume  # Simplified
        }
    
    def _calculate_volatility_factor(self, current_bar: OHLCV) -> float:
        """Calculate volatility factor for the current bar"""
        if current_bar.close <= 0:
            return 1.0
        
        price_range = (current_bar.high - current_bar.low) / current_bar.close
        return 1.0 + min(price_range * 5, 3.0)  # Cap at 4x base cost
    
    def _calculate_volume_factor(self, current_bar: OHLCV) -> float:
        """Calculate volume factor for the current bar"""
        if current_bar.volume <= 0:
            return 2.0  # High cost for zero volume
        
        # Normalize volume (assuming typical volume is 1M)
        normalized_volume = current_bar.volume / 1000000
        return 1.0 + (1.0 / max(normalized_volume, 0.1))


class PortfolioManager:
    """Manages portfolio positions and executions"""
    
    def __init__(self, 
                 initial_capital: float,
                 executor: OrderExecutor = None,
                 max_positions: int = 5):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
            executor: Order executor
            max_positions: Maximum number of positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.executor = executor or MarketOrderExecutor()
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.trade_counter = 0
    
    def process_signal(self, signal: Signal, current_bar: OHLCV) -> Optional[Trade]:
        """Process a trading signal and execute if appropriate"""
        
        # Get current position
        position = self.positions.get(signal.symbol, Position(signal.symbol, 0.0, 0.0))
        
        # Check position limits
        if not self._can_open_position(signal, position):
            return None
        
        # Execute order
        result = self.executor.execute_order(
            signal, current_bar, position, self.current_capital
        )
        
        if not result.executed:
            return None
        
        # Update position and capital
        trade = self._update_position(signal, position, result)
        
        # Update equity curve
        self._update_equity_curve(current_bar)
        
        return trade
    
    def _can_open_position(self, signal: Signal, position: Position) -> bool:
        """Check if we can open a new position"""
        
        # Can always close existing positions
        if signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            return True
        
        # Can always adjust existing positions
        if not position.is_flat:
            return True
        
        # Check position limits for new positions
        active_positions = len([p for p in self.positions.values() if not p.is_flat])
        return active_positions < self.max_positions
    
    def _update_position(self, signal: Signal, position: Position, result: OrderResult) -> Trade:
        """Update position based on execution result"""
        
        old_quantity = position.quantity
        old_avg_price = position.avg_price
        
        # Calculate new position
        new_quantity = old_quantity + result.quantity
        
        if new_quantity == 0:
            # Position closed
            new_avg_price = 0.0
            realized_pnl = self._calculate_realized_pnl(
                old_quantity, old_avg_price, result.quantity, result.price
            )
            position.realized_pnl += realized_pnl
        else:
            # Position opened or adjusted
            if old_quantity == 0:
                # New position
                new_avg_price = result.price
                realized_pnl = 0.0
            else:
                # Adjust existing position
                if (old_quantity > 0 and result.quantity > 0) or (old_quantity < 0 and result.quantity < 0):
                    # Same direction - update average price
                    total_cost = (old_quantity * old_avg_price) + (result.quantity * result.price)
                    new_avg_price = total_cost / new_quantity
                    realized_pnl = 0.0
                else:
                    # Opposite direction - partial or full close
                    close_quantity = min(abs(old_quantity), abs(result.quantity))
                    if old_quantity > 0:
                        close_quantity = -close_quantity
                    
                    realized_pnl = self._calculate_realized_pnl(
                        old_quantity, old_avg_price, close_quantity, result.price
                    )
                    
                    if abs(new_quantity) < abs(old_quantity):
                        # Partial close
                        new_avg_price = old_avg_price
                    else:
                        # Full close and reverse
                        new_avg_price = result.price
                    
                    position.realized_pnl += realized_pnl
        
        # Update position
        position.quantity = new_quantity
        position.avg_price = new_avg_price
        self.positions[signal.symbol] = position
        
        # Update capital
        self.current_capital -= (abs(result.quantity) * result.price + result.commission)
        
        # Create trade record
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"trade_{self.trade_counter}",
            entry_time=signal.timestamp,
            exit_time=signal.timestamp if new_quantity == 0 else None,
            symbol=signal.symbol,
            quantity=result.quantity,
            entry_price=result.price,
            exit_price=result.price if new_quantity == 0 else None,
            pnl=realized_pnl if new_quantity == 0 else None,
            commission=result.commission,
            strategy_name=signal.strategy_name,
            signal_metadata=signal.metadata
        )
        
        self.trades.append(trade)
        return trade
    
    def _calculate_realized_pnl(self, old_quantity: float, old_price: float,
                               close_quantity: float, close_price: float) -> float:
        """Calculate realized P&L for position close"""
        if old_quantity == 0 or close_quantity == 0:
            return 0.0
        
        # Ensure close_quantity has opposite sign to old_quantity
        if (old_quantity > 0 and close_quantity > 0) or (old_quantity < 0 and close_quantity < 0):
            close_quantity = -close_quantity
        
        # Calculate P&L
        if old_quantity > 0:
            # Closing long position
            pnl = abs(close_quantity) * (close_price - old_price)
        else:
            # Closing short position
            pnl = abs(close_quantity) * (old_price - close_price)
        
        return pnl
    
    def _update_equity_curve(self, current_bar: OHLCV):
        """Update equity curve with current market values"""
        total_equity = self.current_capital
        
        # Add unrealized P&L from open positions
        for position in self.positions.values():
            if not position.is_flat:
                unrealized_pnl = position.quantity * (current_bar.close - position.avg_price)
                total_equity += unrealized_pnl
        
        self.equity_curve.append(total_equity)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
            'open_positions': len([p for p in self.positions.values() if not p.is_flat]),
            'total_trades': len(self.trades),
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'realized_pnl': pos.realized_pnl
            } for symbol, pos in self.positions.items() if not pos.is_flat}
        }