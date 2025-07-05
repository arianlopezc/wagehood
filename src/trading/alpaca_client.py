"""
Alpaca Trading Client

This module provides a trading client for Alpaca Markets, enabling order
management, portfolio tracking, and account operations through the alpaca-py SDK.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import os

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        StopLimitOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import (
        OrderSide, TimeInForce, OrderType, OrderStatus, PositionSide
    )
    from alpaca.common.exceptions import APIError
    ALPACA_TRADING_AVAILABLE = True
except ImportError:
    ALPACA_TRADING_AVAILABLE = False

from src.core.models import TimeFrame

logger = logging.getLogger(__name__)


class AlpacaOrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class AlpacaOrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class AlpacaTimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class AlpacaTradingClient:
    """
    Alpaca Markets trading client for order management and portfolio tracking.
    
    This client provides a high-level interface for trading operations using
    Alpaca's commission-free stock trading platform. Supports both paper
    trading and live trading environments.
    
    Features:
    - Order management (market, limit, stop orders)
    - Portfolio and position tracking
    - Account information and buying power
    - Real-time trade updates via WebSocket
    - Paper trading for safe testing
    - Risk management and validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alpaca trading client.
        
        Args:
            config: Configuration dictionary with options:
                - api_key: Alpaca API key (or use ALPACA_API_KEY env var)
                - secret_key: Alpaca secret key (or use ALPACA_SECRET_KEY env var)
                - paper: Use paper trading environment (default: True)
                - max_retries: Maximum retry attempts (default: 3)
                - retry_delay: Delay between retries in seconds (default: 1.0)
        """
        if not ALPACA_TRADING_AVAILABLE:
            raise ImportError("alpaca-py is required for AlpacaTradingClient. Install with: pip install alpaca-py")
        
        self.config = config or {}
        
        # Configuration
        self.api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY')
        self.secret_key = self.config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
        self.paper = self.config.get('paper', True)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass api_key and secret_key in config."
            )
        
        # Initialize clients
        self.trading_client = None
        self.trading_stream = None
        
        # Connection state
        self._connected = False
        self._streaming = False
        self._trade_handlers = []
        
        # Account and portfolio cache
        self._account_cache = None
        self._positions_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 10  # seconds
        
        logger.info(f"Initialized AlpacaTradingClient (paper={self.paper})")
    
    async def connect(self) -> bool:
        """
        Establish connection to Alpaca trading services.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            # Initialize trading stream
            self.trading_stream = TradingStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            # Test connection
            await self._test_connection()
            
            self._connected = True
            logger.info("Successfully connected to Alpaca Trading")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca Trading: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Alpaca trading services."""
        try:
            # Stop trading stream
            if self._streaming:
                await self.stop_trade_updates()
            
            self._connected = False
            logger.info("Disconnected from Alpaca Trading")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _test_connection(self) -> None:
        """Test connection by retrieving account information."""
        try:
            account = self.trading_client.get_account()
            logger.debug(f"Connection test successful. Account ID: {account.id}")
        except Exception as e:
            raise ConnectionError(f"Connection test failed: {str(e)}")
    
    # Order Management
    
    async def place_market_order(self, symbol: str, quantity: float, side: AlpacaOrderSide,
                               time_in_force: AlpacaTimeInForce = AlpacaTimeInForce.DAY) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            quantity: Number of shares to trade
            side: Order side (BUY or SELL)
            time_in_force: Time in force (default: DAY)
            
        Returns:
            Order information dictionary
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            # Convert enums to Alpaca format
            alpaca_side = OrderSide.BUY if side == AlpacaOrderSide.BUY else OrderSide.SELL
            alpaca_tif = self._convert_time_in_force(time_in_force)
            
            # Create order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=alpaca_tif
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Convert to standard format
            order_info = self._convert_order_to_dict(order)
            
            logger.info(f"Placed market order: {order_info['id']} - {side.value} {quantity} {symbol}")
            return order_info
            
        except APIError as e:
            logger.error(f"API error placing market order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            raise
    
    async def place_limit_order(self, symbol: str, quantity: float, side: AlpacaOrderSide,
                              limit_price: float, 
                              time_in_force: AlpacaTimeInForce = AlpacaTimeInForce.DAY) -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares to trade
            side: Order side (BUY or SELL)
            limit_price: Limit price for the order
            time_in_force: Time in force (default: DAY)
            
        Returns:
            Order information dictionary
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            # Convert enums to Alpaca format
            alpaca_side = OrderSide.BUY if side == AlpacaOrderSide.BUY else OrderSide.SELL
            alpaca_tif = self._convert_time_in_force(time_in_force)
            
            # Create order request
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=limit_price
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Convert to standard format
            order_info = self._convert_order_to_dict(order)
            
            logger.info(f"Placed limit order: {order_info['id']} - {side.value} {quantity} {symbol} @ ${limit_price}")
            return order_info
            
        except APIError as e:
            logger.error(f"API error placing limit order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            raise
    
    async def place_stop_order(self, symbol: str, quantity: float, side: AlpacaOrderSide,
                             stop_price: float,
                             time_in_force: AlpacaTimeInForce = AlpacaTimeInForce.DAY) -> Dict[str, Any]:
        """
        Place a stop order.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares to trade
            side: Order side (BUY or SELL)
            stop_price: Stop price for the order
            time_in_force: Time in force (default: DAY)
            
        Returns:
            Order information dictionary
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            # Convert enums to Alpaca format
            alpaca_side = OrderSide.BUY if side == AlpacaOrderSide.BUY else OrderSide.SELL
            alpaca_tif = self._convert_time_in_force(time_in_force)
            
            # Create order request
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                stop_price=stop_price
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Convert to standard format
            order_info = self._convert_order_to_dict(order)
            
            logger.info(f"Placed stop order: {order_info['id']} - {side.value} {quantity} {symbol} @ stop ${stop_price}")
            return order_info
            
        except APIError as e:
            logger.error(f"API error placing stop order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
            
        except APIError as e:
            logger.error(f"API error cancelling order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            cancel_statuses = self.trading_client.cancel_orders()
            success_count = sum(1 for status in cancel_statuses if status.http_status == 200)
            
            logger.info(f"Cancelled {success_count} orders")
            return success_count
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order information.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order information dictionary or None if not found
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._convert_order_to_dict(order)
            
        except APIError as e:
            if "404" in str(e):
                return None
            logger.error(f"API error getting order {order_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise
    
    async def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of orders.
        
        Args:
            status: Order status filter (optional)
            limit: Maximum number of orders to return
            
        Returns:
            List of order information dictionaries
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            orders = self.trading_client.get_orders(limit=limit)
            
            order_list = []
            for order in orders:
                if status is None or order.status.value == status:
                    order_list.append(self._convert_order_to_dict(order))
            
            return order_list
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise
    
    # Portfolio and Position Management
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information dictionary
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        # Check cache
        if self._account_cache and self._is_cache_valid():
            return self._account_cache
        
        try:
            account = self.trading_client.get_account()
            
            account_info = {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status.value,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'multiplier': account.multiplier,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'shorting_enabled': account.shorting_enabled,
                'long_market_value': float(account.long_market_value) if account.long_market_value else 0.0,
                'short_market_value': float(account.short_market_value) if account.short_market_value else 0.0,
                'initial_margin': float(account.initial_margin) if account.initial_margin else 0.0,
                'maintenance_margin': float(account.maintenance_margin) if account.maintenance_margin else 0.0
            }
            
            # Cache the result
            self._account_cache = account_info
            self._cache_timestamp = datetime.now()
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions.
        
        Returns:
            List of position information dictionaries
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        # Check cache
        if self._positions_cache and self._is_cache_valid():
            return self._positions_cache
        
        try:
            positions = self.trading_client.get_all_positions()
            
            position_list = []
            for position in positions:
                position_info = {
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': position.side.value,
                    'market_value': float(position.market_value) if position.market_value else 0.0,
                    'cost_basis': float(position.cost_basis) if position.cost_basis else 0.0,
                    'unrealized_pl': float(position.unrealized_pl) if position.unrealized_pl else 0.0,
                    'unrealized_plpc': float(position.unrealized_plpc) if position.unrealized_plpc else 0.0,
                    'unrealized_intraday_pl': float(position.unrealized_intraday_pl) if position.unrealized_intraday_pl else 0.0,
                    'unrealized_intraday_plpc': float(position.unrealized_intraday_plpc) if position.unrealized_intraday_plpc else 0.0,
                    'current_price': float(position.current_price) if position.current_price else 0.0,
                    'lastday_price': float(position.lastday_price) if position.lastday_price else 0.0,
                    'change_today': float(position.change_today) if position.change_today else 0.0,
                    'avg_entry_price': float(position.avg_entry_price) if position.avg_entry_price else 0.0
                }
                position_list.append(position_info)
            
            # Cache the result
            self._positions_cache = position_list
            self._cache_timestamp = datetime.now()
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position information dictionary or None if no position
        """
        positions = await self.get_positions()
        
        for position in positions:
            if position['symbol'] == symbol:
                return position
        
        return None
    
    async def close_position(self, symbol: str, qty: Optional[float] = None) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            qty: Quantity to close (None for full position)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            if qty is None:
                # Close entire position
                self.trading_client.close_position(symbol)
                logger.info(f"Closed entire position for {symbol}")
            else:
                # Close partial position by placing opposite order
                position = await self.get_position(symbol)
                if not position:
                    logger.warning(f"No position found for {symbol}")
                    return False
                
                # Determine order side (opposite of position)
                if float(position['qty']) > 0:
                    side = AlpacaOrderSide.SELL
                else:
                    side = AlpacaOrderSide.BUY
                
                # Place market order to close
                await self.place_market_order(symbol, abs(qty), side)
                logger.info(f"Closed {qty} shares of {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    async def close_all_positions(self) -> int:
        """
        Close all positions.
        
        Returns:
            Number of positions closed
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            cancel_statuses = self.trading_client.close_all_positions(cancel_orders=True)
            success_count = sum(1 for status in cancel_statuses if status.http_status == 200)
            
            logger.info(f"Closed {success_count} positions")
            return success_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    # Real-time Trade Updates
    
    async def start_trade_updates(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Start receiving real-time trade updates.
        
        Args:
            handler: Function to handle trade update events
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Trading")
        
        try:
            # Add handler to list
            self._trade_handlers.append(handler)
            
            # Subscribe to trade updates
            self.trading_stream.subscribe_trade_updates(self._handle_trade_update)
            
            # Start streaming if not already started
            if not self._streaming:
                asyncio.create_task(self.trading_stream.run())
                self._streaming = True
                logger.info("Started trade update streaming")
            
        except Exception as e:
            logger.error(f"Error starting trade updates: {e}")
            raise
    
    async def stop_trade_updates(self) -> None:
        """Stop receiving real-time trade updates."""
        try:
            if self._streaming:
                await self.trading_stream.stop_ws()
                self._streaming = False
                self._trade_handlers.clear()
                logger.info("Stopped trade update streaming")
        except Exception as e:
            logger.error(f"Error stopping trade updates: {e}")
    
    async def _handle_trade_update(self, trade_update) -> None:
        """Handle incoming trade update."""
        try:
            # Convert trade update to dictionary
            update_data = {
                'event': trade_update.event.value,
                'timestamp': trade_update.timestamp,
                'order_id': trade_update.order.id,
                'symbol': trade_update.order.symbol,
                'side': trade_update.order.side.value,
                'qty': float(trade_update.order.qty),
                'filled_qty': float(trade_update.order.filled_qty) if trade_update.order.filled_qty else 0.0,
                'status': trade_update.order.status.value,
                'order_type': trade_update.order.order_type.value,
                'filled_avg_price': float(trade_update.order.filled_avg_price) if trade_update.order.filled_avg_price else None
            }
            
            # Call all registered handlers
            for handler in self._trade_handlers:
                try:
                    await handler(update_data)
                except Exception as e:
                    logger.error(f"Error in trade update handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    # Utility Methods
    
    def _convert_time_in_force(self, tif: AlpacaTimeInForce):
        """Convert internal time in force to Alpaca format."""
        mapping = {
            AlpacaTimeInForce.DAY: TimeInForce.DAY,
            AlpacaTimeInForce.GTC: TimeInForce.GTC,
            AlpacaTimeInForce.IOC: TimeInForce.IOC,
            AlpacaTimeInForce.FOK: TimeInForce.FOK
        }
        return mapping.get(tif, TimeInForce.DAY)
    
    def _convert_order_to_dict(self, order) -> Dict[str, Any]:
        """Convert Alpaca order object to dictionary."""
        return {
            'id': order.id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'qty': float(order.qty),
            'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
            'status': order.status.value,
            'time_in_force': order.time_in_force.value,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'stop_price': float(order.stop_price) if order.stop_price else None,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'created_at': order.created_at,
            'updated_at': order.updated_at,
            'submitted_at': order.submitted_at,
            'filled_at': order.filled_at,
            'expired_at': order.expired_at,
            'canceled_at': order.canceled_at,
            'failed_at': order.failed_at,
            'replaced_at': order.replaced_at
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp:
            return False
        
        return (datetime.now() - self._cache_timestamp).total_seconds() < self._cache_ttl
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information."""
        return {
            'connected': self._connected,
            'streaming': self._streaming,
            'paper_trading': self.paper,
            'handlers_count': len(self._trade_handlers)
        }