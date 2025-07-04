"""
Transaction cost models for realistic backtesting
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class TransactionCost:
    """Container for transaction costs"""
    commission: float
    slippage: float
    spread: float
    total: float


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models"""
    
    @abstractmethod
    def calculate_cost(self, quantity: float, price: float, side: str, 
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """
        Calculate transaction costs for a trade
        
        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Execution price
            side: 'buy' or 'sell'
            symbol: Trading symbol
            market_conditions: Additional market data for cost calculation
            
        Returns:
            TransactionCost object with breakdown of costs
        """
        pass


class CommissionFreeModel(TransactionCostModel):
    """Commission-free cost model (default) - zero commission trading"""
    
    def __init__(self, slippage_rate: float = 0.0):
        """
        Initialize commission-free cost model
        
        Args:
            slippage_rate: Optional slippage rate (defaults to 0.0 for true commission-free)
        """
        self.slippage_rate = slippage_rate
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate commission-free transaction costs"""
        abs_quantity = abs(quantity)
        
        # No commission
        commission = 0.0
        
        # Optional slippage (default is 0.0 for true commission-free)
        slippage_cost = 0.0
        if self.slippage_rate > 0:
            base_slippage = price * self.slippage_rate
            if market_conditions:
                volatility_factor = market_conditions.get('volatility_factor', 1.0)
                volume_factor = market_conditions.get('volume_factor', 1.0)
                slippage = base_slippage * volatility_factor * volume_factor
            else:
                slippage = base_slippage
            slippage_cost = abs_quantity * slippage
        
        # No spread cost in commission-free model
        spread_cost = 0.0
        
        total_cost = commission + slippage_cost + spread_cost
        
        return TransactionCost(
            commission=commission,
            slippage=slippage_cost,
            spread=spread_cost,
            total=total_cost
        )


class SimpleCommissionModel(TransactionCostModel):
    """Simple fixed commission model"""
    
    def __init__(self, commission_per_share: float = 0.0):
        """
        Initialize simple commission model
        
        Args:
            commission_per_share: Fixed commission per share (default: 0.0 for commission-free)
        """
        self.commission_per_share = commission_per_share
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate simple commission cost"""
        abs_quantity = abs(quantity)
        commission = abs_quantity * self.commission_per_share
        
        return TransactionCost(
            commission=commission,
            slippage=0.0,
            spread=0.0,
            total=commission
        )


class PercentageCommissionModel(TransactionCostModel):
    """Percentage-based commission model"""
    
    def __init__(self, commission_rate: float = 0.0, min_commission: float = 0.0):
        """
        Initialize percentage commission model
        
        Args:
            commission_rate: Commission as percentage of trade value (default: 0.0 for commission-free)
            min_commission: Minimum commission per trade (default: 0.0 for commission-free)
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate percentage-based commission cost"""
        abs_quantity = abs(quantity)
        trade_value = abs_quantity * price
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        return TransactionCost(
            commission=commission,
            slippage=0.0,
            spread=0.0,
            total=commission
        )


class RealisticCostModel(TransactionCostModel):
    """Realistic cost model with commission, slippage, and spread"""
    
    def __init__(self, 
                 commission_rate: float = 0.0,
                 min_commission: float = 0.0,
                 slippage_rate: float = 0.001,
                 spread_rate: float = 0.0005):
        """
        Initialize realistic cost model
        
        Args:
            commission_rate: Commission as percentage of trade value (default: 0.0 for commission-free)
            min_commission: Minimum commission per trade (default: 0.0 for commission-free)
            slippage_rate: Slippage as percentage of price
            spread_rate: Bid-ask spread as percentage of price
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_rate = slippage_rate
        self.spread_rate = spread_rate
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate realistic transaction costs"""
        abs_quantity = abs(quantity)
        trade_value = abs_quantity * price
        
        # Commission calculation
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Slippage calculation (varies with market conditions)
        base_slippage = price * self.slippage_rate
        if market_conditions:
            volatility_factor = market_conditions.get('volatility_factor', 1.0)
            volume_factor = market_conditions.get('volume_factor', 1.0)
            slippage = base_slippage * volatility_factor * volume_factor
        else:
            slippage = base_slippage
        
        slippage_cost = abs_quantity * slippage
        
        # Spread calculation
        spread = price * self.spread_rate
        spread_cost = abs_quantity * spread * 0.5  # Pay half the spread
        
        total_cost = commission + slippage_cost + spread_cost
        
        return TransactionCost(
            commission=commission,
            slippage=slippage_cost,
            spread=spread_cost,
            total=total_cost
        )


class TieredCommissionModel(TransactionCostModel):
    """Tiered commission model based on trade size"""
    
    def __init__(self, tiers: Dict[float, float] = None):
        """
        Initialize tiered commission model
        
        Args:
            tiers: Dictionary mapping trade value thresholds to commission rates
        """
        self.tiers = tiers or {
            0: 0.003,      # 0.3% for trades under $1000
            1000: 0.002,   # 0.2% for trades $1000-$10000
            10000: 0.001,  # 0.1% for trades $10000-$100000
            100000: 0.0005 # 0.05% for trades over $100000
        }
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate tiered commission cost"""
        abs_quantity = abs(quantity)
        trade_value = abs_quantity * price
        
        # Find appropriate tier
        commission_rate = 0.003  # Default rate
        for threshold in sorted(self.tiers.keys(), reverse=True):
            if trade_value >= threshold:
                commission_rate = self.tiers[threshold]
                break
        
        commission = trade_value * commission_rate
        
        return TransactionCost(
            commission=commission,
            slippage=0.0,
            spread=0.0,
            total=commission
        )


class MarketImpactModel(TransactionCostModel):
    """Market impact model for large trades"""
    
    def __init__(self, 
                 commission_rate: float = 0.0,
                 min_commission: float = 0.0,
                 impact_coefficient: float = 0.1):
        """
        Initialize market impact model
        
        Args:
            commission_rate: Base commission rate (default: 0.0 for commission-free)
            min_commission: Minimum commission per trade (default: 0.0 for commission-free)
            impact_coefficient: Market impact coefficient
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.impact_coefficient = impact_coefficient
    
    def calculate_cost(self, quantity: float, price: float, side: str,
                      symbol: str, market_conditions: Dict[str, Any] = None) -> TransactionCost:
        """Calculate costs including market impact"""
        abs_quantity = abs(quantity)
        trade_value = abs_quantity * price
        
        # Base commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Market impact (square root model)
        if market_conditions:
            avg_volume = market_conditions.get('avg_volume', 1000000)
            participation_rate = abs_quantity / avg_volume
            impact = self.impact_coefficient * math.sqrt(participation_rate) * price
        else:
            # Default impact for medium-sized trades
            impact = price * 0.001
        
        impact_cost = abs_quantity * impact
        
        total_cost = commission + impact_cost
        
        return TransactionCost(
            commission=commission,
            slippage=impact_cost,
            spread=0.0,
            total=total_cost
        )