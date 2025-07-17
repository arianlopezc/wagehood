"""
Data Requirements Calculator

This module provides centralized calculation of minimum data requirements
for all trading strategies and timeframes to ensure reliable signal generation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import math


class DataRequirementsCalculator:
    """
    Centralized calculator for minimum data requirements across all strategies.
    
    This ensures consistent validation between input validation and data fetching,
    accounting for trading days vs calendar days and providing adequate buffers.
    """
    
    # Trading days per week (Monday-Friday)
    TRADING_DAYS_PER_WEEK = 5
    CALENDAR_DAYS_PER_WEEK = 7
    
    # Average trading days per month/year (accounting for holidays)
    TRADING_DAYS_PER_MONTH = 22
    TRADING_DAYS_PER_YEAR = 252
    
    # Safety buffers for TA-Lib calculations
    TALIB_BUFFER = 20  # Extra periods for TA-Lib stability
    STRATEGY_BUFFER = 10  # Extra periods for strategy-specific logic
    
    @classmethod
    def calculate_macd_rsi_requirements(cls, parameters: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate minimum data requirements for MACD+RSI strategy.
        
        Args:
            parameters: Strategy parameters dict
            
        Returns:
            Dictionary with minimum requirements
        """
        macd_fast = parameters.get("macd_fast", 12)
        macd_slow = parameters.get("macd_slow", 26)
        macd_signal = parameters.get("macd_signal", 9)
        rsi_period = parameters.get("rsi_period", 14)
        
        # MACD needs slow + signal periods for full calculation
        macd_minimum = macd_slow + macd_signal
        
        # RSI needs its period
        rsi_minimum = rsi_period
        
        # Total requirement: max of individual requirements + buffers
        technical_minimum = max(macd_minimum, rsi_minimum)
        total_minimum = technical_minimum + cls.TALIB_BUFFER + cls.STRATEGY_BUFFER
        
        return {
            "macd_minimum": macd_minimum,
            "rsi_minimum": rsi_minimum,
            "technical_minimum": technical_minimum,
            "total_minimum": total_minimum,
            "recommended_minimum": total_minimum + 20  # Extra safety margin
        }
    
    @classmethod
    def calculate_rsi_trend_requirements(cls, parameters: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate minimum data requirements for RSI Trend strategy.
        
        Args:
            parameters: Strategy parameters dict
            
        Returns:
            Dictionary with minimum requirements
        """
        rsi_period = parameters.get("rsi_period", 14)
        trend_confirmation_periods = parameters.get("trend_confirmation_periods", 10)
        
        # RSI trend needs RSI period + trend confirmation
        technical_minimum = rsi_period + trend_confirmation_periods
        total_minimum = technical_minimum + cls.TALIB_BUFFER + cls.STRATEGY_BUFFER
        
        return {
            "rsi_minimum": rsi_period,
            "trend_minimum": trend_confirmation_periods,
            "technical_minimum": technical_minimum,
            "total_minimum": total_minimum,
            "recommended_minimum": total_minimum + 15
        }
    
    @classmethod
    def calculate_bollinger_requirements(cls, parameters: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate minimum data requirements for Bollinger Breakout strategy.
        
        Args:
            parameters: Strategy parameters dict
            
        Returns:
            Dictionary with minimum requirements
        """
        bb_period = parameters.get("bb_period", 20)
        consolidation_periods = parameters.get("consolidation_periods", 10)
        
        # Bollinger Bands need the BB period + consolidation analysis
        technical_minimum = bb_period + consolidation_periods
        total_minimum = technical_minimum + cls.TALIB_BUFFER + cls.STRATEGY_BUFFER
        
        return {
            "bb_minimum": bb_period,
            "consolidation_minimum": consolidation_periods,
            "technical_minimum": technical_minimum,
            "total_minimum": total_minimum,
            "recommended_minimum": total_minimum + 15
        }
    
    @classmethod
    def calculate_sr_requirements(cls, parameters: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate minimum data requirements for Support/Resistance strategy.
        
        Args:
            parameters: Strategy parameters dict
            
        Returns:
            Dictionary with minimum requirements
        """
        lookback_periods = parameters.get("lookback_periods", 50)
        consolidation_periods = parameters.get("consolidation_periods", 10)
        
        # S/R needs lookback periods for level identification
        technical_minimum = lookback_periods + consolidation_periods
        total_minimum = technical_minimum + cls.TALIB_BUFFER + cls.STRATEGY_BUFFER
        
        return {
            "lookback_minimum": lookback_periods,
            "consolidation_minimum": consolidation_periods,
            "technical_minimum": technical_minimum,
            "total_minimum": total_minimum,
            "recommended_minimum": total_minimum + 20
        }
    
    @classmethod
    def calculate_calendar_days_needed(cls, periods_required: int, timeframe: str) -> int:
        """
        Calculate calendar days needed to get the required periods.
        
        Args:
            periods_required: Number of periods needed (hours for 1h, days for 1d)
            timeframe: Data timeframe ('1h' or '1d')
            
        Returns:
            Number of calendar days needed
        """
        if timeframe == "1h":
            # For hourly data, periods_required is in HOURS
            # Convert hours to trading days: ~6.5 trading hours per day
            trading_hours_per_day = 6.5
            trading_days_needed = math.ceil(periods_required / trading_hours_per_day)
            
            # Convert trading days to calendar days (account for weekends)
            calendar_days = math.ceil(trading_days_needed * cls.CALENDAR_DAYS_PER_WEEK / cls.TRADING_DAYS_PER_WEEK)
            
            # Add small buffer and ensure we stay within Alpaca's 30-day limit
            calendar_days_with_buffer = min(calendar_days + 5, 25)  # Cap at 25 days for safety
            
            return calendar_days_with_buffer
        
        elif timeframe == "1d":
            # For daily data, periods_required is in TRADING DAYS
            # Rule of thumb: 252 trading days per year, 365 calendar days per year
            # So multiply by ~1.45 to account for weekends and holidays
            calendar_days = math.ceil(periods_required * 1.45)
            # Add buffer for holidays and market closures
            return calendar_days + 14
        
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    @classmethod
    def validate_date_range(cls, start_date: datetime, end_date: datetime, 
                          strategy_type: str, strategy_parameters: Dict[str, Any],
                          timeframe: str) -> Dict[str, Any]:
        """
        Validate if date range provides sufficient data for strategy.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            strategy_type: Type of strategy ('macd_rsi', 'rsi_trend', 'bollinger', 'sr')
            strategy_parameters: Strategy parameters
            timeframe: Data timeframe ('1h' or '1d')
            
        Returns:
            Dictionary with validation result
        """
        # Calculate requirements based on strategy type
        if strategy_type == "macd_rsi":
            requirements = cls.calculate_macd_rsi_requirements(strategy_parameters)
        elif strategy_type == "rsi_trend":
            requirements = cls.calculate_rsi_trend_requirements(strategy_parameters)
        elif strategy_type == "bollinger":
            requirements = cls.calculate_bollinger_requirements(strategy_parameters)
        elif strategy_type == "sr":
            requirements = cls.calculate_sr_requirements(strategy_parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Use recommended minimum for validation
        periods_needed = requirements["recommended_minimum"]
        calendar_days_needed = cls.calculate_calendar_days_needed(periods_needed, timeframe)
        
        # Check if date range is sufficient
        actual_calendar_days = (end_date - start_date).days
        
        if actual_calendar_days < calendar_days_needed:
            periods_type = "hours" if timeframe == "1h" else "trading days"
            return {
                "valid": False,
                "error": f"Date range too small. Need {calendar_days_needed} calendar days "
                        f"(~{periods_needed} {periods_type}) for {strategy_type} strategy with {timeframe} timeframe. "
                        f"Got {actual_calendar_days} calendar days.",
                "required_calendar_days": calendar_days_needed,
                "required_periods": periods_needed,
                "actual_calendar_days": actual_calendar_days
            }
        
        return {
            "valid": True,
            "error": None,
            "required_calendar_days": calendar_days_needed,
            "required_periods": periods_needed,
            "actual_calendar_days": actual_calendar_days
        }
    
    @classmethod
    def validate_data_sufficiency(cls, data_points: int, strategy_type: str, 
                                strategy_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if fetched data has sufficient points for strategy.
        
        Args:
            data_points: Number of data points fetched
            strategy_type: Type of strategy
            strategy_parameters: Strategy parameters
            
        Returns:
            Dictionary with validation result
        """
        # Calculate requirements based on strategy type
        if strategy_type == "macd_rsi":
            requirements = cls.calculate_macd_rsi_requirements(strategy_parameters)
        elif strategy_type == "rsi_trend":
            requirements = cls.calculate_rsi_trend_requirements(strategy_parameters)
        elif strategy_type == "bollinger":
            requirements = cls.calculate_bollinger_requirements(strategy_parameters)
        elif strategy_type == "sr":
            requirements = cls.calculate_sr_requirements(strategy_parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        minimum_required = requirements["total_minimum"]
        
        if data_points < minimum_required:
            return {
                "valid": False,
                "error": f"Insufficient data for {strategy_type} strategy. "
                        f"Got {data_points} data points, need minimum {minimum_required}. "
                        f"Technical minimum: {requirements['technical_minimum']}, "
                        f"with buffers: {minimum_required}",
                "data_points": data_points,
                "minimum_required": minimum_required,
                "technical_minimum": requirements["technical_minimum"]
            }
        
        return {
            "valid": True,
            "error": None,
            "data_points": data_points,
            "minimum_required": minimum_required,
            "technical_minimum": requirements["technical_minimum"]
        }