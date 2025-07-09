"""Custom exceptions for the trading system"""


class WagehoodError(Exception):
    """Base exception for all Wagehood-related errors"""

    pass


class DataError(WagehoodError):
    """Raised when there are issues with market data"""

    pass


class CalculationError(WagehoodError):
    """Raised when indicator calculations fail"""

    pass


class StrategyError(WagehoodError):
    """Raised when strategy execution fails"""

    pass


class BacktestError(WagehoodError):
    """Raised when backtesting fails"""

    pass


class ValidationError(WagehoodError):
    """Raised when data validation fails"""

    pass


class InsufficientDataError(DataError):
    """Raised when there isn't enough data for calculations"""

    pass
