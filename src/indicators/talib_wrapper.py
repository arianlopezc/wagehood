"""
TA-Lib Wrapper for Technical Indicators

This module provides a wrapper around TA-Lib functions to maintain API compatibility
with the existing custom implementations while using industry-standard calculations.
"""

import numpy as np
import talib
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Constants from the original momentum.py
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def calculate_rsi(
    data: Union[np.ndarray, list], period: int = RSI_PERIOD
) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) using TA-Lib.

    RSI measures the speed and change of price movements, oscillating between 0 and 100.
    Values above 70 typically indicate overbought conditions, below 30 indicate oversold.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods for RSI calculation (default from constants)

    Returns:
        np.ndarray: RSI values with NaN for insufficient data points

    Raises:
        ValueError: If period is less than 1 or greater than data length
        TypeError: If data cannot be converted to numpy array
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(data_array)}"
        )

    try:
        # Use TA-Lib for RSI calculation
        rsi = talib.RSI(data_array, timeperiod=period)
        return rsi
    except Exception as e:
        logger.error(f"TA-Lib RSI calculation failed: {e}")
        raise


def calculate_macd(
    data: Union[np.ndarray, list],
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence) using TA-Lib.

    MACD consists of:
    - MACD Line: Difference between fast EMA and slow EMA
    - Signal Line: EMA of the MACD line
    - Histogram: Difference between MACD line and signal line

    Args:
        data: Price data (numpy array or list)
        fast: Fast EMA period (default from constants)
        slow: Slow EMA period (default from constants)
        signal: Signal line EMA period (default from constants)

    Returns:
        tuple: (macd_line, signal_line, histogram)

    Raises:
        ValueError: If fast >= slow or periods are invalid
        TypeError: If data cannot be converted to numpy array
    """
    if fast >= slow:
        raise ValueError("Fast period must be less than slow period")

    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("All periods must be at least 1")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < slow:
        raise ValueError(
            f"Insufficient data: need {slow} points, got {len(data_array)}"
        )

    try:
        # Use TA-Lib for MACD calculation
        macd_line, signal_line, histogram = talib.MACD(
            data_array, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"TA-Lib MACD calculation failed: {e}")
        raise


def calculate_stochastic(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator using TA-Lib.

    The Stochastic Oscillator compares a security's closing price to its price range
    over a specific period, oscillating between 0 and 100.

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        k_period: Period for %K calculation (default 14)
        d_period: Period for %D smoothing (default 3)
        smooth_k: Period for %K smoothing (default 3)

    Returns:
        tuple: (%K, %D) values

    Raises:
        ValueError: If arrays have different lengths or periods are invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if k_period < 1 or d_period < 1 or smooth_k < 1:
        raise ValueError("All periods must be at least 1")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("High, low, and close arrays must have the same length")

    if len(close_array) < k_period:
        raise ValueError(
            f"Insufficient data: need {k_period} points, got {len(close_array)}"
        )

    try:
        # Use TA-Lib for Stochastic calculation
        k_percent, d_percent = talib.STOCH(
            high_array,
            low_array,
            close_array,
            fastk_period=k_period,
            slowk_period=smooth_k,
            slowk_matype=0,  # Simple moving average
            slowd_period=d_period,
            slowd_matype=0,  # Simple moving average
        )
        return k_percent, d_percent
    except Exception as e:
        logger.error(f"TA-Lib Stochastic calculation failed: {e}")
        raise


def calculate_williams_r(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Williams %R using TA-Lib.

    Williams %R is a momentum indicator that measures overbought and oversold levels,
    oscillating between -100 and 0. Values above -20 indicate overbought conditions,
    below -80 indicate oversold conditions.

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        period: Lookback period (default 14)

    Returns:
        np.ndarray: Williams %R values

    Raises:
        ValueError: If arrays have different lengths or period is invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("High, low, and close arrays must have the same length")

    if len(close_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(close_array)}"
        )

    try:
        # Use TA-Lib for Williams %R calculation
        williams_r = talib.WILLR(high_array, low_array, close_array, timeperiod=period)
        return williams_r
    except Exception as e:
        logger.error(f"TA-Lib Williams %R calculation failed: {e}")
        raise


def calculate_cci(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    period: int = 20,
) -> np.ndarray:
    """
    Calculate Commodity Channel Index (CCI) using TA-Lib.

    CCI measures the variation of a security's price from its statistical mean.
    Values above +100 may indicate overbought conditions, below -100 may indicate oversold.

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        period: Period for CCI calculation (default 20)

    Returns:
        np.ndarray: CCI values

    Raises:
        ValueError: If arrays have different lengths or period is invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("High, low, and close arrays must have the same length")

    if len(close_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(close_array)}"
        )

    try:
        # Use TA-Lib for CCI calculation
        cci = talib.CCI(high_array, low_array, close_array, timeperiod=period)
        return cci
    except Exception as e:
        logger.error(f"TA-Lib CCI calculation failed: {e}")
        raise


def calculate_momentum(data: Union[np.ndarray, list], period: int = 10) -> np.ndarray:
    """
    Calculate Price Momentum using TA-Lib.

    Momentum compares the current price to the price n periods ago.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods to look back (default 10)

    Returns:
        np.ndarray: Momentum values

    Raises:
        ValueError: If period is less than 1 or greater than data length
        TypeError: If data cannot be converted to numpy array
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(data_array)}"
        )

    try:
        # Use TA-Lib for Momentum calculation
        momentum = talib.MOM(data_array, timeperiod=period)
        return momentum
    except Exception as e:
        logger.error(f"TA-Lib Momentum calculation failed: {e}")
        raise


def calculate_roc(data: Union[np.ndarray, list], period: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change (ROC) using TA-Lib.

    ROC measures the percentage change between the current price and the price n periods ago.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods to look back (default 10)

    Returns:
        np.ndarray: ROC values as percentages

    Raises:
        ValueError: If period is less than 1 or greater than data length
        TypeError: If data cannot be converted to numpy array
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(data_array)}"
        )

    try:
        # Use TA-Lib for ROC calculation
        roc = talib.ROC(data_array, timeperiod=period)
        return roc
    except Exception as e:
        logger.error(f"TA-Lib ROC calculation failed: {e}")
        raise


# Additional TA-Lib functions that weren't in the original implementation
def calculate_bb(
    data: Union[np.ndarray, list], period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands using TA-Lib.

    Args:
        data: Price data (numpy array or list)
        period: Period for moving average (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    try:
        data_array = np.asarray(data, dtype=float)
        upper, middle, lower = talib.BBANDS(
            data_array, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )
        return upper, middle, lower
    except Exception as e:
        logger.error(f"TA-Lib Bollinger Bands calculation failed: {e}")
        raise


def calculate_atr(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) using TA-Lib.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation (default 14)

    Returns:
        np.ndarray: ATR values
    """
    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)

        atr = talib.ATR(high_array, low_array, close_array, timeperiod=period)
        return atr
    except Exception as e:
        logger.error(f"TA-Lib ATR calculation failed: {e}")
        raise
