"""
Moving Average Indicators

This module implements various moving average calculations including Simple Moving Average (SMA),
Exponential Moving Average (EMA), Weighted Moving Average (WMA), and Volume Weighted Moving Average (VWMA).
"""

import numpy as np
from typing import Union, Optional

# Constants - previously imported from ..core.constants
MA_FAST = 12
MA_SLOW = 26


def calculate_sma(data: Union[np.ndarray, list], period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).

    The SMA is calculated as the average of the last n periods of data.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods for the moving average

    Returns:
        np.ndarray: SMA values with NaN for insufficient data points

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

    if len(data_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(data_array)}"
        )

    # Initialize result array with NaN
    result = np.full(len(data_array), np.nan)

    # Calculate SMA using vectorized operations
    for i in range(period - 1, len(data_array)):
        result[i] = np.mean(data_array[i - period + 1 : i + 1])

    return result


def calculate_ema(
    data: Union[np.ndarray, list], period: int, alpha: Optional[float] = None
) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    The EMA gives more weight to recent prices, making it more responsive to new information.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods for the moving average
        alpha: Smoothing factor (if None, calculated as 2/(period+1))

    Returns:
        np.ndarray: EMA values with NaN for insufficient data points

    Raises:
        ValueError: If period is less than 1 or alpha is not between 0 and 1
        TypeError: If data cannot be converted to numpy array
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    if alpha is None:
        alpha = 2.0 / (period + 1)
    elif not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(data_array)}"
        )

    # Initialize result array with NaN
    result = np.full(len(data_array), np.nan)

    # Start with SMA for the first EMA value
    result[period - 1] = np.mean(data_array[:period])

    # Calculate EMA recursively
    for i in range(period, len(data_array)):
        result[i] = alpha * data_array[i] + (1 - alpha) * result[i - 1]

    return result


def calculate_wma(data: Union[np.ndarray, list], period: int) -> np.ndarray:
    """
    Calculate Weighted Moving Average (WMA).

    WMA gives linearly decreasing weights to older data points.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods for the moving average

    Returns:
        np.ndarray: WMA values with NaN for insufficient data points

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

    if len(data_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(data_array)}"
        )

    # Initialize result array with NaN
    result = np.full(len(data_array), np.nan)

    # Create weights array (1, 2, 3, ..., period)
    weights = np.arange(1, period + 1)
    weights_sum = np.sum(weights)

    # Calculate WMA
    for i in range(period - 1, len(data_array)):
        window_data = data_array[i - period + 1 : i + 1]
        result[i] = np.sum(window_data * weights) / weights_sum

    return result


def calculate_vwma(
    data: Union[np.ndarray, list], volume: Union[np.ndarray, list], period: int
) -> np.ndarray:
    """
    Calculate Volume Weighted Moving Average (VWMA).

    VWMA weights prices by their corresponding volume, giving more importance
    to prices with higher trading volume.

    Args:
        data: Price data (numpy array or list)
        volume: Volume data (numpy array or list)
        period: Number of periods for the moving average

    Returns:
        np.ndarray: VWMA values with NaN for insufficient data points

    Raises:
        ValueError: If period is less than 1 or data lengths don't match
        TypeError: If data cannot be converted to numpy array
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        data_array = np.asarray(data, dtype=float)
        volume_array = np.asarray(volume, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) != len(volume_array):
        raise ValueError("Data and volume arrays must have the same length")

    if len(data_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(data_array)}"
        )

    # Initialize result array with NaN
    result = np.full(len(data_array), np.nan)

    # Calculate VWMA
    for i in range(period - 1, len(data_array)):
        window_data = data_array[i - period + 1 : i + 1]
        window_volume = volume_array[i - period + 1 : i + 1]

        # Avoid division by zero
        if np.sum(window_volume) == 0:
            result[i] = np.nan
        else:
            result[i] = np.sum(window_data * window_volume) / np.sum(window_volume)

    return result


def calculate_ma_crossover(fast_ma: np.ndarray, slow_ma: np.ndarray) -> np.ndarray:
    """
    Calculate moving average crossover signals.

    Args:
        fast_ma: Fast moving average values
        slow_ma: Slow moving average values

    Returns:
        np.ndarray: Crossover signals (1 for bullish, -1 for bearish, 0 for none)

    Raises:
        ValueError: If arrays have different lengths
    """
    if len(fast_ma) != len(slow_ma):
        raise ValueError("Fast and slow MA arrays must have the same length")

    # Calculate the difference between fast and slow MA
    diff = fast_ma - slow_ma

    # Initialize signals array
    signals = np.zeros(len(diff))

    # Find crossover points
    for i in range(1, len(diff)):
        if np.isnan(diff[i - 1]) or np.isnan(diff[i]):
            continue

        # Bullish crossover: fast MA crosses above slow MA
        if diff[i - 1] <= 0 and diff[i] > 0:
            signals[i] = 1
        # Bearish crossover: fast MA crosses below slow MA
        elif diff[i - 1] >= 0 and diff[i] < 0:
            signals[i] = -1

    return signals


def calculate_ma_envelope(
    data: Union[np.ndarray, list], period: int, envelope_pct: float = 0.025
) -> tuple:
    """
    Calculate Moving Average Envelope.

    MA Envelope consists of upper and lower bands around a moving average.

    Args:
        data: Price data (numpy array or list)
        period: Number of periods for the moving average
        envelope_pct: Percentage for envelope bands (default 2.5%)

    Returns:
        tuple: (upper_band, middle_band, lower_band)

    Raises:
        ValueError: If envelope_pct is negative
    """
    if envelope_pct < 0:
        raise ValueError("Envelope percentage must be non-negative")

    # Calculate the base moving average
    ma = calculate_sma(data, period)

    # Calculate envelope bands
    upper_band = ma * (1 + envelope_pct)
    lower_band = ma * (1 - envelope_pct)

    return upper_band, ma, lower_band
