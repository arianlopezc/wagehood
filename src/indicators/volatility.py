"""
Volatility Indicators

This module implements volatility-based technical indicators including Bollinger Bands,
Average True Range (ATR), Keltner Channels, and Donchian Channels.
"""

import numpy as np
from typing import Union, Tuple, Optional
from ..core.constants import BB_PERIOD, BB_STD_DEV
from .moving_averages import calculate_sma, calculate_ema


def calculate_bollinger_bands(
    data: Union[np.ndarray, list], period: int = BB_PERIOD, std_dev: float = BB_STD_DEV
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of:
    - Middle Band: Simple Moving Average
    - Upper Band: SMA + (standard deviation * multiplier)
    - Lower Band: SMA - (standard deviation * multiplier)

    Args:
        data: Price data (numpy array or list)
        period: Period for moving average and standard deviation (default from constants)
        std_dev: Standard deviation multiplier (default from constants)

    Returns:
        tuple: (upper_band, middle_band, lower_band)

    Raises:
        ValueError: If period is less than 2 or std_dev is negative
        TypeError: If data cannot be converted to numpy array
    """
    if period < 2:
        raise ValueError("Period must be at least 2 for standard deviation calculation")

    if std_dev < 0:
        raise ValueError("Standard deviation multiplier must be non-negative")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(data_array)}"
        )

    # Calculate middle band (SMA)
    middle_band = calculate_sma(data_array, period)

    # Initialize result arrays
    upper_band = np.full(len(data_array), np.nan)
    lower_band = np.full(len(data_array), np.nan)

    # Calculate bands
    for i in range(period - 1, len(data_array)):
        # Calculate standard deviation for the period
        window_data = data_array[i - period + 1 : i + 1]
        std = np.std(window_data, ddof=0)  # Population standard deviation

        # Calculate upper and lower bands
        upper_band[i] = middle_band[i] + (std * std_dev)
        lower_band[i] = middle_band[i] - (std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_atr(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by calculating the average of true ranges over a specified period.
    True Range is the maximum of:
    - Current High - Current Low
    - Absolute value of Current High - Previous Close
    - Absolute value of Current Low - Previous Close

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        period: Period for ATR calculation (default 14)

    Returns:
        np.ndarray: ATR values

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

    if len(close_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(close_array)}"
        )

    # Calculate True Range
    true_range = np.full(len(close_array), np.nan)

    # First TR is just High - Low
    true_range[0] = high_array[0] - low_array[0]

    # Calculate TR for remaining periods
    for i in range(1, len(close_array)):
        hl = high_array[i] - low_array[i]
        hc = abs(high_array[i] - close_array[i - 1])
        lc = abs(low_array[i] - close_array[i - 1])

        true_range[i] = max(hl, hc, lc)

    # Calculate ATR using Wilder's smoothing method
    atr = np.full(len(close_array), np.nan)

    # First ATR is simple average of first 'period' TRs
    if len(true_range) >= period:
        atr[period - 1] = np.mean(true_range[:period])

        # Calculate subsequent ATR values using Wilder's method
        for i in range(period, len(close_array)):
            atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period

    return atr


def calculate_keltner_channels(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    period: int = 20,
    multiplier: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Keltner Channels.

    Keltner Channels consist of:
    - Middle Line: Exponential Moving Average of close prices
    - Upper Channel: EMA + (ATR * multiplier)
    - Lower Channel: EMA - (ATR * multiplier)

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        period: Period for EMA and ATR calculation (default 20)
        multiplier: ATR multiplier (default 2.0)

    Returns:
        tuple: (upper_channel, middle_line, lower_channel)

    Raises:
        ValueError: If arrays have different lengths or parameters are invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    if multiplier < 0:
        raise ValueError("Multiplier must be non-negative")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("High, low, and close arrays must have the same length")

    if len(close_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(close_array)}"
        )

    # Calculate middle line (EMA of close)
    middle_line = calculate_ema(close_array, period)

    # Calculate ATR
    atr = calculate_atr(high_array, low_array, close_array, period)

    # Calculate upper and lower channels
    upper_channel = middle_line + (atr * multiplier)
    lower_channel = middle_line - (atr * multiplier)

    return upper_channel, middle_line, lower_channel


def calculate_donchian_channels(
    high: Union[np.ndarray, list], low: Union[np.ndarray, list], period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Donchian Channels.

    Donchian Channels consist of:
    - Upper Channel: Highest high over the period
    - Lower Channel: Lowest low over the period
    - Middle Channel: Average of upper and lower channels

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        period: Lookback period (default 20)

    Returns:
        tuple: (upper_channel, middle_channel, lower_channel)

    Raises:
        ValueError: If arrays have different lengths or period is invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if len(high_array) != len(low_array):
        raise ValueError("High and low arrays must have the same length")

    if len(high_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(high_array)}"
        )

    # Initialize result arrays
    upper_channel = np.full(len(high_array), np.nan)
    lower_channel = np.full(len(high_array), np.nan)
    middle_channel = np.full(len(high_array), np.nan)

    # Calculate channels
    for i in range(period - 1, len(high_array)):
        # Calculate highest high and lowest low over the period
        upper_channel[i] = np.max(high_array[i - period + 1 : i + 1])
        lower_channel[i] = np.min(low_array[i - period + 1 : i + 1])

        # Calculate middle channel
        middle_channel[i] = (upper_channel[i] + lower_channel[i]) / 2.0

    return upper_channel, middle_channel, lower_channel


def calculate_volatility(
    data: Union[np.ndarray, list],
    period: int = 20,
    annualize: bool = False,
    trading_periods: int = 252,
) -> np.ndarray:
    """
    Calculate Historical Volatility.

    Historical volatility measures the rate of price change for a security over a specific period.

    Args:
        data: Price data (numpy array or list)
        period: Period for volatility calculation (default 20)
        annualize: Whether to annualize the volatility (default False)
        trading_periods: Number of trading periods per year for annualization (default 252)

    Returns:
        np.ndarray: Volatility values

    Raises:
        ValueError: If period is less than 2 or trading_periods is invalid
        TypeError: If data cannot be converted to numpy array
    """
    if period < 2:
        raise ValueError("Period must be at least 2")

    if trading_periods <= 0:
        raise ValueError("Trading periods must be positive")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < period + 1:
        raise ValueError(
            f"Insufficient data: need {period + 1} points, got {len(data_array)}"
        )

    # Calculate log returns
    log_returns = np.log(data_array[1:] / data_array[:-1])

    # Initialize result array
    volatility = np.full(len(data_array), np.nan)

    # Calculate rolling volatility
    for i in range(period, len(data_array)):
        # Get returns for the period
        period_returns = log_returns[i - period : i]

        # Calculate standard deviation
        vol = np.std(period_returns, ddof=1)  # Sample standard deviation

        # Annualize if requested
        if annualize:
            vol *= np.sqrt(trading_periods)

        volatility[i] = vol

    return volatility


def calculate_bb_width(
    upper_band: np.ndarray, lower_band: np.ndarray, middle_band: np.ndarray
) -> np.ndarray:
    """
    Calculate Bollinger Band Width.

    BB Width measures the width of the Bollinger Bands relative to the middle band.

    Args:
        upper_band: Upper Bollinger Band values
        lower_band: Lower Bollinger Band values
        middle_band: Middle Bollinger Band values (SMA)

    Returns:
        np.ndarray: Bollinger Band Width values

    Raises:
        ValueError: If arrays have different lengths
    """
    if not (len(upper_band) == len(lower_band) == len(middle_band)):
        raise ValueError("All band arrays must have the same length")

    # Calculate BB Width
    bb_width = np.where(
        middle_band != 0, (upper_band - lower_band) / middle_band, np.nan
    )

    return bb_width


def calculate_bb_percent(
    data: Union[np.ndarray, list], upper_band: np.ndarray, lower_band: np.ndarray
) -> np.ndarray:
    """
    Calculate Bollinger Band Percent (%B).

    %B indicates where the price is relative to the Bollinger Bands.
    Values above 1 indicate price is above the upper band,
    values below 0 indicate price is below the lower band.

    Args:
        data: Price data (numpy array or list)
        upper_band: Upper Bollinger Band values
        lower_band: Lower Bollinger Band values

    Returns:
        np.ndarray: %B values

    Raises:
        ValueError: If arrays have different lengths
        TypeError: If data cannot be converted to numpy array
    """
    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if not (len(data_array) == len(upper_band) == len(lower_band)):
        raise ValueError("All arrays must have the same length")

    # Calculate %B
    band_diff = upper_band - lower_band
    percent_b = np.where(band_diff != 0, (data_array - lower_band) / band_diff, np.nan)

    return percent_b


def calculate_price_channels(
    high: Union[np.ndarray, list], low: Union[np.ndarray, list], period: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Price Channels (Highest High and Lowest Low).

    Price channels show the highest high and lowest low over a specified period.

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        period: Lookback period (default 20)

    Returns:
        tuple: (highest_high, lowest_low)

    Raises:
        ValueError: If arrays have different lengths or period is invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    if period < 1:
        raise ValueError("Period must be at least 1")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if len(high_array) != len(low_array):
        raise ValueError("High and low arrays must have the same length")

    if len(high_array) < period:
        raise ValueError(
            f"Insufficient data: need {period} points, got {len(high_array)}"
        )

    # Initialize result arrays
    highest_high = np.full(len(high_array), np.nan)
    lowest_low = np.full(len(high_array), np.nan)

    # Calculate channels
    for i in range(period - 1, len(high_array)):
        highest_high[i] = np.max(high_array[i - period + 1 : i + 1])
        lowest_low[i] = np.min(low_array[i - period + 1 : i + 1])

    return highest_high, lowest_low
