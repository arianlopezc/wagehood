"""
Momentum Indicators

This module implements momentum-based technical indicators including RSI, MACD, 
Stochastic Oscillator, Williams %R, and Commodity Channel Index (CCI).
"""

import numpy as np
from typing import Union, Tuple, Optional
from ..core.constants import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from .moving_averages import calculate_ema, calculate_sma


def calculate_rsi(data: Union[np.ndarray, list], period: int = RSI_PERIOD) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
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
        raise ValueError(f"Insufficient data: need {period + 1} points, got {len(data_array)}")
    
    # Calculate price changes
    delta = np.diff(data_array)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Initialize result array with NaN
    result = np.full(len(data_array), np.nan)
    
    # Calculate initial average gain and loss
    if len(gains) >= period:
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate RSI for the first valid point
        if avg_loss == 0 and avg_gain == 0:
            result[period] = 50.0  # Neutral for no price movement
        elif avg_loss == 0:
            result[period] = 100.0  # Only gains, no losses
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1 + rs))
        
        # Calculate RSI for subsequent points using smoothed averages
        for i in range(period + 1, len(data_array)):
            gain_idx = i - 1
            loss_idx = i - 1
            
            # Smoothed average calculation (Wilder's method)
            avg_gain = (avg_gain * (period - 1) + gains[gain_idx]) / period
            avg_loss = (avg_loss * (period - 1) + losses[loss_idx]) / period
            
            if avg_loss == 0 and avg_gain == 0:
                result[i] = 50.0  # Neutral for no price movement
            elif avg_loss == 0:
                result[i] = 100.0  # Only gains, no losses
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1 + rs))
    
    return result


def calculate_macd(data: Union[np.ndarray, list], fast: int = MACD_FAST, 
                   slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
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
        raise ValueError(f"Insufficient data: need {slow} points, got {len(data_array)}")
    
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(data_array, fast)
    slow_ema = calculate_ema(data_array, slow)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD line)
    # Need to handle NaN values in MACD line
    valid_macd_start = slow - 1  # First valid MACD value index
    signal_line = np.full(len(data_array), np.nan)
    
    if len(data_array) > valid_macd_start + signal - 1:
        # Extract valid MACD values for signal calculation
        valid_macd = macd_line[valid_macd_start:]
        valid_macd_clean = valid_macd[~np.isnan(valid_macd)]
        
        if len(valid_macd_clean) >= signal:
            signal_ema = calculate_ema(valid_macd_clean, signal)
            signal_line[valid_macd_start:valid_macd_start + len(signal_ema)] = signal_ema
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(high: Union[np.ndarray, list], low: Union[np.ndarray, list], 
                        close: Union[np.ndarray, list], k_period: int = 14, 
                        d_period: int = 3, smooth_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator.
    
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
        raise ValueError(f"Insufficient data: need {k_period} points, got {len(close_array)}")
    
    # Initialize result arrays
    k_percent = np.full(len(close_array), np.nan)
    d_percent = np.full(len(close_array), np.nan)
    
    # Calculate %K
    for i in range(k_period - 1, len(close_array)):
        # Get the highest high and lowest low for the period
        period_high = np.max(high_array[i - k_period + 1:i + 1])
        period_low = np.min(low_array[i - k_period + 1:i + 1])
        
        # Calculate raw %K
        if period_high == period_low:
            k_percent[i] = 50.0  # Avoid division by zero
        else:
            k_percent[i] = ((close_array[i] - period_low) / (period_high - period_low)) * 100.0
    
    # Smooth %K if requested
    if smooth_k > 1:
        k_percent = calculate_sma(k_percent, smooth_k)
    
    # Calculate %D (moving average of %K)
    d_percent = calculate_sma(k_percent, d_period)
    
    return k_percent, d_percent


def calculate_williams_r(high: Union[np.ndarray, list], low: Union[np.ndarray, list], 
                        close: Union[np.ndarray, list], period: int = 14) -> np.ndarray:
    """
    Calculate Williams %R.
    
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
        raise ValueError(f"Insufficient data: need {period} points, got {len(close_array)}")
    
    # Initialize result array
    williams_r = np.full(len(close_array), np.nan)
    
    # Calculate Williams %R
    for i in range(period - 1, len(close_array)):
        # Get the highest high and lowest low for the period
        period_high = np.max(high_array[i - period + 1:i + 1])
        period_low = np.min(low_array[i - period + 1:i + 1])
        
        # Calculate Williams %R
        if period_high == period_low:
            williams_r[i] = -50.0  # Avoid division by zero
        else:
            williams_r[i] = ((period_high - close_array[i]) / (period_high - period_low)) * -100.0
    
    return williams_r


def calculate_cci(high: Union[np.ndarray, list], low: Union[np.ndarray, list], 
                  close: Union[np.ndarray, list], period: int = 20) -> np.ndarray:
    """
    Calculate Commodity Channel Index (CCI).
    
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
        raise ValueError(f"Insufficient data: need {period} points, got {len(close_array)}")
    
    # Calculate Typical Price (TP)
    typical_price = (high_array + low_array + close_array) / 3.0
    
    # Initialize result array
    cci = np.full(len(close_array), np.nan)
    
    # Calculate CCI
    for i in range(period - 1, len(close_array)):
        # Calculate Simple Moving Average of Typical Price
        sma_tp = np.mean(typical_price[i - period + 1:i + 1])
        
        # Calculate Mean Deviation
        mean_deviation = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp))
        
        # Calculate CCI
        if mean_deviation == 0:
            cci[i] = 0.0
        else:
            cci[i] = (typical_price[i] - sma_tp) / (0.015 * mean_deviation)
    
    return cci


def calculate_momentum(data: Union[np.ndarray, list], period: int = 10) -> np.ndarray:
    """
    Calculate Price Momentum.
    
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
        raise ValueError(f"Insufficient data: need {period + 1} points, got {len(data_array)}")
    
    # Initialize result array
    momentum = np.full(len(data_array), np.nan)
    
    # Calculate momentum
    for i in range(period, len(data_array)):
        momentum[i] = data_array[i] - data_array[i - period]
    
    return momentum


def calculate_roc(data: Union[np.ndarray, list], period: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change (ROC).
    
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
        raise ValueError(f"Insufficient data: need {period + 1} points, got {len(data_array)}")
    
    # Initialize result array
    roc = np.full(len(data_array), np.nan)
    
    # Calculate ROC
    for i in range(period, len(data_array)):
        if data_array[i - period] != 0:
            roc[i] = ((data_array[i] - data_array[i - period]) / data_array[i - period]) * 100.0
        else:
            roc[i] = np.nan
    
    return roc