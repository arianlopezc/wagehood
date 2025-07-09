"""
Support and Resistance Levels

This module implements algorithms for detecting support and resistance levels,
pivot points, and Fibonacci retracements in price data.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from scipy.signal import argrelextrema
from ..core.constants import MIN_DATA_POINTS


def calculate_support_resistance(
    data: Union[np.ndarray, list],
    lookback: int = 20,
    min_touches: int = 3,
    tolerance: float = 0.001,
) -> Dict[str, List[float]]:
    """
    Calculate Support and Resistance levels.

    Identifies significant price levels where the price has repeatedly bounced off,
    indicating potential support (price floor) or resistance (price ceiling) levels.

    Args:
        data: Price data (numpy array or list)
        lookback: Period for finding local extrema (default 20)
        min_touches: Minimum number of touches required for a level (default 3)
        tolerance: Price tolerance for grouping similar levels (default 0.1%)

    Returns:
        dict: Dictionary containing 'support' and 'resistance' lists of levels

    Raises:
        ValueError: If parameters are invalid or insufficient data
        TypeError: If data cannot be converted to numpy array
    """
    if lookback < 1:
        raise ValueError("Lookback period must be at least 1")

    if min_touches < 2:
        raise ValueError("Minimum touches must be at least 2")

    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < lookback * 2:
        raise ValueError(
            f"Insufficient data: need at least {lookback * 2} points, got {len(data_array)}"
        )

    # Find local minima and maxima
    local_minima_idx = argrelextrema(data_array, np.less, order=lookback)[0]
    local_maxima_idx = argrelextrema(data_array, np.greater, order=lookback)[0]

    # Get the actual price levels
    support_candidates = data_array[local_minima_idx]
    resistance_candidates = data_array[local_maxima_idx]

    # Group similar levels and count touches
    support_levels = _group_and_count_levels(support_candidates, tolerance, min_touches)
    resistance_levels = _group_and_count_levels(
        resistance_candidates, tolerance, min_touches
    )

    return {"support": support_levels, "resistance": resistance_levels}


def _group_and_count_levels(
    levels: np.ndarray, tolerance: float, min_touches: int
) -> List[float]:
    """
    Group similar price levels and filter by minimum touches.

    Args:
        levels: Array of price levels
        tolerance: Price tolerance for grouping
        min_touches: Minimum number of touches required

    Returns:
        List of significant levels
    """
    if len(levels) == 0:
        return []

    # Sort levels
    sorted_levels = np.sort(levels)

    # Group similar levels
    groups = []
    current_group = [sorted_levels[0]]

    for i in range(1, len(sorted_levels)):
        # Check if current level is within tolerance of the group
        if (
            abs(sorted_levels[i] - np.mean(current_group)) / np.mean(current_group)
            <= tolerance
        ):
            current_group.append(sorted_levels[i])
        else:
            # Finish current group and start new one
            groups.append(current_group)
            current_group = [sorted_levels[i]]

    # Add the last group
    groups.append(current_group)

    # Filter groups by minimum touches and calculate average level
    significant_levels = []
    for group in groups:
        if len(group) >= min_touches:
            significant_levels.append(np.mean(group))

    return significant_levels


def calculate_pivot_points(
    high: Union[np.ndarray, list],
    low: Union[np.ndarray, list],
    close: Union[np.ndarray, list],
    method: str = "standard",
) -> Dict[str, float]:
    """
    Calculate Pivot Points.

    Pivot points are technical analysis indicators used to determine the overall trend
    and potential support/resistance levels.

    Args:
        high: High prices (numpy array or list)
        low: Low prices (numpy array or list)
        close: Close prices (numpy array or list)
        method: Calculation method ('standard', 'fibonacci', 'woodie', 'camarilla')

    Returns:
        dict: Dictionary containing pivot point and support/resistance levels

    Raises:
        ValueError: If arrays have different lengths or method is invalid
        TypeError: If data cannot be converted to numpy arrays
    """
    valid_methods = ["standard", "fibonacci", "woodie", "camarilla"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of: {valid_methods}")

    try:
        high_array = np.asarray(high, dtype=float)
        low_array = np.asarray(low, dtype=float)
        close_array = np.asarray(close, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy arrays of floats")

    if not (len(high_array) == len(low_array) == len(close_array)):
        raise ValueError("High, low, and close arrays must have the same length")

    if len(high_array) == 0:
        raise ValueError("Arrays cannot be empty")

    # Use the most recent period for calculation
    high_val = high_array[-1]
    low_val = low_array[-1]
    close_val = close_array[-1]

    # Calculate pivot point based on method
    if method == "standard":
        return _calculate_standard_pivots(high_val, low_val, close_val)
    elif method == "fibonacci":
        return _calculate_fibonacci_pivots(high_val, low_val, close_val)
    elif method == "woodie":
        return _calculate_woodie_pivots(high_val, low_val, close_val)
    elif method == "camarilla":
        return _calculate_camarilla_pivots(high_val, low_val, close_val)


def _calculate_standard_pivots(
    high: float, low: float, close: float
) -> Dict[str, float]:
    """Calculate standard pivot points."""
    pivot = (high + low + close) / 3

    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "r2": pivot + (high - low),
        "r3": high + 2 * (pivot - low),
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
        "s3": low - 2 * (high - pivot),
    }


def _calculate_fibonacci_pivots(
    high: float, low: float, close: float
) -> Dict[str, float]:
    """Calculate Fibonacci pivot points."""
    pivot = (high + low + close) / 3
    range_val = high - low

    return {
        "pivot": pivot,
        "r1": pivot + 0.382 * range_val,
        "r2": pivot + 0.618 * range_val,
        "r3": pivot + range_val,
        "s1": pivot - 0.382 * range_val,
        "s2": pivot - 0.618 * range_val,
        "s3": pivot - range_val,
    }


def _calculate_woodie_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate Woodie's pivot points."""
    pivot = (high + low + 2 * close) / 4

    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "r2": pivot + high - low,
        "r3": high + 2 * (pivot - low),
        "s1": 2 * pivot - high,
        "s2": pivot - high + low,
        "s3": low - 2 * (high - pivot),
    }


def _calculate_camarilla_pivots(
    high: float, low: float, close: float
) -> Dict[str, float]:
    """Calculate Camarilla pivot points."""
    range_val = high - low

    return {
        "pivot": close,
        "r1": close + range_val * 1.1 / 12,
        "r2": close + range_val * 1.1 / 6,
        "r3": close + range_val * 1.1 / 4,
        "r4": close + range_val * 1.1 / 2,
        "s1": close - range_val * 1.1 / 12,
        "s2": close - range_val * 1.1 / 6,
        "s3": close - range_val * 1.1 / 4,
        "s4": close - range_val * 1.1 / 2,
    }


def calculate_fibonacci_retracements(
    high: float, low: float, trend: str = "up"
) -> Dict[str, float]:
    """
    Calculate Fibonacci Retracement levels.

    Fibonacci retracements are horizontal lines that indicate potential support
    and resistance levels based on Fibonacci ratios.

    Args:
        high: High price of the move
        low: Low price of the move
        trend: Direction of the trend ('up' or 'down')

    Returns:
        dict: Dictionary containing Fibonacci retracement levels

    Raises:
        ValueError: If trend is invalid or high <= low
    """
    if trend not in ["up", "down"]:
        raise ValueError("Trend must be 'up' or 'down'")

    if high <= low:
        raise ValueError("High must be greater than low")

    range_val = high - low

    # Fibonacci ratios
    fib_ratios = {
        "0.0": 0.0,
        "23.6": 0.236,
        "38.2": 0.382,
        "50.0": 0.5,
        "61.8": 0.618,
        "78.6": 0.786,
        "100.0": 1.0,
    }

    levels = {}

    if trend == "up":
        # For uptrend, retracements are calculated from high
        levels["100.0"] = high
        levels["78.6"] = high - range_val * fib_ratios["78.6"]
        levels["61.8"] = high - range_val * fib_ratios["61.8"]
        levels["50.0"] = high - range_val * fib_ratios["50.0"]
        levels["38.2"] = high - range_val * fib_ratios["38.2"]
        levels["23.6"] = high - range_val * fib_ratios["23.6"]
        levels["0.0"] = low
    else:
        # For downtrend, retracements are calculated from low
        levels["0.0"] = high
        levels["23.6"] = low + range_val * fib_ratios["23.6"]
        levels["38.2"] = low + range_val * fib_ratios["38.2"]
        levels["50.0"] = low + range_val * fib_ratios["50.0"]
        levels["61.8"] = low + range_val * fib_ratios["61.8"]
        levels["78.6"] = low + range_val * fib_ratios["78.6"]
        levels["100.0"] = low

    return levels


def calculate_fibonacci_extensions(
    high: float, low: float, retrace: float, trend: str = "up"
) -> Dict[str, float]:
    """
    Calculate Fibonacci Extension levels.

    Fibonacci extensions are used to project potential price targets beyond
    the original move.

    Args:
        high: High price of the initial move
        low: Low price of the initial move
        retrace: Retracement level
        trend: Direction of the trend ('up' or 'down')

    Returns:
        dict: Dictionary containing Fibonacci extension levels

    Raises:
        ValueError: If trend is invalid or parameters are inconsistent
    """
    if trend not in ["up", "down"]:
        raise ValueError("Trend must be 'up' or 'down'")

    if high <= low:
        raise ValueError("High must be greater than low")

    if trend == "up" and (retrace < low or retrace > high):
        raise ValueError("Retrace must be between low and high for uptrend")

    if trend == "down" and (retrace > high or retrace < low):
        raise ValueError("Retrace must be between low and high for downtrend")

    # Calculate the original range and retracement range
    original_range = high - low

    # Extension ratios
    ext_ratios = {
        "127.2": 1.272,
        "161.8": 1.618,
        "200.0": 2.0,
        "261.8": 2.618,
        "361.8": 3.618,
    }

    levels = {}

    if trend == "up":
        # For uptrend, extensions are calculated from retracement level
        for ratio_name, ratio in ext_ratios.items():
            levels[ratio_name] = retrace + original_range * ratio
    else:
        # For downtrend, extensions are calculated from retracement level
        for ratio_name, ratio in ext_ratios.items():
            levels[ratio_name] = retrace - original_range * ratio

    return levels


def detect_breakouts(
    data: Union[np.ndarray, list],
    levels: List[float],
    tolerance: float = 0.001,
    min_volume_ratio: float = 1.5,
    volume: Optional[Union[np.ndarray, list]] = None,
) -> List[Dict]:
    """
    Detect breakouts from support/resistance levels.

    Args:
        data: Price data (numpy array or list)
        levels: List of support/resistance levels
        tolerance: Price tolerance for level breach (default 0.1%)
        min_volume_ratio: Minimum volume ratio for confirmation (default 1.5)
        volume: Volume data for confirmation (optional)

    Returns:
        List of breakout events with details

    Raises:
        ValueError: If parameters are invalid
        TypeError: If data cannot be converted to numpy array
    """
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative")

    if min_volume_ratio < 0:
        raise ValueError("Minimum volume ratio must be non-negative")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < 2:
        raise ValueError("Need at least 2 data points")

    if volume is not None:
        try:
            volume_array = np.asarray(volume, dtype=float)
        except (ValueError, TypeError):
            raise TypeError("Volume must be convertible to numpy array of floats")

        if len(volume_array) != len(data_array):
            raise ValueError("Volume and price arrays must have the same length")
    else:
        volume_array = None

    breakouts = []

    for level in levels:
        for i in range(1, len(data_array)):
            # Check for breakout
            prev_price = data_array[i - 1]
            curr_price = data_array[i]

            # Determine if breakout occurred
            breakout_type = None

            if prev_price <= level and curr_price > level * (1 + tolerance):
                breakout_type = "bullish"
            elif prev_price >= level and curr_price < level * (1 - tolerance):
                breakout_type = "bearish"

            if breakout_type:
                # Check volume confirmation if available
                volume_confirmed = True
                if volume_array is not None and i > 0:
                    avg_volume = np.mean(volume_array[max(0, i - 10) : i])
                    if avg_volume > 0:
                        volume_ratio = volume_array[i] / avg_volume
                        volume_confirmed = volume_ratio >= min_volume_ratio

                breakout = {
                    "index": i,
                    "price": curr_price,
                    "level": level,
                    "type": breakout_type,
                    "volume_confirmed": volume_confirmed,
                }

                if volume_array is not None:
                    breakout["volume"] = volume_array[i]

                breakouts.append(breakout)

    return breakouts


def calculate_trend_lines(
    data: Union[np.ndarray, list], window: int = 20, min_points: int = 3
) -> Dict[str, List[Dict]]:
    """
    Calculate trend lines from price data.

    Args:
        data: Price data (numpy array or list)
        window: Window size for trend line calculation (default 20)
        min_points: Minimum points required for trend line (default 3)

    Returns:
        dict: Dictionary containing 'uptrend' and 'downtrend' lines

    Raises:
        ValueError: If parameters are invalid
        TypeError: If data cannot be converted to numpy array
    """
    if window < min_points:
        raise ValueError("Window must be at least as large as min_points")

    if min_points < 2:
        raise ValueError("Minimum points must be at least 2")

    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Data must be convertible to numpy array of floats")

    if len(data_array) < window:
        raise ValueError(
            f"Insufficient data: need {window} points, got {len(data_array)}"
        )

    uptrend_lines = []
    downtrend_lines = []

    # Find local extrema
    local_minima_idx = argrelextrema(data_array, np.less, order=5)[0]
    local_maxima_idx = argrelextrema(data_array, np.greater, order=5)[0]

    # Calculate uptrend lines from local minima
    for i in range(len(local_minima_idx) - min_points + 1):
        points_idx = local_minima_idx[i : i + min_points]
        points_price = data_array[points_idx]

        # Fit line
        slope, intercept = np.polyfit(points_idx, points_price, 1)

        # Check if it's actually an uptrend
        if slope > 0:
            uptrend_lines.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "start_idx": points_idx[0],
                    "end_idx": points_idx[-1],
                    "r_squared": _calculate_r_squared(
                        points_idx, points_price, slope, intercept
                    ),
                }
            )

    # Calculate downtrend lines from local maxima
    for i in range(len(local_maxima_idx) - min_points + 1):
        points_idx = local_maxima_idx[i : i + min_points]
        points_price = data_array[points_idx]

        # Fit line
        slope, intercept = np.polyfit(points_idx, points_price, 1)

        # Check if it's actually a downtrend
        if slope < 0:
            downtrend_lines.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "start_idx": points_idx[0],
                    "end_idx": points_idx[-1],
                    "r_squared": _calculate_r_squared(
                        points_idx, points_price, slope, intercept
                    ),
                }
            )

    return {"uptrend": uptrend_lines, "downtrend": downtrend_lines}


def _calculate_r_squared(
    x: np.ndarray, y: np.ndarray, slope: float, intercept: float
) -> float:
    """Calculate R-squared for trend line quality."""
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - (ss_res / ss_tot)
