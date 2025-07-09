#!/usr/bin/env python3
"""
Additional mathematical tests for edge cases and validation.
"""

import numpy as np
from src.indicators.moving_averages import calculate_sma, calculate_ema
from src.indicators.momentum import calculate_rsi, calculate_macd
from src.indicators.volatility import calculate_bollinger_bands, calculate_atr

def test_edge_cases():
    """Test mathematical edge cases."""
    print("Testing Mathematical Edge Cases...")
    
    # Test 1: Constant prices (no volatility)
    constant_prices = np.array([100.0] * 50)
    print("\n1. Constant Prices Test:")
    
    rsi_constant = calculate_rsi(constant_prices, 14)
    print(f"   RSI for constant prices: {rsi_constant[20]:.1f} (should be ~50)")
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(constant_prices, 20, 2.0)
    print(f"   Bollinger Band width: {bb_upper[30] - bb_lower[30]:.6f} (should be ~0)")
    
    # Test 2: Perfect uptrend
    uptrend_prices = np.array([100 + i for i in range(50)])
    print("\n2. Perfect Uptrend Test:")
    
    rsi_up = calculate_rsi(uptrend_prices, 14)
    print(f"   RSI for uptrend: {rsi_up[-1]:.1f} (should be high)")
    
    sma_up = calculate_sma(uptrend_prices, 10)
    ema_up = calculate_ema(uptrend_prices, 10)
    print(f"   EMA > SMA for uptrend: {ema_up[-1] > sma_up[-1]} (should be True)")
    
    # Test 3: Perfect downtrend
    downtrend_prices = np.array([150 - i for i in range(50)])
    print("\n3. Perfect Downtrend Test:")
    
    rsi_down = calculate_rsi(downtrend_prices, 14)
    print(f"   RSI for downtrend: {rsi_down[-1]:.1f} (should be low)")
    
    # Test 4: High volatility scenario
    np.random.seed(42)
    volatile_prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.05, 100)))
    print("\n4. High Volatility Test:")
    
    bb_upper_vol, bb_middle_vol, bb_lower_vol = calculate_bollinger_bands(volatile_prices, 20, 2.0)
    bb_width = bb_upper_vol[-1] - bb_lower_vol[-1]
    print(f"   Bollinger Band width: {bb_width:.2f} (should be significant)")
    
    atr_vol = calculate_atr(volatile_prices, volatile_prices * 1.02, volatile_prices * 0.98, 14)
    print(f"   ATR: {atr_vol[-1]:.4f} (should be positive)")
    
    # Test 5: MACD crossover detection
    trend_change_prices = np.concatenate([
        100 + np.linspace(0, 10, 50),  # Uptrend
        110 - np.linspace(0, 20, 50)   # Downtrend
    ])
    print("\n5. MACD Trend Change Test:")
    
    macd_line, signal_line, histogram = calculate_macd(trend_change_prices, 12, 26, 9)
    
    # Find crossovers in histogram
    crossovers = 0
    for i in range(1, len(histogram)):
        if not np.isnan(histogram[i]) and not np.isnan(histogram[i-1]):
            if (histogram[i-1] >= 0 and histogram[i] < 0) or (histogram[i-1] <= 0 and histogram[i] > 0):
                crossovers += 1
    
    print(f"   MACD crossovers detected: {crossovers} (should be > 0)")
    
    print("\n✅ All edge case tests completed")

def test_mathematical_properties():
    """Test mathematical properties and relationships."""
    print("\nTesting Mathematical Properties...")
    
    # Generate test data
    np.random.seed(42)
    prices = 100 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 200)))
    
    # Test 1: Moving Average Ordering
    sma_10 = calculate_sma(prices, 10)
    sma_20 = calculate_sma(prices, 20)
    sma_50 = calculate_sma(prices, 50)
    
    # In trending markets, shorter MAs should be more responsive
    price_trend = np.mean(np.diff(prices[-20:]))  # Recent trend
    
    if price_trend > 0:  # Uptrend
        recent_ordering = sma_10[-1] >= sma_20[-1] >= sma_50[-1]
        print(f"1. Uptrend MA ordering (SMA10 ≥ SMA20 ≥ SMA50): {recent_ordering}")
    else:  # Downtrend
        recent_ordering = sma_10[-1] <= sma_20[-1] <= sma_50[-1]
        print(f"1. Downtrend MA ordering (SMA10 ≤ SMA20 ≤ SMA50): {recent_ordering}")
    
    # Test 2: RSI Symmetry Property
    rsi = calculate_rsi(prices, 14)
    
    # RSI should be roughly symmetric around 50 for random walk
    rsi_valid = rsi[~np.isnan(rsi)]
    rsi_mean = np.mean(rsi_valid)
    rsi_symmetry = abs(rsi_mean - 50) < 10  # Within 10 points of 50
    print(f"2. RSI symmetry around 50: {rsi_symmetry} (mean: {rsi_mean:.1f})")
    
    # Test 3: Bollinger Band Coverage
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices, 20, 2.0)
    
    valid_indices = ~(np.isnan(bb_upper) | np.isnan(bb_lower))
    prices_in_bands = np.sum((prices[valid_indices] >= bb_lower[valid_indices]) & 
                            (prices[valid_indices] <= bb_upper[valid_indices]))
    total_valid = np.sum(valid_indices)
    coverage_pct = (prices_in_bands / total_valid) * 100
    
    # ~95% of prices should be within 2-sigma bands
    coverage_good = 90 <= coverage_pct <= 98
    print(f"3. Bollinger Band coverage: {coverage_good} ({coverage_pct:.1f}% within bands)")
    
    # Test 4: MACD Signal Lag
    macd_line, signal_line, histogram = calculate_macd(prices, 12, 26, 9)
    
    # Signal line should lag MACD line
    valid_indices = ~(np.isnan(macd_line) | np.isnan(signal_line))
    if np.sum(valid_indices) > 10:
        # Calculate correlation with lag
        macd_changes = np.diff(macd_line[valid_indices])
        signal_changes = np.diff(signal_line[valid_indices])
        
        if len(macd_changes) > 1 and len(signal_changes) > 1:
            correlation = np.corrcoef(macd_changes[:-1], signal_changes[1:])[0, 1]
            signal_lags = correlation > 0.7  # Strong positive correlation with lag
            print(f"4. MACD signal lag property: {signal_lags} (correlation: {correlation:.3f})")
        else:
            print("4. MACD signal lag property: Insufficient data")
    
    print("\n✅ Mathematical property tests completed")

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\nTesting Numerical Stability...")
    
    # Test 1: Very large numbers
    large_prices = np.array([1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3, 1e6 + 4] * 20)
    sma_large = calculate_sma(large_prices, 5)
    stable_large = not np.any(np.isnan(sma_large[4:]))
    print(f"1. Large number stability: {stable_large}")
    
    # Test 2: Very small numbers
    small_prices = np.array([1e-6, 1.1e-6, 1.2e-6, 1.3e-6, 1.4e-6] * 20)
    sma_small = calculate_sma(small_prices, 5)
    stable_small = not np.any(np.isnan(sma_small[4:]))
    print(f"2. Small number stability: {stable_small}")
    
    # Test 3: Mixed precision
    mixed_prices = np.array([100.123456789, 100.234567890, 100.345678901, 100.456789012] * 25)
    rsi_mixed = calculate_rsi(mixed_prices, 14)
    stable_mixed = not np.any(np.isnan(rsi_mixed[14:]))
    print(f"3. Mixed precision stability: {stable_mixed}")
    
    print("\n✅ Numerical stability tests completed")

if __name__ == "__main__":
    test_edge_cases()
    test_mathematical_properties()
    test_numerical_stability()
    print("\n🎉 All additional mathematical tests completed successfully!")