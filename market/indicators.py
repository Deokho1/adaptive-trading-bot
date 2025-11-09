"""
Technical indicators for market analysis.

This module provides pure functions to compute standard technical indicators
from candle data without external dependencies.
"""

import math
from typing import Sequence
from exchange.models import Candle


def compute_atr(candles: Sequence[Candle], period: int = 14) -> list[float]:
    """
    Compute Average True Range (ATR) for a sequence of candles.
    
    Args:
        candles: Sequence of Candle objects
        period: Period for ATR calculation (default 14)
    
    Returns:
        List of ATR values aligned with candles (same length)
        First period-1 values will be 0.0 or simple average
    """
    if len(candles) < 2:
        return [0.0] * len(candles)
    
    true_ranges = []
    atr_values = []
    
    # Calculate True Range for each candle
    for i in range(len(candles)):
        if i == 0:
            # First candle: TR = high - low
            tr = candles[i].high - candles[i].low
        else:
            # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
            hl = candles[i].high - candles[i].low
            hc = abs(candles[i].high - candles[i-1].close)
            lc = abs(candles[i].low - candles[i-1].close)
            tr = max(hl, hc, lc)
        
        true_ranges.append(tr)
    
    # Calculate ATR
    for i in range(len(candles)):
        if i < period - 1:
            # Not enough data, use simple average of available TRs
            if i == 0:
                atr_values.append(true_ranges[0])
            else:
                avg_tr = sum(true_ranges[:i+1]) / (i + 1)
                atr_values.append(avg_tr)
        elif i == period - 1:
            # First ATR calculation: simple average
            atr = sum(true_ranges[:period]) / period
            atr_values.append(atr)
        else:
            # Smoothed ATR: (previous_atr * (period-1) + current_tr) / period
            prev_atr = atr_values[i-1]
            current_tr = true_ranges[i]
            atr = (prev_atr * (period - 1) + current_tr) / period
            atr_values.append(atr)
    
    return atr_values


def compute_adx(candles: Sequence[Candle], period: int = 14) -> list[float]:
    """
    Compute Average Directional Index (ADX) for trend strength.
    
    Args:
        candles: Sequence of Candle objects
        period: Period for ADX calculation (default 14)
    
    Returns:
        List of ADX values aligned with candles
    """
    if len(candles) < period + 1:
        return [0.0] * len(candles)
    
    # Calculate directional movements and true range
    plus_dm = []
    minus_dm = []
    true_ranges = []
    
    for i in range(len(candles)):
        if i == 0:
            plus_dm.append(0.0)
            minus_dm.append(0.0)
            true_ranges.append(candles[i].high - candles[i].low)
        else:
            # Directional movements
            up_move = candles[i].high - candles[i-1].high
            down_move = candles[i-1].low - candles[i].low
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0.0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0.0)
            
            # True range
            hl = candles[i].high - candles[i].low
            hc = abs(candles[i].high - candles[i-1].close)
            lc = abs(candles[i].low - candles[i-1].close)
            true_ranges.append(max(hl, hc, lc))
    
    # Smooth the values using Wilder's smoothing
    smoothed_plus_dm = _wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = _wilders_smoothing(minus_dm, period)
    smoothed_tr = _wilders_smoothing(true_ranges, period)
    
    # Calculate DI+ and DI-
    plus_di = []
    minus_di = []
    
    for i in range(len(candles)):
        if smoothed_tr[i] != 0:
            plus_di.append(100 * smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di.append(100 * smoothed_minus_dm[i] / smoothed_tr[i])
        else:
            plus_di.append(0.0)
            minus_di.append(0.0)
    
    # Calculate DX
    dx_values = []
    for i in range(len(candles)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            di_diff = abs(plus_di[i] - minus_di[i])
            dx = 100 * di_diff / di_sum
            dx_values.append(dx)
        else:
            dx_values.append(0.0)
    
    # Calculate ADX (smoothed DX)
    adx_values = _wilders_smoothing(dx_values, period)
    
    return adx_values


def compute_bollinger_bands(
    closes: Sequence[float], 
    period: int = 20, 
    num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute Bollinger Bands.
    
    Args:
        closes: Sequence of closing prices
        period: Period for moving average (default 20)
        num_std: Number of standard deviations (default 2.0)
    
    Returns:
        Tuple of (middle, upper, lower) band lists
    """
    if len(closes) < period:
        zeros = [0.0] * len(closes)
        return zeros, zeros, zeros
    
    middle = []
    upper = []
    lower = []
    
    for i in range(len(closes)):
        if i < period - 1:
            # Not enough data
            middle.append(closes[i])
            upper.append(closes[i])
            lower.append(closes[i])
        else:
            # Calculate SMA and standard deviation
            window = closes[i - period + 1:i + 1]
            sma = sum(window) / period
            
            # Calculate standard deviation
            variance = sum((x - sma) ** 2 for x in window) / period
            std_dev = math.sqrt(variance)
            
            middle.append(sma)
            upper.append(sma + num_std * std_dev)
            lower.append(sma - num_std * std_dev)
    
    return middle, upper, lower


def compute_rsi(closes: Sequence[float], period: int = 14) -> list[float]:
    """
    Compute Relative Strength Index (RSI) using Wilder's smoothing.
    
    Args:
        closes: Sequence of closing prices
        period: Period for RSI calculation (default 14)
    
    Returns:
        List of RSI values aligned with closes
    """
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    
    # Calculate price changes
    gains = []
    losses = []
    
    for i in range(len(closes)):
        if i == 0:
            gains.append(0.0)
            losses.append(0.0)
        else:
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
    
    # Calculate RSI
    rsi_values = []
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(len(closes)):
        if i < period:
            rsi_values.append(50.0)  # Default neutral value
            if i == period - 1:
                # First calculation: simple average
                avg_gain = sum(gains[:period]) / period
                avg_loss = sum(losses[:period]) / period
        else:
            # Wilder's smoothing
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
    
    return rsi_values


def _wilders_smoothing(values: list[float], period: int) -> list[float]:
    """
    Apply Wilder's smoothing to a list of values.
    
    Args:
        values: List of values to smooth
        period: Smoothing period
    
    Returns:
        List of smoothed values
    """
    if len(values) < period:
        return values[:]
    
    smoothed = []
    
    for i in range(len(values)):
        if i < period - 1:
            smoothed.append(values[i])
        elif i == period - 1:
            # First smoothed value: simple average
            avg = sum(values[:period]) / period
            smoothed.append(avg)
        else:
            # Wilder's smoothing: (previous_smooth * (period-1) + current_value) / period
            prev_smooth = smoothed[i-1]
            current_value = values[i]
            smooth = (prev_smooth * (period - 1) + current_value) / period
            smoothed.append(smooth)
    
    return smoothed