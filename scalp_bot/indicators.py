"""
Technical indicators for scalp bot

Helper functions that work on pandas DataFrame with OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Union


def compute_pct_change(df: pd.DataFrame, window_bars: int, 
                      price_col: str = 'close') -> pd.Series:
    """
    Compute percentage change over a rolling window
    
    Args:
        df: DataFrame with OHLCV data
        window_bars: Number of bars to look back
        price_col: Column to use for calculation
    
    Returns:
        Series with percentage changes
    """
    if len(df) < window_bars:
        return pd.Series([0.0] * len(df), index=df.index)
    
    current_price = df[price_col]
    past_price = df[price_col].shift(window_bars)
    
    pct_change = ((current_price - past_price) / past_price * 100).fillna(0.0)
    return pct_change


def compute_rolling_low(df: pd.DataFrame, window_bars: int,
                       price_col: str = 'low') -> pd.Series:
    """
    Compute rolling minimum over a window
    
    Args:
        df: DataFrame with OHLCV data  
        window_bars: Number of bars for rolling window
        price_col: Column to use for calculation
        
    Returns:
        Series with rolling minimums
    """
    return df[price_col].rolling(window=window_bars, min_periods=1).min()


def compute_rolling_high(df: pd.DataFrame, window_bars: int,
                        price_col: str = 'high') -> pd.Series:
    """
    Compute rolling maximum over a window
    
    Args:
        df: DataFrame with OHLCV data
        window_bars: Number of bars for rolling window  
        price_col: Column to use for calculation
        
    Returns:
        Series with rolling maximums
    """
    return df[price_col].rolling(window=window_bars, min_periods=1).max()


def compute_atr(df: pd.DataFrame, window_bars: int = 14) -> pd.Series:
    """
    Compute Average True Range
    
    Args:
        df: DataFrame with OHLCV data (must have high, low, close)
        window_bars: Period for ATR calculation
        
    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low'] 
    close = df['close']
    prev_close = close.shift(1)
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Average True Range
    atr = true_range.rolling(window=window_bars, min_periods=1).mean()
    return atr.fillna(0.0)


def compute_volume_ratio(df: pd.DataFrame, window_bars: int = 20,
                        volume_col: str = 'volume') -> pd.Series:
    """
    Compute current volume / rolling mean volume ratio
    
    Args:
        df: DataFrame with OHLCV data
        window_bars: Period for volume average
        volume_col: Column name for volume
        
    Returns:
        Series with volume ratios
    """
    current_volume = df[volume_col]
    avg_volume = df[volume_col].rolling(window=window_bars, min_periods=1).mean()
    
    # Avoid division by zero
    volume_ratio = np.where(avg_volume > 0, 
                           current_volume / avg_volume, 
                           1.0)
    
    return pd.Series(volume_ratio, index=df.index).fillna(1.0)


def compute_rsi(df: pd.DataFrame, window_bars: int = 14,
               price_col: str = 'close') -> pd.Series:
    """
    Compute Relative Strength Index
    
    Args:
        df: DataFrame with OHLCV data
        window_bars: Period for RSI calculation
        price_col: Column to use for calculation
        
    Returns:
        Series with RSI values (0-100)
    """
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window_bars, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_bars, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50.0)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()
    
    # Percentage changes
    df['pct_change_5m'] = compute_pct_change(df, 5)
    df['pct_change_15m'] = compute_pct_change(df, 15)
    df['pct_change_1m'] = compute_pct_change(df, 1)
    
    # Rolling extremes
    df['rolling_low_5m'] = compute_rolling_low(df, 5)
    df['rolling_high_5m'] = compute_rolling_high(df, 5)
    df['rolling_low_15m'] = compute_rolling_low(df, 15)
    df['rolling_high_15m'] = compute_rolling_high(df, 15)
    
    # Volatility and momentum
    df['atr'] = compute_atr(df, 14)
    df['volume_ratio'] = compute_volume_ratio(df, 20)
    df['rsi'] = compute_rsi(df, 14)
    
    return df