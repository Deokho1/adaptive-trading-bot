"""
MarketAnalyzer Module

Analyzes market conditions using technical indicators:
- ADX (Average Directional Index) for trend strength
- ATR (Average True Range) for volatility
- Bollinger Bands for price range
- RSI (Relative Strength Index) for momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging


class MarketAnalyzer:
    """Analyzes market data to determine trading conditions"""
    
    def __init__(self, config: Dict):
        """
        Initialize MarketAnalyzer
        
        Args:
            config: Configuration dictionary with market analysis parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.atr_period = config.get('atr_period', 14)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.rsi_period = config.get('rsi_period', 14)
    
    def calculate_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default from config)
            
        Returns:
            Series with RSI values
        """
        if period is None:
            period = self.rsi_period
            
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        middle = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with ADX values
        """
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx
    
    def analyze_market_condition(self, df: pd.DataFrame) -> str:
        """
        Determine if market is trending or ranging
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            'trend' or 'range' based on market conditions
        """
        try:
            # Calculate ADX
            adx = self.calculate_adx(df)
            current_adx = adx.iloc[-1]
            
            # Calculate Bollinger Bands
            bb = self.calculate_bollinger_bands(df)
            bb_width = (bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / bb['middle'].iloc[-1]
            
            # Calculate ATR
            atr = self.calculate_atr(df)
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(window=20).mean().iloc[-1]
            
            self.logger.info(f"Market Analysis - ADX: {current_adx:.2f}, BB Width: {bb_width:.4f}, ATR: {current_atr:.2f}")
            
            # Decision logic
            if current_adx > self.adx_threshold and current_atr > avg_atr * 0.8:
                self.logger.info("Market Condition: TREND (High ADX and ATR)")
                return 'trend'
            else:
                self.logger.info("Market Condition: RANGE (Low ADX or ATR)")
                return 'range'
                
        except Exception as e:
            self.logger.error(f"Error analyzing market condition: {e}")
            # Default to range strategy in case of error
            return 'range'
    
    def get_all_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {
            'rsi': self.calculate_rsi(df),
            'adx': self.calculate_adx(df),
            'atr': self.calculate_atr(df),
            'bollinger_bands': self.calculate_bollinger_bands(df)
        }
        
        return indicators
