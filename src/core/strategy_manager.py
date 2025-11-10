"""
StrategyManager Module

Manages dual-mode trading strategies:
1. Trend Strategy: Volatility Breakout
2. Range Strategy: RSI Mean Reversion
"""

import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime


class StrategyManager:
    """Manages trading strategies and generates signals"""
    
    def __init__(self, config: Dict, market_analyzer):
        """
        Initialize StrategyManager
        
        Args:
            config: Configuration dictionary with strategy parameters
            market_analyzer: MarketAnalyzer instance
        """
        self.config = config
        self.market_analyzer = market_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Trend strategy parameters
        self.k_value = config.get('trend', {}).get('k_value', 0.5)
        
        # Range strategy parameters
        self.rsi_period = config.get('range', {}).get('rsi_period', 14)
        self.rsi_oversold = config.get('range', {}).get('rsi_oversold', 30)
        self.rsi_overbought = config.get('range', {}).get('rsi_overbought', 70)
        
        self.current_strategy = None
        self.last_strategy_change = None
    
    def trend_strategy_signal(self, df: pd.DataFrame, current_price: float) -> Tuple[str, Dict]:
        """
        Volatility Breakout Strategy
        
        Buy when price breaks above (yesterday's high + k * (high - low))
        Sell at market close or when position becomes unfavorable
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            
        Returns:
            Tuple of (signal, details) where signal is 'buy', 'sell', or 'hold'
        """
        try:
            # Need at least 2 days of data
            if len(df) < 2:
                return 'hold', {'reason': 'Insufficient data'}
            
            # Get yesterday's data
            yesterday = df.iloc[-2]
            yesterday_high = yesterday['high']
            yesterday_low = yesterday['low']
            yesterday_close = yesterday['close']
            
            # Calculate breakout price
            volatility_range = yesterday_high - yesterday_low
            target_price = yesterday_close + (self.k_value * volatility_range)
            
            details = {
                'strategy': 'trend',
                'target_price': target_price,
                'current_price': current_price,
                'yesterday_high': yesterday_high,
                'yesterday_low': yesterday_low,
                'k_value': self.k_value
            }
            
            # Buy signal if price breaks above target
            if current_price >= target_price:
                self.logger.info(f"Trend Strategy: BUY signal at {current_price} (target: {target_price})")
                return 'buy', details
            else:
                return 'hold', details
                
        except Exception as e:
            self.logger.error(f"Error in trend strategy: {e}")
            return 'hold', {'error': str(e)}
    
    def range_strategy_signal(self, df: pd.DataFrame, current_price: float, has_position: bool) -> Tuple[str, Dict]:
        """
        RSI Mean Reversion Strategy
        
        Buy when RSI < oversold threshold
        Sell when RSI > overbought threshold
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            has_position: Whether currently holding a position
            
        Returns:
            Tuple of (signal, details) where signal is 'buy', 'sell', or 'hold'
        """
        try:
            # Calculate RSI
            rsi = self.market_analyzer.calculate_rsi(df, self.rsi_period)
            
            if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
                return 'hold', {'reason': 'Insufficient data for RSI'}
            
            current_rsi = rsi.iloc[-1]
            
            details = {
                'strategy': 'range',
                'current_rsi': current_rsi,
                'current_price': current_price,
                'oversold': self.rsi_oversold,
                'overbought': self.rsi_overbought
            }
            
            # Generate signals based on RSI
            if current_rsi < self.rsi_oversold and not has_position:
                self.logger.info(f"Range Strategy: BUY signal at RSI {current_rsi:.2f}")
                return 'buy', details
            elif current_rsi > self.rsi_overbought and has_position:
                self.logger.info(f"Range Strategy: SELL signal at RSI {current_rsi:.2f}")
                return 'sell', details
            else:
                return 'hold', details
                
        except Exception as e:
            self.logger.error(f"Error in range strategy: {e}")
            return 'hold', {'error': str(e)}
    
    def get_trading_signal(self, df: pd.DataFrame, current_price: float, has_position: bool) -> Tuple[str, Dict]:
        """
        Get trading signal based on current market condition
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            has_position: Whether currently holding a position
            
        Returns:
            Tuple of (signal, details) where signal is 'buy', 'sell', or 'hold'
        """
        # Determine market condition
        market_condition = self.market_analyzer.analyze_market_condition(df)
        
        # Track strategy changes
        if self.current_strategy != market_condition:
            self.logger.info(f"Strategy switch: {self.current_strategy} -> {market_condition}")
            self.current_strategy = market_condition
            self.last_strategy_change = datetime.now()
        
        # Execute appropriate strategy
        if market_condition == 'trend':
            return self.trend_strategy_signal(df, current_price)
        else:  # range
            return self.range_strategy_signal(df, current_price, has_position)
    
    def get_current_strategy(self) -> Optional[str]:
        """Get the currently active strategy"""
        return self.current_strategy
