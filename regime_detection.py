"""
Simple Regime Detection Module
Extracted from existing codebase for dual-core strategy
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple

class MarketMode(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    NEUTRAL = "neutral"

class RegimeDetector:
    """Simple regime detection for dual-core strategy"""
    
    def __init__(self):
        self.price_history = []
        self.volume_history = []
        
    def detect_regime(self, price: float, volume: float, index: int) -> Tuple[str, Dict]:
        """
        Detect current market regime
        
        Args:
            price: Current price
            volume: Current volume
            index: Current bar index
            
        Returns:
            Tuple of (market_mode, regime_data)
        """
        # Store history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only recent data
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
            self.volume_history = self.volume_history[-100:]
        
        # Need minimum data for analysis
        if len(self.price_history) < 20:
            return MarketMode.NEUTRAL.value, {
                'adx': 20,
                'trend_strength': 0.5,
                'volatility': 0.05
            }
        
        # Calculate simple trend indicators
        prices = np.array(self.price_history)
        
        # Simple moving averages for trend
        if len(prices) >= 20:
            sma_10 = np.mean(prices[-10:])
            sma_20 = np.mean(prices[-20:])
            
            # Price momentum
            price_change_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            price_change_20 = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
            
            # Volatility
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) >= 20 else 0.05
            
            # Simple ADX approximation
            price_changes = np.diff(prices[-14:]) if len(prices) >= 14 else np.array([0])
            adx = min(50, max(10, np.std(price_changes) * 1000))  # Rough ADX estimate
            
            # Trend strength
            trend_strength = abs(price_change_20) * 10  # Scale to 0-1 range
            trend_strength = min(1.0, max(0.0, trend_strength))
            
            # Regime classification
            if sma_10 > sma_20 * 1.005 and price_change_10 > 0.02:  # Strong uptrend
                market_mode = MarketMode.TREND_UP.value
            elif sma_10 < sma_20 * 0.995 and price_change_10 < -0.02:  # Strong downtrend
                market_mode = MarketMode.TREND_DOWN.value
            elif abs(price_change_10) < 0.01 and volatility < 0.03:  # Low volatility range
                market_mode = MarketMode.RANGE.value
            else:  # Everything else
                market_mode = MarketMode.NEUTRAL.value
                
        else:
            # Default values for insufficient data
            market_mode = MarketMode.NEUTRAL.value
            adx = 20
            trend_strength = 0.5
            volatility = 0.05
        
        regime_data = {
            'adx': adx,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'price_change_10': price_change_10 if len(prices) >= 10 else 0,
            'price_change_20': price_change_20 if len(prices) >= 20 else 0
        }
        
        return market_mode, regime_data