"""
ETH Mean Reversion Engine
Specialized for mean-reversion behavior using RSI + Bollinger Bands
Ignores EMA signals, focuses on range-bound trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

class ETHMeanReversionEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # ETH-specific parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.max_exposure = self.config.get('max_exposure', 45.0)
        
        # RSI thresholds
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        # Regime-based exposure settings (RANGE focus)
        self.regime_exposures = {
            'range': 25.0,          # Primary focus - range trading
            'neutral': 15.0,        # Secondary focus
            'trend_up': 8.0,        # Limited exposure in trends
            'trend_down': 3.0       # Minimal exposure in downtrends
        }
        
        # Mean reversion signal strengths
        self.signal_strength = self.config.get('signal_strength', 10.0)  # ±10% per signal
        
        self.logger = logging.getLogger(f'{__class__.__name__}')
        self.logger.info(f"ETH Range-Only Mean Reversion Engine initialized: RSI({self.rsi_period}), BB({self.bb_period},{self.bb_std})")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ETH-specific mean reversion indicators"""
        data = df.copy()
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=self.bb_period).mean()
        bb_std = data['close'].rolling(window=self.bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * self.bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std * self.bb_std)
        
        # BB position (where price is within bands)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Price distance from BB bands
        data['distance_upper'] = (data['bb_upper'] - data['close']) / data['close']
        data['distance_lower'] = (data['close'] - data['bb_lower']) / data['close']
        
        # Volatility measures
        data['volatility'] = data['close'].rolling(window=10).std() / data['close'].rolling(window=10).mean()
        
        # Price momentum (for mean reversion detection)
        data['momentum_2'] = data['close'].pct_change(2)
        data['momentum_5'] = data['close'].pct_change(5)
        
        return data
    
    def get_regime_signal(self, market_mode: str, regime_data: Dict) -> float:
        """Get regime-based base exposure (RANGE focus)"""
        base_exposure = self.regime_exposures.get(market_mode, 5.0)
        
        # Adjust based on market conditions
        volatility = regime_data.get('volatility', 0.05)
        
        if market_mode == 'range':
            # Increase exposure in stable range markets
            if volatility < 0.03:  # Low volatility = good for mean reversion
                base_exposure *= 1.3
            elif volatility < 0.05:
                base_exposure *= 1.1
        elif market_mode == 'trend_up':
            # Very limited exposure in uptrends
            base_exposure = min(base_exposure, 25.0)  # Cap at 25%
        
        return min(base_exposure, self.max_exposure)
    
    def get_mean_reversion_signal(self, row: pd.Series) -> Tuple[str, float]:
        """Get strict RSI + Bollinger Bands mean reversion signal - RANGE ONLY"""
        rsi = row['rsi']
        bb_position = row['bb_position']
        close = row['close']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        volatility = row['volatility']
        momentum_2 = row['momentum_2']
        
        signal = "HOLD"
        signal_strength = 0.0
        
        # STRICT CONDITIONS: Both RSI AND Bollinger Band must trigger
        
        # Strong Buy: RSI <= 30 AND price touches lower band
        if rsi <= 30 and close <= bb_lower:
            signal = "BUY"
            signal_strength = 10.0  # Fixed +10% as requested
            
        # Strong Sell: RSI >= 70 AND price touches upper band
        elif rsi >= 70 and close >= bb_upper:
            signal = "SELL"
            signal_strength = -10.0  # Fixed -10% as requested
            
        # No other conditions - only extreme range reversals
        
        return signal, signal_strength
    
    def generate_signal(self, df: pd.DataFrame, idx: int, market_mode: str, regime_data: Dict, current_dd: float = 0.0) -> Tuple[float, str, Dict]:
        """Generate range-only ETH mean reversion signal"""
        
        if idx < max(self.rsi_period, self.bb_period):
            return 0.0, "HOLD", {"regime": 0.0, "mean_reversion": 0.0, "engines": []}
        
        # RANGE ONLY: Return 0 exposure if not in range mode
        if market_mode != "range":
            return 0.0, "HOLD", {"regime": 0.0, "mean_reversion": 0.0, "engines": [f"Range-only mode: {market_mode} -> HOLD"]}
        
        row = df.iloc[idx]
        
        # 1. Get mean reversion signal (only active in range mode)
        mr_signal, mr_strength = self.get_mean_reversion_signal(row)
        
        # 2. RANGE ONLY: Simple exposure calculation
        # Base exposure is 0, only add/subtract on strong signals
        total_exposure = 0.0
        
        if mr_signal == "BUY":
            total_exposure = 10.0  # +10% on RSI<=30 & BB lower touch
        elif mr_signal == "SELL":
            total_exposure = -10.0  # -10% on RSI>=70 & BB upper touch
            
        # 3. Apply drawdown adjustment (reduce risk during losses)
        if current_dd > 10.0:
            total_exposure *= 0.5  # Halve exposure during drawdown
        elif current_dd > 5.0:
            total_exposure *= 0.8  # Reduce exposure during minor drawdown
        
        # 4. Ensure exposure is within bounds
        total_exposure = max(-self.max_exposure, min(total_exposure, self.max_exposure))
        
        detail = {
            "regime": 0.0,  # No base regime exposure
            "mean_reversion": mr_strength,
            "engines": ["range_only_mean_reversion"] if abs(mr_strength) > 0 else ["range_hold"],
            "rsi": row['rsi'],
            "bb_position": row['bb_position'],
            "market_mode": market_mode
        }
        
        signal_type = mr_signal  # Direct signal pass-through
        
        self.logger.debug(f"ETH Range Signal: {signal_type} {total_exposure:.1f}% | Mode: {market_mode} | RSI: {row['rsi']:.1f} | BB: {row['bb_position']:.2f}")
        
        return total_exposure, signal_type, detail
    
    def get_strategy_info(self) -> Dict:
        """Return strategy configuration info"""
        return {
            "name": "ETH Mean Reversion",
            "type": "mean_reversion", 
            "parameters": {
                "rsi_period": self.rsi_period,
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "max_exposure": self.max_exposure,
                "regime_exposures": self.regime_exposures,
                "signal_strength": self.signal_strength
            }
        }