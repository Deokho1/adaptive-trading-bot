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
        self.logger.info(f"ETH Mean Reversion Engine initialized: RSI({self.rsi_period}), BB({self.bb_period},{self.bb_std})")
    
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
        """Get RSI + Bollinger Bands mean reversion signal"""
        rsi = row['rsi']
        bb_position = row['bb_position']
        close = row['close']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        volatility = row['volatility']
        momentum_2 = row['momentum_2']
        
        signal = "HOLD"
        signal_strength = 0.0
        
        # Oversold condition: RSI < 30 AND price <= lower band
        if rsi < self.rsi_oversold and close <= bb_lower:
            # Strong oversold signal
            oversold_strength = (self.rsi_oversold - rsi) / self.rsi_oversold  # 0-1 scale
            bb_breach = max(0, (bb_lower - close) / close)  # How far below lower band
            
            signal = "BUY"
            signal_strength = self.signal_strength * (1 + oversold_strength + bb_breach * 2)
            signal_strength = min(signal_strength, 20.0)  # Cap at +20%
            
        # Overbought condition: RSI > 70 AND price >= upper band
        elif rsi > self.rsi_overbought and close >= bb_upper:
            # Strong overbought signal
            overbought_strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            bb_breach = max(0, (close - bb_upper) / close)
            
            signal = "SELL"
            signal_strength = -self.signal_strength * (1 + overbought_strength + bb_breach * 2)
            signal_strength = max(signal_strength, -15.0)  # Cap at -15%
            
        # Moderate oversold: RSI < 35 OR price near lower band
        elif rsi < 35 or bb_position < 0.2:
            if momentum_2 < -0.02:  # Recent downward momentum
                signal = "BUY"
                signal_strength = self.signal_strength * 0.5  # Half strength
                
        # Moderate overbought: RSI > 65 OR price near upper band  
        elif rsi > 65 or bb_position > 0.8:
            if momentum_2 > 0.02:  # Recent upward momentum
                signal = "SELL" 
                signal_strength = -self.signal_strength * 0.4  # 40% strength
        
        # Adjust for volatility - reduce signals in high volatility
        if volatility > 0.07:
            signal_strength *= 0.7
        
        return signal, signal_strength
    
    def generate_signal(self, df: pd.DataFrame, idx: int, market_mode: str, regime_data: Dict, current_dd: float = 0.0) -> Tuple[float, str, Dict]:
        """Generate comprehensive ETH mean reversion signal"""
        
        if idx < max(self.rsi_period, self.bb_period):
            return 0.0, "HOLD", {"regime": 0.0, "mean_reversion": 0.0, "engines": []}
        
        row = df.iloc[idx]
        
        # 1. Get regime-based base exposure (RANGE focus)
        regime_exposure = self.get_regime_signal(market_mode, regime_data)
        
        # 2. Get mean reversion signal (RSI + BB)
        mr_signal, mr_strength = self.get_mean_reversion_signal(row)
        
        # 3. Combine signals based on market mode
        if market_mode == 'range':
            # Primary mode - use full mean reversion signals with cap
            total_exposure = min(regime_exposure + mr_strength, self.max_exposure)
        elif market_mode == 'neutral':
            # Secondary mode - moderate mean reversion with cap
            total_exposure = min(regime_exposure + (mr_strength * 0.7), self.max_exposure)
        elif market_mode == 'trend_up':
            # Limited mode - only strong oversold signals with cap
            if mr_signal == "BUY" and mr_strength > 8.0:
                total_exposure = min(regime_exposure + (mr_strength * 0.5), self.max_exposure)
            else:
                total_exposure = regime_exposure
        else:  # trend_down
            # Minimal mode - very conservative with cap
            if mr_signal == "BUY" and mr_strength > 12.0:
                total_exposure = min(regime_exposure + (mr_strength * 0.3), self.max_exposure)
            else:
                total_exposure = regime_exposure
        
        # 4. Apply drawdown adjustment
        if current_dd > 10.0:
            dd_factor = 0.5 if current_dd < 15.0 else 0.3
            total_exposure *= dd_factor
        elif current_dd > 5.0:
            total_exposure *= 0.8
        
        # 5. Cap exposure
        total_exposure = max(0.0, min(total_exposure, self.max_exposure))
        
        engines_used = ["regime"]
        if abs(mr_strength) > 2.0:
            engines_used.append("mean_reversion")
        
        detail = {
            "regime": regime_exposure,
            "mean_reversion": mr_strength,
            "engines": engines_used,
            "rsi": row['rsi'],
            "bb_position": row['bb_position']
        }
        
        signal_type = "BUY" if mr_signal == "BUY" and mr_strength > 5.0 else "HOLD"
        if mr_signal == "SELL" and mr_strength < -5.0:
            signal_type = "SELL"
        
        self.logger.debug(f"ETH Signal: {signal_type} {total_exposure:.1f}% | Regime: {regime_exposure:.1f}% + MR: {mr_strength:+.1f}% | RSI: {row['rsi']:.1f}")
        
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