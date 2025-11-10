"""
BTC Trend Following Engine
Specialized for trend-following behavior with regime-based exposure control
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

class BTCTrendEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # BTC-specific parameters
        self.ema_fast = self.config.get('ema_fast', 15)
        self.ema_slow = self.config.get('ema_slow', 60)
        self.volume_threshold = self.config.get('volume_threshold', 1.20)
        self.max_exposure = self.config.get('max_exposure', 100.0)
        
        # Regime-based exposure settings
        self.regime_exposures = {
            'trend_up': 100.0,      # Full exposure in uptrends
            'trend_down': 5.0,      # Minimal exposure in downtrends
            'range': 20.0,          # Conservative in range
            'neutral': 10.0         # Very conservative in neutral
        }
        
        # Drawdown-based risk management
        self.dd_thresholds = {
            'low': (0.0, 5.0),      # DD < 5%: risk_factor = 1.0
            'medium': (5.0, 12.0),  # 5-12% DD: reduce exposure
            'high': (12.0, float('inf'))  # >12% DD: heavy reduction
        }
        
        self.logger = logging.getLogger(f'{__class__.__name__}')
        self.logger.info(f"BTC Trend Engine initialized: EMA({self.ema_fast},{self.ema_slow}), Vol:{self.volume_threshold}x")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BTC-specific technical indicators"""
        data = df.copy()
        
        # EMA crossover system
        data[f'ema_{self.ema_fast}'] = data['close'].ewm(span=self.ema_fast).mean()
        data[f'ema_{self.ema_slow}'] = data['close'].ewm(span=self.ema_slow).mean()
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Trend strength
        data['ema_diff'] = data[f'ema_{self.ema_fast}'] - data[f'ema_{self.ema_slow}']
        data['ema_diff_pct'] = (data['ema_diff'] / data['close']) * 100
        
        # Price momentum
        data['price_change'] = data['close'].pct_change()
        data['momentum_3'] = data['close'].pct_change(3)
        data['momentum_7'] = data['close'].pct_change(7)
        
        return data
    
    def get_regime_signal(self, market_mode: str, regime_data: Dict) -> float:
        """Get regime-based base exposure"""
        base_exposure = self.regime_exposures.get(market_mode, 10.0)
        
        # Adjust based on regime strength
        adx = regime_data.get('adx', 20)
        trend_strength = regime_data.get('trend_strength', 0.5)
        
        if market_mode == 'trend_up':
            # Increase exposure for strong uptrends
            if adx > 30 and trend_strength > 0.7:
                base_exposure *= 1.2
            elif adx > 25 and trend_strength > 0.5:
                base_exposure *= 1.1
        elif market_mode == 'trend_down':
            # Reduce exposure more in strong downtrends
            if adx > 30:
                base_exposure *= 0.5
        
        return min(base_exposure, self.max_exposure)
    
    def get_ema_signal(self, row: pd.Series) -> Tuple[str, float]:
        """Get EMA crossover signal"""
        ema_fast = row[f'ema_{self.ema_fast}']
        ema_slow = row[f'ema_{self.ema_slow}']
        volume_ratio = row['volume_ratio']
        price_change = row['price_change']
        
        # EMA crossover detection
        ema_diff = ema_fast - ema_slow
        ema_diff_pct = (ema_diff / row['close']) * 100
        
        signal = "HOLD"
        signal_strength = 0.0
        
        # Bullish crossover
        if ema_diff_pct > 0.1 and volume_ratio >= self.volume_threshold:
            if price_change > 0:
                signal = "BUY"
                signal_strength = min(abs(ema_diff_pct) * 2.0, 15.0)  # Max +15%
        
        # Bearish crossover
        elif ema_diff_pct < -0.1 and volume_ratio >= self.volume_threshold:
            if price_change < 0:
                signal = "SELL"
                signal_strength = -min(abs(ema_diff_pct) * 2.0, 10.0)  # Max -10%
        
        return signal, signal_strength
    
    def apply_drawdown_adjustment(self, exposure: float, current_dd: float) -> float:
        """Apply drawdown-based risk adjustment"""
        if current_dd < 5.0:
            # Low DD: maintain full exposure
            risk_factor = 1.0
        elif current_dd < 12.0:
            # Medium DD: progressive reduction
            if current_dd < 8.0:
                risk_factor = 0.6
            else:
                risk_factor = 0.4
        else:
            # High DD: severe reduction
            risk_factor = 0.2
        
        adjusted_exposure = exposure * risk_factor
        
        self.logger.debug(f"DD adjustment: {current_dd:.1f}% DD → risk_factor={risk_factor:.1f} → exposure={adjusted_exposure:.1f}%")
        return adjusted_exposure
    
    def generate_signal(self, df: pd.DataFrame, idx: int, market_mode: str, regime_data: Dict, current_dd: float = 0.0) -> Tuple[float, str, Dict]:
        """Generate comprehensive BTC trading signal"""
        
        if idx < max(self.ema_fast, self.ema_slow):
            return 0.0, "HOLD", {"regime": 0.0, "ema": 0.0, "engines": []}
        
        row = df.iloc[idx]
        
        # 1. Get regime-based base exposure
        regime_exposure = self.get_regime_signal(market_mode, regime_data)
        
        # 2. Get EMA signal
        ema_signal, ema_strength = self.get_ema_signal(row)
        
        # 3. Combine signals with proper capping
        if market_mode == 'trend_up' and ema_signal == "BUY":
            # Strong buy in uptrend - cap at max_exposure
            total_exposure = min(regime_exposure + ema_strength, self.max_exposure)
        elif market_mode == 'trend_down' and ema_signal == "SELL":
            # Reduce exposure in downtrend
            total_exposure = max(0, regime_exposure + ema_strength)
        elif market_mode == 'range':
            # Conservative in range - only strong signals
            if abs(ema_strength) > 8.0:
                total_exposure = min(regime_exposure + (ema_strength * 0.5), self.max_exposure)
            else:
                total_exposure = regime_exposure
        else:
            # Default to regime exposure
            total_exposure = regime_exposure
        
        # 4. Apply drawdown adjustment
        total_exposure = self.apply_drawdown_adjustment(total_exposure, current_dd)
        
        # 5. Cap exposure
        total_exposure = max(0.0, min(total_exposure, self.max_exposure))
        
        engines_used = ["regime"]
        if abs(ema_strength) > 1.0:
            engines_used.append("ema")
        
        detail = {
            "regime": regime_exposure,
            "ema": ema_strength,
            "engines": engines_used,
            "dd_factor": current_dd
        }
        
        signal_type = "BUY" if total_exposure > regime_exposure else "HOLD"
        
        self.logger.debug(f"BTC Signal: {signal_type} {total_exposure:.1f}% | Regime: {regime_exposure:.1f}% + EMA: {ema_strength:+.1f}% | DD: {current_dd:.1f}%")
        
        return total_exposure, signal_type, detail
    
    def get_strategy_info(self) -> Dict:
        """Return strategy configuration info"""
        return {
            "name": "BTC Trend Following",
            "type": "trend_follow",
            "parameters": {
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "volume_threshold": self.volume_threshold,
                "max_exposure": self.max_exposure,
                "regime_exposures": self.regime_exposures
            }
        }