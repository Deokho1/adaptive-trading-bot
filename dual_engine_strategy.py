"""
Dual-Engine Trading Strategy Architecture
=========================================

This module implements a sophisticated dual-engine trading strategy that combines:
1. RegimeEngine: Long-term regime detection and baseline risk management
2. SignalEngine: Short-term momentum signals for aggressive entry/exit
3. StrategyManager: Intelligent integration of both engines

The system is designed with asset-specific parameters:
- ETH (Defensive): Strict drawdown control, regime transition protection
- BTC (Aggressive): Higher exposure in uptrends, trend continuation logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from market.market_analyzer import MarketAnalyzer, MarketMode
from exchange.models import Candle

logger = logging.getLogger(__name__)

# ============================================================================
# Asset Type Detection
# ============================================================================

class AssetType(Enum):
    """Asset type for parameter differentiation"""
    ETH = "ETH"  # Defensive mode
    BTC = "BTC"  # Aggressive mode
    
def detect_asset_type(symbol: str) -> AssetType:
    """Detect asset type from trading symbol"""
    if "ETH" in symbol.upper():
        return AssetType.ETH
    elif "BTC" in symbol.upper():
        return AssetType.BTC
    else:
        return AssetType.BTC  # Default to BTC mode

# ============================================================================
# Enums and Data Classes
# ============================================================================

class SignalType(Enum):
    """Signal types for momentum-based entries/exits"""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"

@dataclass
class SignalResult:
    """Result from SignalEngine analysis"""
    signal: SignalType
    strength: float  # 0.0 to 1.0
    reason: str
    additive_exposure: float  # Additional exposure to add/subtract
    
@dataclass
class RegimeResult:
    """Result from RegimeEngine analysis"""
    mode: MarketMode
    baseline_exposure: float  # Base exposure level
    risk_factor: float  # Risk adjustment factor
    
@dataclass
class StrategySignal:
    """Combined signal from both engines"""
    final_exposure: float
    regime_exposure: float
    signal_exposure: float
    active_engines: List[str]
    reasoning: str
    regime_transition_brake: bool = False  # New: tracks if transition protection is active

# ============================================================================
# SignalEngine: Short-term momentum and volume-based signals
# ============================================================================

class SignalEngine:
    """
    Generates aggressive entry/exit signals based on short-term momentum
    and volume spikes. Parameters adapt based on asset type (ETH vs BTC).
    """
    
    def __init__(self, config: Dict, asset_type: AssetType):
        self.config = config
        self.asset_type = asset_type
        
        # Asset-specific EMA parameters
        if asset_type == AssetType.BTC:
            # BTC (Ultra-Aggressive): Optimized EMA parameters for better signals
            self.ema_short = config.get('ema_short_btc', 5)   # Reverted from 7 to 5 for higher sensitivity
            self.ema_long = config.get('ema_long_btc', 15)    # Reverted from 21 to 15 for faster signals
            self.volume_threshold = config.get('volume_threshold_btc', 1.12)  # Reduced from 1.18 to 1.12
            self.max_additive_exposure = config.get('max_additive_exposure_btc', 0.21)  # 21% (reduced from 25%)
            self.pullback_tolerance = config.get('pullback_tolerance_btc', 0.05)  # Changed from 3% to 5%
        else:
            # ETH (Enhanced Defensive): Ultra-conservative EMA parameters
            self.ema_short = config.get('ema_short_eth', 10)  # Changed from 12 to 10
            self.ema_long = config.get('ema_long_eth', 24)    # Changed from 26 to 24
            self.volume_threshold = config.get('volume_threshold_eth', 1.2)  # Changed from 1.5 to 1.2
            self.max_additive_exposure = config.get('max_additive_exposure_eth', 0.16)  # 16% (increased from 14% to allow 110% total)
            self.pullback_tolerance = 0.0  # No pullback tolerance for ETH
            
        self.volume_lookback = config.get('volume_lookback', 20)
        self.min_additive_exposure = config.get('min_additive_exposure', 0.05)  # 5%
        
        # State tracking for BTC trend continuation
        self.last_buy_price = None
        self.trend_continuation_active = False
        
        # Cache for indicators
        self._ema_cache = {}
        self._volume_cache = {}
        
    def analyze(self, candles: List[Candle], current_regime: MarketMode, 
                regime_transition_brake: bool = False) -> SignalResult:
        """
        Analyze short-term momentum and generate signals
        
        Args:
            candles: Recent price data (at least 50 candles recommended)
            current_regime: Current market regime from RegimeEngine
            regime_transition_brake: Whether regime transition protection is active
            
        Returns:
            SignalResult with signal type and additive exposure
        """
        if len(candles) < max(self.ema_long, self.volume_lookback) + 10:
            return SignalResult(SignalType.HOLD, 0.0, "Insufficient data", 0.0)
            
        # Asset-specific regime filtering
        if not self._is_regime_favorable(current_regime, regime_transition_brake):
            reason = f"Inactive in {current_regime.value}"
            if regime_transition_brake:
                reason += " + transition brake"
            return SignalResult(SignalType.HOLD, 0.0, reason, 0.0)
            
        # Convert to DataFrame for easier analysis
        df = self._candles_to_df(candles)
        
        # Calculate indicators
        ema_short = self._calculate_ema(df['close'], self.ema_short)
        ema_long = self._calculate_ema(df['close'], self.ema_long)
        volume_avg = df['volume'].rolling(self.volume_lookback).mean()
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_ema_short = ema_short.iloc[-1]
        current_ema_long = ema_long.iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = volume_avg.iloc[-1]
        
        # Previous values for crossover detection
        prev_ema_short = ema_short.iloc[-2] if len(ema_short) > 1 else current_ema_short
        prev_ema_long = ema_long.iloc[-2] if len(ema_long) > 1 else current_ema_long
        
        # Crossover detection
        bullish_cross = (prev_ema_short <= prev_ema_long) and (current_ema_short > current_ema_long)
        bearish_cross = (prev_ema_short >= prev_ema_long) and (current_ema_short < current_ema_long)
        
        # Volume spike detection
        volume_spike = current_volume > (avg_volume * self.volume_threshold)
        
        # Price momentum
        price_above_short = current_price > current_ema_short
        price_above_long = current_price > current_ema_long
        
        # RANGE mode mean reversion indicators
        rsi = None
        bb_lower_touch = False
        if current_regime == MarketMode.RANGE:
            # Calculate RSI(14) for oversold detection
            rsi = self._calculate_rsi(df['close'], 14)
            current_rsi = rsi.iloc[-1] if rsi is not None else 50.0
            
            # Calculate Bollinger Bands for mean reversion
            bb_middle = df['close'].rolling(20).mean().iloc[-1]
            bb_std = df['close'].rolling(20).std().iloc[-1]
            bb_lower = bb_middle - (2 * bb_std)
            bb_lower_touch = current_price <= bb_lower * 1.005  # 0.5% tolerance
        else:
            current_rsi = 50.0  # Neutral value for non-range modes
        
        # BTC-specific trend continuation logic
        if self.asset_type == AssetType.BTC:
            self._update_trend_continuation(current_price, bullish_cross, bearish_cross, current_regime)
        
        # Generate signals
        signal, strength, reason, additive_exposure = self._generate_signal(
            bullish_cross, bearish_cross, volume_spike, 
            price_above_short, price_above_long, current_regime, 
            current_price, current_rsi, bb_lower_touch
        )
        
        logger.info(f"[SignalEngine-{self.asset_type.value}] {reason} -> {signal.value} (+{additive_exposure:.1%})")
        
        return SignalResult(signal, strength, reason, additive_exposure)
    
    def _is_regime_favorable(self, regime: MarketMode, regime_transition_brake: bool) -> bool:
        """Check if current regime is favorable for signal generation"""
        if regime_transition_brake:
            return False
            
        if self.asset_type == AssetType.ETH:
            # ETH (Enhanced Defensive): More restrictive signal generation
            # Disable in TREND_DOWN and NEUTRAL for more conservative approach
            return regime in [MarketMode.TREND_UP, MarketMode.RANGE]
        else:
            # BTC (Aggressive): Disable only in TREND_DOWN
            return regime != MarketMode.TREND_DOWN
    
    def _update_trend_continuation(self, current_price: float, bullish_cross: bool, 
                                 bearish_cross: bool, regime: MarketMode):
        """Update BTC trend continuation state"""
        if bullish_cross and regime == MarketMode.TREND_UP:
            self.last_buy_price = current_price
            self.trend_continuation_active = True
            
        elif bearish_cross or regime != MarketMode.TREND_UP:
            self.trend_continuation_active = False
            self.last_buy_price = None
    
    def _should_ignore_pullback(self, current_price: float) -> bool:
        """Check if BTC pullback should be ignored for trend continuation"""
        if (self.asset_type == AssetType.BTC and 
            self.trend_continuation_active and 
            self.last_buy_price is not None):
            
            pullback_pct = (self.last_buy_price - current_price) / self.last_buy_price
            return pullback_pct <= self.pullback_tolerance
            
        return False
    
    def _generate_signal(self, bullish_cross: bool, bearish_cross: bool, 
                        volume_spike: bool, price_above_short: bool, 
                        price_above_long: bool, regime: MarketMode, 
                        current_price: float, current_rsi: float = 50.0,
                        bb_lower_touch: bool = False) -> Tuple[SignalType, float, str, float]:
        """Generate trading signal based on technical conditions"""
        
        # RANGE mode: Special mean reversion logic (ignore EMA signals)
        if regime == MarketMode.RANGE:
            # Only trade on strong oversold + Bollinger Band touch
            if current_rsi <= 30.0 and bb_lower_touch:
                # Limited mean reversion opportunity (max 25% total exposure)
                exposure = min(0.10, self.max_additive_exposure * 0.4)
                return (SignalType.BUY, 0.6, "Range mean reversion (RSI30+BB)", exposure)
            else:
                # No trades in range mode otherwise
                return (SignalType.HOLD, 0.0, "Range mode: waiting for oversold", 0.0)
        
        # BTC trend continuation: ignore bearish signals during small pullbacks
        if (self.asset_type == AssetType.BTC and 
            self._should_ignore_pullback(current_price) and
            bearish_cross):
            return (SignalType.HOLD, 0.0, "Ignoring pullback in BTC uptrend", 0.0)
        
        # Strong BUY signals
        if bullish_cross and volume_spike:
            exposure = self.max_additive_exposure
            # BTC max 90% total exposure in TREND_UP (reduced from 100%)
            if self.asset_type == AssetType.BTC and regime == MarketMode.TREND_UP:
                exposure = min(0.40, self.max_additive_exposure * 2.0)  # Up to 40% additive (allow 90% total)
            return (SignalType.BUY, 0.9, "EMA crossover + volume spike", exposure)
        
        if bullish_cross and price_above_long:
            exposure = self.max_additive_exposure * 0.8
            if self.asset_type == AssetType.BTC and regime == MarketMode.TREND_UP:
                exposure = min(0.35, self.max_additive_exposure * 1.6)  # Reduced max exposure
            return (SignalType.BUY, 0.8, "EMA crossover + uptrend", exposure)
        
        # Medium BUY signals  
        if price_above_short and price_above_long and volume_spike:
            exposure = self.min_additive_exposure * 1.5
            if self.asset_type == AssetType.BTC and regime == MarketMode.TREND_UP:
                exposure = min(0.30, self.min_additive_exposure * 3.0)
            return (SignalType.BUY, 0.7, "Strong momentum + volume", exposure)
        
        if bullish_cross:
            exposure = self.min_additive_exposure
            return (SignalType.BUY, 0.6, "EMA crossover", exposure)
        
        # SELL signals
        if bearish_cross:
            return (SignalType.SELL, 0.8, "Bearish EMA crossover", -self.max_additive_exposure)
        
        # ETH-specific: More conservative in NEUTRAL
        if (self.asset_type == AssetType.ETH and 
            not price_above_short and regime == MarketMode.NEUTRAL):
            return (SignalType.SELL, 0.6, "ETH: Price below EMA in neutral", 
                   -self.min_additive_exposure)
        
        # Default HOLD
        return (SignalType.HOLD, 0.0, "No clear signal", 0.0)
    
    def _candles_to_df(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candle data to DataFrame"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high, 
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        return pd.DataFrame(data)
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# ============================================================================
# RegimeEngine: Long-term regime detection and risk management
# ============================================================================

class RegimeEngine:
    """
    Wraps the existing MarketAnalyzer to provide regime-based baseline exposure
    with asset-specific parameters (ETH defensive vs BTC aggressive).
    """
    
    def __init__(self, config: Dict, asset_type: AssetType):
        self.config = config
        self.asset_type = asset_type
        self.market_analyzer = MarketAnalyzer(config)
        
        # Asset-specific exposure mapping
        if asset_type == AssetType.ETH:
            # ETH (Defensive mode): Lower exposures, strict risk control
            self.exposure_map = {
                MarketMode.TREND_UP: 0.55,     # 50-60% -> 55%
                MarketMode.NEUTRAL: 0.275,     # 25-30% -> 27.5%
                MarketMode.RANGE: 0.475,       # 45-50% -> 47.5%
                MarketMode.TREND_DOWN: 0.08    # Changed from 6% to 8% max for ETH in downtrend
            }
        else:
            # BTC (Aggressive-Return mode): Maximum TREND_UP with restored profitable trades
            self.exposure_map = {
                MarketMode.TREND_UP: 1.00,     # 100% (maximum exposure for bull markets)
                MarketMode.NEUTRAL: 0.10,      # 10% (restored for momentum continuation)
                MarketMode.RANGE: 0.20,        # 20% (restored for mean reversion profits)
                MarketMode.TREND_DOWN: 0.05    # 5% (increased for short-term rebounds)
            }
        
        # Risk adjustment factors - Enhanced DD Control
        self.max_drawdown_threshold = 0.05  # Level 1: Start reducing exposure at 5% DD (changed from 7%)
        self.volatility_threshold = 0.05    # High volatility threshold
        
        # Drawdown adaptive control for BTC (Tighter control)
        self.adaptive_drawdown_threshold = 0.12  # Level 2: 12% threshold for adaptive control (changed from 18%)
        self.adaptive_drawdown_max = 0.20        # 20% max for adaptive control (reduced from 22%)
        self.adaptive_multiplier = 0.8           # 0.8x exposure multiplier (more aggressive)
        self.adaptive_release_bars = 6           # Release after 6 bars (reduced from 10)
        self.adaptive_counter = 0                # Counter for adaptive control
        
        # Regime transition tracking for ETH protection (Enhanced)
        self.previous_regime = None
        self.transition_brake_counter = 0
        self.transition_brake_duration = 10 if asset_type == AssetType.ETH else 4  # 10 bars for ETH, 4 for BTC
        
    def analyze(self, candles: List[Candle], current_drawdown: float = 0.0, 
                signal_strength: float = 0.0) -> Tuple[RegimeResult, bool]:
        """
        Analyze market regime and determine baseline exposure
        
        Args:
            candles: Price data for regime analysis
            current_drawdown: Current portfolio drawdown
            signal_strength: Signal strength from SignalEngine for weighted exposure
            
        Returns:
            Tuple of (RegimeResult, regime_transition_brake)
        """
        if len(candles) < 50:
            return RegimeResult(MarketMode.NEUTRAL, 0.2, 1.0), False
            
        # Get regime from existing MarketAnalyzer
        regime = self.market_analyzer.update_mode(candles, candles[-1].timestamp)
        
        # Check for regime transition (ETH only)
        regime_transition_brake = self._check_regime_transition(regime)
        
        # Base exposure from regime
        baseline_exposure = self.exposure_map.get(regime, 0.3)
        
        # Apply regime transition brake for ETH (Enhanced protection)
        if regime_transition_brake and self.asset_type == AssetType.ETH:
            baseline_exposure *= 0.5  # Cut exposure by 50% (was 50%, keeping aggressive protection)
            logger.info(f"[REGIME_TRANSITION_PROTECTION] ETH exposure cut: {baseline_exposure:.1%}")
        
        # Risk adjustment based on drawdown
        risk_factor = self._calculate_risk_factor(candles, current_drawdown, regime)
        adjusted_exposure = baseline_exposure * risk_factor
        
        # Apply drawdown adaptive control for BTC
        if self.asset_type == AssetType.BTC:
            adaptive_factor = self._calculate_adaptive_drawdown_control(current_drawdown)
            adjusted_exposure *= adaptive_factor
            
            # Signal-weighted exposure boost for BTC
            signal_boost = self._calculate_signal_weighted_boost(signal_strength)
            adjusted_exposure += signal_boost
            
            # Cap total exposure at 100% (enhanced from 105%)
            adjusted_exposure = min(adjusted_exposure, 1.00)
        
        logger.info(f"[RegimeEngine-{self.asset_type.value}] {regime.value} -> {adjusted_exposure:.1%} "
                   f"(risk factor: {risk_factor:.2f}, brake: {regime_transition_brake})")
        
        return RegimeResult(regime, adjusted_exposure, risk_factor), regime_transition_brake
    
    def _check_regime_transition(self, current_regime: MarketMode) -> bool:
        """Check for regime transitions and manage brake counter"""
        regime_transition_brake = False
        
        if self.asset_type == AssetType.ETH:
            # Check for dangerous transitions: TREND_UP -> TREND_DOWN/NEUTRAL
            if (self.previous_regime == MarketMode.TREND_UP and 
                current_regime in [MarketMode.TREND_DOWN, MarketMode.NEUTRAL]):
                self.transition_brake_counter = self.transition_brake_duration
                regime_transition_brake = True
                logger.info(f"[REGIME_TRANSITION_PROTECTION] ETH brake activated: "
                           f"{self.previous_regime.value} -> {current_regime.value}")
            
            # Countdown brake counter
            elif self.transition_brake_counter > 0:
                self.transition_brake_counter -= 1
                regime_transition_brake = True
                logger.info(f"[REGIME_TRANSITION_PROTECTION] ETH brake countdown: {self.transition_brake_counter}")
        
        self.previous_regime = current_regime
        return regime_transition_brake
    
    def _calculate_adaptive_drawdown_control(self, current_drawdown: float) -> float:
        """Calculate adaptive drawdown control multiplier for BTC"""
        if self.asset_type != AssetType.BTC:
            return 1.0
            
        # Check if we're in adaptive drawdown range (18-22%)
        if self.adaptive_drawdown_threshold <= current_drawdown <= self.adaptive_drawdown_max:
            self.adaptive_counter = self.adaptive_release_bars
            logger.info(f"[ADAPTIVE_DRAWDOWN] BTC drawdown {current_drawdown:.1%} - applying {self.adaptive_multiplier:.2f}x multiplier")
            return self.adaptive_multiplier
        
        # Count down and release adaptive control
        elif self.adaptive_counter > 0:
            self.adaptive_counter -= 1
            if self.adaptive_counter == 0:
                logger.info(f"[ADAPTIVE_DRAWDOWN] BTC adaptive control released after 10 bars")
            return self.adaptive_multiplier
            
        return 1.0
    
    def _calculate_risk_factor(self, candles: List[Candle], current_drawdown: float, 
                              current_regime: MarketMode = None) -> float:
        """Calculate risk adjustment factor based on market conditions and drawdown"""
        risk_factor = 1.0
        
        # Full throttle for TREND_UP when drawdown is low
        if (current_regime == MarketMode.TREND_UP and 
            current_drawdown < 0.05):  # DD < 5%
            return 1.0  # Full throttle, skip all other risk adjustments
        
        # ETH NEUTRAL mode additional risk reduction
        if (self.asset_type == AssetType.ETH and 
            current_regime == MarketMode.NEUTRAL):
            risk_factor *= 0.70  # Apply 0.70x multiplier for enhanced risk control (changed from 0.85)
            logger.info(f"[RegimeEngine] ETH NEUTRAL mode adjustment: 0.70")
        
        # Enhanced Drawdown-based adjustment with two levels
        if current_drawdown > self.max_drawdown_threshold:
            # Level 1: 5% DD threshold - reduce by 40% (factor 0.60)
            dd_factor = 0.60
            if current_drawdown > self.adaptive_drawdown_threshold:
                # Level 2: 12% DD threshold - reduce by 60% (factor 0.40)
                dd_factor = 0.40
            risk_factor *= dd_factor
            logger.info(f"[RegimeEngine] Drawdown adjustment: {dd_factor:.2f} (DD: {current_drawdown:.1%})")
        
        # Enhanced volatility-based adjustment with ATR check for ETH
        if len(candles) >= 20:
            prices = [c.close for c in candles[-20:]]
            volatility = np.std(prices) / np.mean(prices)
            
            # Calculate ATR for ETH
            if self.asset_type == AssetType.ETH and len(candles) >= 14:
                atr_values = []
                for i in range(max(1, len(candles) - 14), len(candles)):
                    high = candles[i].high
                    low = candles[i].low
                    prev_close = candles[i-1].close
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    atr_values.append(tr)
                
                if atr_values:
                    atr = sum(atr_values) / len(atr_values)
                    current_price = candles[-1].close
                    atr_percentage = (atr / current_price) * 100
                    
                    # Apply volatility brake if ATR > 1.8%
                    if atr_percentage > 1.8:
                        atr_factor = 0.8
                        risk_factor *= atr_factor
                        logger.info(f"[RegimeEngine] ETH ATR brake: {atr_factor:.2f} (ATR: {atr_percentage:.2f}%)")
            
            # Standard volatility adjustment
            if volatility > self.volatility_threshold:
                vol_factor = max(0.5, 1.0 - (volatility - self.volatility_threshold) * 5)
                risk_factor *= vol_factor
                logger.info(f"[RegimeEngine] Volatility adjustment: {vol_factor:.2f} (Vol: {volatility:.3f})")
        
        return max(0.1, risk_factor)  # Never go below 10% of baseline
    
    def _calculate_signal_weighted_boost(self, signal_strength: float) -> float:
        """Calculate signal-weighted exposure boost for BTC"""
        if self.asset_type != AssetType.BTC or signal_strength <= 0:
            return 0.0
        
        # Strong signal: 0.7-1.0 strength -> +10% boost
        if signal_strength >= 0.7:
            return 0.10
        # Weak signal: 0.3-0.7 strength -> +5% boost  
        elif signal_strength >= 0.3:
            return 0.05
        # Very weak signal: no boost
        else:
            return 0.0

# ============================================================================
# StrategyManager: Integration of both engines
# ============================================================================

class StrategyManager:
    """
    Intelligent integration of RegimeEngine and SignalEngine with asset-specific
    parameters for ETH (defensive) and BTC (aggressive) trading modes.
    """
    
    def __init__(self, config: Dict, symbol: str = "KRW-BTC", btc_mode: str = "dual_engine"):
        self.config = config
        self.symbol = symbol
        self.asset_type = detect_asset_type(symbol)
        self.btc_mode = btc_mode  # "regime_only" or "dual_engine"
        
        # Initialize engines with asset-specific parameters
        self.regime_engine = RegimeEngine(config, self.asset_type)
        self.signal_engine = SignalEngine(config, self.asset_type)
        
        # Global constraints (asset-specific)
        if self.asset_type == AssetType.BTC:
            self.max_total_exposure = config.get('max_total_exposure_btc', 1.0)  # 100% for BTC
        else:
            self.max_total_exposure = config.get('max_total_exposure_eth', 0.8)  # 80% for ETH
        self.min_total_exposure = config.get('min_total_exposure', 0.0)  # 0%
        
        # Performance tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'regime_overrides': 0,
            'signal_activations': 0,
            'regime_transitions': 0  # New: track transition protections
        }
        
        logger.info(f"[StrategyManager] Initialized for {self.asset_type.value} mode: {symbol}")
        
    def analyze(self, candles: List[Candle], current_drawdown: float = 0.0) -> StrategySignal:
        """
        Generate combined trading signal from both engines
        
        Args:
            candles: Price history for analysis
            current_drawdown: Current portfolio drawdown
            
        Returns:
            StrategySignal with final exposure and detailed reasoning
        """
        if len(candles) < 50:
            return StrategySignal(0.2, 0.2, 0.0, ["regime"], "Insufficient data - minimal exposure")
        
        # BTC regime-only mode: bypass SignalEngine completely
        if self.asset_type == AssetType.BTC and self.btc_mode == "regime_only":
            # Only use RegimeEngine for BTC regime-only mode
            regime_result, regime_transition_brake = self.regime_engine.analyze(
                candles, current_drawdown, 0.0)  # No signal strength
            
            strategy_signal = StrategySignal(
                final_exposure=regime_result.baseline_exposure,
                regime_exposure=regime_result.baseline_exposure,
                signal_exposure=0.0,
                active_engines=["regime"],
                reasoning=f"Regime-only mode: {regime_result.mode.value} -> {regime_result.baseline_exposure:.1%}",
                regime_transition_brake=regime_transition_brake
            )
            
            # Track performance for regime-only
            self._update_stats_regime_only(regime_result, strategy_signal, regime_transition_brake)
            
            # Log regime-only decision
            logger.info(f"[StrategyManager-{self.asset_type.value}][REGIME_ONLY] Final: {regime_result.baseline_exposure:.1%} | "
                       f"Regime: {regime_result.baseline_exposure:.1%} + Signal: +0.0% | "
                       f"Engines: ['regime'] | Brake: {regime_transition_brake}")
            
            return strategy_signal
        
        # Dual-engine mode (default for ETH, optional for BTC)
        # Get signals from both engines (two-pass approach for signal weighting)
        # First pass: get signal strength
        signal_result = self.signal_engine.analyze(candles, MarketMode.NEUTRAL, False)
        
        # Second pass: get regime with signal strength
        regime_result, regime_transition_brake = self.regime_engine.analyze(
            candles, current_drawdown, signal_result.strength)
        
        # Final pass: get refined signal with actual regime
        signal_result = self.signal_engine.analyze(candles, regime_result.mode, regime_transition_brake)
        
        # Combine exposures
        final_exposure, active_engines, reasoning = self._combine_signals(
            regime_result, signal_result, regime_transition_brake
        )
        
        # Create combined signal
        strategy_signal = StrategySignal(
            final_exposure=final_exposure,
            regime_exposure=regime_result.baseline_exposure,
            signal_exposure=signal_result.additive_exposure,
            active_engines=active_engines,
            reasoning=reasoning,
            regime_transition_brake=regime_transition_brake
        )
        
        # Track performance
        self._update_stats(regime_result, signal_result, strategy_signal, regime_transition_brake)
        
        # Log combined decision
        logger.info(f"[StrategyManager-{self.asset_type.value}] Final: {final_exposure:.1%} | "
                   f"Regime: {regime_result.baseline_exposure:.1%} + "
                   f"Signal: {signal_result.additive_exposure:+.1%} | "
                   f"Engines: {active_engines} | Brake: {regime_transition_brake}")
        
        return strategy_signal
    
    def _combine_signals(self, regime: RegimeResult, signal: SignalResult, 
                        regime_transition_brake: bool) -> Tuple[float, List[str], str]:
        """Intelligently combine signals from both engines with asset-specific logic"""
        
        active_engines = ["regime"]
        reasoning_parts = [f"Regime: {regime.mode.value} -> {regime.baseline_exposure:.1%}"]
        
        if regime_transition_brake:
            reasoning_parts.append("[TRANSITION_BRAKE]")
        
        # Start with regime baseline
        total_exposure = regime.baseline_exposure
        
        # Add signal engine contribution with asset-specific rules
        if signal.signal != SignalType.HOLD and signal.additive_exposure != 0:
            # ETH-specific signal filtering
            if self.asset_type == AssetType.ETH:
                # Disable signal in NEUTRAL, cap at +5% exposure in NEUTRAL
                if regime.mode == MarketMode.NEUTRAL:
                    capped_exposure = min(signal.additive_exposure, 0.05)
                    if capped_exposure != signal.additive_exposure:
                        reasoning_parts.append(f"Signal capped in NEUTRAL: {capped_exposure:+.1%}")
                    total_exposure += capped_exposure
                elif regime.mode != MarketMode.TREND_DOWN:
                    total_exposure += signal.additive_exposure
                    active_engines.append("signal")
                    reasoning_parts.append(f"Signal: {signal.reason} -> {signal.additive_exposure:+.1%}")
            else:
                # BTC: More permissive signal usage
                total_exposure += signal.additive_exposure
                active_engines.append("signal")
                reasoning_parts.append(f"Signal: {signal.reason} -> {signal.additive_exposure:+.1%}")
        else:
            reasoning_parts.append(f"Signal: {signal.reason}")
        
        # Apply global constraints
        total_exposure = max(self.min_total_exposure, 
                           min(self.max_total_exposure, total_exposure))
        
        reasoning = " | ".join(reasoning_parts)
        
        return total_exposure, active_engines, reasoning
    
    def _update_stats(self, regime: RegimeResult, signal: SignalResult, 
                     final: StrategySignal, regime_transition_brake: bool):
        """Update performance tracking statistics"""
        self.performance_stats['total_signals'] += 1
        
        if signal.signal != SignalType.HOLD:
            self.performance_stats['signal_activations'] += 1
            
        if len(final.active_engines) == 1:  # Only regime active
            self.performance_stats['regime_overrides'] += 1
            
        if regime_transition_brake:
            self.performance_stats['regime_transitions'] += 1
            
        # Store recent history (keep last 100)
        self.signal_history.append({
            'regime_mode': regime.mode.value,
            'regime_exposure': regime.baseline_exposure,
            'signal_type': signal.signal.value,
            'signal_exposure': signal.additive_exposure,
            'final_exposure': final.final_exposure,
            'active_engines': final.active_engines,
            'transition_brake': regime_transition_brake
        })
        
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
    
    def _update_stats_regime_only(self, regime: RegimeResult, 
                                 final: StrategySignal, regime_transition_brake: bool):
        """Update performance tracking statistics for regime-only mode"""
        self.performance_stats['total_signals'] += 1
        self.performance_stats['regime_overrides'] += 1  # Always regime-only
            
        if regime_transition_brake:
            self.performance_stats['regime_transitions'] += 1
            
        # Store recent history (keep last 100)
        self.signal_history.append({
            'regime_mode': regime.mode.value,
            'regime_exposure': regime.baseline_exposure,
            'signal_type': 'HOLD',  # No signals in regime-only mode
            'signal_exposure': 0.0,
            'final_exposure': final.final_exposure,
            'active_engines': final.active_engines,
            'transition_brake': regime_transition_brake
        })
        
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of strategy performance and engine utilization"""
        stats = self.performance_stats.copy()
        stats['asset_type'] = self.asset_type.value
        
        if stats['total_signals'] > 0:
            stats['signal_activation_rate'] = stats['signal_activations'] / stats['total_signals']
            stats['regime_override_rate'] = stats['regime_overrides'] / stats['total_signals']
            stats['transition_protection_rate'] = stats['regime_transitions'] / stats['total_signals']
            
        return stats

# ============================================================================
# Utility Functions
# ============================================================================

def create_dual_engine_config() -> Dict:
    """Create default configuration for dual-engine strategy with asset-specific parameters"""
    return {
        # RegimeEngine config (inherits from MarketAnalyzer)
        'adx_period': 14,
        'adx_threshold': 25,
        'atr_period': 14,
        'bb_period': 20,
        'bb_std': 2,
        
        # SignalEngine config - ETH (Defensive)
        'ema_short_eth': 8,     # Changed from 4 to 8 for smoother signals
        'ema_long_eth': 24,     # Changed from 12 to 24 for smoother signals
        'volume_threshold_eth': 1.25,    # Increased to 1.25 for higher quality signals
        'max_additive_exposure_eth': 0.15,  # 15% max additive
        
        # SignalEngine config - BTC (Aggressive)  
        'ema_short_btc': 15,    # Changed from 8 to 15 for slower, more reliable signals
        'ema_long_btc': 60,     # Extended from 50 to 60 for better Sharpe ratio
        'volume_threshold_btc': 1.20,    # Lowered to 1.20 for improved reactivity
        'max_additive_exposure_btc': 0.25,  # 25% max additive
        'pullback_tolerance_btc': 0.05,     # 5% pullback tolerance
        
        # Common SignalEngine config
        'volume_lookback': 20,
        'min_additive_exposure': 0.05,     # 5% min additive
        
        # StrategyManager config - Asset specific
        'max_total_exposure_eth': 0.45,     # 45% max for ETH (reduced from 100%)
        'max_total_exposure_btc': 1.00,     # 100% max for BTC (increased for returns)
        'min_total_exposure': 0.0,
    }

def get_current_parameters() -> Dict:
    """Get current strategy parameters for display"""
    config = create_dual_engine_config()
    return {
        'ema_periods': {
            'btc': f"({config['ema_short_btc']}, {config['ema_long_btc']})",
            'eth': f"({config['ema_short_eth']}, {config['ema_long_eth']})"
        },
        'volume_thresholds': {
            'btc': f"{config['volume_threshold_btc']:.2f}×",
            'eth': f"{config['volume_threshold_eth']:.2f}×"
        },
        'exposure_limits': {
            'btc': f"{config['max_total_exposure_btc']*100:.0f}%",
            'eth': f"{config['max_total_exposure_eth']*100:.0f}%"
        },
        'signal_sensitivity': 'Very High (EMA 4,12)',
        'risk_factor': '0.85× (Enhanced DD reduction)',
        'rebalancing_frequency': '4 hours'
    }

if __name__ == "__main__":
    # Example usage and testing
    config = create_dual_engine_config()
    
    # Test both asset types
    eth_strategy = StrategyManager(config, "KRW-ETH")
    btc_strategy = StrategyManager(config, "KRW-BTC")
    
    print("🚀 Enhanced Dual-Engine Trading Strategy Initialized")
    print(f"📊 ETH (Defensive): Lower risk, regime transition protection")
    print(f"⚡ BTC (Aggressive): Higher exposure, trend continuation logic")
    print(f"🎯 Goals: ETH <25% MDD, BTC 130-160% returns with <20% MDD")