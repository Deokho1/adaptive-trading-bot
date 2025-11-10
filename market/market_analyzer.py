"""
Market analyzer for classifying market conditions.

This module provides the MarketAnalyzer class that uses technical indicators
to classify market conditions into TREND, RANGE, or NEUTRAL modes.
"""

from datetime import datetime, timedelta
from typing import Dict, List
import logging

from core.types import MarketMode
from exchange.models import Candle
from .indicators import compute_atr, compute_adx, compute_bollinger_bands

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """
    Analyzes market conditions using technical indicators.
    
    Classifies market into TREND, RANGE, or NEUTRAL based on:
    - ATR (Average True Range) for volatility
    - ADX (Average Directional Index) for trend strength
    - Bollinger Band width for market expansion/contraction
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the market analyzer.
        
        Args:
            config: Configuration dictionary containing market_analyzer settings
        """
        analyzer_config = config.get("market_analyzer", {})
        
        # Indicator periods
        self.adx_period = analyzer_config.get("adx_period", 14)
        self.atr_period = analyzer_config.get("atr_period", 14)
        self.bb_period = analyzer_config.get("bb_period", 20)
        self.bb_stddev = analyzer_config.get("bb_stddev", 2.0)
        
        # TREND thresholds (KEEP AS-IS - working well)
        self.adx_trend_enter = analyzer_config.get("adx_trend_enter", 22.0)  # was 25.0
        self.adx_trend_exit = analyzer_config.get("adx_trend_exit", 18.0)    # was 20.0
        self.atr_trend_min = analyzer_config.get("atr_trend_min", 1.0)       # was 2.0
        
        # RANGE thresholds (UPDATED - much more relaxed for 10-30% coverage)
        self.adx_range_enter = analyzer_config.get("adx_range_enter", 30.0)  # was 22.0
        self.adx_range_exit = analyzer_config.get("adx_range_exit", 35.0)    # was 25.0
        self.bw_range_enter = analyzer_config.get("bw_range_enter", 12.0)    # was 8.0
        self.bw_range_exit = analyzer_config.get("bw_range_exit", 15.0)      # was 10.0
        self.atr_range_max = analyzer_config.get("atr_range_max", 6.0)       # was 4.5
        
        # Mode persistence
        self.cooldown_hours = analyzer_config.get("cooldown_hours", 2)       # was 4
        
        # Trend direction parameters
        self.ma_period = analyzer_config.get("ma_period", 30)                # Moving average for trend direction (더 반응성 향상)
        self.slope_lookback = analyzer_config.get("slope_lookback", 8)       # Lookback for slope calculation (changed from 3 to 8 for smoother detection)
        
        # State
        self.current_mode: MarketMode = MarketMode.NEUTRAL
        self.last_mode_change: datetime | None = None
    
    def _compute_metrics(self, btc_candles: List[Candle]) -> Dict[str, float]:
        """
        Compute technical indicator metrics from candle data.
        
        Args:
            btc_candles: List of BTC candles (should be sufficient for indicators)
        
        Returns:
            Dictionary containing computed metrics:
            - atr: Latest ATR value
            - atr_ratio: ATR as percentage of price 
            - adx: Latest ADX value
            - bandwidth: Bollinger Band width as percentage
        """
        if len(btc_candles) < max(self.adx_period, self.atr_period, self.bb_period) + 1:
            return {
                "atr": 0.0,
                "atr_ratio": 0.0,
                "adx": 0.0,
                "bandwidth": 0.0
            }
        
        # Compute indicators
        atr_values = compute_atr(btc_candles, self.atr_period)
        adx_values = compute_adx(btc_candles, self.adx_period)
        
        # Extract closing prices for Bollinger Bands
        closes = [candle.close for candle in btc_candles]
        bb_result = compute_bollinger_bands(
            closes, self.bb_period, self.bb_stddev
        )
        
        # bb_result is (middle, upper, lower)
        bb_middle, bb_upper, bb_lower = bb_result
        
        # Get latest values
        latest_atr = atr_values[-1] if atr_values else 0.0
        latest_adx = adx_values[-1] if adx_values else 0.0
        latest_close = btc_candles[-1].close
        
        # Calculate ATR ratio (ATR as percentage of close price)
        atr_ratio = (latest_atr / latest_close * 100) if latest_close > 0 else 0.0
        
        # Calculate Bollinger Band width
        if bb_middle and bb_upper and bb_lower:
            latest_middle = bb_middle[-1]
            latest_upper = bb_upper[-1]
            latest_lower = bb_lower[-1]
            
            if latest_middle > 0:
                bandwidth = ((latest_upper - latest_lower) / latest_middle) * 100
            else:
                bandwidth = 0.0
        else:
            bandwidth = 0.0
        
        return {
            "atr": latest_atr,
            "atr_ratio": atr_ratio,
            "adx": latest_adx,
            "bandwidth": bandwidth
        }
    
    def _detect_trend_direction(self, btc_candles: List[Candle]) -> str:
        """
        Detect trend direction using moving average slope.
        
        Args:
            btc_candles: List of BTC candles
        
        Returns:
            Trend direction: "UP", "DOWN", or "FLAT"
        """
        if len(btc_candles) < max(self.ma_period, self.slope_lookback + 1):
            return "FLAT"
        
        # Extract closing prices
        closes = [candle.close for candle in btc_candles]
        
        # Calculate simple moving average
        if len(closes) < self.ma_period:
            return "FLAT"
        
        ma_values = []
        for i in range(self.ma_period - 1, len(closes)):
            ma_window = closes[i - self.ma_period + 1:i + 1]
            ma_values.append(sum(ma_window) / len(ma_window))
        
        if len(ma_values) < self.slope_lookback + 1:
            return "FLAT"
        
        # Get current price and MA
        current_close = closes[-1]
        current_ma = ma_values[-1]
        
        # Calculate slope (recent MA change)
        slope = ma_values[-1] - ma_values[-self.slope_lookback]
        
        # Debug log for slope calculation
        logger.debug(f"[DEBUG][bot] Slope={slope:.4f}, MA period={self.ma_period}, lookback={self.slope_lookback}")
        
        # Determine trend direction
        if current_close > current_ma and slope > 0:
            return "UP"
        elif current_close < current_ma and slope < 0:
            return "DOWN"
        else:
            return "FLAT"
    
    def update_mode(self, btc_candles: List[Candle], now: datetime) -> MarketMode:
        """
        Update and return the current market mode based on candle analysis.
        
        Args:
            btc_candles: List of BTC candles for analysis
            now: Current datetime for cooldown calculation
        
        Returns:
            Current market mode after analysis
        """
        # Check if we have enough data
        min_candles = max(self.adx_period, self.atr_period, self.bb_period) + 1
        if len(btc_candles) < min_candles:
            return self.current_mode
        
        # Check cooldown period
        if (self.cooldown_hours > 0 and 
            self.last_mode_change and 
            now - self.last_mode_change < timedelta(hours=self.cooldown_hours)):
            return self.current_mode
        
        # Compute current metrics
        metrics = self._compute_metrics(btc_candles)
        atr_value = metrics["atr"]
        atr_ratio = metrics["atr_ratio"]
        adx_value = metrics["adx"]
        bandwidth = metrics["bandwidth"]
        
        # Store latest ADX for external access
        self.latest_adx = adx_value
        
        # Detect trend direction
        trend_direction = self._detect_trend_direction(btc_candles)
        
        # Determine new mode based on current mode and thresholds
        new_mode = self._classify_market_mode_with_direction(
            adx_value, atr_ratio, bandwidth, trend_direction, self.current_mode
        )
        
        # Update state if mode changed
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.last_mode_change = now
            
            # Log mode changes with trend direction info
            import logging
            logger = logging.getLogger(__name__)
            if new_mode in [MarketMode.TREND_UP, MarketMode.TREND_DOWN]:
                logger.info(
                    f"[INFO][bot] MarketMode: {new_mode.value.upper()} (slope={trend_direction}, ADX={adx_value:.1f})"
                )
        
        # Debug log for RANGE detection (DEBUG level only)
        if new_mode == MarketMode.RANGE:
            import logging
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[RANGE] ts=%s ADX=%.1f BW=%.2f%% ATR_ratio=%.2f%%",
                    now.strftime('%Y-%m-%d %H:%M'), adx_value, bandwidth, atr_ratio
                )
        
        return self.current_mode
    
    def _classify_market_mode_with_direction(
        self, 
        adx_value: float, 
        atr_ratio: float, 
        bandwidth: float,
        trend_direction: str,
        current_mode: MarketMode
    ) -> MarketMode:
        """
        Classify market mode with trend direction awareness.
        
        Args:
            adx_value: ADX indicator value
            atr_ratio: ATR as percentage of price
            bandwidth: Bollinger Band width percentage
            trend_direction: "UP", "DOWN", or "FLAT"
            current_mode: Current market mode for hysteresis
        
        Returns:
            New market mode
        """
        # Check for TREND conditions first
        if current_mode in [MarketMode.TREND_UP, MarketMode.TREND_DOWN, MarketMode.TREND]:
            # Exit thresholds for TREND
            if (adx_value < self.adx_trend_exit or 
                atr_ratio < self.atr_trend_min):
                # Fall back to RANGE or NEUTRAL
                pass  # Continue to RANGE/NEUTRAL checks
            else:
                # Stay in TREND, but update direction
                if trend_direction == "UP":
                    return MarketMode.TREND_UP
                elif trend_direction == "DOWN":
                    return MarketMode.TREND_DOWN
                else:
                    return MarketMode.NEUTRAL  # Flat trend
        
        # Check for TREND entry
        if (adx_value >= self.adx_trend_enter and 
            atr_ratio >= self.atr_trend_min):
            if trend_direction == "UP":
                return MarketMode.TREND_UP
            elif trend_direction == "DOWN":
                return MarketMode.TREND_DOWN
            # If trend strength is high but direction is flat, use NEUTRAL
            return MarketMode.NEUTRAL
        
        # Check for RANGE conditions
        if current_mode == MarketMode.RANGE:
            # Exit thresholds for RANGE
            if (adx_value > self.adx_range_exit or 
                bandwidth > self.bw_range_exit or 
                atr_ratio > self.atr_range_max):
                # Fall back to NEUTRAL
                return MarketMode.NEUTRAL
            else:
                return MarketMode.RANGE
        else:
            # Entry thresholds for RANGE
            if (adx_value <= self.adx_range_enter and 
                bandwidth <= self.bw_range_enter and 
                atr_ratio <= self.atr_range_max):
                return MarketMode.RANGE
        
        # Default to NEUTRAL if no clear trend or range
        return MarketMode.NEUTRAL
    
    def _classify_market_mode(
        self, 
        adx_value: float, 
        atr_ratio: float, 
        bandwidth: float,
        current_mode: MarketMode
    ) -> MarketMode:
        """
        Classify market mode based on indicator values with hysteresis.
        
        Args:
            adx_value: Current ADX value
            atr_ratio: Current ATR ratio
            bandwidth: Current Bollinger Band width
            current_mode: Current market mode for hysteresis
        
        Returns:
            Classified market mode
        """
        # TREND mode logic
        if current_mode == MarketMode.TREND:
            # Use exit thresholds for staying in TREND (hysteresis)
            if (adx_value >= self.adx_trend_exit and 
                atr_ratio >= self.atr_trend_min * 0.8):
                return MarketMode.TREND
        else:
            # Use enter thresholds for entering TREND
            if (adx_value >= self.adx_trend_enter and 
                atr_ratio >= self.atr_trend_min):
                return MarketMode.TREND
        
        # RANGE mode logic
        if current_mode == MarketMode.RANGE:
            # Use exit thresholds for staying in RANGE (hysteresis)
            if (adx_value <= self.adx_range_exit and 
                bandwidth <= self.bw_range_exit):
                return MarketMode.RANGE
        else:
            # Use enter thresholds for entering RANGE
            if (adx_value <= self.adx_range_enter and 
                bandwidth <= self.bw_range_enter and 
                atr_ratio <= self.atr_range_max):
                return MarketMode.RANGE
        
        # Default to NEUTRAL if no clear trend or range
        return MarketMode.NEUTRAL
    
    def get_current_mode(self) -> MarketMode:
        """Get the current market mode without updating."""
        return self.current_mode
    
    def get_last_mode_change(self) -> datetime | None:
        """Get the timestamp of the last mode change."""
        return self.last_mode_change