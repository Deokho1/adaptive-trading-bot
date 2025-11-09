"""
Volatility Breakout Strategy for TREND market mode.

This strategy implements a trend-following approach based on ATR volatility breakouts,
volume confirmation, and multiple exit conditions including trailing stops and RSI.
"""

from typing import List
from datetime import timedelta

from core.types import MarketMode, OrderSide
from strategy.base import Strategy, StrategyContext, TradeSignal


class VolatilityBreakoutStrategy(Strategy):
    """
    Trend-following strategy based on volatility breakouts.
    
    Entry Conditions:
    - Market mode is TREND
    - Price breaks above previous close + (ATR * k_atr)
    - Current volume > average volume * min_volume_factor
    
    Exit Conditions:
    - Fixed stop loss (stop_loss_pct)
    - Trailing stop from peak price (trail_stop_pct)
    - RSI overbought exit (rsi_exit) when in profit
    - Maximum hold time (max_hold_hours)
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the volatility breakout strategy.
        
        Args:
            config: Strategy configuration containing trend-specific parameters
        """
        super().__init__(config)
        
        # Extract trend strategy config with defaults
        self.k_atr = config.get("k_atr", 0.5)
        self.min_volume_factor = config.get("min_volume_factor", 1.2)
        self.per_trade_pct = config.get("per_trade_pct", 0.05)
        self.max_positions = config.get("max_positions", 3)
        self.rsi_period = config.get("rsi_period", 14)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.03)
        self.trail_stop_pct = config.get("trail_stop_pct", 0.05)
        self.rsi_exit = config.get("rsi_exit", 70.0)
        self.max_hold_hours = config.get("max_hold_hours", 24)
    
    def generate_signals(self, context: StrategyContext) -> List[TradeSignal]:
        """
        Generate trading signals based on volatility breakout logic.
        
        Args:
            context: Current market context with candles, indicators, and position data
            
        Returns:
            List of TradeSignal objects (empty if no signals)
        """
        signals: List[TradeSignal] = []
        
        # Only active in TREND mode
        if context.mode != MarketMode.TREND:
            return signals
        
        candles = context.candles
        if len(candles) < 2:  # Need at least 2 candles for breakout logic
            return signals
        
        latest_candle = candles[-1]
        previous_candle = candles[-2]
        
        # Extract required indicators
        atr_values = context.indicators.get("atr", [])
        rsi_values = context.indicators.get("rsi", [])
        volume_values = context.indicators.get("volume", [])
        
        if not atr_values or not rsi_values or not volume_values:
            return signals
        
        # Get latest values
        atr = atr_values[-1]
        rsi = rsi_values[-1]
        current_volume = volume_values[-1]
        current_price = latest_candle.close
        
        # Calculate average volume over recent period (default 20 candles)
        volume_window = min(20, len(volume_values))
        if volume_window > 0:
            avg_volume = sum(volume_values[-volume_window:]) / volume_window
        else:
            avg_volume = current_volume
        
        # Entry logic: No existing position
        if context.position is None:
            return self._check_entry_conditions(
                context, previous_candle, latest_candle, atr, current_volume, avg_volume
            )
        
        # Exit logic: Existing position
        else:
            return self._check_exit_conditions(
                context, current_price, rsi, atr
            )
    
    def _check_entry_conditions(
        self,
        context: StrategyContext,
        previous_candle,
        latest_candle,
        atr: float,
        current_volume: float,
        avg_volume: float
    ) -> List[TradeSignal]:
        """
        Check entry conditions for new positions.
        
        Args:
            context: Strategy context
            previous_candle: Previous candle data
            latest_candle: Current candle data
            atr: Current ATR value
            current_volume: Current volume
            avg_volume: Average volume over recent period
            
        Returns:
            List containing BUY signal if conditions met, empty otherwise
        """
        signals: List[TradeSignal] = []
        
        # Calculate breakout level
        prev_close = previous_candle.close
        target_price = prev_close + (atr * self.k_atr)
        
        # Check volume filter
        if current_volume < avg_volume * self.min_volume_factor:
            return signals
        
        # Check price breakout
        if latest_candle.close >= target_price:
            # Calculate position size
            amount_krw = context.portfolio_value * self.per_trade_pct
            
            if amount_krw > 0:
                signals.append(
                    TradeSignal(
                        symbol=context.symbol,
                        side=OrderSide.BUY,
                        mode=MarketMode.TREND,
                        reason=f"volatility_breakout (target: {target_price:,.0f}, close: {latest_candle.close:,.0f})",
                        amount_krw=amount_krw,
                    )
                )
        
        return signals
    
    def _check_exit_conditions(
        self,
        context: StrategyContext,
        current_price: float,
        rsi: float,
        atr: float
    ) -> List[TradeSignal]:
        """
        Check exit conditions for existing positions.
        
        Args:
            context: Strategy context
            current_price: Current market price
            rsi: Current RSI value
            atr: Current ATR value
            
        Returns:
            List containing SELL signal if exit condition met, empty otherwise
        """
        signals: List[TradeSignal] = []
        position = context.position
        
        if not position:
            return signals
        
        # 1. Fixed stop loss
        stop_loss_price = position.entry_price * (1 - self.stop_loss_pct)
        if current_price <= stop_loss_price:
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.TREND,
                    reason=f"trend_stop_loss (stop: {stop_loss_price:,.0f}, current: {current_price:,.0f})",
                    size=position.size,
                )
            )
            return signals
        
        # 2. Trailing stop (using peak price)
        trail_stop_price = position.peak_price * (1 - self.trail_stop_pct)
        if current_price <= trail_stop_price:
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.TREND,
                    reason=f"trend_trailing_stop (trail: {trail_stop_price:,.0f}, peak: {position.peak_price:,.0f})",
                    size=position.size,
                )
            )
            return signals
        
        # 3. RSI exit (if overbought and in profit)
        if rsi >= self.rsi_exit and current_price > position.entry_price:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.TREND,
                    reason=f"trend_rsi_exit (RSI: {rsi:.1f}, profit: {pnl_pct:+.1f}%)",
                    size=position.size,
                )
            )
            return signals
        
        # 4. Maximum hold time
        hold_duration = context.now - position.entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        if hold_hours >= self.max_hold_hours:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.TREND,
                    reason=f"trend_time_exit (held: {hold_hours:.1f}h, profit: {pnl_pct:+.1f}%)",
                    size=position.size,
                )
            )
            return signals
        
        return signals
    
    def __str__(self) -> str:
        return f"VolatilityBreakoutStrategy(k_atr={self.k_atr}, per_trade={self.per_trade_pct:.1%})"