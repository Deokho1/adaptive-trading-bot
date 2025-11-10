"""
RSI Mean Reversion Strategy for RANGE market mode.

This strategy implements a mean-reversion approach based on RSI oversold conditions
combined with Bollinger Band support, designed for sideways market conditions.
"""

from typing import List

from core.types import MarketMode, OrderSide
from strategy.base import Strategy, StrategyContext, TradeSignal


class RSIMeanReversionStrategy(Strategy):
    """
    Mean-reversion strategy based on RSI and Bollinger Bands.
    
    Entry Conditions:
    - Market mode is RANGE
    - RSI <= rsi_entry (oversold, typically 30)
    - Price <= Bollinger Band lower (additional support confirmation)
    
    Exit Conditions:
    - Take profit at fixed percentage gain (take_profit_pct)
    - RSI returns to neutral level (rsi_exit) when in profit
    - Stop loss: maximum of fixed stop or ATR-based dynamic stop
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the RSI mean reversion strategy.
        
        Args:
            config: Strategy configuration containing range-specific parameters
        """
        super().__init__(config)
        
        # Extract range strategy config with defaults
        self.per_trade_pct = config.get("per_trade_pct", 0.03)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_entry = config.get("rsi_entry", 30.0)
        self.rsi_exit = config.get("rsi_exit", 50.0)
        self.take_profit_pct = config.get("take_profit_pct", 0.04)
        self.fixed_stop_pct = config.get("fixed_stop_pct", 0.02)
        self.atr_stop_n = config.get("atr_stop_n", 1.0)
    
    def generate_signals(self, context: StrategyContext) -> List[TradeSignal]:
        """
        Generate trading signals based on RSI mean reversion logic.
        
        Args:
            context: Current market context with candles, indicators, and position data
            
        Returns:
            List of TradeSignal objects (empty if no signals)
        """
        signals: List[TradeSignal] = []
        
        # Only active in RANGE mode
        if context.mode != MarketMode.RANGE:
            return signals
        
        candles = context.candles
        if not candles:
            return signals
        
        latest_candle = candles[-1]
        current_price = latest_candle.close
        
        # Extract required indicators
        rsi_values = context.indicators.get("rsi", [])
        atr_values = context.indicators.get("atr", [])
        bb_lower = context.indicators.get("bb_lower", [])
        bb_middle = context.indicators.get("bb_middle", [])
        bb_upper = context.indicators.get("bb_upper", [])
        
        if not rsi_values or not atr_values or not bb_lower:
            return signals
        
        # Get latest values
        rsi = rsi_values[-1]
        atr = atr_values[-1]
        lower_band = bb_lower[-1]
        
        # Entry logic: No existing position
        if context.position is None:
            return self._check_entry_conditions(
                context, current_price, rsi, lower_band
            )
        
        # Exit logic: Existing position
        else:
            return self._check_exit_conditions(
                context, current_price, rsi, atr
            )
    
    def _check_entry_conditions(
        self,
        context: StrategyContext,
        current_price: float,
        rsi: float,
        lower_band: float
    ) -> List[TradeSignal]:
        """
        Check entry conditions for new positions.
        
        Args:
            context: Strategy context
            current_price: Current market price
            rsi: Current RSI value
            lower_band: Bollinger Band lower value
            
        Returns:
            List containing BUY signal if conditions met, empty otherwise
        """
        signals: List[TradeSignal] = []
        
        # Entry condition: RSI oversold AND price at/below lower Bollinger Band
        if rsi <= self.rsi_entry and current_price <= lower_band:
            # Calculate position size
            amount_krw = context.portfolio_value * self.per_trade_pct
            
            if amount_krw > 0:
                signals.append(
                    TradeSignal(
                        symbol=context.symbol,
                        side=OrderSide.BUY,
                        mode=MarketMode.RANGE,
                        reason=f"range_rsi_entry (RSI: {rsi:.1f}, price: {current_price:,.0f}, BB lower: {lower_band:,.0f})",
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
        
        entry_price = position.entry_price
        pnl_ratio = (current_price - entry_price) / entry_price
        pnl_pct = pnl_ratio * 100
        
        # 1. Take profit at target percentage
        if pnl_ratio >= self.take_profit_pct:
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.RANGE,
                    reason=f"range_take_profit (target: {self.take_profit_pct:.1%}, actual: {pnl_pct:+.1f}%)",
                    size=position.size,
                )
            )
            return signals
        
        # 2. RSI exit (back to neutral) when in profit
        if rsi >= self.rsi_exit and current_price > entry_price:
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.RANGE,
                    reason=f"range_rsi_exit (RSI: {rsi:.1f}, profit: {pnl_pct:+.1f}%)",
                    size=position.size,
                )
            )
            return signals
        
        # 3. Stop loss: maximum of fixed stop and ATR-based stop
        fixed_stop_price = entry_price * (1 - self.fixed_stop_pct)
        atr_stop_price = entry_price - (self.atr_stop_n * atr)
        stop_price = max(fixed_stop_price, atr_stop_price)
        
        if current_price <= stop_price:
            # Determine which stop was triggered
            if stop_price == fixed_stop_price:
                stop_type = "fixed"
                reference = f"{self.fixed_stop_pct:.1%}"
            else:
                stop_type = "ATR"
                reference = f"{self.atr_stop_n}x ATR"
            
            signals.append(
                TradeSignal(
                    symbol=context.symbol,
                    side=OrderSide.SELL,
                    mode=MarketMode.RANGE,
                    reason=f"range_stop_loss ({stop_type}: {stop_price:,.0f}, loss: {pnl_pct:.1f}%)",
                    size=position.size,
                )
            )
            return signals
        
        return signals
    
    def get_bollinger_band_position(self, price: float, bb_lower: float, bb_middle: float, bb_upper: float) -> str:
        """
        Determine position relative to Bollinger Bands.
        
        Args:
            price: Current price
            bb_lower: Lower Bollinger Band
            bb_middle: Middle Bollinger Band  
            bb_upper: Upper Bollinger Band
            
        Returns:
            String describing position: "below", "lower", "middle", "upper", "above"
        """
        if price <= bb_lower:
            return "below"
        elif price <= bb_middle:
            return "lower"
        elif price <= bb_upper:
            return "upper"
        else:
            return "above"
    
    def __str__(self) -> str:
        return f"RSIMeanReversionStrategy(rsi_entry={self.rsi_entry}, per_trade={self.per_trade_pct:.1%})"