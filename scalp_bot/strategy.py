"""
Dip scalping strategy implementation

Core strategy logic for detecting dips and trading rebounds.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

from .config import ScalpConfig
from .state import MarketState, PositionState, SymbolState, Position
from .risk import RiskManager


class DipScalpStrategy:
    """Dip scalping strategy for short-term rebounds"""
    
    def __init__(self, config: ScalpConfig, risk_manager: RiskManager):
        """
        Initialize strategy
        
        Args:
            config: Strategy configuration
            risk_manager: Risk management instance
        """
        self.config = config
        self.risk_manager = risk_manager
        
        # Per-symbol state tracking
        self.symbol_states: Dict[str, SymbolState] = {}
        
        # Event logging for debugging
        self.events: List[Dict] = []
        
        # Performance tracking
        self.stats = {
            "total_signals": 0,
            "entries": 0,
            "exits": 0,
            "spike_detections": 0
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.symbol_states.clear()
        self.events.clear()
        self.stats = {k: 0 for k in self.stats}
    
    def _get_symbol_state(self, symbol: str) -> SymbolState:
        """Get or create symbol state"""
        if symbol not in self.symbol_states:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)
        return self.symbol_states[symbol]
    
    def _log_event(self, timestamp: datetime, symbol: str, event_type: str,
                   old_state: str = "", new_state: str = "", **kwargs) -> None:
        """Log strategy events for debugging"""
        event = {
            "timestamp": timestamp,
            "symbol": symbol,
            "event_type": event_type,
            "old_state": old_state,
            "new_state": new_state,
            **kwargs
        }
        self.events.append(event)
    
    def on_bar(self, symbol: str, bar_index: int, bar_row: pd.Series,
               df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        Process new bar and generate signals
        
        Args:
            symbol: Trading symbol
            bar_index: Current bar index
            bar_row: Current bar data
            df_slice: Historical data slice up to current bar
            current_time: Current timestamp
            
        Returns:
            Signal dictionary or None
        """
        state = self._get_symbol_state(symbol)
        self.stats["total_signals"] += 1
        
        # Skip if insufficient data
        if len(df_slice) < 20:
            return None
        
        # Update market state classification
        old_market_state = state.market_state
        self._update_market_state(state, bar_row, current_time)
        
        # Log state changes
        if old_market_state != state.market_state:
            self._log_event(
                current_time, symbol, "STATE_CHANGE",
                old_state=old_market_state.value,
                new_state=state.market_state.value,
                price=bar_row['close'],
                pct_change_5m=bar_row.get('pct_change_5m', 0),
                pct_change_15m=bar_row.get('pct_change_15m', 0),
                volume_ratio=bar_row.get('volume_ratio', 1)
            )
        
        # Update open position if we have one
        signal = None
        if state.open_position is not None:
            signal = self._check_exit_conditions(state, bar_row, current_time)
            if signal:
                return signal
        
        # Check for new entry opportunities
        if state.position_state == PositionState.FLAT:
            signal = self._check_entry_conditions(state, bar_row, df_slice, current_time)
        
        return signal
    
    def _update_market_state(self, state: SymbolState, bar_row: pd.Series,
                           current_time: datetime) -> None:
        """Update market state based on price changes"""
        pct_change_5m = bar_row.get('pct_change_5m', 0)
        pct_change_15m = bar_row.get('pct_change_15m', 0)
        
        # Handle cooldown state
        if state.market_state == MarketState.COOLDOWN:
            state.cooldown_bars_remaining -= 1
            if state.cooldown_bars_remaining <= 0:
                state.market_state = MarketState.NORMAL
                state.reset_spike_tracking()
                return
        
        # Check for spikes
        is_spike_down = (pct_change_5m <= self.config.spike_down_threshold_5m or
                        pct_change_15m <= self.config.spike_down_threshold_15m)
        
        is_spike_up = (pct_change_5m >= self.config.spike_up_threshold_5m or
                      pct_change_15m >= self.config.spike_up_threshold_15m)
        
        current_price = bar_row['close']
        
        # State transitions
        if is_spike_down and state.market_state not in [MarketState.SPIKE_DOWN, MarketState.COOLDOWN]:
            state.market_state = MarketState.SPIKE_DOWN
            state.last_spike_timestamp = current_time
            state.recent_low_since_spike = current_price
            state.recent_high_since_spike = current_price
            
            self._log_event(
                current_time, state.symbol, "SPIKE_DETECTED",
                new_state="SPIKE_DOWN",
                price=current_price,
                pct_change_5m=pct_change_5m,
                pct_change_15m=pct_change_15m
            )
            self.stats["spike_detections"] += 1
            
        elif is_spike_up and state.market_state not in [MarketState.SPIKE_UP, MarketState.COOLDOWN]:
            state.market_state = MarketState.SPIKE_UP
            state.last_spike_timestamp = current_time
            state.cooldown_bars_remaining = self.config.cooldown_bars_after_spike
            
            # Transition directly to cooldown after spike up
            if state.cooldown_bars_remaining > 0:
                state.market_state = MarketState.COOLDOWN
            
        # Update extremes during spike
        if state.market_state == MarketState.SPIKE_DOWN:
            if state.recent_low_since_spike is None or current_price < state.recent_low_since_spike:
                state.recent_low_since_spike = current_price
            if state.recent_high_since_spike is None or current_price > state.recent_high_since_spike:
                state.recent_high_since_spike = current_price
            
            # Check if spike is over (price recovering)
            if pct_change_5m > 0:  # Positive change
                state.market_state = MarketState.COOLDOWN
                state.cooldown_bars_remaining = self.config.cooldown_bars_after_spike
    
    def _check_entry_conditions(self, state: SymbolState, bar_row: pd.Series,
                               df_slice: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """Check if entry conditions are met - MULTI-STRATEGY APPROACH"""
        
        # 🎯 REALISTIC HIGH-FREQUENCY SCALPING with multiple strategies
        if not self.risk_manager.can_open_new_position(state.symbol, current_time.date()):
            return None
        
        current_price = bar_row['close']
        volume_ratio = bar_row.get('volume_ratio', 1.0)
        
        # Calculate technical indicators
        rsi = self._calculate_rsi(df_slice, period=14)
        ema_20 = self._calculate_ema(df_slice, period=20)
        atr_pct = self._calculate_atr_percent(df_slice, period=14)
        
        # Price distance from EMA
        ema_diff_pct = ((current_price - ema_20) / ema_20) * 100 if ema_20 > 0 else 0
        
        # EMA slope (trend direction)
        ema_slope = self._calculate_ema_slope(df_slice, period=20)
        
        # 🚀 STRATEGY 1: DIP_BUY - Buy oversold dips (BALANCED)
        dip_buy_conditions = {
            "rsi_oversold": rsi < 32,  # Moderately oversold
            "below_ema": ema_diff_pct < -0.15,  # Moderately below EMA
            "volume_ok": volume_ratio >= 1.2,  # Moderate volume requirement
            "not_in_strong_downtrend": ema_slope > -0.4,  # Allow some downtrend
            "atr_filter": atr_pct > 0.12  # Moderate volatility filter
        }
        
        # 🚀 STRATEGY 2: BREAKOUT_BUY - Buy momentum breakouts (BALANCED)
        breakout_buy_conditions = {
            "rsi_momentum": 58 < rsi < 72,  # Good momentum range
            "above_ema": ema_diff_pct > 0.08,  # Clear but not extreme breakout
            "high_volume": volume_ratio >= 1.3,  # Good volume
            "ema_rising": ema_slope > 0.15,  # Clear uptrend
            "atr_filter": atr_pct > 0.15  # Moderate volatility requirement
        }
        
        # 🚀 STRATEGY 3: BREAKDOWN_SHORT - Short overbought breakdowns (BALANCED)
        breakdown_short_conditions = {
            "rsi_overbought": rsi > 68,  # Moderately overbought
            "ema_turning_down": ema_slope < -0.15,  # Clear reversal signal
            "volume_confirmation": volume_ratio >= 1.25,  # Good volume
            "price_near_ema": abs(ema_diff_pct) < 0.25,  # Near EMA
            "atr_filter": atr_pct > 0.14  # Moderate volatility requirement
        }
        
        # Check which strategy triggers
        trade_type = None
        side = None
        
        if all(dip_buy_conditions.values()):
            trade_type = "dip_buy"
            side = "LONG"
        elif all(breakout_buy_conditions.values()):
            trade_type = "breakout_buy" 
            side = "LONG"
        elif all(breakdown_short_conditions.values()):
            trade_type = "breakdown_short"
            side = "SHORT"
        
        if trade_type is None:
            # Log rejected signals for debugging
            self._log_event(
                current_time, state.symbol, "ENTRY_REJECTED",
                note=f"NO_STRATEGY_MATCHED",
                price=current_price,
                rsi=rsi,
                ema_diff_pct=ema_diff_pct,
                volume_ratio=volume_ratio,
                atr_pct=atr_pct
            )
            return None
        
        # 🎯 ADAPTIVE POSITION SIZING based on volatility
        base_position_size_pct = self.risk_manager.calculate_position_size(
            state.symbol, current_price, current_price * (1 - self.config.stop_loss_pct / 100)
        )
        
        # Volatility-based scaling
        if atr_pct > 0.3:  # High volatility
            position_size_pct = base_position_size_pct * 1.5
        elif atr_pct < 0.1:  # Low volatility  
            position_size_pct = base_position_size_pct * 0.5
        else:
            position_size_pct = base_position_size_pct
        
        # Cap position size
        position_size_pct = min(position_size_pct, self.config.per_trade_risk_pct)
        
        if position_size_pct <= 0:
            return None
        
        # Calculate stop loss and take profit
        if side == "LONG":
            stop_loss_price = current_price * (1 - self.config.stop_loss_pct / 100)
            take_profit_price = current_price * (1 + self.config.take_profit_pct / 100)
        else:  # SHORT
            stop_loss_price = current_price * (1 + self.config.stop_loss_pct / 100)  
            take_profit_price = current_price * (1 - self.config.take_profit_pct / 100)
        
        # Create position
        position = Position(
            symbol=state.symbol,
            side=side,
            entry_price=current_price,
            size_pct=position_size_pct,
            entry_time=current_time,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price
        )
        
        # Store additional info for RSI-based exit
        position.entry_rsi = rsi
        position.entry_ema_diff = ema_diff_pct
        position.entry_volume_ratio = volume_ratio
        position.entry_atr_pct = atr_pct
        position.trade_type = trade_type
        
        # Update state
        state.position_state = PositionState.LONG if side == "LONG" else PositionState.SHORT
        state.open_position = position
        self.risk_manager.add_position(position)
        
        self.stats["entries"] += 1
        
        # Enhanced logging for multi-strategy
        self._log_event(
            current_time, state.symbol, "ENTRY_SIGNAL",
            note=f"{trade_type.upper()}_EXECUTED",
            price=current_price,
            size_pct=position_size_pct,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            rsi=rsi,
            ema_diff_pct=ema_diff_pct,
            volume_ratio=volume_ratio,
            atr_pct=atr_pct,
            trade_type=trade_type
        )
        
        return {
            "action": "BUY" if side == "LONG" else "SELL",
            "symbol": state.symbol,
            "price": current_price,
            "size_pct": position_size_pct,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "reason": f"{trade_type}_entry",
            "market_state": state.market_state.value,
            "trade_type": trade_type,
            "rsi_at_entry": rsi,
            "ema_diff_at_entry": ema_diff_pct,
            "volume_ratio_at_entry": volume_ratio,
            "atr_at_entry": atr_pct
        }
    
    def _check_exit_conditions(self, state: SymbolState, bar_row: pd.Series,
                              current_time: datetime) -> Optional[Dict]:
        """Check exit conditions for open position - RSI-BASED EXITS"""
        
        if state.open_position is None:
            return None
        
        position = state.open_position
        current_price = bar_row['close']
        
        # Update position metrics
        position.update_excursions(current_price)
        position.holding_bars += 1
        
        # Calculate current RSI for exit decision
        # Note: We need the full df_slice for RSI calculation, but it's not available here
        # For now, use simple price-based exits, but this should be enhanced
        
        # Check exit conditions
        exit_reason = None
        
        # 🎯 RSI-BASED EXIT LOGIC (Enhanced)
        # For now, use time-based proxy for RSI exit until we can access df_slice
        rsi_exit_triggered = False
        
        if hasattr(position, 'trade_type'):
            if position.trade_type in ['dip_buy', 'breakout_buy'] and position.side == "LONG":
                # For longs: exit when RSI crosses back to 50+ (from oversold/momentum)
                # Simplified: use time and price movement as proxy
                price_gain = (current_price - position.entry_price) / position.entry_price * 100
                if price_gain > 0.1 and position.holding_bars > 5:  # Proxy for RSI recovery
                    rsi_exit_triggered = True
                    
            elif position.trade_type == 'breakdown_short' and position.side == "SHORT":
                # For shorts: exit when RSI crosses back below 50 (from overbought)
                price_loss = (position.entry_price - current_price) / position.entry_price * 100
                if price_loss > 0.1 and position.holding_bars > 5:  # Proxy for RSI recovery
                    rsi_exit_triggered = True
        
        # Exit priority: SL > TP > RSI > Time
        if position.side == "LONG":
            if current_price <= position.stop_loss:
                exit_reason = "stop_loss"
            elif current_price >= position.take_profit:
                exit_reason = "take_profit"
            elif rsi_exit_triggered:
                exit_reason = "rsi_exit"
            elif position.holding_bars >= self.config.max_holding_bars:
                exit_reason = "time_exit"
        else:  # SHORT
            if current_price >= position.stop_loss:
                exit_reason = "stop_loss"
            elif current_price <= position.take_profit:
                exit_reason = "take_profit"
            elif rsi_exit_triggered:
                exit_reason = "rsi_exit"
            elif position.holding_bars >= self.config.max_holding_bars:
                exit_reason = "time_exit"
        
        if exit_reason is None:
            return None
        
        # Calculate P&L based on position side
        if position.side == "LONG":
            pnl_abs = (current_price - position.entry_price) * (position.size_pct / 100)
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        else:  # SHORT
            pnl_abs = (position.entry_price - current_price) * (position.size_pct / 100)
            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100
        
        # Update risk manager
        self.risk_manager.remove_position(state.symbol)
        self.risk_manager.update_daily_pnl(current_time.date(), pnl_abs)
        
        # Check for drawdown-based exposure reduction
        if self.risk_manager.get_max_drawdown() > 2.0:  # 2% MDD threshold
            self.risk_manager.reduce_exposure(0.8)  # Reduce by 20%
        
        # Update state
        state.position_state = PositionState.FLAT
        exit_position = state.open_position
        state.open_position = None
        
        self.stats["exits"] += 1
        
        # Enhanced logging with trade type and entry conditions
        self._log_event(
            current_time, state.symbol, "EXIT_SIGNAL",
            note=f"{exit_reason.upper()}_{getattr(position, 'trade_type', 'unknown').upper()}",
            price=current_price,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            holding_bars=position.holding_bars,
            mfe_pct=position.max_favorable_excursion,
            mae_pct=position.max_adverse_excursion,
            trade_type=getattr(position, 'trade_type', 'unknown'),
            entry_rsi=getattr(position, 'entry_rsi', 0),
            entry_ema_diff=getattr(position, 'entry_ema_diff', 0),
            entry_volume_ratio=getattr(position, 'entry_volume_ratio', 1.0),
            entry_atr_pct=getattr(position, 'entry_atr_pct', 0)
        )
        
        return {
            "action": "SELL" if position.side == "LONG" else "BUY",
            "symbol": state.symbol,
            "price": current_price,
            "size_pct": position.size_pct,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "reason": exit_reason,
            "holding_bars": position.holding_bars,
            "entry_price": position.entry_price,
            "max_favorable_excursion": position.max_favorable_excursion,
            "max_adverse_excursion": position.max_adverse_excursion,
            "trade_type": getattr(position, 'trade_type', 'unknown'),
            "realized_pnl_pct": pnl_pct
        }
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(df) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
            
        closes = df['close'].values
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_ema(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate EMA (Exponential Moving Average)"""
        if len(df) < period:
            return df['close'].iloc[-1] if len(df) > 0 else 0.0
        
        closes = df['close'].values
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema = np.mean(closes[:period])
        
        # Calculate EMA
        for i in range(period, len(closes)):
            ema = (closes[i] * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def _calculate_atr_percent(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price"""
        if len(df) < period + 1:
            return 0.1  # Default low volatility
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range calculation
        tr1 = high[1:] - low[1:]  # High - Low
        tr2 = np.abs(high[1:] - close[:-1])  # |High - Previous Close|
        tr3 = np.abs(low[1:] - close[:-1])   # |Low - Previous Close|
        
        true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
        
        if len(true_ranges) < period:
            return 0.1
        
        atr = np.mean(true_ranges[-period:])
        current_price = close[-1]
        
        atr_percent = (atr / current_price) * 100 if current_price > 0 else 0.1
        
        return float(atr_percent)
    
    def _calculate_ema_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate EMA slope to determine trend direction"""
        if len(df) < period + 5:
            return 0.0
        
        # Get EMAs for last 5 periods to calculate slope
        ema_values = []
        for i in range(5):
            subset = df.iloc[:-i] if i > 0 else df
            ema = self._calculate_ema(subset, period)
            ema_values.append(ema)
        
        ema_values.reverse()  # Oldest to newest
        
        if len(ema_values) < 2:
            return 0.0
        
        # Calculate slope (change per period)
        slope = (ema_values[-1] - ema_values[0]) / len(ema_values)
        slope_percent = (slope / ema_values[-1]) * 100 if ema_values[-1] > 0 else 0
        
        return float(slope_percent)
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy statistics"""
        return {
            **self.stats,
            "open_positions": len([s for s in self.symbol_states.values() 
                                 if s.position_state == PositionState.LONG]),
            "symbols_tracked": len(self.symbol_states)
        }