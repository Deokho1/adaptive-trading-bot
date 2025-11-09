"""
Risk manager for controlling trading risk.

This module provides functionality to manage trading risk through
capital allocation limits, daily loss limits, and position count controls.
"""

from datetime import datetime
from typing import List

from core.types import MarketMode
from exchange.models import Position


class RiskManager:
    """
    Manages trading risk through various limits and controls.
    
    Provides capital allocation limits per market mode, daily loss limits,
    and maximum position count controls to prevent excessive risk exposure.
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the risk manager with configuration.
        
        Args:
            config: Configuration dictionary containing risk settings
        """
        rconf = config.get("risk", {})
        
        # Capital allocation limits (as percentage of portfolio)
        self.max_trend_cap_pct = rconf.get("max_trend_cap_pct", 30.0)
        self.max_range_cap_pct = rconf.get("max_range_cap_pct", 20.0)
        
        # Daily loss limit (as percentage of portfolio)
        self.daily_loss_limit_pct = rconf.get("daily_loss_limit_pct", 5.0)
        
        # Maximum total positions
        self.max_positions_total = rconf.get("max_positions_total", 3)
        
        # Daily tracking state
        self.daily_start_value: float | None = None
        self.daily_start_date: str | None = None  # "YYYY-MM-DD"
        self.kill_switch_triggered: bool = False
    
    def reset_daily_start(self, portfolio_value: float, now: datetime) -> None:
        """
        Reset starting portfolio value for the new day.
        
        Args:
            portfolio_value: Current portfolio value in KRW
            now: Current datetime
        """
        today = now.strftime("%Y-%m-%d")
        
        if self.daily_start_date != today:
            # New day - reset tracking
            self.daily_start_value = portfolio_value
            self.daily_start_date = today
            self.kill_switch_triggered = False
    
    def update_daily_pnl(self, portfolio_value: float, now: datetime) -> float:
        """
        Update daily P&L and trigger kill switch if loss exceeds limit.
        
        Args:
            portfolio_value: Current portfolio value in KRW
            now: Current datetime
            
        Returns:
            Current daily P&L ratio (negative means loss)
        """
        self.reset_daily_start(portfolio_value, now)
        
        if self.daily_start_value is None or self.daily_start_value == 0:
            return 0.0
        
        # Calculate daily P&L ratio
        daily_pnl_ratio = ((portfolio_value - self.daily_start_value) / self.daily_start_value) * 100
        
        # Check if daily loss limit exceeded
        if daily_pnl_ratio <= -self.daily_loss_limit_pct:
            self.kill_switch_triggered = True
        
        return daily_pnl_ratio
    
    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is currently active.
        
        Returns:
            True if kill switch is triggered (stop all trading)
        """
        return self.kill_switch_triggered
    
    def can_open_trade(
        self,
        mode: MarketMode,
        portfolio_value: float,
        existing_positions: List[Position],
        new_trade_amount_krw: float,
        symbol: str,
    ) -> bool:
        """
        Check if opening a new trade is allowed based on risk limits.
        
        Args:
            mode: Market mode for the new trade
            portfolio_value: Current portfolio value in KRW
            existing_positions: List of current positions
            new_trade_amount_krw: KRW amount for the new trade
            symbol: Trading pair symbol
            
        Returns:
            True if the new trade is allowed, False otherwise
        """
        # Check kill switch
        if self.kill_switch_triggered:
            return False
        
        # Check if position already exists for this symbol
        for position in existing_positions:
            if position.symbol == symbol:
                return False  # Don't allow multiple positions for same symbol
        
        # Check total position count
        if len(existing_positions) >= self.max_positions_total:
            return False
        
        # Check mode-specific capital allocation
        if mode == MarketMode.TREND:
            return self._check_trend_capital_limit(
                portfolio_value, existing_positions, new_trade_amount_krw
            )
        elif mode == MarketMode.RANGE:
            return self._check_range_capital_limit(
                portfolio_value, existing_positions, new_trade_amount_krw
            )
        else:
            # NEUTRAL mode - be more conservative
            return False
    
    def _check_trend_capital_limit(
        self,
        portfolio_value: float,
        existing_positions: List[Position],
        new_trade_amount_krw: float
    ) -> bool:
        """
        Check if new TREND trade fits within capital allocation limit.
        
        Args:
            portfolio_value: Current portfolio value in KRW
            existing_positions: List of current positions
            new_trade_amount_krw: KRW amount for the new trade
            
        Returns:
            True if trade is within TREND capital limit
        """
        # Calculate current TREND position value (approximate)
        current_trend_value = sum(
            pos.entry_price * pos.size  # Rough approximation using entry value
            for pos in existing_positions
            if pos.mode == MarketMode.TREND
        )
        
        # Check if new trade would exceed limit
        max_trend_value = portfolio_value * (self.max_trend_cap_pct / 100)
        total_after_trade = current_trend_value + new_trade_amount_krw
        
        return total_after_trade <= max_trend_value
    
    def _check_range_capital_limit(
        self,
        portfolio_value: float,
        existing_positions: List[Position],
        new_trade_amount_krw: float
    ) -> bool:
        """
        Check if new RANGE trade fits within capital allocation limit.
        
        Args:
            portfolio_value: Current portfolio value in KRW
            existing_positions: List of current positions
            new_trade_amount_krw: KRW amount for the new trade
            
        Returns:
            True if trade is within RANGE capital limit
        """
        # Calculate current RANGE position value (approximate)
        current_range_value = sum(
            pos.entry_price * pos.size  # Rough approximation using entry value
            for pos in existing_positions
            if pos.mode == MarketMode.RANGE
        )
        
        # Check if new trade would exceed limit
        max_range_value = portfolio_value * (self.max_range_cap_pct / 100)
        total_after_trade = current_range_value + new_trade_amount_krw
        
        return total_after_trade <= max_range_value
    
    def get_available_capital_for_mode(
        self,
        mode: MarketMode,
        portfolio_value: float,
        existing_positions: List[Position]
    ) -> float:
        """
        Calculate available capital for a specific market mode.
        
        Args:
            mode: Market mode to check
            portfolio_value: Current portfolio value in KRW
            existing_positions: List of current positions
            
        Returns:
            Available capital in KRW for the specified mode
        """
        if mode == MarketMode.TREND:
            max_capital = portfolio_value * (self.max_trend_cap_pct / 100)
            used_capital = sum(
                pos.entry_price * pos.size
                for pos in existing_positions
                if pos.mode == MarketMode.TREND
            )
        elif mode == MarketMode.RANGE:
            max_capital = portfolio_value * (self.max_range_cap_pct / 100)
            used_capital = sum(
                pos.entry_price * pos.size
                for pos in existing_positions
                if pos.mode == MarketMode.RANGE
            )
        else:
            return 0.0  # No capital allocation for NEUTRAL
        
        return max(0.0, max_capital - used_capital)
    
    def get_risk_status(self) -> dict:
        """
        Get current risk manager status.
        
        Returns:
            Dictionary with risk status information
        """
        return {
            "kill_switch_active": self.kill_switch_triggered,
            "daily_start_date": self.daily_start_date,
            "daily_start_value": self.daily_start_value,
            "max_trend_cap_pct": self.max_trend_cap_pct,
            "max_range_cap_pct": self.max_range_cap_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_positions_total": self.max_positions_total,
        }