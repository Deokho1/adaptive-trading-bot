"""
Risk management for scalp bot

Handles position sizing, exposure limits, and daily loss limits.
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, List
from dataclasses import dataclass

from .config import ScalpConfig
from .state import Position


@dataclass
class DailyPnL:
    """Track daily P&L"""
    date: date
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    

class RiskManager:
    """Manages risk for the scalp trading bot"""
    
    def __init__(self, config: ScalpConfig, initial_equity: float = 10000.0):
        """
        Initialize risk manager
        
        Args:
            config: Scalp bot configuration
            initial_equity: Starting equity amount
        """
        self.config = config
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        
        # Track positions and P&L
        self.open_positions: Dict[str, Position] = {}
        self.daily_pnl: Dict[date, DailyPnL] = {}
        self.total_realized_pnl = 0.0
        
        # Risk tracking
        self.daily_loss_limit_hit = False
        self.current_date: Optional[date] = None
    
    def update_equity(self, new_equity: float) -> None:
        """Update current equity"""
        self.current_equity = new_equity
    
    def can_open_new_position(self, symbol: str, current_date: date) -> bool:
        """
        Check if we can open a new position
        
        Args:
            symbol: Symbol to check
            current_date: Current trading date
            
        Returns:
            True if position can be opened, False otherwise
        """
        # Update current date
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_loss_limit_hit = False  # Reset daily limit
        
        # Check daily loss limit
        if self._check_daily_loss_limit(current_date):
            return False
        
        # Check if already have position in this symbol
        if symbol in self.open_positions:
            return False  # max_position_per_symbol = 1
        
        # Check total exposure limit
        if self._get_total_exposure_pct() >= self.config.max_total_exposure_pct:
            return False
        
        # Check max positions per symbol
        symbol_positions = sum(1 for pos in self.open_positions.values() 
                             if pos.symbol == symbol)
        if symbol_positions >= self.config.max_position_per_symbol:
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Symbol being traded
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size as percentage of equity
        """
        # Risk amount (% of equity we're willing to lose)
        risk_amount = self.current_equity * (self.config.per_trade_risk_pct / 100)
        
        # Price risk per unit
        price_risk_per_unit = abs(entry_price - stop_loss_price)
        
        if price_risk_per_unit == 0:
            return 0.0
        
        # Position value based on risk
        position_value = risk_amount / (price_risk_per_unit / entry_price)
        
        # Convert to percentage of equity
        position_size_pct = (position_value / self.current_equity) * 100
        
        # Cap at reasonable limits
        max_single_position = min(20.0, self.config.max_total_exposure_pct / 2)
        position_size_pct = min(position_size_pct, max_single_position)
        
        return position_size_pct
    
    def apply_fees_and_slippage(self, order_notional: float) -> float:
        """
        Calculate effective notional after fees and slippage
        
        Args:
            order_notional: Gross order amount
            
        Returns:
            Net order amount after costs
        """
        fees = order_notional * self.config.fee_rate
        slippage = order_notional * self.config.slippage_rate
        
        return order_notional - fees - slippage
    
    def add_position(self, position: Position) -> None:
        """Add a new position"""
        self.open_positions[position.symbol] = position
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove and return a position"""
        return self.open_positions.pop(symbol, None)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.open_positions.get(symbol)
    
    def update_daily_pnl(self, trade_date: date, realized_pnl: float) -> None:
        """
        Update daily P&L tracking
        
        Args:
            trade_date: Date of the trade
            realized_pnl: Realized P&L from the trade
        """
        if trade_date not in self.daily_pnl:
            self.daily_pnl[trade_date] = DailyPnL(date=trade_date)
        
        self.daily_pnl[trade_date].realized_pnl += realized_pnl
        self.daily_pnl[trade_date].total_pnl = self.daily_pnl[trade_date].realized_pnl
        
        self.total_realized_pnl += realized_pnl
        self.current_equity += realized_pnl
    
    def _check_daily_loss_limit(self, current_date: date) -> bool:
        """Check if daily loss limit has been hit"""
        if current_date not in self.daily_pnl:
            return False
        
        daily_loss_limit = self.current_equity * (self.config.daily_loss_limit_pct / 100)
        daily_pnl = self.daily_pnl[current_date].total_pnl
        
        if daily_pnl <= -daily_loss_limit:
            self.daily_loss_limit_hit = True
            return True
        
        return False
    
    def _get_total_exposure_pct(self) -> float:
        """Calculate current total exposure as percentage"""
        total_exposure = sum(pos.size_pct for pos in self.open_positions.values())
        return total_exposure
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            "current_equity": self.current_equity,
            "total_realized_pnl": self.total_realized_pnl,
            "total_pnl_pct": (self.total_realized_pnl / self.initial_equity) * 100,
            "open_positions": len(self.open_positions),
            "total_exposure_pct": self._get_total_exposure_pct(),
            "daily_loss_limit_hit": self.daily_loss_limit_hit,
            "max_drawdown_pct": self._calculate_max_drawdown()
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.daily_pnl:
            return 0.0
        
        equity_values = [self.initial_equity]
        running_equity = self.initial_equity
        
        for daily in sorted(self.daily_pnl.values(), key=lambda x: x.date):
            running_equity += daily.realized_pnl
            equity_values.append(running_equity)
        
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown percentage"""
        return self._calculate_max_drawdown()