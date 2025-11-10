"""
State management for scalp bot

Defines market states and position states for the dip scalping strategy.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class MarketState(Enum):
    """Market condition states for scalp trading"""
    NORMAL = "normal"
    SPIKE_DOWN = "spike_down" 
    SPIKE_UP = "spike_up"
    COOLDOWN = "cooldown"


class PositionState(Enum):
    """Position states for the bot"""
    FLAT = "flat"
    LONG = "long"


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str  # "LONG" for now
    entry_price: float
    size_pct: float  # percentage of portfolio
    entry_time: datetime
    stop_loss: float
    take_profit: float
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    holding_bars: int = 0
    
    def update_excursions(self, current_price: float) -> None:
        """Update MFE and MAE based on current price"""
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        
        if pnl_pct > self.max_favorable_excursion:
            self.max_favorable_excursion = pnl_pct
            
        if pnl_pct < self.max_adverse_excursion:
            self.max_adverse_excursion = pnl_pct


@dataclass 
class SymbolState:
    """Per-symbol state tracking"""
    symbol: str
    market_state: MarketState = MarketState.NORMAL
    position_state: PositionState = PositionState.FLAT
    last_spike_timestamp: Optional[datetime] = None
    recent_low_since_spike: Optional[float] = None
    recent_high_since_spike: Optional[float] = None
    cooldown_bars_remaining: int = 0
    open_position: Optional[Position] = None
    
    def reset_spike_tracking(self) -> None:
        """Reset spike-related tracking variables"""
        self.recent_low_since_spike = None
        self.recent_high_since_spike = None
        self.last_spike_timestamp = None
        self.cooldown_bars_remaining = 0