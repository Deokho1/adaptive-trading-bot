"""
Data models for the exchange layer.

This module defines the basic data structures used throughout
the trading bot for representing market data and trading positions.
"""

from dataclasses import dataclass
from datetime import datetime

from core.types import MarketMode


@dataclass
class Candle:
    """
    Represents a single candle (OHLCV) data point.
    
    All timestamps are in UTC for consistency.
    Prices and volume are represented as floats.
    """
    symbol: str
    timestamp: datetime  # candle close time in UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __str__(self) -> str:
        return (
            f"Candle({self.symbol} @ {self.timestamp.isoformat()}: "
            f"O:{self.open} H:{self.high} L:{self.low} C:{self.close} V:{self.volume})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Position:
    """
    Represents an open trading position.
    
    Tracks entry details and current status for risk management
    and profit/loss calculations.
    """
    symbol: str
    mode: MarketMode     # Market mode when position was opened
    entry_price: float   # Price at which position was entered
    size: float          # Amount of cryptocurrency held
    entry_time: datetime # When the position was opened
    peak_price: float    # Highest price seen since entry (for trailing stops)
    
    def __str__(self) -> str:
        return (
            f"Position({self.symbol} {self.mode} @ {self.entry_price:,.0f}: "
            f"size={self.size:.6f}, peak={self.peak_price:,.0f}, "
            f"entry={self.entry_time.isoformat()})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def current_value_krw(self, current_price: float) -> float:
        """Calculate current position value in KRW."""
        return self.size * current_price
    
    def unrealized_pnl_krw(self, current_price: float) -> float:
        """Calculate unrealized P&L in KRW."""
        current_value = self.current_value_krw(current_price)
        entry_value = self.size * self.entry_price
        return current_value - entry_value
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100