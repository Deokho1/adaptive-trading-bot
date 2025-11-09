"""
Data models for the exchange layer.

This module defines the basic data structures used throughout
the trading bot for representing market data.
"""

from dataclasses import dataclass
from datetime import datetime


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