"""
Type definitions for the trading bot.

This module contains enums and type definitions used throughout
the trading bot system.
"""

from enum import Enum


class TradingMode(str, Enum):
    """
    Trading mode selection for bot operation.
    
    - PAPER: Virtual trading using real market data but simulated orders
    - LIVE: Real trading with actual orders and money
    - BACKTEST: Historical simulation using past data
    """
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class MarketMode(Enum):
    """
    Market mode classification for trading strategy selection.
    
    - TREND: Strong directional movement, suitable for trend-following strategies
    - RANGE: Sideways movement, suitable for mean-reversion strategies  
    - NEUTRAL: Uncertain conditions, cautious or no trading
    """
    TREND = "trend"
    RANGE = "range"
    NEUTRAL = "neutral"
    
    def __str__(self) -> str:
        return self.value.upper()
    
    def __repr__(self) -> str:
        return f"MarketMode.{self.name}"


class OrderSide(Enum):
    """
    Order side for trading operations.
    
    - BUY: Purchase order
    - SELL: Sale order
    """
    BUY = "buy"
    SELL = "sell"
    
    def __str__(self) -> str:
        return self.value.upper()
    
    def __repr__(self) -> str:
        return f"OrderSide.{self.name}"