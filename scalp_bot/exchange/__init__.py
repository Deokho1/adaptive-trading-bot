"""
Exchange module for scalp bot

Provides unified interface for backtest and live trading.
"""

from .base import (
    Candle,
    Order,
    Position,
    TradeFill,
    Balance,
    ExchangeClient,
)

__all__ = [
    "Candle",
    "Order", 
    "Position",
    "TradeFill",
    "Balance",
    "ExchangeClient",
]