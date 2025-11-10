"""
Strategy package for the adaptive trading bot.

This package provides base classes and interfaces for implementing
trading strategies.
"""

from .base import StrategyContext, TradeSignal, Strategy
from .trend_vol_breakout import VolatilityBreakoutStrategy
from .range_rsi_meanrev import RSIMeanReversionStrategy
from .strategy_manager import StrategyManager

__all__ = [
    "StrategyContext",
    "TradeSignal",
    "Strategy",
    "VolatilityBreakoutStrategy",
    "RSIMeanReversionStrategy",
    "StrategyManager",
]