"""
Strategy package for the adaptive trading bot.

This package provides base classes and interfaces for implementing
trading strategies.
"""

from .base import StrategyContext, TradeSignal, Strategy

__all__ = [
    "StrategyContext",
    "TradeSignal",
    "Strategy",
]