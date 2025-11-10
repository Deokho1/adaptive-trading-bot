"""
Scalp Bot - High-frequency dip trading bot

A simple, always-on scalp/dip bot that monitors the market 24/7
and trades short-term dips with tight profit targets and stops.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"

from .state import MarketState, PositionState
from .config import ScalpConfig
from .strategy import DipScalpStrategy
from .risk import RiskManager
from .backtest import ScalpBacktester

__all__ = [
    "MarketState",
    "PositionState", 
    "ScalpConfig",
    "DipScalpStrategy",
    "RiskManager",
    "ScalpBacktester"
]