"""
Base exchange interface and data models

Defines common data structures and protocol for exchange implementations.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Protocol, Literal
import pandas as pd


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass  
class Order:
    """Order information"""
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["limit", "market"]
    price: float
    amount: float
    filled: float
    status: Literal["open", "closed", "canceled", "rejected"]
    created_at: datetime
    fee: float = 0.0


@dataclass
class Position:
    """Position information"""
    symbol: str
    side: Literal["long", "short"]
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float = 1.0


@dataclass
class TradeFill:
    """Trade execution details"""
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    fee: float
    timestamp: datetime


@dataclass
class Balance:
    """Account balance information"""
    currency: str
    free: float
    used: float
    total: float


class ExchangeClient(Protocol):
    """
    Unified exchange interface for backtest and live trading
    
    This protocol ensures identical signatures between BacktestExchangeClient
    and LiveExchangeClient implementations.
    """
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch OHLCV candle data
        
        Args:
            symbol: Trading symbol (e.g. "BTC/KRW")
            timeframe: Candle timeframe ("1m", "5m", "1h", etc.)
            since: Unix timestamp to fetch from (None for recent)
            limit: Maximum number of candles to return
            
        Returns:
            List of Candle objects
        """
        ...

    def fetch_balance(self) -> dict[str, float]:
        """
        Get account balances
        
        Returns:
            Dictionary mapping currency to available balance
        """
        ...

    def fetch_ticker(self, symbol: str) -> dict:
        """
        Get current ticker information
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker data (last, bid, ask, etc.)
        """
        ...

    def create_order(
        self,
        symbol: str,
        type: Literal["limit", "market"],
        side: Literal["buy", "sell"],
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> Order:
        """
        Create a new order
        
        Args:
            symbol: Trading symbol
            type: Order type
            side: Buy or sell
            amount: Order quantity
            price: Limit price (for limit orders)
            params: Additional parameters
            
        Returns:
            Created Order object
        """
        ...

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """
        Cancel an existing order
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
        """
        ...

    def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get open orders
        
        Args:
            symbol: Filter by symbol (None for all symbols)
            
        Returns:
            List of open Order objects
        """
        ...

    def fetch_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions
        
        Args:
            symbol: Filter by symbol (None for all symbols)
            
        Returns:
            List of Position objects
        """
        ...

    def get_portfolio_value(self, quote_currency: str = "KRW") -> float:
        """
        Get total portfolio value in quote currency
        
        Args:
            quote_currency: Currency to value portfolio in
            
        Returns:
            Total portfolio value
        """
        ...