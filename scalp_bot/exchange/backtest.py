"""
Backtest exchange implementation

Simulates exchange operations using historical OHLCV data.
"""

from datetime import datetime
from typing import Literal
import pandas as pd
import uuid

from .base import (
    Candle,
    Order,
    Position,
    TradeFill,
    Balance,
    ExchangeClient,
)


class BacktestExchangeClient:
    """
    Exchange client for backtesting with historical data
    
    Simulates real exchange operations using OHLCV DataFrames.
    """
    
    def __init__(
        self, 
        initial_balances: dict[str, float],
        ohlcv_data: dict[str, pd.DataFrame],
        fee_rate: float = 0.0005,  # 0.05% fee
        slippage_rate: float = 0.0001,  # 0.01% slippage
    ):
        """
        Initialize backtest exchange
        
        Args:
            initial_balances: Starting balances (e.g. {"KRW": 10_000_000})
            ohlcv_data: Historical data per symbol
                       Each DataFrame must have columns: timestamp, open, high, low, close, volume
            fee_rate: Trading fee rate (default 0.05%)
            slippage_rate: Price slippage rate (default 0.01%)
        """
        self.initial_balances = initial_balances.copy()
        self.balances = initial_balances.copy()
        self.ohlcv_data = ohlcv_data
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        
        # State tracking
        self.current_time_index: dict[str, int] = {
            symbol: 0 for symbol in ohlcv_data.keys()
        }
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.trade_fills: list[TradeFill] = []
        
        # Validate data format
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate OHLCV data format"""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        for symbol, df in self.ohlcv_data.items():
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Symbol {symbol} missing columns: {missing}")
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                self.ohlcv_data[symbol]["timestamp"] = pd.to_datetime(df["timestamp"])
    
    def advance_time_step(self, symbol: str | None = None) -> None:
        """
        Advance to next time step
        
        Args:
            symbol: Symbol to advance (None for all symbols)
        """
        if symbol is None:
            # Advance all symbols
            for sym in self.current_time_index:
                if self.current_time_index[sym] < len(self.ohlcv_data[sym]) - 1:
                    self.current_time_index[sym] += 1
        else:
            # Advance specific symbol
            if symbol in self.current_time_index:
                if self.current_time_index[symbol] < len(self.ohlcv_data[symbol]) - 1:
                    self.current_time_index[symbol] += 1
    
    def get_current_candle(self, symbol: str) -> Candle | None:
        """
        Get current candle for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current Candle or None if no data
        """
        if symbol not in self.ohlcv_data:
            return None
            
        idx = self.current_time_index[symbol]
        if idx >= len(self.ohlcv_data[symbol]):
            return None
            
        row = self.ohlcv_data[symbol].iloc[idx]
        return Candle(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"], 
            close=row["close"],
            volume=row["volume"]
        )
    
    def get_current_time(self) -> datetime | None:
        """Get current simulation time"""
        if not self.current_time_index:
            return None
            
        # Return time from first symbol
        first_symbol = next(iter(self.current_time_index))
        candle = self.get_current_candle(first_symbol)
        return candle.timestamp if candle else None
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch OHLCV data up to current time
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (ignored in backtest)
            since: Start timestamp (ignored in backtest)
            limit: Number of candles to return
            
        Returns:
            List of historical Candle objects
        """
        if symbol not in self.ohlcv_data:
            return []
            
        current_idx = self.current_time_index[symbol]
        start_idx = max(0, current_idx - limit + 1)
        end_idx = current_idx + 1
        
        candles = []
        for i in range(start_idx, end_idx):
            if i < len(self.ohlcv_data[symbol]):
                row = self.ohlcv_data[symbol].iloc[i]
                candles.append(Candle(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                ))
        
        return candles
    
    def fetch_balance(self) -> dict[str, float]:
        """Get current balances"""
        return self.balances.copy()
    
    def fetch_ticker(self, symbol: str) -> dict:
        """
        Get current ticker (last price)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker info
        """
        candle = self.get_current_candle(symbol)
        if not candle:
            return {}
            
        return {
            "symbol": symbol,
            "last": candle.close,
            "bid": candle.close * (1 - self.slippage_rate),
            "ask": candle.close * (1 + self.slippage_rate),
            "timestamp": candle.timestamp
        }
    
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
        Create and immediately fill order (simplified for backtest)
        
        Args:
            symbol: Trading symbol
            type: Order type (market orders filled immediately)
            side: Buy or sell
            amount: Order quantity
            price: Limit price (ignored for market orders)
            params: Additional parameters
            
        Returns:
            Filled Order object
        """
        current_candle = self.get_current_candle(symbol)
        if not current_candle:
            # No data available - reject order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                type=type,
                price=price or 0.0,
                amount=amount,
                filled=0.0,
                status="rejected",
                created_at=datetime.now(),
                fee=0.0
            )
            self.orders.append(order)
            return order
        
        # For backtest, fill immediately at current close price with slippage
        if side == "buy":
            fill_price = current_candle.close * (1 + self.slippage_rate)
        else:
            fill_price = current_candle.close * (1 - self.slippage_rate)
        
        # Calculate fee
        trade_value = amount * fill_price
        fee = trade_value * self.fee_rate
        
        # Check if we have enough balance
        if side == "buy":
            # Need quote currency (e.g. KRW for BTC/KRW)
            quote_currency = symbol.split("/")[-1] if "/" in symbol else "KRW"
            required_balance = trade_value + fee
            
            if self.balances.get(quote_currency, 0) < required_balance:
                # Insufficient balance - reject order
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    type=type,
                    price=fill_price,
                    amount=amount,
                    filled=0.0,
                    status="rejected", 
                    created_at=current_candle.timestamp,
                    fee=0.0
                )
                self.orders.append(order)
                return order
            
            # Execute buy order
            base_currency = symbol.split("/")[0] if "/" in symbol else symbol
            self.balances[quote_currency] = self.balances.get(quote_currency, 0) - required_balance
            self.balances[base_currency] = self.balances.get(base_currency, 0) + amount
            
        else:  # sell
            # Need base currency (e.g. BTC for BTC/KRW)
            base_currency = symbol.split("/")[0] if "/" in symbol else symbol
            
            if self.balances.get(base_currency, 0) < amount:
                # Insufficient balance - reject order
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    type=type,
                    price=fill_price,
                    amount=amount,
                    filled=0.0,
                    status="rejected",
                    created_at=current_candle.timestamp,
                    fee=0.0
                )
                self.orders.append(order)
                return order
            
            # Execute sell order
            quote_currency = symbol.split("/")[-1] if "/" in symbol else "KRW"
            self.balances[base_currency] = self.balances.get(base_currency, 0) - amount
            self.balances[quote_currency] = self.balances.get(quote_currency, 0) + trade_value - fee
        
        # Create successful order
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type=type,
            price=fill_price,
            amount=amount,
            filled=amount,
            status="closed",
            created_at=current_candle.timestamp,
            fee=fee
        )
        
        # Record trade fill
        trade_fill = TradeFill(
            order_id=order.id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=fill_price,
            fee=fee,
            timestamp=current_candle.timestamp
        )
        
        self.orders.append(order)
        self.trade_fills.append(trade_fill)
        
        # Update position (simplified - just track long positions)
        if side == "buy":
            if symbol in self.positions:
                # Average existing position
                existing = self.positions[symbol]
                total_cost = (existing.size * existing.entry_price) + (amount * fill_price)
                total_size = existing.size + amount
                avg_price = total_cost / total_size
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side="long",
                    size=total_size,
                    entry_price=avg_price,
                    unrealized_pnl=0.0,  # Will be calculated when needed
                    leverage=1.0
                )
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side="long", 
                    size=amount,
                    entry_price=fill_price,
                    unrealized_pnl=0.0,
                    leverage=1.0
                )
        else:  # sell
            if symbol in self.positions:
                existing = self.positions[symbol]
                if existing.size <= amount:
                    # Close entire position
                    del self.positions[symbol]
                else:
                    # Reduce position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side="long",
                        size=existing.size - amount,
                        entry_price=existing.entry_price,
                        unrealized_pnl=0.0,
                        leverage=1.0
                    )
        
        return order
    
    def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel order (not implemented for backtest)"""
        # In backtest, orders are filled immediately, so nothing to cancel
        pass
    
    def fetch_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders (empty for backtest)"""
        # In backtest, orders are filled immediately
        return []
    
    def fetch_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get current positions
        
        Args:
            symbol: Filter by symbol
            
        Returns:
            List of Position objects
        """
        positions = list(self.positions.values())
        
        if symbol is not None:
            positions = [pos for pos in positions if pos.symbol == symbol]
        
        # Update unrealized PnL
        for position in positions:
            current_candle = self.get_current_candle(position.symbol)
            if current_candle:
                position.unrealized_pnl = (
                    (current_candle.close - position.entry_price) * position.size
                )
        
        return positions
    
    def get_portfolio_value(self, quote_currency: str = "KRW") -> float:
        """
        Calculate total portfolio value
        
        Args:
            quote_currency: Currency to value portfolio in
            
        Returns:
            Total portfolio value
        """
        total_value = self.balances.get(quote_currency, 0)
        
        # Add value of all positions
        for symbol, position in self.positions.items():
            current_candle = self.get_current_candle(symbol)
            if current_candle:
                position_value = position.size * current_candle.close
                total_value += position_value
        
        # Add value of other currency balances (simplified - assume 1:1 for now)
        for currency, balance in self.balances.items():
            if currency != quote_currency and balance > 0:
                # TODO: Add proper currency conversion
                total_value += balance
        
        return total_value