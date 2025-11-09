"""
Virtual portfolio for backtesting.

This module provides a virtual portfolio implementation that tracks
cash, positions, and equity for backtesting purposes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from exchange.models import Position
from core.types import OrderSide, MarketMode

logger = logging.getLogger("bot")


@dataclass
class PortfolioSnapshot:
    """
    Snapshot of portfolio state at a specific time.
    """
    ts: datetime
    equity: float          # Total account value (cash + positions_value)
    cash: float           # Available cash
    positions_value: float # Current value of all positions
    unrealized_pnl: float # Unrealized profit/loss


@dataclass
class BacktestPortfolio:
    """
    Virtual portfolio for backtesting.
    
    Tracks cash, positions, and equity over time during backtesting.
    """
    initial_cash: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[PortfolioSnapshot] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize portfolio after creation."""
        logger.info(f"BacktestPortfolio initialized with {self.initial_cash:,.0f} KRW")
    
    def apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        size: float,
        ts: datetime,
    ) -> None:
        """
        Apply a filled trade to the portfolio.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            price: Fill price
            size: Trade size
            ts: Transaction timestamp
        """
        trade_value = price * size
        
        if side == OrderSide.BUY:
            # Check if we have enough cash
            if self.cash < trade_value:
                logger.warning(
                    f"Insufficient cash for BUY {symbol}: "
                    f"need {trade_value:,.0f}, have {self.cash:,.0f}"
                )
                return
            
            # Reduce cash
            self.cash -= trade_value
            
            # Add or update position
            if symbol in self.positions:
                # Average into existing position
                existing_pos = self.positions[symbol]
                total_size = existing_pos.size + size
                avg_price = ((existing_pos.entry_price * existing_pos.size) + (price * size)) / total_size
                
                # Update position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    mode=existing_pos.mode,  # Keep existing mode
                    entry_price=avg_price,
                    size=total_size,
                    entry_time=existing_pos.entry_time,  # Keep original entry time
                    peak_price=max(existing_pos.peak_price, price)
                )
                
                logger.debug(
                    f"Averaged into {symbol}: new size={total_size:.6f}, "
                    f"avg_price={avg_price:.0f}"
                )
            else:
                # Create new position (use TREND as default mode for backtest)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    mode=MarketMode.TREND,  # Default mode for backtest
                    entry_price=price,
                    size=size,
                    entry_time=ts,
                    peak_price=price
                )
                
                logger.debug(
                    f"Opened position {symbol}: size={size:.6f}, "
                    f"price={price:.0f}, value={trade_value:.0f}"
                )
        
        elif side == OrderSide.SELL:
            # Check if we have the position
            if symbol not in self.positions:
                logger.warning(f"Cannot SELL {symbol}: no position exists")
                return
            
            position = self.positions[symbol]
            
            # For simplicity, assume full position close
            if size > position.size:
                logger.warning(
                    f"SELL size {size:.6f} exceeds position size {position.size:.6f} "
                    f"for {symbol}, selling full position"
                )
                size = position.size
            
            # Calculate P&L
            pnl = (price - position.entry_price) * size
            
            # Add cash from sale
            self.cash += trade_value
            
            # Update or remove position
            if size >= position.size:
                # Full close
                del self.positions[symbol]
                logger.debug(
                    f"Closed position {symbol}: size={size:.6f}, "
                    f"price={price:.0f}, P&L={pnl:+.0f}"
                )
            else:
                # Partial close
                remaining_size = position.size - size
                self.positions[symbol] = Position(
                    symbol=symbol,
                    mode=position.mode,
                    entry_price=position.entry_price,
                    size=remaining_size,
                    entry_time=position.entry_time,
                    peak_price=position.peak_price
                )
                
                logger.debug(
                    f"Partial close {symbol}: sold={size:.6f}, "
                    f"remaining={remaining_size:.6f}, P&L={pnl:+.0f}"
                )
    
    def update_equity(self, prices: Dict[str, float], ts: datetime) -> None:
        """
        Update portfolio equity and record snapshot.
        
        Args:
            prices: Current market prices for all symbols
            ts: Current timestamp
        """
        # Calculate positions value
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = prices.get(symbol, position.entry_price)
            position_value = current_price * position.size
            positions_value += position_value
            
            # Update peak price
            if current_price > position.peak_price:
                position.peak_price = current_price
        
        # Calculate metrics
        equity = self.cash + positions_value
        unrealized_pnl = equity - self.initial_cash
        
        # Create snapshot
        snapshot = PortfolioSnapshot(
            ts=ts,
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl
        )
        
        self.history.append(snapshot)
    
    def get_current_equity(self, prices: Dict[str, float]) -> float:
        """
        Get current portfolio equity.
        
        Args:
            prices: Current market prices
            
        Returns:
            Current total equity value
        """
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = prices.get(symbol, position.entry_price)
            positions_value += current_price * position.size
        
        return self.cash + positions_value
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    def get_cash_utilization(self) -> float:
        """
        Get cash utilization ratio.
        
        Returns:
            Ratio of used cash to initial cash (0.0 to 1.0+)
        """
        if self.initial_cash == 0:
            return 0.0
        
        used_cash = self.initial_cash - self.cash
        return used_cash / self.initial_cash
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get portfolio performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.history:
            return {}
        
        latest = self.history[-1]
        
        total_return = (latest.equity - self.initial_cash) / self.initial_cash
        
        # Calculate max drawdown
        max_equity = self.initial_cash
        max_drawdown = 0.0
        
        for snapshot in self.history:
            if snapshot.equity > max_equity:
                max_equity = snapshot.equity
            
            drawdown = (max_equity - snapshot.equity) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "initial_cash": self.initial_cash,
            "final_equity": latest.equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "cash_utilization": self.get_cash_utilization(),
            "position_count": self.get_position_count(),
            "history_length": len(self.history),
        }
    
    def save_history_csv(self, path: Path) -> None:
        """
        포트폴리오 스냅샷을 CSV 파일로 저장합니다.
        
        Args:
            path: CSV 파일 저장 경로
        """
        if not self.history:
            logger.warning("포트폴리오 히스토리가 비어있어 CSV를 저장할 수 없습니다.")
            return
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                # CSV 헤더 작성
                f.write("ts,equity,cash,positions_value,unrealized_pnl\n")
                
                # 데이터 작성 (오래된 것부터 최신 순서)
                for snapshot in self.history:
                    f.write(
                        f"{snapshot.ts.isoformat()},"
                        f"{snapshot.equity},"
                        f"{snapshot.cash},"
                        f"{snapshot.positions_value},"
                        f"{snapshot.unrealized_pnl}\n"
                    )
            
            logger.info(f"포트폴리오 히스토리 CSV 저장 완료: {path} ({len(self.history)}개 스냅샷)")
            
        except Exception as e:
            logger.error(f"CSV 저장 실패 {path}: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of portfolio."""
        if self.history:
            latest = self.history[-1]
            return (
                f"BacktestPortfolio(equity={latest.equity:,.0f}, "
                f"cash={self.cash:,.0f}, positions={len(self.positions)})"
            )
        else:
            return (
                f"BacktestPortfolio(cash={self.cash:,.0f}, "
                f"positions={len(self.positions)})"
            )