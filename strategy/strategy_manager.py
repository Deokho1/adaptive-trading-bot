"""
Strategy Manager for orchestrating trading strategies.

This module manages the execution of different trading strategies based on market mode,
handles mode transitions, and coordinates position management across strategies.
"""

from datetime import datetime
from typing import Any, Dict, List

from core.types import MarketMode, OrderSide
from exchange.models import Candle
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.base import StrategyContext, TradeSignal
from strategy.trend_vol_breakout import VolatilityBreakoutStrategy
from strategy.range_rsi_meanrev import RSIMeanReversionStrategy


class StrategyManager:
    """
    Orchestrates trading strategies based on market conditions.
    
    Responsibilities:
    - Detect market mode changes and force-close incompatible positions
    - Route strategy execution based on current market mode
    - Coordinate between trend and range strategies
    - Generate comprehensive trading signals for all symbols
    """
    
    def __init__(
        self,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        trend_strategy: VolatilityBreakoutStrategy,
        range_strategy: RSIMeanReversionStrategy,
    ) -> None:
        """
        Initialize the strategy manager.
        
        Args:
            position_manager: Manages open positions
            risk_manager: Handles risk controls
            trend_strategy: Strategy for TREND market mode
            range_strategy: Strategy for RANGE market mode
        """
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        
        # Track current market mode for mode change detection
        self.current_mode: MarketMode = MarketMode.NEUTRAL
        self.mode_change_count = 0
        
    def on_new_candle(
        self,
        market_mode: MarketMode,
        market_data: Dict[str, List[Candle]],
        indicators: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        available_krw: float,
        now: datetime,
    ) -> List[TradeSignal]:
        """
        Process new candle data and generate trading signals.
        
        Called once per 4-hour candle close to evaluate all symbols and generate
        appropriate trading signals based on current market conditions.
        
        Args:
            market_mode: Current global market mode from MarketAnalyzer
            market_data: Dictionary mapping symbols to their candle data
            indicators: Dictionary mapping symbols to their technical indicators
            portfolio_value: Total portfolio value in KRW
            available_krw: Available KRW for new positions
            now: Current timestamp
            
        Returns:
            List of TradeSignal objects for execution
        """
        signals: List[TradeSignal] = []
        
        # 1. Handle market mode changes
        mode_change_signals = self._handle_mode_change(market_mode, now)
        signals.extend(mode_change_signals)
        
        # 2. Generate strategy signals for each symbol
        strategy_signals = self._generate_strategy_signals(
            market_mode, market_data, indicators, portfolio_value, available_krw, now
        )
        signals.extend(strategy_signals)
        
        # 3. Update current mode
        self.current_mode = market_mode
        
        return signals
    
    def _handle_mode_change(self, new_mode: MarketMode, now: datetime) -> List[TradeSignal]:
        """
        Handle market mode transitions by closing incompatible positions.
        
        When market mode changes, positions opened in the previous mode
        should be closed as they may no longer be appropriate for the new
        market conditions.
        
        Args:
            new_mode: New market mode
            now: Current timestamp
            
        Returns:
            List of SELL signals for positions that need to be closed
        """
        signals: List[TradeSignal] = []
        
        if new_mode == self.current_mode:
            return signals  # No mode change
        
        # Mode change detected
        self.mode_change_count += 1
        prev_mode = self.current_mode
        
        # Get positions that belong to the previous mode
        prev_mode_positions = self.position_manager.get_positions_by_mode(prev_mode)
        
        # Force close all positions from previous mode
        for position in prev_mode_positions:
            hold_duration = now - position.entry_time
            hold_hours = hold_duration.total_seconds() / 3600
            
            signals.append(
                TradeSignal(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    mode=prev_mode,  # Keep original mode for tracking
                    reason=f"mode_change_exit ({prev_mode} -> {new_mode}, held: {hold_hours:.1f}h)",
                    size=position.size,
                )
            )
        
        return signals
    
    def _generate_strategy_signals(
        self,
        market_mode: MarketMode,
        market_data: Dict[str, List[Candle]],
        indicators: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        available_krw: float,
        now: datetime,
    ) -> List[TradeSignal]:
        """
        Generate trading signals from active strategies.
        
        Routes signal generation to the appropriate strategy based on
        current market mode and symbol-specific data.
        
        Args:
            market_mode: Current market mode
            market_data: Symbol candle data
            indicators: Symbol indicator data
            portfolio_value: Total portfolio value
            available_krw: Available capital
            now: Current timestamp
            
        Returns:
            List of signals from active strategies
        """
        signals: List[TradeSignal] = []
        
        # Process each symbol
        for symbol, candles in market_data.items():
            if not candles:
                continue
                
            # Get symbol-specific data
            symbol_indicators = indicators.get(symbol, {})
            position = self.position_manager.get_position(symbol)
            
            # Build strategy context
            context = StrategyContext(
                symbol=symbol,
                candles=candles,
                indicators=symbol_indicators,
                mode=market_mode,
                position=position,
                portfolio_value=portfolio_value,
                available_krw=available_krw,
                now=now,
            )
            
            # Route to appropriate strategy
            if market_mode == MarketMode.TREND:
                trend_signals = self.trend_strategy.generate_signals(context)
                signals.extend(trend_signals)
                
            elif market_mode == MarketMode.RANGE:
                range_signals = self.range_strategy.generate_signals(context)
                signals.extend(range_signals)
                
            else:  # MarketMode.NEUTRAL
                # In NEUTRAL mode, only allow exits, no new entries
                # Strategies should handle this internally, but we can add
                # additional logic here if needed
                pass
        
        return signals
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """
        Get current strategy manager status.
        
        Returns:
            Dictionary containing strategy status information
        """
        positions = self.position_manager.get_positions()
        
        status = {
            "current_mode": self.current_mode,
            "mode_change_count": self.mode_change_count,
            "total_positions": len(positions),
            "trend_positions": len([p for p in positions if p.mode == MarketMode.TREND]),
            "range_positions": len([p for p in positions if p.mode == MarketMode.RANGE]),
            "strategies": {
                "trend": str(self.trend_strategy),
                "range": str(self.range_strategy),
            }
        }
        
        return status
    
    def get_position_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get summary of current positions with P&L information.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            Dictionary containing position summary
        """
        positions = self.position_manager.get_positions()
        
        summary = {
            "positions": [],
            "total_positions": len(positions),
            "total_unrealized_pnl_krw": 0.0,
        }
        
        for position in positions:
            current_price = current_prices.get(position.symbol, position.entry_price)
            unrealized_pnl_krw = position.unrealized_pnl_krw(current_price)
            unrealized_pnl_pct = position.unrealized_pnl_pct(current_price)
            
            position_info = {
                "symbol": position.symbol,
                "mode": position.mode.value,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "size": position.size,
                "entry_time": position.entry_time.isoformat(),
                "unrealized_pnl_krw": unrealized_pnl_krw,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            }
            
            summary["positions"].append(position_info)
            summary["total_unrealized_pnl_krw"] += unrealized_pnl_krw
        
        return summary
    
    def __str__(self) -> str:
        return f"StrategyManager(mode={self.current_mode}, positions={self.position_manager.get_position_count()})"