"""
Backtest runner for historical trading simulation.

This module provides the main backtest runner that orchestrates
historical simulation of trading strategies.
"""

import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass

from exchange.models import Candle
from market.market_analyzer import MarketAnalyzer
from market.indicators import compute_atr, compute_rsi, compute_bollinger_bands
from strategy.base import TradeSignal
from core.types import OrderSide, MarketMode
from .data_loader import BacktestDataLoader
from .portfolio import BacktestPortfolio


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    portfolio: BacktestPortfolio
    start_time: datetime
    end_time: datetime
    num_steps: int
    notes: str = ""


logger = logging.getLogger("bot")


class BacktestRunner:
    """
    Main backtest runner for historical trading simulation.
    
    Orchestrates the backtesting process by:
    1. Loading historical data
    2. Running strategies on historical candles 
    3. Executing trades in virtual portfolio
    4. Tracking performance
    """
    
    def __init__(
        self,
        data_dir: Path,
        config: Dict,
        base_symbol: str = "KRW-BTC",
        symbols: List[str] = None,
        initial_cash: float = 1_000_000,  # 1M KRW default
    ):
        """
        Initialize backtest runner.
        
        Args:
            data_dir: Directory containing historical CSV data
            config: Configuration dictionary
            base_symbol: Primary symbol for market mode analysis
            symbols: List of symbols to trade
            initial_cash: Starting cash for portfolio
        """
        self.data_dir = data_dir
        self.config = config
        self.base_symbol = base_symbol
        self.symbols = symbols or ["KRW-BTC", "KRW-ETH"]
        self.initial_cash = initial_cash
        
        # Components
        self.data_loader = BacktestDataLoader(data_dir)
        self.market_analyzer = MarketAnalyzer(config)
        
        logger.info(f"BacktestRunner initialized with {initial_cash:,.0f} KRW")
        logger.info(f"Base symbol: {base_symbol}, Trading symbols: {self.symbols}")
    
    def run(self) -> BacktestResult:
        """
        Run the backtest simulation.
        
        Returns:
            BacktestResult with portfolio and performance data
        """
        logger.info("Starting backtest simulation...")
        
        # 1. Load data
        all_data = self.data_loader.load_all(self.symbols)
        
        if self.base_symbol not in all_data or not all_data[self.base_symbol]:
            logger.warning(f"No data available for base symbol {self.base_symbol}")
            return BacktestResult(
                portfolio=BacktestPortfolio(self.initial_cash, self.initial_cash),
                start_time=datetime.now(),
                end_time=datetime.now(),
                num_steps=0,
                notes="No data available"
            )
        
        # Determine common length (take minimum across all symbols)
        max_len = min(len(candles) for candles in all_data.values() if candles)
        
        if max_len == 0:
            logger.warning("No common data found across symbols")
            return BacktestResult(
                portfolio=BacktestPortfolio(self.initial_cash, self.initial_cash),
                start_time=datetime.now(),
                end_time=datetime.now(), 
                num_steps=0,
                notes="No common data"
            )
        
        # 2. Warmup period calculation
        warmup = max(
            self.config["market_analyzer"]["atr_period"],
            self.config["market_analyzer"]["bb_period"], 
            self.config.get("strategies", {}).get("trend", {}).get("rsi_period", 14),
            self.config.get("strategies", {}).get("range", {}).get("rsi_period", 14),
        )
        
        start_idx = warmup
        end_idx = max_len
        
        if end_idx <= start_idx:
            logger.warning(f"Not enough data after warmup: need {warmup}, have {max_len}")
            return BacktestResult(
                portfolio=BacktestPortfolio(self.initial_cash, self.initial_cash),
                start_time=datetime.now(),
                end_time=datetime.now(),
                num_steps=0,
                notes="Insufficient data after warmup"
            )
        
        # 3. Create portfolio
        portfolio = BacktestPortfolio(
            initial_cash=self.initial_cash,
            cash=self.initial_cash,
        )
        
        base_candles = all_data[self.base_symbol]
        start_time = base_candles[start_idx].timestamp
        end_time = base_candles[end_idx - 1].timestamp
        
        logger.info(f"Running backtest from {start_time} to {end_time}")
        logger.info(f"Processing {end_idx - start_idx} time steps")
        
        # 4. Main backtest loop
        for t in range(start_idx, end_idx):
            try:
                ts = base_candles[t].timestamp
                
                # Build visible history for each symbol (up to current time)
                visible_data = {}
                for symbol, candles in all_data.items():
                    if t < len(candles):
                        visible_data[symbol] = candles[:t + 1]
                
                # Get market mode from base symbol
                btc_candles = visible_data[self.base_symbol]
                market_mode = self.market_analyzer.update_mode(btc_candles, ts)
                
                # Compute indicators for each symbol
                indicators_by_symbol = {}
                for symbol, candles in visible_data.items():
                    if len(candles) < warmup:
                        continue
                        
                    closes = [c.close for c in candles]
                    highs = [c.high for c in candles]
                    lows = [c.low for c in candles]
                    volumes = [c.volume for c in candles]
                    
                    # Compute technical indicators
                    atr_period = self.config["market_analyzer"]["atr_period"]
                    bb_period = self.config["market_analyzer"]["bb_period"]
                    bb_stddev = self.config["market_analyzer"]["bb_stddev"]
                    
                    atr = compute_atr(candles, period=atr_period)
                    rsi = compute_rsi(closes, period=14)
                    bb_middle, bb_upper, bb_lower = compute_bollinger_bands(closes, period=bb_period, num_std=bb_stddev)
                    
                    indicators_by_symbol[symbol] = {
                        "atr": atr,
                        "rsi": rsi,
                        "bb_middle": bb_middle,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "volume": volumes,
                    }
                
                # Get portfolio metrics
                portfolio_value = portfolio.initial_cash
                if portfolio.history:
                    portfolio_value = portfolio.history[-1].equity
                available_krw = portfolio.cash
                
                # Generate simple signals (placeholder for now - we don't have StrategyManager set up)
                signals = self._generate_simple_signals(
                    market_mode, visible_data, indicators_by_symbol, portfolio_value, available_krw, ts, portfolio
                )
                
                # Apply signals to portfolio
                for signal in signals:
                    self._apply_signal_to_portfolio(signal, visible_data, portfolio, ts)
                
                # Update portfolio equity
                prices = {symbol: candles[-1].close for symbol, candles in visible_data.items()}
                portfolio.update_equity(prices, ts)
                
                # Progress logging
                if t % 100 == 0:
                    progress = ((t - start_idx) / (end_idx - start_idx)) * 100
                    logger.debug(f"Backtest progress: {progress:.1f}%")
                    
            except Exception as e:
                logger.error(f"Error at step {t} (time {ts}): {e}")
                continue
        
        logger.info(f"Backtest completed: {end_idx - start_idx} steps processed")
        
        # Save backtest results to CSV
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = end_time.strftime("%Y%m%d_%H%M%S")
            csv_path = results_dir / f"backtest_history_{timestamp}.csv"
            
            portfolio.save_history_csv(csv_path)
            logger.info(f"백테스트 결과가 저장되었습니다: {csv_path}")
            
        except Exception as e:
            logger.error(f"백테스트 결과 저장 중 오류 발생: {e}")
        
        return BacktestResult(
            portfolio=portfolio,
            start_time=start_time,
            end_time=end_time,
            num_steps=end_idx - start_idx,
            notes="simple backtest run"
        )
    
    def _generate_simple_signals(
        self,
        market_mode: MarketMode,
        visible_data: Dict[str, List[Candle]],
        indicators_by_symbol: Dict[str, Dict],
        portfolio_value: float,
        available_krw: float,
        ts: datetime,
        portfolio: BacktestPortfolio,
    ) -> List[TradeSignal]:
        """
        Generate simple trading signals based on basic rules.
        This is a placeholder until we integrate with StrategyManager.
        """
        signals = []
        
        # Simple strategy: RSI-based for demonstration
        for symbol, candles in visible_data.items():
            if symbol not in indicators_by_symbol:
                continue
                
            indicators = indicators_by_symbol[symbol]
            current_price = candles[-1].close
            
            if "rsi" not in indicators or not indicators["rsi"]:
                continue
                
            current_rsi = indicators["rsi"][-1]
            
            # Simple rules
            if market_mode == MarketMode.TREND:
                # Trend following: buy on RSI oversold
                if current_rsi < 30 and symbol not in portfolio.positions:
                    # Buy signal
                    amount_krw = portfolio_value * 0.02  # 2% of equity
                    if amount_krw <= available_krw:
                        signals.append(TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            amount_krw=amount_krw,
                            reason="RSI oversold in TREND mode"
                        ))
                        
                elif current_rsi > 70 and symbol in portfolio.positions:
                    # Sell signal
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        size=None,  # Full position
                        reason="RSI overbought in TREND mode"
                    ))
                    
            elif market_mode == MarketMode.RANGE:
                # Mean reversion: buy low, sell high
                if current_rsi < 25 and symbol not in portfolio.positions:
                    amount_krw = portfolio_value * 0.015  # 1.5% of equity
                    if amount_krw <= available_krw:
                        signals.append(TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            amount_krw=amount_krw,
                            reason="RSI oversold in RANGE mode"
                        ))
                        
                elif current_rsi > 75 and symbol in portfolio.positions:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        size=None,
                        reason="RSI overbought in RANGE mode"
                    ))
        
        return signals
    
    def _apply_signal_to_portfolio(
        self,
        signal: TradeSignal,
        visible_data: Dict[str, List[Candle]],
        portfolio: BacktestPortfolio,
        ts: datetime,
    ) -> None:
        """Apply a trading signal to the portfolio."""
        if signal.symbol not in visible_data:
            return
            
        current_price = visible_data[signal.symbol][-1].close
        
        if signal.side == OrderSide.BUY:
            amount = signal.amount_krw
            if amount is None:
                amount = portfolio.cash * 0.02  # Default 2%
                
            size = amount / current_price
            if size > 0:
                portfolio.apply_fill(signal.symbol, OrderSide.BUY, current_price, size, ts)
                logger.debug(f"BUY {signal.symbol}: {size:.6f} @ {current_price:.0f} ({signal.reason})")
                
        elif signal.side == OrderSide.SELL:
            if signal.symbol not in portfolio.positions:
                return
                
            position = portfolio.positions[signal.symbol]
            size = signal.size if signal.size is not None else position.size
            size = min(size, position.size)  # Clamp to available size
            
            if size > 0:
                portfolio.apply_fill(signal.symbol, OrderSide.SELL, current_price, size, ts)
                logger.debug(f"SELL {signal.symbol}: {size:.6f} @ {current_price:.0f} ({signal.reason})")