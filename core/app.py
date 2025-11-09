"""
BotApp - Main orchestration class for the adaptive trading bot.

This module contains the BotApp class that wires together all components
and provides the main execution loop for the trading bot.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from core.config_loader import load_config
from core.types import MarketMode
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient
from exchange.models import Candle
from market.market_analyzer import MarketAnalyzer
from market.indicators import (
    compute_atr,
    compute_rsi,
    compute_bollinger_bands,
)
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.trend_vol_breakout import VolatilityBreakoutStrategy
from strategy.range_rsi_meanrev import RSIMeanReversionStrategy
from strategy.strategy_manager import StrategyManager
from execution.execution_engine import ExecutionEngine

logger = logging.getLogger("bot")


class BotApp:
    """
    Main orchestration class for the adaptive trading bot.
    
    Wires together all components and provides the main execution loop.
    Runs in dry-run mode using only public API endpoints.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize BotApp with all required components.
        
        Args:
            config: Configuration dictionary loaded from config.yaml
        """
        self.config = config
        
        # Basic settings
        self.timezone_name: str = config["app"]["timezone"]
        self.dry_run: bool = config["app"]["dry_run"]
        self.symbols: list[str] = config["strategies"]["symbols"]
        
        logger.info(f"Initializing BotApp (dry_run={self.dry_run}, symbols={self.symbols})")
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
        )
        
        # Upbit client (PUBLIC only for now)
        self.client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=self.rate_limiter,
        )
        
        # Market analyzer
        self.market_analyzer = MarketAnalyzer(config)
        
        # Position & risk management
        self.position_manager = PositionManager(
            positions_file=config["persistence"]["positions_file"],
        )
        self.risk_manager = RiskManager(config)
        
        # Strategies
        self.trend_strategy = VolatilityBreakoutStrategy(config["strategies"]["trend"])
        self.range_strategy = RSIMeanReversionStrategy(config["strategies"]["range"])
        self.strategy_manager = StrategyManager(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            trend_strategy=self.trend_strategy,
            range_strategy=self.range_strategy,
        )
        
        # Execution engine (dry_run)
        self.execution_engine = ExecutionEngine(
            upbit_client=self.client,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            dry_run=self.dry_run,
        )
        
        logger.info("BotApp initialized successfully")
        logger.info(f"Components: {self.strategy_manager}, {self.execution_engine}")
    
    def __str__(self) -> str:
        """String representation of BotApp."""
        mode = "DRY_RUN" if self.dry_run else "LIVE"
        positions = len(self.position_manager.get_positions())
        return f"BotApp(mode={mode}, symbols={len(self.symbols)}, positions={positions})"
    
    def _get_now(self) -> datetime:
        """
        Return current UTC time.
        
        Returns:
            Current datetime in UTC timezone
        """
        return datetime.now(timezone.utc)
    
    def _fetch_candles_for_all_symbols(self, count: int = 200) -> dict[str, list[Candle]]:
        """
        Fetch 4h candles for each symbol in config.
        
        Args:
            count: Number of candles to fetch per symbol
            
        Returns:
            Dictionary mapping symbol to list of candles
        """
        market_data: dict[str, list[Candle]] = {}
        
        for symbol in self.symbols:
            try:
                candles = self.client.get_candles_4h(symbol, count=count)
                if candles:
                    # Ensure sorted oldest -> newest
                    candles = list(sorted(candles, key=lambda c: c.timestamp))
                market_data[symbol] = candles
                logger.debug(f"Fetched {len(candles)} candles for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch candles for {symbol}: {e}")
                market_data[symbol] = []
        
        total_candles = sum(len(candles) for candles in market_data.values())
        logger.info(f"Fetched candles for {len(market_data)} symbols, total {total_candles} candles")
        
        return market_data
    
    def _compute_indicators_for_symbol(self, candles: list[Candle]) -> dict[str, list[float]]:
        """
        Compute ATR, RSI, and Bollinger Bands for a single symbol.
        
        Args:
            candles: List of candle data for the symbol
            
        Returns:
            Dictionary of computed indicators
        """
        if not candles:
            return {}
        
        try:
            closes = [c.close for c in candles]
            volumes = [c.volume for c in candles]
            
            # Get indicator periods from config
            atr_period = self.config["market_analyzer"]["atr_period"]
            rsi_period = self.config["strategies"]["trend"]["rsi_period"]
            bb_period = self.config["market_analyzer"]["bb_period"]
            bb_stddev = self.config["market_analyzer"]["bb_stddev"]
            
            # Compute indicators
            atr_values = compute_atr(candles, period=atr_period)
            rsi_values = compute_rsi(closes, period=rsi_period)
            bb_mid, bb_upper, bb_lower = compute_bollinger_bands(
                closes,
                period=bb_period,
                num_std=bb_stddev,
            )
            
            return {
                "atr": atr_values,
                "rsi": rsi_values,
                "bb_middle": bb_mid,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "volume": volumes,
                "closes": closes,
            }
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return {}
    
    def _compute_all_indicators(self, market_data: dict[str, list[Candle]]) -> dict[str, dict[str, list[float]]]:
        """
        Compute indicators for all symbols.
        
        Args:
            market_data: Dictionary mapping symbol to candle data
            
        Returns:
            Dictionary mapping symbol to indicator data
        """
        indicators: dict[str, dict[str, list[float]]] = {}
        
        for symbol, candles in market_data.items():
            indicators[symbol] = self._compute_indicators_for_symbol(candles)
            
            if indicators[symbol]:
                logger.debug(f"Computed indicators for {symbol}: {list(indicators[symbol].keys())}")
            else:
                logger.warning(f"No indicators computed for {symbol}")
        
        return indicators
    
    def _log_market_summary(self, market_mode: MarketMode, market_data: dict[str, list[Candle]]) -> None:
        """
        Log market summary information.
        
        Args:
            market_mode: Current market mode
            market_data: Market candle data
        """
        logger.info(f"📊 Market Analysis Summary:")
        logger.info(f"  Mode: {market_mode.name}")
        
        # Log current prices for each symbol
        for symbol, candles in market_data.items():
            if candles:
                latest = candles[-1]
                logger.info(f"  {symbol}: {latest.close:,.0f} KRW (volume: {latest.volume:,.0f})")
    
    def _log_execution_summary(self) -> None:
        """Log execution summary after signal processing."""
        positions = self.position_manager.get_positions()
        
        if positions:
            logger.info(f"📈 Position Summary ({len(positions)} open):")
            
            total_exposure = 0.0
            for i, position in enumerate(positions, 1):
                try:
                    current_price = self.client.get_ticker_price(position.symbol)
                    current_value = current_price * position.size
                    total_exposure += current_value
                    
                    pnl = (current_price - position.entry_price) * position.size
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    logger.info(
                        f"  {i}. {position.symbol} ({position.mode.name}): "
                        f"entry={position.entry_price:,.0f}, "
                        f"current={current_price:,.0f}, "
                        f"size={position.size:.6f}, "
                        f"P&L={pnl:+,.0f} KRW ({pnl_pct:+.2f}%)"
                    )
                except Exception as e:
                    logger.warning(f"Error getting current price for {position.symbol}: {e}")
                    
                    # Use entry price as fallback
                    fallback_value = position.entry_price * position.size
                    total_exposure += fallback_value
                    
                    logger.info(
                        f"  {i}. {position.symbol} ({position.mode.name}): "
                        f"entry={position.entry_price:,.0f}, "
                        f"size={position.size:.6f} "
                        f"(price unavailable)"
                    )
            
            logger.info(f"  Total exposure: {total_exposure:,.0f} KRW")
            
        else:
            logger.info("📈 No open positions")
    
    def run_once(self) -> None:
        """
        Run one full 4h candle close cycle:
        - Fetch candles
        - Analyze market (BTC)
        - Build indicators
        - Generate TradeSignals
        - Execute signals (dry_run)
        - Log summary
        """
        logger.info("🚀 Starting bot cycle...")
        cycle_start = self._get_now()
        
        try:
            # 1. Fetch 4h candles for all symbols
            logger.info("📊 Fetching market data...")
            market_data = self._fetch_candles_for_all_symbols(count=200)
            
            # 2. Determine market mode based on BTC
            btc_symbol = "KRW-BTC"
            btc_candles = market_data.get(btc_symbol, [])
            if not btc_candles:
                logger.warning("No BTC candles available, skipping cycle")
                return
            
            market_mode = self.market_analyzer.update_mode(btc_candles, cycle_start)
            
            # 3. Compute indicators for each symbol
            logger.info("🔧 Computing technical indicators...")
            indicators = self._compute_all_indicators(market_data)
            
            # 4. Log market summary
            self._log_market_summary(market_mode, market_data)
            
            # 5. Compute portfolio value / available KRW (mock for now)
            # Later we will use real balances via private API
            portfolio_value = 1_000_000.0  # 1M KRW
            available_krw = 1_000_000.0    # Full amount available
            
            # Update risk manager daily tracking
            self.risk_manager.update_daily_pnl(portfolio_value, cycle_start)
            
            # 6. Generate signals via StrategyManager
            logger.info("🎯 Generating trading signals...")
            signals = self.strategy_manager.on_new_candle(
                market_mode=market_mode,
                market_data=market_data,
                indicators=indicators,
                portfolio_value=portfolio_value,
                available_krw=available_krw,
                now=cycle_start,
            )
            
            if signals:
                logger.info(f"📋 Generated {len(signals)} trading signals:")
                for i, signal in enumerate(signals, 1):
                    logger.info(
                        f"  {i}. {signal.side.name} {signal.symbol} "
                        f"({signal.mode.name}): {signal.reason}"
                        f"{f' - {signal.amount_krw:,.0f} KRW' if signal.amount_krw else ''}"
                    )
            else:
                logger.info("📋 No trading signals generated")
            
            # 7. Execute signals (dry_run)
            if signals:
                logger.info("⚡ Executing trading signals...")
                self.execution_engine.process_signals(signals, portfolio_value, cycle_start)
            
            # 8. Update position prices
            self.execution_engine.process_price_tick(cycle_start)
            
            # 9. Log execution summary
            self._log_execution_summary()
            
            # Calculate cycle duration
            cycle_end = self._get_now()
            duration = (cycle_end - cycle_start).total_seconds()
            
            logger.info(f"✅ Bot cycle completed in {duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error during bot cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def run_forever(self, sleep_seconds: int = 60 * 5) -> None:
        """
        Run the bot continuously with a simple loop.
        
        Args:
            sleep_seconds: Sleep duration between cycles (default: 5 minutes)
        
        Note:
            This is a simple implementation. Later we can align with
            exact 4h candle close times for more precise timing.
        """
        logger.info(f"🔄 Starting continuous bot operation (cycle every {sleep_seconds} seconds)")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"📅 Starting cycle #{cycle_count}")
                
                self.run_once()
                
                logger.info(f"💤 Sleeping for {sleep_seconds} seconds until next cycle...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt, stopping bot...")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous loop: {e}")
                logger.info(f"💤 Sleeping {sleep_seconds} seconds before retry...")
                time.sleep(sleep_seconds)
        
        logger.info("🏁 Bot stopped")
    
    def cleanup(self) -> None:
        """
        Clean up resources before shutdown.
        """
        logger.info("🧹 Cleaning up BotApp resources...")
        
        try:
            if hasattr(self, 'client'):
                self.client.close()
                logger.info("Closed Upbit client")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
        
        logger.info("BotApp cleanup completed")
    
    def get_status(self) -> dict[str, Any]:
        """
        Get current bot status.
        
        Returns:
            Dictionary with bot status information
        """
        positions = self.position_manager.get_positions()
        
        return {
            "dry_run": self.dry_run,
            "symbols": self.symbols,
            "current_mode": self.market_analyzer.get_current_mode(),
            "total_positions": len(positions),
            "kill_switch_active": self.risk_manager.is_kill_switch_active(),
            "execution_status": self.execution_engine.get_execution_status(),
        }