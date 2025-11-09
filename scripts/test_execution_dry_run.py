#!/usr/bin/env python3
"""
Test script for Step 6 - Execution Engine (Dry Run).

This script tests the complete trading pipeline:
- MarketAnalyzer: determines market mode
- StrategyManager: generates TradeSignals based on mode
- ExecutionEngine: processes signals and simulates order execution

Usage:
    python scripts/test_execution_dry_run.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from core.types import MarketMode
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient
from market.market_analyzer import MarketAnalyzer
from market.indicators import compute_atr, compute_rsi, compute_bollinger_bands
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.trend_vol_breakout import VolatilityBreakoutStrategy
from strategy.range_rsi_meanrev import RSIMeanReversionStrategy
from strategy.strategy_manager import StrategyManager
from execution.execution_engine import ExecutionEngine


def compute_symbol_indicators(candles, config):
    """
    Compute technical indicators for a symbol.
    
    Args:
        candles: List of Candle objects
        config: Configuration for indicator periods
        
    Returns:
        Dictionary of computed indicators
    """
    if len(candles) < 20:
        return {}
    
    # Extract price and volume data
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    
    # Compute indicators
    atr_period = config.get("market_analyzer", {}).get("atr_period", 14)
    rsi_period = config.get("strategies", {}).get("trend", {}).get("rsi_period", 14)
    bb_period = config.get("market_analyzer", {}).get("bb_period", 20)
    
    atr_values = compute_atr(candles, atr_period)
    rsi_values = compute_rsi(closes, rsi_period)
    bb_middle, bb_upper, bb_lower = compute_bollinger_bands(closes, bb_period)
    
    return {
        "atr": atr_values,
        "rsi": rsi_values,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "volume": volumes,
        "closes": closes,
    }


def log_market_summary(symbol, candles, indicators, logger):
    """Log current market conditions for a symbol."""
    if not candles or not indicators:
        return
    
    latest_candle = candles[-1]
    
    # Get latest indicator values
    atr = indicators["atr"][-1] if indicators["atr"] else 0
    rsi = indicators["rsi"][-1] if indicators["rsi"] else 50
    bb_middle = indicators["bb_middle"][-1] if indicators["bb_middle"] else latest_candle.close
    bb_upper = indicators["bb_upper"][-1] if indicators["bb_upper"] else latest_candle.close
    bb_lower = indicators["bb_lower"][-1] if indicators["bb_lower"] else latest_candle.close
    
    # Calculate metrics
    atr_ratio = (atr / latest_candle.close) * 100 if latest_candle.close > 0 else 0
    bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
    
    logger.info(f"{symbol}: Price={latest_candle.close:,.0f}, ATR={atr_ratio:.2f}%, RSI={rsi:.1f}, BB_width={bb_width:.2f}%")


def log_position_summary(position_manager, logger):
    """Log current position summary."""
    positions = position_manager.get_positions()
    
    if not positions:
        logger.info("No open positions")
        return
    
    logger.info(f"Open positions ({len(positions)}):")
    for i, pos in enumerate(positions, 1):
        logger.info(
            f"  {i}. {pos.symbol} {pos.mode.name}: "
            f"entry={pos.entry_price:,.0f}, peak={pos.peak_price:,.0f}, "
            f"size={pos.size:.6f}"
        )


def simulate_multiple_cycles(
    config, logger, client, analyzer, strategy_manager, execution_engine,
    portfolio_value, available_krw
):
    """
    Simulate multiple trading cycles to see position changes.
    
    This simulates what would happen if the bot ran over time with
    different market conditions.
    """
    logger.info("\n--- Simulating Multiple Trading Cycles ---")
    
    symbols = config["strategies"]["symbols"]
    now = datetime.now(timezone.utc)
    
    # Simulate 3 cycles with slight variations
    for cycle in range(1, 4):
        logger.info(f"\n=== Cycle {cycle} ===")
        
        # Fetch fresh market data (simulating time passage)
        market_data = {}
        indicators = {}
        
        for symbol in symbols:
            candles = client.get_candles_4h(symbol, count=100)
            market_data[symbol] = candles
            
            symbol_indicators = compute_symbol_indicators(candles, config)
            indicators[symbol] = symbol_indicators
            
            log_market_summary(symbol, candles, symbol_indicators, logger)
        
        # Analyze market mode
        btc_candles = market_data.get("KRW-BTC", [])
        if btc_candles:
            market_mode = analyzer.update_mode(btc_candles, now)
        else:
            market_mode = MarketMode.NEUTRAL
        
        logger.info(f"Market mode: {market_mode}")
        
        # Generate signals
        signals = strategy_manager.on_new_candle(
            market_mode=market_mode,
            market_data=market_data,
            indicators=indicators,
            portfolio_value=portfolio_value,
            available_krw=available_krw,
            now=now,
        )
        
        logger.info(f"Generated {len(signals)} signals:")
        for signal in signals:
            logger.info(f"  📊 {signal}")
        
        # Execute signals
        execution_engine.process_signals(signals, portfolio_value, now)
        
        # Update prices (simulate market movement)
        execution_engine.process_price_tick(now)
        
        # Log position status
        log_position_summary(strategy_manager.position_manager, logger)
        
        # Simulate time passage and slight mode changes for next cycle
        if cycle < 3:
            # Force a different mode for demonstration
            if market_mode == MarketMode.TREND:
                analyzer._current_mode = MarketMode.RANGE
            elif market_mode == MarketMode.RANGE:
                analyzer._current_mode = MarketMode.TREND
            else:
                analyzer._current_mode = MarketMode.TREND


def main() -> None:
    """Test the complete execution pipeline."""
    
    # Setup
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        config["persistence"]["log_level"]
    )
    
    logger.info("=" * 70)
    logger.info("Testing Execution Engine - Dry Run Mode (Step 6)")
    logger.info("=" * 70)
    
    try:
        # Create core components
        logger.info("\n--- Initializing Components ---")
        
        # Rate limiter and client
        rl = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
        )
        
        client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=rl,
        )
        
        # Market analysis
        analyzer = MarketAnalyzer(config)
        
        # Risk management
        position_manager = PositionManager(config["persistence"]["positions_file"])
        risk_manager = RiskManager(config)
        
        # Strategies
        trend_strategy = VolatilityBreakoutStrategy(config["strategies"]["trend"])
        range_strategy = RSIMeanReversionStrategy(config["strategies"]["range"])
        
        strategy_manager = StrategyManager(
            position_manager=position_manager,
            risk_manager=risk_manager,
            trend_strategy=trend_strategy,
            range_strategy=range_strategy,
        )
        
        # Execution engine
        execution_engine = ExecutionEngine(
            upbit_client=client,
            position_manager=position_manager,
            risk_manager=risk_manager,
            dry_run=True,
        )
        
        logger.info(f"Created: {execution_engine}")
        logger.info(f"Created: {strategy_manager}")
        
        # Clear any existing positions for clean test
        position_manager.clear_positions()
        logger.info("Cleared existing positions for clean test")
        
        # Fetch market data and compute indicators
        logger.info("\n--- Fetching Market Data ---")
        symbols = config["strategies"]["symbols"]
        market_data = {}
        indicators = {}
        
        for symbol in symbols:
            logger.info(f"Fetching {symbol} candles...")
            candles = client.get_candles_4h(symbol, count=100)
            market_data[symbol] = candles
            
            # Compute indicators
            symbol_indicators = compute_symbol_indicators(candles, config)
            indicators[symbol] = symbol_indicators
            
            log_market_summary(symbol, candles, symbol_indicators, logger)
        
        # Determine market mode
        logger.info("\n--- Market Mode Analysis ---")
        btc_candles = market_data.get("KRW-BTC", [])
        now = datetime.now(timezone.utc)
        
        if btc_candles:
            market_mode = analyzer.update_mode(btc_candles, now)
            logger.info(f"Detected market mode: {market_mode}")
            
            # Get analysis metrics
            if len(btc_candles) >= 20:
                metrics = analyzer._compute_metrics(btc_candles)
                logger.info(f"Analysis metrics: ADX={metrics['adx']:.1f}, ATR_ratio={metrics['atr_ratio']:.2f}%, BB_width={metrics['bandwidth']:.2f}%")
        else:
            market_mode = MarketMode.NEUTRAL
            logger.warning("No BTC data available, using NEUTRAL mode")
        
        # Portfolio setup
        portfolio_value = 1_000_000.0  # 1M KRW
        available_krw = 1_000_000.0    # Full amount available initially
        
        logger.info(f"\nPortfolio: {portfolio_value:,.0f} KRW total, {available_krw:,.0f} KRW available")
        
        # Generate signals
        logger.info("\n--- Signal Generation ---")
        signals = strategy_manager.on_new_candle(
            market_mode=market_mode,
            market_data=market_data,
            indicators=indicators,
            portfolio_value=portfolio_value,
            available_krw=available_krw,
            now=now,
        )
        
        logger.info(f"Generated {len(signals)} trade signals:")
        for i, signal in enumerate(signals, 1):
            logger.info(f"  {i}. {signal}")
        
        # Execute signals
        logger.info("\n--- Signal Execution ---")
        execution_engine.process_signals(signals, portfolio_value, now)
        
        # Update prices (simulate real-time price updates)
        logger.info("\n--- Price Updates ---")
        execution_engine.process_price_tick(now)
        
        # Show final positions
        logger.info("\n--- Final Position Summary ---")
        log_position_summary(position_manager, logger)
        
        # Show execution status
        status = execution_engine.get_execution_status()
        logger.info(f"\nExecution Status:")
        logger.info(f"  Mode: {'DRY RUN' if status['dry_run'] else 'LIVE'}")
        logger.info(f"  Total positions: {status['total_positions']}")
        logger.info(f"  Total exposure: {status['total_exposure_krw']:,.0f} KRW")
        logger.info(f"  Kill switch: {'ACTIVE' if status['kill_switch_active'] else 'INACTIVE'}")
        
        # Run multiple cycles simulation
        simulate_multiple_cycles(
            config, logger, client, analyzer, strategy_manager, execution_engine,
            portfolio_value, available_krw
        )
        
        # Final cleanup
        client.close()
        
        logger.info("\n" + "=" * 70)
        logger.info("Execution Engine testing completed successfully!")
        logger.info("All components working correctly in dry-run mode.")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Execution testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()