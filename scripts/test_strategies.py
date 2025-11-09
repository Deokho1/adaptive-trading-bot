#!/usr/bin/env python3
"""
Test script for Step 5 - Trading Strategies.

This script tests the complete strategy system:
- VolatilityBreakoutStrategy (TREND mode)
- RSIMeanReversionStrategy (RANGE mode)
- StrategyManager orchestration
- Signal generation with real market data

Usage:
    python scripts/test_strategies.py
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


def compute_symbol_indicators(candles, config):
    """
    Compute technical indicators for a symbol.
    
    Args:
        candles: List of Candle objects
        config: Configuration for indicator periods
        
    Returns:
        Dictionary of computed indicators
    """
    if len(candles) < 20:  # Minimum for meaningful indicators
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


def analyze_market_condition(candles, indicators, logger):
    """
    Analyze and log current market conditions.
    
    Args:
        candles: List of candles
        indicators: Computed indicators
        logger: Logger instance
    """
    if not candles or not indicators:
        return
    
    latest_candle = candles[-1]
    
    # Get latest indicator values
    atr = indicators["atr"][-1] if indicators["atr"] else 0
    rsi = indicators["rsi"][-1] if indicators["rsi"] else 50
    bb_middle = indicators["bb_middle"][-1] if indicators["bb_middle"] else latest_candle.close
    bb_upper = indicators["bb_upper"][-1] if indicators["bb_upper"] else latest_candle.close
    bb_lower = indicators["bb_lower"][-1] if indicators["bb_lower"] else latest_candle.close
    
    # Calculate additional metrics
    atr_ratio = (atr / latest_candle.close) * 100 if latest_candle.close > 0 else 0
    bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
    
    # Determine price position in Bollinger Bands
    if latest_candle.close <= bb_lower:
        bb_position = "Below Lower"
    elif latest_candle.close <= bb_middle:
        bb_position = "Lower Half"
    elif latest_candle.close <= bb_upper:
        bb_position = "Upper Half"
    else:
        bb_position = "Above Upper"
    
    logger.info(f"  Price: {latest_candle.close:,.0f} KRW")
    logger.info(f"  ATR: {atr:,.0f} ({atr_ratio:.2f}%)")
    logger.info(f"  RSI: {rsi:.1f}")
    logger.info(f"  BB Position: {bb_position}")
    logger.info(f"  BB Width: {bb_width:.2f}%")
    logger.info(f"  Volume: {latest_candle.volume:,.0f}")


def test_individual_strategies(config, logger, market_data, indicators, market_mode):
    """
    Test individual strategies in isolation.
    
    Args:
        config: Configuration
        logger: Logger instance
        market_data: Market candle data
        indicators: Computed indicators
        market_mode: Current market mode
    """
    logger.info("\n--- Testing Individual Strategies ---")
    
    # Create strategies
    trend_strategy = VolatilityBreakoutStrategy(config["strategies"]["trend"])
    range_strategy = RSIMeanReversionStrategy(config["strategies"]["range"])
    
    logger.info(f"Trend Strategy: {trend_strategy}")
    logger.info(f"Range Strategy: {range_strategy}")
    
    # Mock portfolio data
    portfolio_value = 1_000_000.0  # 1M KRW
    available_krw = 800_000.0      # 800K available
    now = datetime.now(timezone.utc)
    
    # Test each symbol
    for symbol, candles in market_data.items():
        if not candles:
            continue
            
        logger.info(f"\n=== Testing {symbol} ===")
        symbol_indicators = indicators.get(symbol, {})
        
        # Create strategy context (no position initially)
        from strategy.base import StrategyContext
        context = StrategyContext(
            symbol=symbol,
            candles=candles,
            indicators=symbol_indicators,
            mode=market_mode,
            position=None,
            portfolio_value=portfolio_value,
            available_krw=available_krw,
            now=now,
        )
        
        # Test trend strategy
        logger.info(f"\nTrend Strategy (mode={market_mode}):")
        trend_signals = trend_strategy.generate_signals(context)
        if trend_signals:
            for signal in trend_signals:
                logger.info(f"  📈 {signal}")
        else:
            logger.info("  No trend signals")
        
        # Test range strategy  
        logger.info(f"\nRange Strategy (mode={market_mode}):")
        range_signals = range_strategy.generate_signals(context)
        if range_signals:
            for signal in range_signals:
                logger.info(f"  📊 {signal}")
        else:
            logger.info("  No range signals")


def test_strategy_manager(config, logger, market_data, indicators, market_mode):
    """
    Test the integrated StrategyManager.
    
    Args:
        config: Configuration
        logger: Logger instance  
        market_data: Market candle data
        indicators: Computed indicators
        market_mode: Current market mode
    """
    logger.info("\n--- Testing StrategyManager ---")
    
    # Create components
    position_manager = PositionManager(config["persistence"]["positions_file"])
    risk_manager = RiskManager(config)
    trend_strategy = VolatilityBreakoutStrategy(config["strategies"]["trend"])
    range_strategy = RSIMeanReversionStrategy(config["strategies"]["range"])
    
    # Create strategy manager
    strategy_manager = StrategyManager(
        position_manager=position_manager,
        risk_manager=risk_manager,
        trend_strategy=trend_strategy,
        range_strategy=range_strategy,
    )
    
    logger.info(f"Created: {strategy_manager}")
    
    # Mock portfolio data
    portfolio_value = 1_000_000.0
    available_krw = 800_000.0
    now = datetime.now(timezone.utc)
    
    # Test signal generation
    logger.info(f"\nGenerating signals for market mode: {market_mode}")
    
    signals = strategy_manager.on_new_candle(
        market_mode=market_mode,
        market_data=market_data,
        indicators=indicators,
        portfolio_value=portfolio_value,
        available_krw=available_krw,
        now=now,
    )
    
    logger.info(f"Generated {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        logger.info(f"  {i}. {signal}")
    
    # Test strategy status
    status = strategy_manager.get_strategy_status()
    logger.info(f"\nStrategy Status: {status}")
    
    return strategy_manager, signals


def main() -> None:
    """Test the complete strategy system."""
    
    # Setup
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        config["persistence"]["log_level"]
    )
    
    logger.info("=" * 60)
    logger.info("Testing Trading Strategies (Step 5)")
    logger.info("=" * 60)
    
    try:
        # Create exchange client
        rl = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
        )
        
        client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=rl,
        )
        
        # Get symbols to test
        test_symbols = config["strategies"]["symbols"]
        logger.info(f"Testing symbols: {test_symbols}")
        
        # Fetch market data
        logger.info("\n--- Fetching Market Data ---")
        market_data = {}
        indicators = {}
        
        for symbol in test_symbols:
            logger.info(f"Fetching {symbol} candles...")
            candles = client.get_candles_4h(symbol, count=100)
            market_data[symbol] = candles
            
            # Compute indicators
            symbol_indicators = compute_symbol_indicators(candles, config)
            indicators[symbol] = symbol_indicators
            
            logger.info(f"  Fetched {len(candles)} candles")
            analyze_market_condition(candles, symbol_indicators, logger)
        
        # Analyze market mode (using BTC as reference)
        logger.info("\n--- Market Mode Analysis ---")
        analyzer = MarketAnalyzer(config)
        btc_candles = market_data.get("KRW-BTC", [])
        
        if btc_candles:
            now = datetime.now(timezone.utc)
            market_mode = analyzer.update_mode(btc_candles, now)
            logger.info(f"Current market mode: {market_mode}")
            
            # Get analysis metrics
            if len(btc_candles) >= 20:
                metrics = analyzer._compute_metrics(btc_candles)
                logger.info(f"Market analysis metrics: {metrics}")
        else:
            market_mode = MarketMode.NEUTRAL
            logger.warning("No BTC data available, using NEUTRAL mode")
        
        # Test individual strategies
        test_individual_strategies(config, logger, market_data, indicators, market_mode)
        
        # Test integrated strategy manager
        strategy_manager, signals = test_strategy_manager(config, logger, market_data, indicators, market_mode)
        
        # Test mode change scenario
        logger.info("\n--- Testing Mode Change ---")
        logger.info("Simulating mode change from current to opposite mode...")
        
        if market_mode == MarketMode.TREND:
            new_mode = MarketMode.RANGE
        elif market_mode == MarketMode.RANGE:
            new_mode = MarketMode.TREND
        else:
            new_mode = MarketMode.TREND
        
        logger.info(f"Changing mode: {market_mode} -> {new_mode}")
        
        mode_change_signals = strategy_manager.on_new_candle(
            market_mode=new_mode,
            market_data=market_data,
            indicators=indicators,
            portfolio_value=1_000_000.0,
            available_krw=800_000.0,
            now=datetime.now(timezone.utc),
        )
        
        logger.info(f"Mode change generated {len(mode_change_signals)} signals:")
        for signal in mode_change_signals:
            logger.info(f"  {signal}")
        
        # Clean up
        client.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("Strategy testing completed successfully!")
        logger.info(f"Total signals generated: {len(signals) + len(mode_change_signals)}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Strategy testing failed: {e}")
        raise


if __name__ == "__main__":
    main()