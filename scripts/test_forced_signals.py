#!/usr/bin/env python3
"""
Quick test to force signal generation for ExecutionEngine validation.

This script forces TREND mode to generate buy signals and test execution.
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
    if len(candles) < 20:
        return {}
    
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    
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


def main():
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        "DEBUG"  # Use DEBUG level for detailed logging
    )
    
    logger.info("=" * 50)
    logger.info("Quick ExecutionEngine Test - Forced Signals")
    logger.info("=" * 50)
    
    try:
        # Create components
        rl = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
        )
        
        client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=rl,
        )
        
        position_manager = PositionManager(config["persistence"]["positions_file"])
        risk_manager = RiskManager(config)
        
        # Clear positions
        position_manager.clear_positions()
        
        # Create strategies with more aggressive parameters for testing
        trend_config = config["strategies"]["trend"].copy()
        trend_config["k_atr"] = 0.1  # Lower threshold for easier entry
        trend_strategy = VolatilityBreakoutStrategy(trend_config)
        
        range_strategy = RSIMeanReversionStrategy(config["strategies"]["range"])
        
        strategy_manager = StrategyManager(
            position_manager=position_manager,
            risk_manager=risk_manager,
            trend_strategy=trend_strategy,
            range_strategy=range_strategy,
        )
        
        execution_engine = ExecutionEngine(
            upbit_client=client,
            position_manager=position_manager,
            risk_manager=risk_manager,
            dry_run=True,
        )
        
        logger.info(f"Created: {execution_engine}")
        
        # Fetch market data
        symbols = ["KRW-BTC"]  # Test with just BTC
        market_data = {}
        indicators = {}
        
        for symbol in symbols:
            logger.info(f"Fetching {symbol} data...")
            candles = client.get_candles_4h(symbol, count=100)
            market_data[symbol] = candles
            
            symbol_indicators = compute_symbol_indicators(candles, config)
            indicators[symbol] = symbol_indicators
            
            logger.info(f"  Price: {candles[-1].close:,.0f} KRW")
        
        # Force TREND mode
        market_mode = MarketMode.TREND
        logger.info(f"Forcing market mode: {market_mode}")
        
        # Temporarily disable kill switch for testing
        risk_manager.kill_switch_triggered = False
        logger.info("Disabled kill switch for testing")
        
        # Generate signals
        portfolio_value = 1_000_000.0
        available_krw = 1_000_000.0
        now = datetime.now(timezone.utc)
        
        # Reset daily start for risk manager
        risk_manager.reset_daily_start(portfolio_value, now)
        
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
        
        if not signals:
            # Force create a buy signal for testing
            from strategy.base import TradeSignal
            from core.types import OrderSide
            
            test_signal = TradeSignal(
                symbol="KRW-BTC",
                side=OrderSide.BUY,
                mode=MarketMode.TREND,
                reason="forced_test_signal",
                amount_krw=50000.0  # 50K KRW
            )
            signals = [test_signal]
            logger.info(f"Created forced test signal: {test_signal}")
        
        # Execute signals
        logger.info("\n--- Executing Signals ---")
        execution_engine.process_signals(signals, portfolio_value, now)
        
        # Check positions
        positions = position_manager.get_positions()
        logger.info(f"\nOpen positions after execution: {len(positions)}")
        for i, pos in enumerate(positions, 1):
            logger.info(f"  {i}. {pos.symbol} {pos.mode.name}: entry={pos.entry_price:,.0f}, size={pos.size:.6f}")
        
        # Test price update
        logger.info("\n--- Testing Price Updates ---")
        execution_engine.process_price_tick(now)
        
        # Test sell signal if we have positions
        if positions:
            logger.info("\n--- Testing Sell Signal ---")
            sell_signal = TradeSignal(
                symbol=positions[0].symbol,
                side=OrderSide.SELL,
                mode=positions[0].mode,
                reason="test_exit_signal"
            )
            
            logger.info(f"Created sell signal: {sell_signal}")
            execution_engine.process_signals([sell_signal], portfolio_value, now)
            
            # Check positions after sell
            final_positions = position_manager.get_positions()
            logger.info(f"Positions after sell: {len(final_positions)}")
        
        # Show final status
        status = execution_engine.get_execution_status()
        logger.info(f"\nFinal Status:")
        logger.info(f"  Total positions: {status['total_positions']}")
        logger.info(f"  Total exposure: {status['total_exposure_krw']:,.0f} KRW")
        
        client.close()
        
        logger.info("\n" + "=" * 50)
        logger.info("ExecutionEngine test completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()