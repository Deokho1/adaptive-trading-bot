#!/usr/bin/env python3
"""
Test to verify position persistence - creates positions and leaves them open.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from core.types import MarketMode, OrderSide
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.base import TradeSignal
from execution.execution_engine import ExecutionEngine


def main():
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        "INFO"
    )
    
    logger.info("=" * 60)
    logger.info("Testing Position Persistence")
    logger.info("=" * 60)
    
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
        
        execution_engine = ExecutionEngine(
            upbit_client=client,
            position_manager=position_manager,
            risk_manager=risk_manager,
            dry_run=True,
        )
        
        logger.info(f"Created: {execution_engine}")
        
        # Clear positions first
        position_manager.clear_positions()
        logger.info("Cleared existing positions")
        
        # Disable kill switch for testing
        risk_manager.kill_switch_triggered = False
        
        # Reset daily start for risk manager
        portfolio_value = 1_000_000.0
        now = datetime.now(timezone.utc)
        risk_manager.reset_daily_start(portfolio_value, now)
        
        # Create multiple test signals
        test_signals = [
            TradeSignal(
                symbol="KRW-BTC",
                side=OrderSide.BUY,
                mode=MarketMode.TREND,
                reason="test_btc_position",
                amount_krw=100000.0  # 100K KRW
            ),
            TradeSignal(
                symbol="KRW-ETH",
                side=OrderSide.BUY,
                mode=MarketMode.RANGE,
                reason="test_eth_position",
                amount_krw=80000.0   # 80K KRW
            )
        ]
        
        logger.info(f"Created {len(test_signals)} test signals:")
        for i, signal in enumerate(test_signals, 1):
            logger.info(f"  {i}. {signal}")
        
        # Execute signals
        logger.info("\n--- Executing Buy Signals ---")
        execution_engine.process_signals(test_signals, portfolio_value, now)
        
        # Check positions
        positions = position_manager.get_positions()
        logger.info(f"\n--- Position Summary ---")
        logger.info(f"Total open positions: {len(positions)}")
        
        if positions:
            total_value = 0.0
            for i, pos in enumerate(positions, 1):
                current_price = client.get_ticker_price(pos.symbol)
                current_value = current_price * pos.size
                total_value += current_value
                
                logger.info(f"  {i}. {pos.symbol} ({pos.mode.name}):")
                logger.info(f"     Entry: {pos.entry_price:,.0f} KRW")
                logger.info(f"     Peak:  {pos.peak_price:,.0f} KRW")
                logger.info(f"     Size:  {pos.size:.6f}")
                logger.info(f"     Current: {current_price:,.0f} KRW")
                logger.info(f"     Value: {current_value:,.0f} KRW")
                
                pnl = (current_price - pos.entry_price) * pos.size
                pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                logger.info(f"     P&L: {pnl:+,.0f} KRW ({pnl_pct:+.2f}%)")
            
            logger.info(f"\nTotal portfolio exposure: {total_value:,.0f} KRW")
        
        # Update prices
        logger.info("\n--- Updating Prices ---")
        execution_engine.process_price_tick(now)
        
        # Show final execution status
        status = execution_engine.get_execution_status()
        logger.info(f"\n--- Execution Status ---")
        logger.info(f"Mode: {'DRY RUN' if status['dry_run'] else 'LIVE'}")
        logger.info(f"Total positions: {status['total_positions']}")
        logger.info(f"Total exposure: {status['total_exposure_krw']:,.0f} KRW")
        logger.info(f"Kill switch: {'ACTIVE' if status['kill_switch_active'] else 'INACTIVE'}")
        
        # Show position file path
        positions_file = Path(config["persistence"]["positions_file"])
        logger.info(f"\nPosition data saved to: {positions_file.absolute()}")
        
        client.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("Position persistence test completed!")
        logger.info(f"Check {positions_file} to see saved positions.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()