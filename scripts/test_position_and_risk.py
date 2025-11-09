#!/usr/bin/env python3
"""
Test script for Step 4 - Position and Risk Management.

This script tests the position and risk management implementation:
- PositionManager with JSON persistence
- RiskManager with capital and loss limits
- Strategy base classes

Usage:
    python scripts/test_position_and_risk.py
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
from exchange.models import Position
from risk.position_manager import PositionManager
from risk.risk_manager import RiskManager
from strategy.base import StrategyContext, TradeSignal, Strategy


def test_position_manager(config, logger):
    """Test PositionManager functionality."""
    logger.info("\n--- Testing PositionManager ---")
    
    # Create position manager
    positions_file = config["persistence"]["positions_file"]
    logger.info(f"Creating PositionManager with file: {positions_file}")
    
    pm = PositionManager(positions_file)
    
    # Clear any existing positions for clean test
    pm.positions = []
    pm.save_positions()
    
    logger.info(f"Initial position count: {pm.get_position_count()}")
    
    # Open a fake TREND position
    now = datetime.now(timezone.utc)
    position = pm.open_position(
        symbol="KRW-BTC",
        mode=MarketMode.TREND,
        entry_price=150_000_000.0,  # 1.5억원
        size=0.001,  # 0.001 BTC
        now=now
    )
    
    logger.info(f"Opened position: {position}")
    logger.info(f"Position count after opening: {pm.get_position_count()}")
    
    # Update peak price
    pm.update_peak_price("KRW-BTC", 155_000_000.0)
    updated_position = pm.get_position("KRW-BTC")
    logger.info(f"Updated peak price: {updated_position.peak_price:,.0f}")
    
    # Test persistence by reloading
    logger.info("Testing persistence - creating new PositionManager...")
    pm2 = PositionManager(positions_file)
    logger.info(f"Reloaded position count: {pm2.get_position_count()}")
    
    if pm2.get_position_count() > 0:
        reloaded_position = pm2.get_position("KRW-BTC")
        logger.info(f"Reloaded position: {reloaded_position}")
    
    # Test filtering by mode
    trend_positions = pm2.get_positions_by_mode(MarketMode.TREND)
    range_positions = pm2.get_positions_by_mode(MarketMode.RANGE)
    logger.info(f"TREND positions: {len(trend_positions)}")
    logger.info(f"RANGE positions: {len(range_positions)}")
    
    return pm2


def test_risk_manager(config, logger, positions):
    """Test RiskManager functionality."""
    logger.info("\n--- Testing RiskManager ---")
    
    # Create risk manager
    rm = RiskManager(config)
    logger.info(f"Created RiskManager with config: {rm.get_risk_status()}")
    
    # Mock portfolio
    portfolio_value = 1_000_000.0  # 100만원
    logger.info(f"Mock portfolio value: {portfolio_value:,.0f} KRW")
    
    # Test daily PnL tracking
    now = datetime.now(timezone.utc)
    rm.reset_daily_start(portfolio_value, now)
    logger.info(f"Reset daily start value: {rm.daily_start_value:,.0f}")
    
    # Simulate some loss
    new_value = portfolio_value * 0.97  # 3% loss
    daily_pnl = rm.update_daily_pnl(new_value, now)
    logger.info(f"Daily P&L after 3% loss: {daily_pnl:.2f}%")
    logger.info(f"Kill switch active: {rm.is_kill_switch_active()}")
    
    # Test capital allocation
    existing_positions = positions.get_positions()
    
    # Test TREND trade
    can_trend = rm.can_open_trade(
        mode=MarketMode.TREND,
        portfolio_value=portfolio_value,
        existing_positions=existing_positions,
        new_trade_amount_krw=200_000.0,  # 20만원
        symbol="KRW-ETH"
    )
    logger.info(f"Can open TREND trade (20만원): {can_trend}")
    
    # Test RANGE trade
    can_range = rm.can_open_trade(
        mode=MarketMode.RANGE,
        portfolio_value=portfolio_value,
        existing_positions=existing_positions,
        new_trade_amount_krw=150_000.0,  # 15만원
        symbol="KRW-ADA"
    )
    logger.info(f"Can open RANGE trade (15만원): {can_range}")
    
    # Test available capital
    trend_capital = rm.get_available_capital_for_mode(
        MarketMode.TREND, portfolio_value, existing_positions
    )
    range_capital = rm.get_available_capital_for_mode(
        MarketMode.RANGE, portfolio_value, existing_positions
    )
    
    logger.info(f"Available TREND capital: {trend_capital:,.0f} KRW")
    logger.info(f"Available RANGE capital: {range_capital:,.0f} KRW")
    
    return rm


def test_strategy_base(logger):
    """Test Strategy base classes."""
    logger.info("\n--- Testing Strategy Base Classes ---")
    
    # Test TradeSignal
    buy_signal = TradeSignal(
        symbol="KRW-BTC",
        side=OrderSide.BUY,
        mode=MarketMode.TREND,
        reason="Strong uptrend detected",
        amount_krw=500_000.0
    )
    
    sell_signal = TradeSignal(
        symbol="KRW-BTC",
        side=OrderSide.SELL,
        mode=MarketMode.RANGE,
        reason="Target reached",
        size=0.001
    )
    
    logger.info(f"Buy signal: {buy_signal}")
    logger.info(f"Sell signal: {sell_signal}")
    
    # Test Strategy base class
    class TestStrategy(Strategy):
        def generate_signals(self, context):
            return [buy_signal]
    
    strategy = TestStrategy({"test_param": 123})
    logger.info(f"Strategy: {strategy}")
    
    # Test StrategyContext (just create one for demonstration)
    context = StrategyContext(
        symbol="KRW-BTC",
        candles=[],  # Empty for test
        indicators={},
        mode=MarketMode.TREND,
        position=None,
        portfolio_value=1_000_000.0,
        available_krw=800_000.0,
        now=datetime.now(timezone.utc)
    )
    
    signals = strategy.generate_signals(context)
    logger.info(f"Generated signals: {len(signals)}")
    for signal in signals:
        logger.info(f"  Signal: {signal}")


def main() -> None:
    """Test position and risk management implementation."""
    
    # Setup
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        config["persistence"]["log_level"]
    )
    
    logger.info("=" * 60)
    logger.info("Testing Position and Risk Management (Step 4)")
    logger.info("=" * 60)
    
    try:
        # Test PositionManager
        position_manager = test_position_manager(config, logger)
        
        # Test RiskManager
        risk_manager = test_risk_manager(config, logger, position_manager)
        
        # Test Strategy base classes
        test_strategy_base(logger)
        
        # Summary
        logger.info("\n--- Test Summary ---")
        logger.info(f"Positions in memory: {position_manager.get_position_count()}")
        logger.info(f"Risk status: {risk_manager.get_risk_status()}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Position and risk management test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Position and risk management test failed: {e}")
        raise


if __name__ == "__main__":
    main()