#!/usr/bin/env python3
"""
Unit tests for individual components
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import MarketAnalyzer, StrategyManager, RiskManager, ExecutionEngine
from src.utils import RateLimiter, PositionTracker


def create_sample_ohlcv_data(periods=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate random price data
    close_prices = 50000000 + np.cumsum(np.random.randn(periods) * 100000)
    high_prices = close_prices + np.random.rand(periods) * 500000
    low_prices = close_prices - np.random.rand(periods) * 500000
    open_prices = close_prices + np.random.randn(periods) * 200000
    volume = np.random.rand(periods) * 1000
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return df


def test_market_analyzer():
    """Test MarketAnalyzer component"""
    print("\n" + "=" * 60)
    print("Testing MarketAnalyzer")
    print("=" * 60)
    
    config = {
        'adx_period': 14,
        'adx_threshold': 25,
        'atr_period': 14,
        'bb_period': 20,
        'bb_std': 2,
        'rsi_period': 14
    }
    
    analyzer = MarketAnalyzer(config)
    df = create_sample_ohlcv_data(100)
    
    # Test RSI
    rsi = analyzer.calculate_rsi(df)
    assert len(rsi) == len(df), "RSI length mismatch"
    assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100, "RSI out of range"
    print(f"✓ RSI calculated: {rsi.iloc[-1]:.2f}")
    
    # Test Bollinger Bands
    bb = analyzer.calculate_bollinger_bands(df)
    assert 'upper' in bb and 'middle' in bb and 'lower' in bb, "BB missing bands"
    assert bb['upper'].iloc[-1] > bb['middle'].iloc[-1] > bb['lower'].iloc[-1], "BB bands invalid"
    print(f"✓ Bollinger Bands calculated: Upper={bb['upper'].iloc[-1]:.0f}, Lower={bb['lower'].iloc[-1]:.0f}")
    
    # Test ATR
    atr = analyzer.calculate_atr(df)
    assert len(atr) == len(df), "ATR length mismatch"
    assert atr.iloc[-1] > 0, "ATR should be positive"
    print(f"✓ ATR calculated: {atr.iloc[-1]:.0f}")
    
    # Test ADX
    adx = analyzer.calculate_adx(df)
    assert len(adx) == len(df), "ADX length mismatch"
    print(f"✓ ADX calculated: {adx.iloc[-1]:.2f}")
    
    # Test market condition analysis
    condition = analyzer.analyze_market_condition(df)
    assert condition in ['trend', 'range'], "Invalid market condition"
    print(f"✓ Market condition: {condition}")
    
    print("✓ All MarketAnalyzer tests passed!")


def test_strategy_manager():
    """Test StrategyManager component"""
    print("\n" + "=" * 60)
    print("Testing StrategyManager")
    print("=" * 60)
    
    market_config = {
        'adx_period': 14,
        'adx_threshold': 25,
        'atr_period': 14,
        'bb_period': 20,
        'bb_std': 2,
        'rsi_period': 14
    }
    
    strategy_config = {
        'trend': {'k_value': 0.5},
        'range': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}
    }
    
    analyzer = MarketAnalyzer(market_config)
    manager = StrategyManager(strategy_config, analyzer)
    
    df = create_sample_ohlcv_data(100)
    current_price = df['close'].iloc[-1]
    
    # Test trend strategy
    signal, details = manager.trend_strategy_signal(df, current_price)
    assert signal in ['buy', 'sell', 'hold'], "Invalid signal"
    assert 'strategy' in details and details['strategy'] == 'trend', "Missing strategy info"
    print(f"✓ Trend strategy signal: {signal}")
    
    # Test range strategy
    signal, details = manager.range_strategy_signal(df, current_price, False)
    assert signal in ['buy', 'sell', 'hold'], "Invalid signal"
    assert 'strategy' in details and details['strategy'] == 'range', "Missing strategy info"
    print(f"✓ Range strategy signal: {signal}")
    
    # Test get trading signal
    signal, details = manager.get_trading_signal(df, current_price, False)
    assert signal in ['buy', 'sell', 'hold'], "Invalid signal"
    print(f"✓ Trading signal: {signal} ({details.get('strategy', 'N/A')})")
    
    print("✓ All StrategyManager tests passed!")


def test_risk_manager():
    """Test RiskManager component"""
    print("\n" + "=" * 60)
    print("Testing RiskManager")
    print("=" * 60)
    
    config = {
        'max_position_size': 0.95,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.10
    }
    
    manager = RiskManager(config)
    
    # Test position size calculation
    available_krw = 1000000
    current_price = 50000000
    position_size = manager.calculate_position_size(available_krw, current_price)
    assert position_size > 0, "Position size should be positive"
    print(f"✓ Position size calculated: {position_size:.8f} units")
    
    # Test stop loss
    entry_price = 50000000
    loss_price = entry_price * 0.90  # 10% loss
    should_stop = manager.check_stop_loss(entry_price, loss_price)
    assert should_stop == True, "Stop loss should trigger"
    print(f"✓ Stop loss triggered at 10% loss")
    
    # Test take profit
    profit_price = entry_price * 1.15  # 15% profit
    should_profit = manager.check_take_profit(entry_price, profit_price)
    assert should_profit == True, "Take profit should trigger"
    print(f"✓ Take profit triggered at 15% gain")
    
    # Test order validation
    is_valid, msg = manager.validate_order('buy', 0.01, 50000000, 1000000)
    assert is_valid == True, "Valid order should pass validation"
    print(f"✓ Order validation passed")
    
    # Test risk metrics
    metrics = manager.get_risk_metrics(entry_price, profit_price, position_size)
    assert 'unrealized_pnl' in metrics, "Missing P/L in metrics"
    assert 'stop_loss_price' in metrics, "Missing stop loss price"
    print(f"✓ Risk metrics calculated: P/L = {metrics['unrealized_pnl_pct']:.2f}%")
    
    print("✓ All RiskManager tests passed!")


def test_position_tracker():
    """Test PositionTracker component"""
    print("\n" + "=" * 60)
    print("Testing PositionTracker")
    print("=" * 60)
    
    tracker = PositionTracker(position_file="/tmp/test_positions.json")
    
    # Test initial state
    assert tracker.has_position() == False, "Should not have position initially"
    print("✓ Initial state: No position")
    
    # Test opening position
    tracker.open_position("KRW-BTC", 50000000, 0.01)
    assert tracker.has_position() == True, "Should have position after opening"
    assert tracker.get_entry_price() == 50000000, "Entry price mismatch"
    print("✓ Position opened successfully")
    
    # Test closing position
    trade = tracker.close_position(55000000, "test")
    assert tracker.has_position() == False, "Should not have position after closing"
    assert trade['profit'] > 0, "Should have profit"
    print(f"✓ Position closed: Profit = {trade['profit']:.2f} ({trade['profit_pct']:.2f}%)")
    
    # Test performance summary
    summary = tracker.get_performance_summary()
    assert summary['total_trades'] == 1, "Should have 1 trade"
    assert summary['winning_trades'] == 1, "Should have 1 winning trade"
    print(f"✓ Performance summary: {summary['total_trades']} trades, {summary['win_rate']:.2f}% win rate")
    
    # Clean up
    if os.path.exists("/tmp/test_positions.json"):
        os.remove("/tmp/test_positions.json")
    
    print("✓ All PositionTracker tests passed!")


def test_rate_limiter():
    """Test RateLimiter component"""
    print("\n" + "=" * 60)
    print("Testing RateLimiter")
    print("=" * 60)
    
    limiter = RateLimiter(requests_per_second=5, requests_per_minute=20)
    
    # Test basic functionality
    import time
    start = time.time()
    for i in range(3):
        limiter.wait_if_needed()
    elapsed = time.time() - start
    
    print(f"✓ Rate limiter allowed 3 requests in {elapsed:.2f}s")
    
    # Test decorator
    @limiter
    def sample_function():
        return "success"
    
    result = sample_function()
    assert result == "success", "Decorated function should work"
    print("✓ Rate limiter decorator works")
    
    print("✓ All RateLimiter tests passed!")


def test_execution_engine():
    """Test ExecutionEngine component (dry-run mode only)"""
    print("\n" + "=" * 60)
    print("Testing ExecutionEngine (Dry-run)")
    print("=" * 60)
    
    engine = ExecutionEngine({}, dry_run=True)
    
    # Test initial balance
    balance = engine.get_balance("KRW")
    assert balance == 1000000, "Initial KRW balance should be 1,000,000"
    print(f"✓ Initial balance: {balance:,.0f} KRW")
    
    # Test buy order
    result = engine.execute_buy("KRW-BTC", 0.01, 50000000)
    assert result['status'] == 'success', "Buy should succeed"
    print(f"✓ Buy executed: {result['amount']:.8f} at {result['price']:,.0f} KRW")
    
    # Check updated balance
    krw_balance = engine.get_balance("KRW")
    crypto_balance = engine.get_balance("BTC")
    assert crypto_balance == 0.01, "Crypto balance should be 0.01"
    print(f"✓ Updated balances: {krw_balance:,.0f} KRW, {crypto_balance:.8f} BTC")
    
    # Test sell order
    result = engine.execute_sell("KRW-BTC", 0.01, 55000000)
    assert result['status'] == 'success', "Sell should succeed"
    assert result['profit'] > 0, "Should have profit"
    print(f"✓ Sell executed: {result['amount']:.8f} at {result['price']:,.0f} KRW")
    print(f"  Profit: {result['profit']:,.0f} KRW ({result['profit_pct']:.2f}%)")
    
    print("✓ All ExecutionEngine tests passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Running Component Tests")
    print("=" * 60)
    
    try:
        test_market_analyzer()
        test_strategy_manager()
        test_risk_manager()
        test_position_tracker()
        test_rate_limiter()
        test_execution_engine()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
