#!/usr/bin/env python3
"""
Quick test script to validate the bot runs without errors
"""

import sys
import os
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot import AdaptiveTradingBot


def timeout_handler(signum, frame):
    """Handle timeout"""
    print("\n✓ Test completed - bot ran successfully for test duration")
    sys.exit(0)


def main():
    print("=" * 60)
    print("Testing Adaptive Trading Bot (5 second test)")
    print("=" * 60)
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second test
    
    try:
        # Create and start bot
        bot = AdaptiveTradingBot(config_path="config.yaml")
        
        # Verify bot is initialized properly
        assert bot.market_analyzer is not None, "MarketAnalyzer not initialized"
        assert bot.strategy_manager is not None, "StrategyManager not initialized"
        assert bot.risk_manager is not None, "RiskManager not initialized"
        assert bot.execution_engine is not None, "ExecutionEngine not initialized"
        assert bot.rate_limiter is not None, "RateLimiter not initialized"
        assert bot.position_tracker is not None, "PositionTracker not initialized"
        
        print("\n✓ All components initialized successfully")
        print("\nRunning bot for 5 seconds...")
        
        # Start bot (will be interrupted by alarm)
        bot.start()
        
    except KeyboardInterrupt:
        print("\n✓ Test completed")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
