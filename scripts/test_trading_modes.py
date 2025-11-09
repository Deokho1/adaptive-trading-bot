#!/usr/bin/env python3
"""
Test script to validate different trading modes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from core.app import BotApp
from core.types import TradingMode


def test_mode(mode: str):
    """Test a specific trading mode."""
    print(f"\n{'='*50}")
    print(f"Testing mode: {mode}")
    print('='*50)
    
    try:
        # Load and modify config
        config = load_config()
        config["app"]["mode"] = mode
        
        # Setup logging
        logger = setup_logger(
            config["persistence"]["logs_dir"],
            "INFO"
        )
        
        logger.info(f"Testing {mode} mode...")
        
        # Create BotApp
        app = BotApp(config)
        logger.info(f"Created: {app}")
        
        if mode == "paper":
            # Test paper mode fully
            logger.info("Running paper mode cycle...")
            app.run_once()
        else:
            # For live/backtest, just test initialization
            logger.info(f"{mode.upper()} mode initialized but not executed (not implemented)")
            
            # Try to get status
            status = app.execution_engine.get_execution_status()
            logger.info(f"Status: {status}")
        
        app.cleanup()
        logger.info(f"✅ {mode.upper()} mode test completed")
        
    except Exception as e:
        logger.error(f"❌ {mode.upper()} mode test failed: {e}")
        return False
    
    return True


def main():
    """Test all trading modes."""
    print("🧪 Testing Multi-Mode Trading Bot")
    
    modes = ["paper", "live", "backtest"]
    results = {}
    
    for mode in modes:
        results[mode] = test_mode(mode)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    
    for mode, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{mode.upper():10} : {status}")
    
    print("\nNote: LIVE and BACKTEST modes are not implemented yet.")
    print("They should initialize but raise NotImplementedError when executed.")


if __name__ == "__main__":
    main()