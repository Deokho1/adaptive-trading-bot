#!/usr/bin/env python3
"""
Main entry point for the Adaptive Dual-Mode Crypto Trading Bot

Usage:
    python main.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bot import AdaptiveTradingBot


def main():
    """Main entry point"""
    print("=" * 60)
    print("Adaptive Dual-Mode Crypto Trading Bot")
    print("=" * 60)
    print()
    
    # Create and start bot
    bot = AdaptiveTradingBot(config_path="config.yaml")
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nShutdown requested... stopping bot")
        bot.stop()
    except Exception as e:
        print(f"\nFatal error: {e}")
        bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
