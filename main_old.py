#!/usr/bin/env python3
"""
Main entry point for the Adaptive Dual-Mode Crypto Trading Bot

This bot implements adaptive trading strategies that switch between
trend-following and mean-reversion based on market conditions.

Usage:
    python main.py              # Run single cycle
    python main.py --forever    # Run continuously (future option)
"""

import sys
import logging
from pathlib import Path

from core.config_loader import load_config
from core.logger import setup_logger
from core.app import BotApp


def main() -> None:
    """Main entry point for the trading bot."""
    try:
        # Load configuration
        config = load_config()
        
        # Setup logging
        logger = setup_logger(
            config["persistence"]["logs_dir"],
            config["persistence"]["log_level"],
        )
        
        logger.info("🚀 Starting Adaptive Trading Bot (Step 7)")
        logger.info(f"Configuration loaded from: {Path('config/config.yaml').absolute()}")
        logger.info(f"Dry run mode: {config['app']['dry_run']}")
        
        # Create and run bot application
        app = BotApp(config)
        logger.info(f"Created: {app}")
        
        # For now, run a single cycle
        # Later we can add CLI options for --forever mode
        logger.info("Executing single bot cycle...")
        app.run_once()
        
        # Clean up resources
        app.cleanup()
        
        logger.info("✅ Bot execution completed successfully")
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        config = load_config("config/config.yaml")
        
        # Setup logger
        logs_dir = config["persistence"]["logs_dir"]
        log_level = config["persistence"]["log_level"]
        logger = setup_logger(logs_dir, log_level)
        
        logger.info("Config loaded")
        
        # Log startup mode
        dry_run = config["app"]["dry_run"]
        if dry_run:
            logger.info("Bot starting in DRY_RUN mode")
        else:
            logger.info("Bot starting in LIVE mode")
        
        # Log some basic config info
        exchange_name = config["exchange"]["name"]
        timezone = config["app"]["timezone"]
        logger.info(f"Exchange: {exchange_name}")
        logger.info(f"Timezone: {timezone}")
        
        logger.info("Adaptive Trading Bot initialized successfully")
        logger.info("Bot startup complete - trading logic will be implemented in next steps")
        
    except FileNotFoundError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Startup error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
