#!/usr/bin/env python3
"""
Main entry point for the Adaptive Dual-Mode Crypto Trading Bot

This is the initial implementation (Step 1) that focuses on:
- Loading configuration from config/config.yaml
- Setting up logging to both console and file
- Basic startup logging

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

from core.config_loader import load_config
from core.logger import setup_logger


def main():
    """Main entry point"""
    try:
        # Load configuration
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
