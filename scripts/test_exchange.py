#!/usr/bin/env python3
"""
Test script for Step 2 - Exchange layer testing.

This script tests the exchange layer implementation:
- RateLimiter functionality
- UpbitClient public API methods
- Data model parsing

Usage:
    python scripts/test_exchange.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from exchange import RateLimiter, UpbitClient


def test_exchange_layer():
    """Test the exchange layer implementation."""
    
    # Setup
    config = load_config()
    logger = setup_logger(config["persistence"]["logs_dir"], config["persistence"]["log_level"])
    
    logger.info("=" * 60)
    logger.info("Testing Exchange Layer (Step 2)")
    logger.info("=" * 60)
    
    try:
        # Create rate limiter from config
        public_limit = config["exchange"]["public_rate_limit"]["max_calls_per_sec"]
        private_limit = config["exchange"]["private_rate_limit"]["max_calls_per_sec"]
        
        logger.info(f"Creating RateLimiter (public: {public_limit}/sec, private: {private_limit}/sec)")
        rate_limiter = RateLimiter(
            max_calls_per_sec_public=public_limit,
            max_calls_per_sec_private=private_limit
        )
        
        # Create Upbit client
        base_url = config["exchange"]["base_url"]
        logger.info(f"Creating UpbitClient with base_url: {base_url}")
        
        client = UpbitClient(base_url=base_url, rate_limiter=rate_limiter)
        
        # Test symbol
        symbol = "KRW-BTC"
        logger.info(f"Testing with symbol: {symbol}")
        
        # Test 1: Get 4-hour candles
        logger.info("\n--- Testing get_candles_4h ---")
        candles = client.get_candles_4h(symbol, count=10)
        
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        if candles:
            first_candle = candles[0]
            last_candle = candles[-1]
            logger.info(f"First candle (oldest): {first_candle.timestamp} - Close: {first_candle.close:,.0f}")
            logger.info(f"Last candle (newest): {last_candle.timestamp} - Close: {last_candle.close:,.0f}")
            logger.info(f"Last close price: {last_candle.close:,.0f} KRW")
        
        # Test 2: Get current ticker price
        logger.info("\n--- Testing get_ticker_price ---")
        current_price = client.get_ticker_price(symbol)
        logger.info(f"Current {symbol} price: {current_price:,.0f} KRW")
        
        # Compare candle vs ticker
        if candles:
            price_diff = current_price - last_candle.close
            price_diff_pct = (price_diff / last_candle.close) * 100
            logger.info(f"Price difference from last candle: {price_diff:+,.0f} KRW ({price_diff_pct:+.2f}%)")
        
        # Test rate limiting (make multiple quick calls)
        logger.info("\n--- Testing rate limiting ---")
        logger.info("Making 3 quick API calls to test rate limiting...")
        
        import time
        start_time = time.time()
        
        for i in range(3):
            price = client.get_ticker_price(symbol)
            elapsed = time.time() - start_time
            logger.info(f"Call {i+1}: {price:,.0f} KRW (elapsed: {elapsed:.2f}s)")
        
        total_elapsed = time.time() - start_time
        logger.info(f"Total time for 3 calls: {total_elapsed:.2f}s")
        
        # Clean up
        client.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("Exchange layer test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Exchange layer test failed: {e}")
        raise


if __name__ == "__main__":
    test_exchange_layer()