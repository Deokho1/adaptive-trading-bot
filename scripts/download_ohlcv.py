#!/usr/bin/env python3
"""
Historical OHLCV data downloader for backtesting.

This script downloads 4-hour candle data from Upbit API for configured symbols
and saves them as CSV files for backtesting purposes.
"""

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient
from exchange.models import Candle


def main():
    """Main function to download historical OHLCV data."""
    
    # Parameters
    LOOKBACK_DAYS = 365
    CANDLES_PER_REQUEST = 200
    REQUEST_DELAY = 0.3  # Additional delay between requests for safety
    
    # Setup logging
    logger = setup_logger("logs", level="INFO")
    
    try:
        # Load configuration
        config = load_config("config/config.yaml")
        logger.info("Configuration loaded successfully")
        
        # Get symbols from config
        symbols = config.get("symbols", ["KRW-BTC", "KRW-ETH"])
        logger.info(f"Will download data for symbols: {symbols}")
        
        # Initialize components
        rate_limiter = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"]
        )
        
        upbit_client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=rate_limiter
        )
        
        # Ensure output directory exists
        output_dir = Path("data/ohlcv")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Download data for each symbol
        for symbol in symbols:
            logger.info(f"Starting download for {symbol}...")
            
            try:
                candles = download_symbol_history(
                    upbit_client, 
                    symbol, 
                    lookback_days=LOOKBACK_DAYS,
                    candles_per_request=CANDLES_PER_REQUEST,
                    request_delay=REQUEST_DELAY,
                    logger=logger
                )
                
                if not candles:
                    logger.warning(f"No candles downloaded for {symbol}, skipping CSV creation")
                    continue
                
                # Save to CSV
                csv_filename = f"{symbol}_240m.csv"
                csv_path = output_dir / csv_filename
                
                save_candles_to_csv(candles, csv_path, logger)
                
                logger.info(f"✅ Successfully downloaded and saved {len(candles)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download data for {symbol}: {e}")
                continue
        
        logger.info("🎉 Historical data download completed!")
        
    except Exception as e:
        logger.error(f"Failed to run downloader: {e}")
        sys.exit(1)
    
    finally:
        # Clean up
        try:
            upbit_client.close()
        except:
            pass


def download_symbol_history(
    client: UpbitClient,
    symbol: str,
    lookback_days: int,
    candles_per_request: int,
    request_delay: float,
    logger: logging.Logger,
) -> List[Candle]:
    """
    Download historical candles for a symbol by paginating backwards.
    
    Args:
        client: UpbitClient instance
        symbol: Trading symbol (e.g., "KRW-BTC")
        lookback_days: Number of days of history to download
        candles_per_request: Number of candles per API request
        request_delay: Additional delay between requests
        logger: Logger instance
        
    Returns:
        List of Candle objects sorted from oldest to newest
    """
    all_candles = []
    to_timestamp = None  # Start with most recent
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    request_count = 0
    
    logger.info(f"Downloading {lookback_days} days of history for {symbol}")
    logger.info(f"Target cutoff time: {cutoff_time}")
    
    while True:
        try:
            # Fetch a page of candles
            request_count += 1
            logger.debug(f"Request #{request_count} for {symbol}, to={to_timestamp}")
            
            candles = client.get_candles_4h_page(
                symbol=symbol,
                count=candles_per_request,
                to=to_timestamp
            )
            
            if not candles:
                logger.info(f"No more candles returned for {symbol}, stopping pagination")
                break
            
            # Log progress
            oldest_in_page = candles[0].timestamp
            newest_in_page = candles[-1].timestamp
            logger.info(
                f"Fetched {len(candles)} candles for {symbol} "
                f"(total: {len(all_candles) + len(candles)}) "
                f"oldest: {oldest_in_page}, newest: {newest_in_page}"
            )
            
            # Add to collection
            all_candles.extend(candles)
            
            # Check if we've gone back far enough
            # Ensure both timestamps are timezone-aware for comparison
            oldest_utc = oldest_in_page
            if oldest_utc.tzinfo is None:
                oldest_utc = oldest_utc.replace(tzinfo=timezone.utc)
            
            if oldest_utc <= cutoff_time:
                logger.info(f"Reached cutoff time for {symbol}, stopping pagination")
                break
            
            # Check if we have enough candles (safety limit)
            if len(all_candles) >= lookback_days * 6 * 1.2:  # 6 candles/day * 1.2 safety factor
                logger.info(f"Collected enough candles for {symbol} ({len(all_candles)}), stopping")
                break
            
            # Prepare for next page - use oldest candle's timestamp minus 1 second
            # to ensure no overlap
            to_timestamp = oldest_in_page - timedelta(seconds=1)
            
            # Add delay between requests
            time.sleep(request_delay)
            
        except Exception as e:
            logger.error(f"Error fetching page for {symbol}: {e}")
            break
    
    if all_candles:
        # Sort from oldest to newest
        all_candles.sort(key=lambda c: c.timestamp)
        
        # Trim to exactly the lookback period if needed
        cutoff_candles = []
        for c in all_candles:
            candle_utc = c.timestamp
            if candle_utc.tzinfo is None:
                candle_utc = candle_utc.replace(tzinfo=timezone.utc)
            
            if candle_utc >= cutoff_time:
                cutoff_candles.append(c)
        
        if cutoff_candles:
            logger.info(
                f"Trimmed {symbol} data to {len(cutoff_candles)} candles within {lookback_days} days "
                f"(from {cutoff_candles[0].timestamp} to {cutoff_candles[-1].timestamp})"
            )
            return cutoff_candles
        else:
            logger.warning(f"No candles within {lookback_days} days for {symbol}")
            return all_candles
    
    return all_candles


def save_candles_to_csv(candles: List[Candle], csv_path: Path, logger: logging.Logger) -> None:
    """
    Save candles to CSV file.
    
    Args:
        candles: List of Candle objects
        csv_path: Path to CSV file
        logger: Logger instance
    """
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("timestamp,open,high,low,close,volume\n")
            
            # Write data
            for candle in candles:
                f.write(
                    f"{candle.timestamp.isoformat()},"
                    f"{candle.open},"
                    f"{candle.high},"
                    f"{candle.low},"
                    f"{candle.close},"
                    f"{candle.volume}\n"
                )
        
        logger.info(f"Saved {len(candles)} candles to {csv_path}")
        
        # Log some stats
        if candles:
            oldest = candles[0].timestamp
            newest = candles[-1].timestamp
            duration = newest - oldest
            logger.info(f"Data range: {oldest} to {newest} ({duration.days} days)")
        
    except Exception as e:
        logger.error(f"Failed to save CSV to {csv_path}: {e}")
        raise


if __name__ == "__main__":
    main()