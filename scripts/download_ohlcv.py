"""#!/usr/bin/env python3

OHLCV Data Download Utility"""

Historical OHLCV data downloader for backtesting.

Downloads historical OHLCV candle data from exchanges via ccxt and saves to CSV.

"""This script downloads 4-hour candle data from Upbit API for configured symbols

and saves them as CSV files for backtesting purposes.

import ccxt"""

import pandas as pd

import argparseimport logging

import osimport sys

import timeimport time

from datetime import datetime, timedeltafrom datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import List

def main():

    """Main function to download OHLCV data"""# Add project root to path

    project_root = Path(__file__).parent.parent

    # Parse command line argumentssys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description="Download OHLCV data from exchanges")

    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name (default: binance)")from core.config_loader import load_config

    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g. BTC/USDT)")from core.logger import setup_logger

    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (default: 5m)")from exchange.rate_limiter import RateLimiter

    parser.add_argument("--days", type=int, default=30, help="Number of days to download (default: 30)")from exchange.upbit_client import UpbitClient

    parser.add_argument("--limit", type=int, default=1000, help="Candles per request (default: 1000)")from exchange.models import Candle

    parser.add_argument("--outfile", type=str, help="Output CSV file path (optional)")

    parser.add_argument("--quote-currency", type=str, default="USDT", help="Quote currency (default: USDT)")

    def main():

    args = parser.parse_args()    """Main function to download historical OHLCV data."""

        

    print(f"[INFO] Starting OHLCV download...")    # Parameters

    print(f"[INFO] Exchange: {args.exchange}, Symbol: {args.symbol}, Timeframe: {args.timeframe}, Days: {args.days}")    LOOKBACK_DAYS = 365

        CANDLES_PER_REQUEST = 200

    try:    REQUEST_DELAY = 0.3  # Additional delay between requests for safety

        # Initialize exchange    

        print(f"[INFO] Initializing {args.exchange} exchange...")    # Setup logging

        exchange_cls = getattr(ccxt, args.exchange)    logger = setup_logger("logs", level="INFO")

        exchange = exchange_cls({"enableRateLimit": True})    

            try:

        # Check if symbol exists        # Load configuration

        markets = exchange.load_markets()        config = load_config("config/config.yaml")

        if args.symbol not in markets:        logger.info("Configuration loaded successfully")

            print(f"[ERROR] Symbol {args.symbol} not found on {args.exchange}")        

            print(f"[INFO] Available symbols: {list(markets.keys())[:10]}...")        # Get symbols from config

            return        symbols = config.get("symbols", ["KRW-BTC", "KRW-ETH"])

                logger.info(f"Will download data for symbols: {symbols}")

        # Compute since timestamp        

        since = exchange.milliseconds() - args.days * 24 * 60 * 60 * 1000        # Initialize components

        current_time = exchange.milliseconds()        rate_limiter = RateLimiter(

                    max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],

        print(f"[INFO] Fetching data from {datetime.utcfromtimestamp(since/1000)} to {datetime.utcfromtimestamp(current_time/1000)}")            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"]

                )

        # Fetch all candles        

        all_candles = []        upbit_client = UpbitClient(

        total_requests = 0            base_url=config["exchange"]["base_url"],

                    rate_limiter=rate_limiter

        while True:        )

            try:        

                print(f"[INFO] Fetching batch {total_requests + 1} (since: {datetime.utcfromtimestamp(since/1000)})")        # Ensure output directory exists

                        output_dir = Path("data/ohlcv")

                # Fetch OHLCV data        output_dir.mkdir(parents=True, exist_ok=True)

                data = exchange.fetch_ohlcv(args.symbol, args.timeframe, since=since, limit=args.limit)        logger.info(f"Output directory: {output_dir}")

                        

                # Break if no data        # Download data for each symbol

                if not data or len(data) == 0:        for symbol in symbols:

                    print(f"[INFO] No more data available")            logger.info(f"Starting download for {symbol}...")

                    break            

                            try:

                # Add to collection                candles = download_symbol_history(

                all_candles.extend(data)                    upbit_client, 

                total_requests += 1                    symbol, 

                                    lookback_days=LOOKBACK_DAYS,

                # Update since to next timestamp                    candles_per_request=CANDLES_PER_REQUEST,

                last_timestamp = data[-1][0]                    request_delay=REQUEST_DELAY,

                since = last_timestamp + 1                    logger=logger

                                )

                print(f"[INFO] Fetched {len(data)} candles, total: {len(all_candles)}")                

                                if not candles:

                # Break if we've reached current time                    logger.warning(f"No candles downloaded for {symbol}, skipping CSV creation")

                if last_timestamp >= current_time:                    continue

                    print(f"[INFO] Reached current time")                

                    break                # Save to CSV

                                csv_filename = f"{symbol}_240m.csv"

                # Rate limiting                csv_path = output_dir / csv_filename

                if exchange.rateLimit:                

                    sleep_time = exchange.rateLimit / 1000                save_candles_to_csv(candles, csv_path, logger)

                    print(f"[INFO] Rate limiting: sleeping {sleep_time}s")                

                    time.sleep(sleep_time)                logger.info(f"✅ Successfully downloaded and saved {len(candles)} candles for {symbol}")

                else:                

                    time.sleep(0.1)  # Default small delay            except Exception as e:

                                logger.error(f"❌ Failed to download data for {symbol}: {e}")

            except ccxt.NetworkError as e:                continue

                print(f"[WARNING] Network error: {e}")        

                print(f"[INFO] Retrying in 5 seconds...")        logger.info("🎉 Historical data download completed!")

                time.sleep(5)        

                continue    except Exception as e:

                        logger.error(f"Failed to run downloader: {e}")

            except ccxt.ExchangeError as e:        sys.exit(1)

                print(f"[ERROR] Exchange error: {e}")    

                if "rate limit" in str(e).lower():    finally:

                    print(f"[INFO] Rate limited, waiting 60 seconds...")        # Clean up

                    time.sleep(60)        try:

                    continue            upbit_client.close()

                else:        except:

                    print(f"[ERROR] Unrecoverable exchange error")            pass

                    return

        

        # Check if we got any datadef download_symbol_history(

        if not all_candles:    client: UpbitClient,

            print(f"[ERROR] No candles were fetched")    symbol: str,

            return    lookback_days: int,

            candles_per_request: int,

        print(f"[INFO] Total requests made: {total_requests}")    request_delay: float,

        print(f"[INFO] Total candles fetched: {len(all_candles)}")    logger: logging.Logger,

        ) -> List[Candle]:

        # Convert to DataFrame    """

        print(f"[INFO] Converting to DataFrame...")    Download historical candles for a symbol by paginating backwards.

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])    

            Args:

        # Convert timestamp to readable format        client: UpbitClient instance

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)        symbol: Trading symbol (e.g., "KRW-BTC")

        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")        lookback_days: Number of days of history to download

                candles_per_request: Number of candles per API request

        # Remove duplicates and sort        request_delay: Additional delay between requests

        initial_count = len(df)        logger: Logger instance

        df.drop_duplicates(subset=["timestamp"], inplace=True)        

        df.sort_values("timestamp", inplace=True)    Returns:

        df.reset_index(drop=True, inplace=True)        List of Candle objects sorted from oldest to newest

            """

        if len(df) < initial_count:    all_candles = []

            print(f"[INFO] Removed {initial_count - len(df)} duplicate candles")    to_timestamp = None  # Start with most recent

            cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Determine output file    request_count = 0

        if args.outfile:    

            outfile = args.outfile    logger.info(f"Downloading {lookback_days} days of history for {symbol}")

        else:    logger.info(f"Target cutoff time: {cutoff_time}")

            symbol_safe = args.symbol.replace("/", "").replace("-", "")    

            today = datetime.utcnow().strftime("%Y%m%d")    while True:

            os.makedirs("data", exist_ok=True)        try:

            outfile = f"data/{args.exchange}_{symbol_safe}_{args.timeframe}_{today}.csv"            # Fetch a page of candles

                    request_count += 1

        # Save to CSV            logger.debug(f"Request #{request_count} for {symbol}, to={to_timestamp}")

        print(f"[INFO] Saving to {outfile}...")            

        df.to_csv(outfile, index=False)            candles = client.get_candles_4h_page(

                        symbol=symbol,

        # Final summary                count=candles_per_request,

        print(f"\n{'='*60}")                to=to_timestamp

        print(f"DOWNLOAD COMPLETED SUCCESSFULLY")            )

        print(f"{'='*60}")            

        print(f"Exchange: {args.exchange}")            if not candles:

        print(f"Symbol: {args.symbol}")                logger.info(f"No more candles returned for {symbol}, stopping pagination")

        print(f"Timeframe: {args.timeframe}")                break

        print(f"Days requested: {args.days}")            

        print(f"Candles fetched: {len(df):,}")            # Log progress

        print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")            oldest_in_page = candles[0].timestamp

        print(f"File saved: {outfile}")            newest_in_page = candles[-1].timestamp

        print(f"File size: {os.path.getsize(outfile) / 1024 / 1024:.2f} MB")            logger.info(

        print(f"{'='*60}")                f"Fetched {len(candles)} candles for {symbol} "

                        f"(total: {len(all_candles) + len(candles)}) "

        # Show sample data                f"oldest: {oldest_in_page}, newest: {newest_in_page}"

        print(f"\nSample data (first 5 rows):")            )

        print(df.head().to_string(index=False))            

                    # Add to collection

    except ccxt.BaseError as e:            all_candles.extend(candles)

        print(f"[ERROR] CCXT Error: {e}")            

        return            # Check if we've gone back far enough

                # Ensure both timestamps are timezone-aware for comparison

    except Exception as e:            oldest_utc = oldest_in_page

        print(f"[ERROR] Unexpected error: {e}")            if oldest_utc.tzinfo is None:

        import traceback                oldest_utc = oldest_utc.replace(tzinfo=timezone.utc)

        traceback.print_exc()            

        return            if oldest_utc <= cutoff_time:

                logger.info(f"Reached cutoff time for {symbol}, stopping pagination")

                break

def list_exchanges():            

    """List available exchanges"""            # Check if we have enough candles (safety limit)

    print("Available exchanges:")            if len(all_candles) >= lookback_days * 6 * 1.2:  # 6 candles/day * 1.2 safety factor

    exchanges = [exchange for exchange in dir(ccxt) if not exchange.startswith('_')]                logger.info(f"Collected enough candles for {symbol} ({len(all_candles)}), stopping")

    for i, exchange in enumerate(exchanges):                break

        if i % 5 == 0:            

            print()            # Prepare for next page - use oldest candle's timestamp minus 1 second

        print(f"{exchange:15}", end="")            # to ensure no overlap

    print()            to_timestamp = oldest_in_page - timedelta(seconds=1)

            

            # Add delay between requests

def test_exchange_connection(exchange_name: str):            time.sleep(request_delay)

    """Test connection to exchange"""            

    try:        except Exception as e:

        exchange_cls = getattr(ccxt, exchange_name)            logger.error(f"Error fetching page for {symbol}: {e}")

        exchange = exchange_cls({"enableRateLimit": True})            break

        markets = exchange.load_markets()    

        print(f"[INFO] Successfully connected to {exchange_name}")    if all_candles:

        print(f"[INFO] Available markets: {len(markets)}")        # Sort from oldest to newest

        print(f"[INFO] Sample symbols: {list(markets.keys())[:10]}")        all_candles.sort(key=lambda c: c.timestamp)

        return True        

    except Exception as e:        # Trim to exactly the lookback period if needed

        print(f"[ERROR] Failed to connect to {exchange_name}: {e}")        cutoff_candles = []

        return False        for c in all_candles:

            candle_utc = c.timestamp

            if candle_utc.tzinfo is None:

if __name__ == "__main__":                candle_utc = candle_utc.replace(tzinfo=timezone.utc)

    # Check for special commands            

    import sys            if candle_utc >= cutoff_time:

                    cutoff_candles.append(c)

    if len(sys.argv) > 1:        

        if sys.argv[1] == "--list-exchanges":        if cutoff_candles:

            list_exchanges()            logger.info(

            sys.exit(0)                f"Trimmed {symbol} data to {len(cutoff_candles)} candles within {lookback_days} days "

        elif sys.argv[1] == "--test-exchange" and len(sys.argv) > 2:                f"(from {cutoff_candles[0].timestamp} to {cutoff_candles[-1].timestamp})"

            test_exchange_connection(sys.argv[2])            )

            sys.exit(0)            return cutoff_candles

            else:

    # Run main download function            logger.warning(f"No candles within {lookback_days} days for {symbol}")

    main()            return all_candles
    
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