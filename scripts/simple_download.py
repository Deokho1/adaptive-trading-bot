"""
Simple OHLCV Data Downloader (no ccxt dependency)

Downloads historical OHLCV candle data directly from Binance API and saves to CSV.
"""

import requests
import pandas as pd
import argparse
import os
import time
from datetime import datetime, timedelta


def get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch kline data from Binance API
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe ('1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Number of candles (max 1000)
    
    Returns:
        List of candle data
    """
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return []


def download_binance_data(symbol, interval, days):
    """
    Download historical data from Binance
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Timeframe
        days: Number of days to download
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"[INFO] Starting download for {symbol} {interval} ({days} days)")
    
    # Calculate start time
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_start = start_time
    batch_count = 0
    
    while current_start < end_time:
        batch_count += 1
        print(f"[INFO] Fetching batch {batch_count}...")
        
        # Get batch of data
        candles = get_binance_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time,
            limit=1000
        )
        
        if not candles:
            print(f"[WARNING] No data received for batch {batch_count}")
            break
        
        # Add to collection
        all_candles.extend(candles)
        
        # Update start time for next batch
        last_timestamp = candles[-1][0]
        current_start = last_timestamp + 1
        
        print(f"[INFO] Batch {batch_count}: {len(candles)} candles")
        
        # Rate limiting
        time.sleep(0.1)
    
    if not all_candles:
        print("[ERROR] No data downloaded")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_volume', 
        'taker_buy_quote_volume', 'ignore'
    ])
    
    # Keep only OHLCV columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # Format timestamp for consistency
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Remove duplicates and sort
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def main():
    """Main function to download OHLCV data"""
    
    parser = argparse.ArgumentParser(description="Download OHLCV data from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="5m", help="Timeframe (default: 5m)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download (default: 30)")
    parser.add_argument("--outfile", type=str, help="Output CSV file path (optional)")
    
    args = parser.parse_args()
    
    print(f"[INFO] Binance OHLCV Downloader")
    print(f"[INFO] Symbol: {args.symbol}, Interval: {args.interval}, Days: {args.days}")
    
    # Validate interval
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if args.interval not in valid_intervals:
        print(f"[ERROR] Invalid interval. Valid options: {valid_intervals}")
        return
    
    # Download data
    try:
        df = download_binance_data(args.symbol, args.interval, args.days)
        
        if df is None or df.empty:
            print("[ERROR] Failed to download data")
            return
        
        # Determine output file
        if args.outfile:
            outfile = args.outfile
        else:
            today = datetime.now().strftime("%Y%m%d")
            os.makedirs("data", exist_ok=True)
            outfile = f"data/binance_{args.symbol}_{args.interval}_{today}.csv"
        
        # Save to CSV
        df.to_csv(outfile, index=False)
        
        # Show results
        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETED")
        print(f"{'='*60}")
        print(f"Symbol: {args.symbol}")
        print(f"Interval: {args.interval}")
        print(f"Days: {args.days}")
        print(f"Candles: {len(df):,}")
        print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"File: {outfile}")
        print(f"Size: {os.path.getsize(outfile) / 1024:.1f} KB")
        print(f"{'='*60}")
        
        # Show sample
        print(f"\nSample data:")
        print(df.head().to_string(index=False))
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()