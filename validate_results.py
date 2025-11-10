"""
Quick data validation and simple backtest verification
"""

import pandas as pd
import numpy as np

def validate_data():
    """Validate the data files"""
    print("=== DATA VALIDATION ===")
    
    # Load data
    btc_df = pd.read_csv('data/ohlcv/KRW-BTC_240m.csv')
    eth_df = pd.read_csv('data/ohlcv/KRW-ETH_240m.csv')
    
    print(f"BTC data: {len(btc_df)} rows")
    print(f"ETH data: {len(eth_df)} rows")
    
    print(f"\nBTC period: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")
    print(f"ETH period: {eth_df['timestamp'].min()} to {eth_df['timestamp'].max()}")
    
    # Convert timestamps
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'])
    
    # Align data
    aligned_data = pd.merge(btc_df, eth_df, on='timestamp', suffixes=('_btc', '_eth'))
    print(f"\nAligned data: {len(aligned_data)} rows")
    
    # Check price ranges
    print(f"\nBTC price range: {btc_df['close'].min():,.0f} to {btc_df['close'].max():,.0f}")
    print(f"ETH price range: {eth_df['close'].min():,.0f} to {eth_df['close'].max():,.0f}")
    
    # Calculate simple returns
    btc_returns = btc_df['close'].pct_change().dropna()
    eth_returns = eth_df['close'].pct_change().dropna()
    
    print(f"\nBTC cumulative return: {(btc_df['close'].iloc[-1] / btc_df['close'].iloc[0] - 1) * 100:.1f}%")
    print(f"ETH cumulative return: {(eth_df['close'].iloc[-1] / eth_df['close'].iloc[0] - 1) * 100:.1f}%")
    
    return aligned_data

def simple_backtest(df):
    """Simple buy-and-hold backtest"""
    print("\n=== SIMPLE BUY-AND-HOLD TEST ===")
    
    initial_capital = 10_000_000
    btc_allocation = 0.7
    eth_allocation = 0.3
    
    # Initial prices
    btc_start_price = df['close_btc'].iloc[0]
    eth_start_price = df['close_eth'].iloc[0]
    
    # Final prices
    btc_end_price = df['close_btc'].iloc[-1]
    eth_end_price = df['close_eth'].iloc[-1]
    
    # Calculate returns
    btc_return = (btc_end_price / btc_start_price - 1) * 100
    eth_return = (eth_end_price / eth_start_price - 1) * 100
    
    # Portfolio return
    portfolio_return = btc_return * btc_allocation + eth_return * eth_allocation
    
    print(f"BTC buy-and-hold: {btc_return:+.1f}%")
    print(f"ETH buy-and-hold: {eth_return:+.1f}%")
    print(f"Portfolio (70:30): {portfolio_return:+.1f}%")
    
    # Final portfolio value
    final_value = initial_capital * (1 + portfolio_return / 100)
    print(f"Final portfolio value: ₩{final_value:,.0f}")

def check_strategy_logic():
    """Check if strategy logic makes sense"""
    print("\n=== STRATEGY LOGIC CHECK ===")
    
    # Load results if available
    try:
        from pathlib import Path
        results_dir = Path('results/dual_core_strategy')
        if results_dir.exists():
            result_files = list(results_dir.glob('equity_curves_*.csv'))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                results_df = pd.read_csv(latest_file)
                
                print(f"Portfolio equity range: ₩{results_df['portfolio_equity'].min():,.0f} to ₩{results_df['portfolio_equity'].max():,.0f}")
                print(f"BTC exposure range: {results_df['btc_exposure'].min():.1f}% to {results_df['btc_exposure'].max():.1f}%")
                print(f"ETH exposure range: {results_df['eth_exposure'].min():.1f}% to {results_df['eth_exposure'].max():.1f}%")
                
                # Check for impossible exposures
                impossible_btc = (results_df['btc_exposure'] > 100).sum()
                impossible_eth = (results_df['eth_exposure'] > 50).sum()  # ETH max should be 45%
                
                print(f"Impossible BTC exposures (>100%): {impossible_btc}")
                print(f"Impossible ETH exposures (>50%): {impossible_eth}")
                
            else:
                print("No equity curve files found")
    except Exception as e:
        print(f"Could not load results: {e}")

if __name__ == "__main__":
    aligned_data = validate_data()
    simple_backtest(aligned_data)
    check_strategy_logic()