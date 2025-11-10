import pandas as pd
import numpy as np
from trend_follow import BTCTrendEngine
from mean_reversion import ETHMeanReversionEngine
from regime_detection import RegimeDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Load data
    btc_data = pd.read_csv('data/ohlcv/KRW-BTC_240m.csv', index_col=0, parse_dates=True)
    eth_data = pd.read_csv('data/ohlcv/KRW-ETH_240m.csv', index_col=0, parse_dates=True)
    
    # Align data
    common_index = btc_data.index.intersection(eth_data.index)
    btc_aligned = btc_data.loc[common_index]
    eth_aligned = eth_data.loc[common_index]
    
    # Initialize engines
    btc_engine = BTCTrendEngine()
    eth_engine = ETHMeanReversionEngine()
    regime_detector = RegimeDetector()
    
    # Track exposures
    btc_exposures = []
    eth_exposures = []
    
    print("Checking exposure calculations...")
    
    for i in range(60, min(200, len(btc_aligned))):  # Check first 140 rows after warmup
        row_btc = btc_aligned.iloc[i]
        row_eth = eth_aligned.iloc[i]
        
        # Get regime
        btc_regime = regime_detector.detect_regime(row_btc['close'], row_btc['volume'], i)
        eth_regime = regime_detector.detect_regime(row_eth['close'], row_eth['volume'], i)
        
        # Get signals
        btc_signal, btc_exposure, btc_detail = btc_engine.generate_signal(row_btc, btc_regime, 0.0)
        eth_signal, eth_exposure, eth_detail = eth_engine.generate_signal(row_eth, eth_regime, 0.0)
        
        btc_exposures.append(btc_exposure)
        eth_exposures.append(eth_exposure)
        
        # Check for violations
        if btc_exposure > 100.0:
            print(f"Row {i}: BTC exposure violation: {btc_exposure:.1f}%")
            print(f"  Regime: {btc_regime['mode']}")
            print(f"  Detail: {btc_detail}")
            print(f"  Row data: Close={row_btc['close']:.0f}, Volume={row_btc['volume']:.0f}")
            print()
            
        if eth_exposure > 45.0:
            print(f"Row {i}: ETH exposure violation: {eth_exposure:.1f}%")
            print(f"  Regime: {eth_regime['mode']}")
            print(f"  Detail: {eth_detail}")
            print()
    
    # Summary statistics
    btc_exposures = np.array(btc_exposures)
    eth_exposures = np.array(eth_exposures)
    
    print(f"\nBTC Exposure Stats:")
    print(f"  Min: {btc_exposures.min():.1f}%")
    print(f"  Max: {btc_exposures.max():.1f}%")
    print(f"  Mean: {btc_exposures.mean():.1f}%")
    print(f"  >100%: {(btc_exposures > 100).sum()} occurrences")
    
    print(f"\nETH Exposure Stats:")
    print(f"  Min: {eth_exposures.min():.1f}%")
    print(f"  Max: {eth_exposures.max():.1f}%")
    print(f"  Mean: {eth_exposures.mean():.1f}%")
    print(f"  >45%: {(eth_exposures > 45).sum()} occurrences")

if __name__ == "__main__":
    main()