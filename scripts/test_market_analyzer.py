#!/usr/bin/env python3
"""
Test script for Step 3 - Market Analysis.

This script tests the market analysis implementation:
- Technical indicators (ATR, ADX, Bollinger Bands)
- MarketAnalyzer classification (TREND/RANGE/NEUTRAL)
- Real BTC data analysis

Usage:
    python scripts/test_market_analyzer.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from core.types import MarketMode
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient
from market.market_analyzer import MarketAnalyzer


def main() -> None:
    """Test the market analyzer implementation."""
    
    # Setup
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        config["persistence"]["log_level"]
    )
    
    logger.info("=" * 60)
    logger.info("Testing Market Analyzer (Step 3)")
    logger.info("=" * 60)
    
    try:
        # Create exchange client
        rl = RateLimiter(
            max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
            max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
        )
        
        client = UpbitClient(
            base_url=config["exchange"]["base_url"],
            rate_limiter=rl,
        )
        
        # Fetch BTC candles (need more for indicators)
        logger.info("Fetching BTC 4h candles for analysis...")
        candles = client.get_candles_4h("KRW-BTC", count=200)
        logger.info(f"Fetched {len(candles)} candles for analysis")
        
        if len(candles) < 50:
            logger.error("Not enough candles for analysis")
            return
        
        # Create market analyzer
        analyzer = MarketAnalyzer(config)
        
        # Analyze current market
        now = datetime.now(timezone.utc)
        current_mode = analyzer.update_mode(candles, now)
        
        # Get the metrics used for classification
        metrics = analyzer._compute_metrics(candles)
        
        # Log detailed analysis
        logger.info("\n--- Market Analysis Results ---")
        logger.info(f"Analysis time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"Latest candle: {candles[-1].timestamp} - Close: {candles[-1].close:,.0f} KRW")
        logger.info("")
        
        logger.info("Technical Indicators:")
        logger.info(f"  ATR: {metrics['atr']:,.0f}")
        logger.info(f"  ATR Ratio: {metrics['atr_ratio']:.2f}%")
        logger.info(f"  ADX: {metrics['adx']:.1f}")
        logger.info(f"  Bollinger Band Width: {metrics['bandwidth']:.2f}%")
        logger.info("")
        
        logger.info("Classification Thresholds:")
        logger.info(f"  TREND Enter: ADX >= {analyzer.adx_trend_enter}, ATR Ratio >= {analyzer.atr_trend_min}%")
        logger.info(f"  TREND Exit: ADX >= {analyzer.adx_trend_exit}, ATR Ratio >= {analyzer.atr_trend_min * 0.8:.1f}%")
        logger.info(f"  RANGE Enter: ADX <= {analyzer.adx_range_enter}, BW <= {analyzer.bw_range_enter}%, ATR <= {analyzer.atr_range_max}%")
        logger.info(f"  RANGE Exit: ADX <= {analyzer.adx_range_exit}, BW <= {analyzer.bw_range_exit}%")
        logger.info("")
        
        # Classification logic explanation
        logger.info("Classification Analysis:")
        
        # Check TREND conditions
        trend_adx = metrics['adx'] >= analyzer.adx_trend_enter
        trend_atr = metrics['atr_ratio'] >= analyzer.atr_trend_min
        logger.info(f"  TREND conditions: ADX {metrics['adx']:.1f} >= {analyzer.adx_trend_enter} [{trend_adx}], ATR {metrics['atr_ratio']:.2f}% >= {analyzer.atr_trend_min}% [{trend_atr}]")
        
        # Check RANGE conditions  
        range_adx = metrics['adx'] <= analyzer.adx_range_enter
        range_bw = metrics['bandwidth'] <= analyzer.bw_range_enter
        range_atr = metrics['atr_ratio'] <= analyzer.atr_range_max
        logger.info(f"  RANGE conditions: ADX {metrics['adx']:.1f} <= {analyzer.adx_range_enter} [{range_adx}], BW {metrics['bandwidth']:.2f}% <= {analyzer.bw_range_enter}% [{range_bw}], ATR {metrics['atr_ratio']:.2f}% <= {analyzer.atr_range_max}% [{range_atr}]")
        logger.info("")
        
        # Final result
        logger.info(f"🎯 Market Mode: {current_mode}")
        
        if analyzer.last_mode_change:
            logger.info(f"   Last change: {analyzer.last_mode_change.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        else:
            logger.info("   Last change: Never (initial state)")
            
        logger.info("")
        
        # Test mode persistence by calling again
        logger.info("--- Testing Mode Persistence ---")
        logger.info("Calling update_mode again immediately...")
        
        second_mode = analyzer.update_mode(candles, now)
        logger.info(f"Second call result: {second_mode}")
        logger.info(f"Mode changed: {current_mode != second_mode}")
        
        # Summary log in requested format
        logger.info("")
        logger.info("--- Summary ---")
        logger.info(f"[INFO] ATR={metrics['atr']:,.0f}, ATR_ratio={metrics['atr_ratio']:.2f}%, ADX={metrics['adx']:.1f}, BandWidth={metrics['bandwidth']:.2f}%, mode={current_mode}")
        
        # Clean up
        client.close()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Market analyzer test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Market analyzer test failed: {e}")
        raise


if __name__ == "__main__":
    main()