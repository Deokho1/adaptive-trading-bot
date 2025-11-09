#!/usr/bin/env python3
"""
Backtest runner script.

This script provides a command-line interface for running backtests
with different strategies and configurations.
"""

import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from backtest.runner import BacktestRunner


def main():
    """Main backtest runner function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run cryptocurrency trading backtest")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ohlcv",
        help="Historical data directory (default: data/ohlcv)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["KRW-BTC", "KRW-ETH"],
        help="Trading symbols (default: KRW-BTC KRW-ETH)"
    )
    
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="Initial cash amount in KRW (default: 1,000,000)"
    )
    
    parser.add_argument(
        "--base-symbol",
        type=str,
        default="KRW-BTC",
        help="Base symbol for market mode analysis (default: KRW-BTC)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("logs", level=args.log_level)
    logger = logging.getLogger("bot")
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Validate data directory
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Please create the data directory and add CSV files:")
            logger.info(f"  mkdir -p {data_dir}")
            logger.info(f"  # Add CSV files like {data_dir}/KRW-BTC.csv")
            sys.exit(1)
        
        # Check for data files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {data_dir}")
            logger.info("Expected CSV format: timestamp,open,high,low,close,volume")
            logger.info(f"Example files: {data_dir}/KRW-BTC.csv, {data_dir}/KRW-ETH.csv")
            sys.exit(1)
        
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        
        # Validate symbols against available data
        available_symbols = set()
        for csv_file in csv_files:
            # Extract symbol from filename (e.g., "KRW-BTC.csv" -> "KRW-BTC")
            symbol = csv_file.stem
            available_symbols.add(symbol)
        
        requested_symbols = set(args.symbols)
        missing_symbols = requested_symbols - available_symbols
        
        if missing_symbols:
            logger.warning(f"Missing data for symbols: {missing_symbols}")
            logger.info(f"Available symbols: {sorted(available_symbols)}")
            
            # Use only available symbols
            valid_symbols = list(requested_symbols & available_symbols)
            if not valid_symbols:
                logger.error("No valid symbols with available data")
                sys.exit(1)
            
            logger.info(f"Using symbols: {valid_symbols}")
        else:
            valid_symbols = args.symbols
        
        # Check base symbol availability
        if args.base_symbol not in available_symbols:
            logger.error(f"Base symbol {args.base_symbol} not found in available data")
            sys.exit(1)
        
        # Print backtest configuration
        logger.info("=" * 60)
        logger.info("BACKTEST CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Base Symbol: {args.base_symbol}")
        logger.info(f"Trading Symbols: {valid_symbols}")
        logger.info(f"Initial Cash: {args.initial_cash:,.0f} KRW")
        logger.info(f"Data Directory: {data_dir}")
        logger.info("=" * 60)
        
        # Initialize backtest runner
        runner = BacktestRunner(
            data_dir=data_dir,
            config=config,
            base_symbol=args.base_symbol,
            symbols=valid_symbols,
            initial_cash=args.initial_cash
        )
        
        # Run backtest
        logger.info("Starting backtest...")
        start_time = datetime.now()
        
        try:
            result = runner.run()
        except NotImplementedError as e:
            logger.error(f"Backtest not implemented: {e}")
            sys.exit(1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Backtest completed in {duration.total_seconds():.1f} seconds")
        
        # Print results summary
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        portfolio = result.portfolio
        
        logger.info(f"Initial Cash: {portfolio.initial_cash:,.0f} KRW")
        
        if portfolio.history:
            final_equity = portfolio.history[-1].equity
            total_return = (final_equity - portfolio.initial_cash) / portfolio.initial_cash
            logger.info(f"Final Equity: {final_equity:,.0f} KRW")
            logger.info(f"Total Return: {total_return:+.2%}")
            logger.info(f"Snapshots: {len(portfolio.history)}")
        else:
            logger.info("Final Equity: No equity data recorded")
            logger.info("Total Return: 0.00%")
            logger.info("Snapshots: 0")
        
        logger.info(f"Steps: {result.num_steps}")
        logger.info(f"Time Range: {result.start_time} to {result.end_time}")
        logger.info(f"Final Positions: {len(portfolio.positions)}")
        logger.info(f"Final Cash: {portfolio.cash:,.0f} KRW")
        
        if result.notes:
            logger.info(f"Notes: {result.notes}")
        
        logger.info("=" * 60)
        
        # Success
        logger.info("Backtest completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()