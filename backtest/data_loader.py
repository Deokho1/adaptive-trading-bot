"""
Data loader for backtest historical OHLCV data.

This module provides functionality to load historical market data
from CSV files for backtesting purposes.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from exchange.models import Candle

logger = logging.getLogger("bot")


class BacktestDataLoader:
    """
    Loader for historical OHLCV data from CSV files.
    
    Expected CSV format:
    timestamp,open,high,low,close,volume
    2023-01-01T00:00:00+09:00,50000000,51000000,49000000,50500000,100
    """
    
    def __init__(self, data_dir: str | Path = "data/ohlcv") -> None:
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files with OHLCV data
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
        else:
            logger.info(f"BacktestDataLoader initialized with data_dir: {self.data_dir}")
    
    def _get_filename(self, symbol: str) -> str:
        """
        Map symbol to CSV filename.
        
        Args:
            symbol: Trading symbol (e.g., "KRW-BTC")
            
        Returns:
            Filename for the symbol (e.g., "KRW-BTC.csv")
        """
        return f"{symbol}.csv"
    
    def load_symbol(self, symbol: str) -> List[Candle]:
        """
        Load OHLCV data for a single symbol from CSV.
        
        Args:
            symbol: Trading symbol to load (e.g., "KRW-BTC")
            
        Returns:
            List of Candle objects sorted from oldest to newest
        """
        filename = self._get_filename(symbol)
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Data file not found for {symbol}: {filepath}")
            return []
        
        candles = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    try:
                        # Parse timestamp
                        timestamp_str = row['timestamp'].strip()
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        # Parse OHLCV values
                        open_price = float(row['open'])
                        high_price = float(row['high'])
                        low_price = float(row['low'])
                        close_price = float(row['close'])
                        volume = float(row['volume'])
                        
                        # Create Candle object
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume
                        )
                        
                        candles.append(candle)
                        
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing row in {filename}: {e}")
                        continue
            
            # Sort by timestamp (oldest to newest)
            candles.sort(key=lambda c: c.timestamp)
            
            logger.info(f"Loaded {len(candles)} candles for {symbol}")
            
            if candles:
                start_time = candles[0].timestamp
                end_time = candles[-1].timestamp
                logger.debug(f"{symbol} data range: {start_time} to {end_time}")
            
            return candles
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol} from {filepath}: {e}")
            return []
    
    def load_all(self, symbols: List[str]) -> Dict[str, List[Candle]]:
        """
        Load OHLCV candles for all given symbols.
        
        Args:
            symbols: List of trading symbols to load
            
        Returns:
            Dictionary mapping symbol to list of Candle objects
        """
        logger.info(f"Loading data for {len(symbols)} symbols: {symbols}")
        
        data = {}
        
        for symbol in symbols:
            data[symbol] = self.load_symbol(symbol)
        
        # Log summary
        total_candles = sum(len(candles) for candles in data.values())
        successful_symbols = [sym for sym, candles in data.items() if candles]
        
        logger.info(f"Data loading complete:")
        logger.info(f"  Total candles loaded: {total_candles}")
        logger.info(f"  Successful symbols: {len(successful_symbols)}/{len(symbols)}")
        
        if len(successful_symbols) < len(symbols):
            failed_symbols = [sym for sym in symbols if sym not in successful_symbols]
            logger.warning(f"  Failed to load: {failed_symbols}")
        
        return data
    
    def get_common_timerange(self, data: Dict[str, List[Candle]]) -> tuple[datetime | None, datetime | None]:
        """
        Get the common time range across all loaded symbols.
        
        Args:
            data: Dictionary of symbol -> candles
            
        Returns:
            Tuple of (start_time, end_time) for the common range
        """
        if not data:
            return None, None
        
        # Filter out empty datasets
        valid_data = {sym: candles for sym, candles in data.items() if candles}
        
        if not valid_data:
            return None, None
        
        # Find latest start time and earliest end time
        start_times = [candles[0].timestamp for candles in valid_data.values()]
        end_times = [candles[-1].timestamp for candles in valid_data.values()]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        if common_start > common_end:
            logger.warning("No common time range found across symbols")
            return None, None
        
        logger.info(f"Common time range: {common_start} to {common_end}")
        return common_start, common_end
    
    def validate_data_consistency(self, data: Dict[str, List[Candle]]) -> bool:
        """
        Validate that all symbols have consistent timestamps.
        
        Args:
            data: Dictionary of symbol -> candles
            
        Returns:
            True if data is consistent, False otherwise
        """
        if not data:
            return True
        
        # Get valid datasets
        valid_data = {sym: candles for sym, candles in data.items() if candles}
        
        if len(valid_data) <= 1:
            return True
        
        # Compare timestamps across symbols
        symbol_names = list(valid_data.keys())
        reference_symbol = symbol_names[0]
        reference_timestamps = [c.timestamp for c in valid_data[reference_symbol]]
        
        inconsistencies = []
        
        for symbol in symbol_names[1:]:
            symbol_timestamps = [c.timestamp for c in valid_data[symbol]]
            
            if len(symbol_timestamps) != len(reference_timestamps):
                inconsistencies.append(f"{symbol} has {len(symbol_timestamps)} candles vs {len(reference_timestamps)}")
                continue
            
            # Check timestamp alignment
            mismatched = sum(1 for a, b in zip(reference_timestamps, symbol_timestamps) if a != b)
            if mismatched > 0:
                inconsistencies.append(f"{symbol} has {mismatched} mismatched timestamps")
        
        if inconsistencies:
            logger.warning("Data consistency issues found:")
            for issue in inconsistencies:
                logger.warning(f"  {issue}")
            return False
        
        logger.info("Data consistency validation passed")
        return True