"""
Data loader for scalp bot

Handles loading and preprocessing of OHLCV data for backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from .indicators import add_all_indicators


class DataLoader:
    """Loads and preprocesses OHLCV data for backtesting"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.loaded_data: Dict[str, pd.DataFrame] = {}
    
    def load_symbol_data(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        """
        Load OHLCV data for a symbol
        
        Args:
            symbol: Symbol to load (e.g., "BTC", "SOL")
            timeframe: Timeframe (e.g., "5m", "1h")
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        # Try different file naming conventions
        possible_files = [
            f"{symbol.lower()}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe}.csv", 
            f"{symbol.lower()}_5min.csv",
            f"{symbol.upper()}_5min.csv",
            f"{symbol.lower()}.csv",
            f"{symbol.upper()}.csv"
        ]
        
        df = None
        for filename in possible_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"Loaded {symbol} data from {filename}")
                    break
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
        
        if df is None:
            raise FileNotFoundError(f"No data file found for {symbol} in {self.data_dir}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Ensure proper datetime index
        df = self._process_timestamps(df)
        
        # Add technical indicators
        df = add_all_indicators(df)
        
        # Cache the data
        cache_key = f"{symbol}_{timeframe}"
        self.loaded_data[cache_key] = df
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase"""
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Timestamp': 'timestamp',
            'Time': 'timestamp',
            'Date': 'timestamp',
            'DateTime': 'timestamp'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process timestamps and set as index"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif df.index.name == 'timestamp' or pd.api.types.is_datetime64_any_dtype(df.index):
            # Index is already datetime
            pass
        else:
            # Create a dummy timestamp index
            print("Warning: No timestamp column found, creating dummy timestamps")
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='5min')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def load_multiple_symbols(self, symbols: List[str], 
                             timeframe: str = "5m") -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols
        
        Args:
            symbols: List of symbols to load
            timeframe: Timeframe for all symbols
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.load_symbol_data(symbol, timeframe)
                print(f"Successfully loaded {symbol}: {len(data_dict[symbol])} bars")
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
        
        return data_dict
    
    def get_aligned_data(self, symbols: List[str], 
                        timeframe: str = "5m") -> Dict[str, pd.DataFrame]:
        """
        Load and align data for multiple symbols to same timeframe
        
        Args:
            symbols: List of symbols
            timeframe: Target timeframe
            
        Returns:
            Dictionary of aligned DataFrames
        """
        data_dict = self.load_multiple_symbols(symbols, timeframe)
        
        if len(data_dict) < 2:
            return data_dict
        
        # Find common time range
        all_indexes = [df.index for df in data_dict.values()]
        common_start = max(idx.min() for idx in all_indexes)
        common_end = min(idx.max() for idx in all_indexes)
        
        # Trim all DataFrames to common range
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_df = df[(df.index >= common_start) & (df.index <= common_end)].copy()
            aligned_data[symbol] = aligned_df
            print(f"{symbol} aligned data: {len(aligned_df)} bars "
                  f"from {common_start} to {common_end}")
        
        return aligned_data