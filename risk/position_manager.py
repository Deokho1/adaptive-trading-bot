"""
Position manager for tracking open trading positions.

This module provides functionality to manage trading positions
with persistence to JSON files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from core.types import MarketMode
from exchange.models import Position


class PositionManager:
    """
    Manages trading positions with JSON persistence.
    
    Keeps track of all open positions in memory and saves them to a JSON file
    for persistence across bot restarts.
    """
    
    def __init__(self, positions_file: str) -> None:
        """
        Initialize the position manager.
        
        Args:
            positions_file: Path to JSON file for storing positions
        """
        self.positions_file = Path(positions_file)
        self.positions: List[Position] = []
        self.load_positions()
    
    def load_positions(self) -> None:
        """
        Load positions from JSON file.
        
        If the file doesn't exist or is invalid, starts with empty list.
        """
        if not self.positions_file.exists():
            self.positions = []
            return
        
        try:
            with open(self.positions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.positions = []
            for item in data:
                position = Position(
                    symbol=item['symbol'],
                    mode=MarketMode(item['mode']),
                    entry_price=float(item['entry_price']),
                    size=float(item['size']),
                    entry_time=datetime.fromisoformat(item['entry_time']),
                    peak_price=float(item['peak_price'])
                )
                self.positions.append(position)
                
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # If file is corrupted, start fresh but log the error
            self.positions = []
            print(f"Warning: Could not load positions from {self.positions_file}: {e}")
    
    def save_positions(self) -> None:
        """
        Save current positions to JSON file.
        
        Creates parent directory if it doesn't exist.
        """
        # Ensure parent directory exists
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert positions to JSON-serializable format
        data = []
        for position in self.positions:
            item = {
                'symbol': position.symbol,
                'mode': position.mode.value,
                'entry_price': position.entry_price,
                'size': position.size,
                'entry_time': position.entry_time.isoformat(),
                'peak_price': position.peak_price
            }
            data.append(item)
        
        # Write to file
        try:
            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error: Could not save positions to {self.positions_file}: {e}")
    
    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_positions_by_mode(self, mode: MarketMode) -> List[Position]:
        """
        Get positions filtered by market mode.
        
        Args:
            mode: Market mode to filter by
            
        Returns:
            List of positions that were opened in the specified mode
        """
        return [pos for pos in self.positions if pos.mode == mode]
    
    def get_position(self, symbol: str) -> Position | None:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position if found, None otherwise
            
        Note:
            Assumes at most one position per symbol
        """
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def open_position(
        self,
        symbol: str,
        mode: MarketMode,
        entry_price: float,
        size: float,
        now: datetime,
    ) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Trading pair symbol
            mode: Market mode when opening position
            entry_price: Entry price for the position
            size: Amount of cryptocurrency
            now: Current datetime
            
        Returns:
            The newly created Position object
        """
        position = Position(
            symbol=symbol,
            mode=mode,
            entry_price=entry_price,
            size=size,
            entry_time=now,
            peak_price=entry_price  # Initialize peak at entry price
        )
        
        self.positions.append(position)
        self.save_positions()
        
        return position
    
    def close_position(self, symbol: str) -> None:
        """
        Close position for a specific symbol.
        
        Args:
            symbol: Trading pair symbol to close
        """
        self.positions = [pos for pos in self.positions if pos.symbol != symbol]
        self.save_positions()
    
    def update_peak_price(self, symbol: str, current_price: float) -> None:
        """
        Update peak price for a position if current price is higher.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        """
        position = self.get_position(symbol)
        if position and current_price > position.peak_price:
            position.peak_price = current_price
            self.save_positions()
    
    def get_total_position_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total value of all positions in KRW.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            Total position value in KRW
        """
        total_value = 0.0
        
        for position in self.positions:
            current_price = current_prices.get(position.symbol, position.entry_price)
            total_value += position.current_value_krw(current_price)
        
        return total_value
    
    def get_position_count(self) -> int:
        """Get total number of open positions."""
        return len(self.positions)
    
    def get_position_count_by_mode(self, mode: MarketMode) -> int:
        """Get number of positions for a specific mode."""
        return len(self.get_positions_by_mode(mode))