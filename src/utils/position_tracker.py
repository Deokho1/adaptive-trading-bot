"""
Position Tracker Module

Tracks current trading position and history
"""

import json
import os
from typing import Dict, Optional, List
from datetime import datetime
import logging


class PositionTracker:
    """Tracks trading positions and trade history"""
    
    def __init__(self, position_file: str = "positions.json"):
        """
        Initialize PositionTracker
        
        Args:
            position_file: Path to file for persisting position data
        """
        self.position_file = position_file
        self.logger = logging.getLogger(__name__)
        
        # Current position
        self.current_position = {
            'has_position': False,
            'entry_price': 0,
            'entry_time': None,
            'amount': 0,
            'ticker': None
        }
        
        # Trade history
        self.trade_history: List[Dict] = []
        
        # Load existing position if available
        self._load_position()
    
    def _load_position(self):
        """Load position from file"""
        if os.path.exists(self.position_file):
            try:
                with open(self.position_file, 'r') as f:
                    data = json.load(f)
                    self.current_position = data.get('current_position', self.current_position)
                    self.trade_history = data.get('trade_history', [])
                self.logger.info(f"Loaded position from {self.position_file}")
            except Exception as e:
                self.logger.error(f"Error loading position file: {e}")
    
    def _save_position(self):
        """Save position to file"""
        try:
            data = {
                'current_position': self.current_position,
                'trade_history': self.trade_history
            }
            with open(self.position_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.debug(f"Saved position to {self.position_file}")
        except Exception as e:
            self.logger.error(f"Error saving position file: {e}")
    
    def open_position(self, ticker: str, entry_price: float, amount: float):
        """
        Open a new position
        
        Args:
            ticker: Trading pair
            entry_price: Entry price
            amount: Position size
        """
        self.current_position = {
            'has_position': True,
            'entry_price': entry_price,
            'entry_time': datetime.now().isoformat(),
            'amount': amount,
            'ticker': ticker
        }
        self._save_position()
        self.logger.info(f"Position opened: {amount:.8f} {ticker} at {entry_price:.2f}")
    
    def close_position(self, exit_price: float, reason: str = "manual") -> Dict:
        """
        Close current position
        
        Args:
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Trade summary
        """
        if not self.current_position['has_position']:
            self.logger.warning("No position to close")
            return {}
        
        # Calculate profit/loss
        entry_price = self.current_position['entry_price']
        amount = self.current_position['amount']
        profit = (exit_price - entry_price) * amount
        profit_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Create trade record
        trade = {
            'ticker': self.current_position['ticker'],
            'entry_price': entry_price,
            'entry_time': self.current_position['entry_time'],
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'amount': amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        }
        
        # Add to history
        self.trade_history.append(trade)
        
        # Reset position
        self.current_position = {
            'has_position': False,
            'entry_price': 0,
            'entry_time': None,
            'amount': 0,
            'ticker': None
        }
        
        self._save_position()
        self.logger.info(f"Position closed: Profit {profit:.2f} ({profit_pct:.2f}%) - Reason: {reason}")
        
        return trade
    
    def has_position(self) -> bool:
        """Check if currently holding a position"""
        return self.current_position['has_position']
    
    def get_position(self) -> Dict:
        """Get current position details"""
        return self.current_position.copy()
    
    def get_entry_price(self) -> float:
        """Get entry price of current position"""
        return self.current_position.get('entry_price', 0)
    
    def get_position_amount(self) -> float:
        """Get amount of current position"""
        return self.current_position.get('amount', 0)
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary from trade history
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0,
                'win_rate': 0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['profit'] > 0)
        losing_trades = sum(1 for t in self.trade_history if t['profit'] < 0)
        total_profit = sum(t['profit'] for t in self.trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit': total_profit,
            'win_rate': win_rate
        }
