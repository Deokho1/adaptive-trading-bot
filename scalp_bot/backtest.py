"""
Backtesting engine for scalp bot

Simulates dip scalping strategy on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import ScalpConfig
from .strategy import DipScalpStrategy
from .risk import RiskManager
from .data_loader import DataLoader
from .logging_utils import ScalpLogger


class ScalpBacktester:
    """Backtester for dip scalping strategy"""
    
    def __init__(self, config: ScalpConfig, initial_equity: float = 10000.0):
        """
        Initialize backtester
        
        Args:
            config: Strategy configuration
            initial_equity: Starting equity
        """
        self.config = config
        self.initial_equity = initial_equity
        
        # Core components
        self.risk_manager = RiskManager(config, initial_equity)
        self.strategy = DipScalpStrategy(config, self.risk_manager)
        self.data_loader = DataLoader()
        self.logger = ScalpLogger()
        
        # Tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.current_time: Optional[datetime] = None
    
    def run_backtest(self, symbols: List[str], start_date: str = None,
                    end_date: str = None, data_dir: str = "data") -> Dict:
        """
        Run backtest on historical data
        
        Args:
            symbols: List of symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            data_dir: Directory with data files
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting scalp backtest for {symbols}")
        print(f"Config: {self.config.candle_interval} timeframe, "
              f"{self.config.take_profit_pct}% TP, {self.config.stop_loss_pct}% SL")
        
        # Reset everything
        self.logger.reset_files()
        self.strategy.reset()
        self.risk_manager = RiskManager(self.config, self.initial_equity)
        self.trades.clear()
        self.equity_curve.clear()
        
        # Load data
        self.data_loader.data_dir = data_dir
        data_dict = self.data_loader.get_aligned_data(symbols, self.config.candle_interval)
        
        if not data_dict:
            raise ValueError(f"No data loaded for symbols: {symbols}")
        
        # Filter date range if specified
        if start_date or end_date:
            data_dict = self._filter_date_range(data_dict, start_date, end_date)
        
        # Run simulation
        results = self._simulate_trading(data_dict)
        
        # Generate outputs
        self._save_results()
        
        print(f"\\nBacktest completed: {len(self.trades)} trades, "
              f"Final equity: {self.risk_manager.current_equity:.2f}")
        
        return results
    
    def _filter_date_range(self, data_dict: Dict[str, pd.DataFrame],
                          start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Filter data by date range"""
        filtered_dict = {}
        
        for symbol, df in data_dict.items():
            filtered_df = df.copy()
            
            if start_date:
                filtered_df = filtered_df[filtered_df.index >= start_date]
            if end_date:
                filtered_df = filtered_df[filtered_df.index <= end_date]
            
            filtered_dict[symbol] = filtered_df
            print(f"{symbol}: {len(filtered_df)} bars from {filtered_df.index[0]} to {filtered_df.index[-1]}")
        
        return filtered_dict
    
    def _simulate_trading(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Run the main trading simulation"""
        
        # Get all unique timestamps and sort
        all_timestamps = set()
        for df in data_dict.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        print(f"Simulating {len(timestamps)} time periods...")
        
        # Track progress
        log_interval = max(100, len(timestamps) // 20)
        
        for i, timestamp in enumerate(timestamps):
            self.current_time = timestamp
            
            # Process each symbol at this timestamp
            for symbol in self.config.symbols:
                if symbol not in data_dict:
                    continue
                
                df = data_dict[symbol]
                
                # Skip if no data at this timestamp
                if timestamp not in df.index:
                    continue
                
                # Get historical slice up to current time
                df_slice = df[df.index <= timestamp].copy()
                if len(df_slice) == 0:
                    continue
                
                current_bar = df_slice.iloc[-1]
                bar_index = len(df_slice) - 1
                
                # Generate signal
                signal = self.strategy.on_bar(
                    symbol=symbol,
                    bar_index=bar_index,
                    bar_row=current_bar,
                    df_slice=df_slice,
                    current_time=timestamp
                )
                
                # Execute signal
                if signal:
                    self._execute_signal(signal, timestamp)
            
            # Log equity curve periodically
            if i % log_interval == 0 or i == len(timestamps) - 1:
                self._log_equity_point(timestamp)
                
                progress = (i / len(timestamps)) * 100
                print(f"Progress: {progress:.1f}% - {timestamp} - "
                      f"Equity: {self.risk_manager.current_equity:.2f}")
        
        # Calculate final results
        return self._calculate_results()
    
    def _execute_signal(self, signal: Dict, timestamp: datetime) -> None:
        """Execute a trading signal"""
        
        if signal["action"] == "BUY":
            # Entry signal - already handled by strategy
            pass
            
        elif signal["action"] == "SELL":
            # Exit signal - record the trade
            trade_record = {
                "timestamp_entry": signal.get("entry_time", timestamp),
                "timestamp_exit": timestamp,
                "symbol": signal["symbol"],
                "side": "LONG",
                "entry_price": signal.get("entry_price", 0),
                "exit_price": signal["price"],
                "size_pct": signal["size_pct"],
                "pnl_abs": signal["pnl_abs"],
                "pnl_pct": signal["pnl_pct"],
                "max_adverse_excursion_pct": signal.get("max_adverse_excursion", 0),
                "max_favorable_excursion_pct": signal.get("max_favorable_excursion", 0),
                "holding_bars": signal.get("holding_bars", 0),
                "reason_entry": "dip_rebound",
                "reason_exit": signal["reason"],
                "market_state_at_entry": signal.get("market_state", "unknown")
            }
            
            self.trades.append(trade_record)
            self.logger.log_trade(trade_record)
    
    def _log_equity_point(self, timestamp: datetime) -> None:
        """Log current equity point"""
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        equity_point = {
            "timestamp": timestamp,
            "equity": risk_metrics["current_equity"],
            "drawdown_pct": risk_metrics["max_drawdown_pct"],
            "open_positions": risk_metrics["open_positions"],
            "total_exposure_pct": risk_metrics["total_exposure_pct"]
        }
        
        self.equity_curve.append(equity_point)
        self.logger.log_equity_point(
            timestamp=timestamp,
            equity=risk_metrics["current_equity"],
            drawdown_pct=risk_metrics["max_drawdown_pct"],
            open_positions=risk_metrics["open_positions"]
        )
    
    def _calculate_results(self) -> Dict:
        """Calculate final backtest results"""
        
        # Basic metrics
        final_equity = self.risk_manager.current_equity
        total_return = (final_equity - self.initial_equity) / self.initial_equity * 100
        
        # Trade analysis
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            return {
                "total_return_pct": total_return,
                "final_equity": final_equity,
                "total_trades": 0,
                "message": "No trades executed"
            }
        
        # Performance metrics
        profitable_trades = len(trades_df[trades_df["pnl_pct"] > 0])
        win_rate = (profitable_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        avg_profit = trades_df["pnl_pct"].mean()
        max_win = trades_df["pnl_pct"].max()
        max_loss = trades_df["pnl_pct"].min()
        
        # Risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Calculate Sharpe ratio (simplified)
        returns = trades_df["pnl_pct"]
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        results = {
            "total_return_pct": total_return,
            "max_drawdown_pct": risk_metrics["max_drawdown_pct"],
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "total_trades": len(trades_df),
            "profitable_trades": profitable_trades,
            "avg_profit_per_trade": avg_profit,
            "max_win": max_win,
            "max_loss": max_loss,
            "final_equity": final_equity,
            "total_fees": self._calculate_total_fees(),
            "strategy_stats": self.strategy.get_strategy_stats()
        }
        
        # Print summary
        self.logger.print_summary(results)
        
        return results
    
    def _calculate_total_fees(self) -> float:
        """Calculate total trading fees"""
        if not self.trades:
            return 0.0
        
        total_fees = 0.0
        for trade in self.trades:
            entry_value = trade["entry_price"] * trade["size_pct"] / 100
            exit_value = trade["exit_price"] * trade["size_pct"] / 100
            
            entry_fees = entry_value * self.config.fee_rate
            exit_fees = exit_value * self.config.fee_rate
            
            total_fees += entry_fees + exit_fees
        
        return total_fees
    
    def _save_results(self) -> None:
        """Save additional result files"""
        
        # Save strategy events
        if self.strategy.events:
            events_df = pd.DataFrame(self.strategy.events)
            self.logger.save_dataframe(events_df, "strategy_events.csv")
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index("timestamp", inplace=True)
            self.logger.save_dataframe(equity_df, "equity_curve_detailed.csv")
        
        # Save trades summary
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df["timestamp_entry"] = pd.to_datetime(trades_df["timestamp_entry"])
            trades_df["timestamp_exit"] = pd.to_datetime(trades_df["timestamp_exit"])
            trades_df.set_index("timestamp_exit", inplace=True)
            self.logger.save_dataframe(trades_df, "trades_detailed.csv")