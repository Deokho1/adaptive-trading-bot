"""
Logging utilities for scalp bot

Handles CSV output generation and console logging.
"""

import os
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ScalpLogger:
    """Handles logging and output generation for scalp bot"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize logger
        
        Args:
            output_dir: Directory for output files (changed to results)
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir()
        
        # File paths - all in results folder
        self.trades_file = self.output_dir / "trades.csv"
        self.equity_file = self.output_dir / "equity_curve.csv" 
        self.events_file = self.output_dir / "debug_events.csv"
        self.adaptive_params_file = self.output_dir / "adaptive_params.csv"
        self.metrics_file = self.output_dir / "metrics.json"
        self.debug_signals_file = self.output_dir / "debug_signals.csv"
        
        # Track if headers written
        self._headers_written = {
            "trades": False,
            "equity": False,
            "events": False,
            "adaptive_params": False,
            "debug_signals": False
        }
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory ready: {self.output_dir}")
    
    def reset_files(self) -> None:
        """Reset/clear all output files"""
        for file_path in [self.trades_file, self.equity_file, self.events_file]:
            if file_path.exists():
                file_path.unlink()
        
        self._headers_written = {k: False for k in self._headers_written}
        print("Output files reset")
    
    def log_adaptive_params(self, params_data: Dict[str, Any]) -> None:
        """
        Log adaptive parameter changes to CSV
        
        Args:
            params_data: Dictionary with parameter data
        """
        columns = [
            "timestamp", "symbol", "parameter_name", "old_value", "new_value", 
            "trigger_reason", "market_condition", "performance_metric"
        ]
        
        # Write header if first time
        if not self._headers_written["adaptive_params"]:
            with open(self.adaptive_params_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            self._headers_written["adaptive_params"] = True
        
        # Write parameter data
        with open(self.adaptive_params_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [params_data.get(col, "") for col in columns]
            writer.writerow(row)
    
    def log_debug_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Log debug signal information
        
        Args:
            signal_data: Dictionary with signal debug data
        """
        columns = [
            "timestamp", "symbol", "signal_type", "signal_strength", "price", 
            "volume", "indicators", "conditions_met", "action_taken", "rejection_reason"
        ]
        
        # Write header if first time
        if not self._headers_written["debug_signals"]:
            with open(self.debug_signals_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            self._headers_written["debug_signals"] = True
        
        # Write signal data
        with open(self.debug_signals_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [signal_data.get(col, "") for col in columns]
            writer.writerow(row)
    
    def save_metrics_json(self, metrics: Dict[str, Any]) -> None:
        """
        Save comprehensive metrics to JSON file
        
        Args:
            metrics: Dictionary containing all metrics
        """
        import json
        from datetime import datetime
        
        # Add metadata
        metrics_with_meta = {
            'generated_at': datetime.now().isoformat(),
            'backtest_version': '1.0',
            'metrics': metrics
        }
        
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)
            print(f"📊 Metrics saved: {self.metrics_file}")
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def reset_files(self) -> None:
        """Reset/clear all output files"""
        for file_path in [
            self.trades_file, self.equity_file, self.events_file,
            self.adaptive_params_file, self.debug_signals_file
        ]:
            if file_path.exists():
                file_path.unlink()
        
        # Remove JSON metrics file
        if self.metrics_file.exists():
            self.metrics_file.unlink()
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Log a completed trade to CSV
        
        Args:
            trade_data: Dictionary with trade information
        """
        # ANALYSIS: 확장된 trades.csv 컬럼 정의
        # Define extended columns with new analysis fields
        columns = [
            "timestamp_entry", "timestamp_exit", "symbol", "side",
            "entry_price", "exit_price", "size_pct", "pnl_abs", "pnl_pct",
            "max_adverse_excursion_pct", "max_favorable_excursion_pct",
            "holding_bars", "reason_entry", "reason_exit", "market_state_at_entry",
            # ANALYSIS: 새로운 분석 컬럼들
            "holding_minutes", "gross_pnl", "net_pnl", "return_pct", 
            "setup_type", "session_label"
        ]
        
        # ANALYSIS: 확장 지표 계산
        # Calculate extended metrics if not already present
        if 'holding_minutes' not in trade_data and 'timestamp_entry' in trade_data and 'timestamp_exit' in trade_data:
            from datetime import datetime
            if isinstance(trade_data['timestamp_entry'], str):
                entry_time = datetime.fromisoformat(trade_data['timestamp_entry'].replace('Z', '+00:00'))
            else:
                entry_time = trade_data['timestamp_entry']
            
            if isinstance(trade_data['timestamp_exit'], str):
                exit_time = datetime.fromisoformat(trade_data['timestamp_exit'].replace('Z', '+00:00'))
            else:
                exit_time = trade_data['timestamp_exit']
            
            # Calculate holding time in minutes
            trade_data['holding_minutes'] = (exit_time - entry_time).total_seconds() / 60
            
            # Calculate gross PnL (before fees/slippage)
            trade_data['gross_pnl'] = trade_data.get('pnl_abs', 0)
            
            # Calculate net PnL (after estimated fees/slippage - 0.07% total cost)
            trade_cost_pct = 0.0007  # 0.07% total cost
            gross_pnl = trade_data.get('pnl_abs', 0)
            trade_data['net_pnl'] = gross_pnl * (1 - trade_cost_pct)
            
            # Return percentage is already in pnl_pct
            trade_data['return_pct'] = trade_data.get('pnl_pct', 0)
            
            # Classify setup type based on entry reason
            reason = str(trade_data.get('reason_entry', '')).lower()
            if 'dip' in reason or 'rebound' in reason:
                trade_data['setup_type'] = 'dip_buy'
            elif 'spike' in reason or 'volume' in reason:
                trade_data['setup_type'] = 'vol_spike'
            elif 'breakout' in reason:
                trade_data['setup_type'] = 'breakout'
            else:
                trade_data['setup_type'] = 'other'
            
            # Classify session based on entry time (UTC)
            hour = entry_time.hour
            if 0 <= hour < 8:
                trade_data['session_label'] = 'asia'
            elif 8 <= hour < 16:
                trade_data['session_label'] = 'eu'
            elif 16 <= hour < 24:
                trade_data['session_label'] = 'us'
            else:
                trade_data['session_label'] = 'other'
        
        # Write header if first time
        if not self._headers_written["trades"]:
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            self._headers_written["trades"] = True
        
        # Write trade data
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [trade_data.get(col, "") for col in columns]
            writer.writerow(row)
    
    def log_equity_point(self, timestamp: datetime, equity: float, 
                        drawdown_pct: float, open_positions: int) -> None:
        """
        Log equity curve point
        
        Args:
            timestamp: Current timestamp
            equity: Current equity value
            drawdown_pct: Current drawdown percentage
            open_positions: Number of open positions
        """
        columns = ["timestamp", "equity", "drawdown_pct", "open_positions_count"]
        
        # Write header if first time
        if not self._headers_written["equity"]:
            with open(self.equity_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            self._headers_written["equity"] = True
        
        # Write equity data
        with open(self.equity_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, equity, drawdown_pct, open_positions])
    
    def log_event(self, event_data: Dict[str, Any]) -> None:
        """
        Log strategy event for debugging
        
        Args:
            event_data: Event information dictionary
        """
        columns = [
            "timestamp", "symbol", "event_type", "old_state", "new_state",
            "price", "pct_change_5m", "pct_change_15m", "volume_ratio", "note"
        ]
        
        # Write header if first time
        if not self._headers_written["events"]:
            with open(self.events_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            self._headers_written["events"] = True
        
        # Write event data
        with open(self.events_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [event_data.get(col, "") for col in columns]
            writer.writerow(row)
    
    def log_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """
        Log multiple events at once
        
        Args:
            events: List of event dictionaries
        """
        for event in events:
            self.log_event(event)
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print backtest summary to console
        
        Args:
            metrics: Summary metrics dictionary
        """
        print("\n" + "="*60)
        print("SCALP BOT BACKTEST SUMMARY")
        print("="*60)
        
        # Performance metrics
        if 'total_return_pct' in metrics:
            print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        if 'max_drawdown_pct' in metrics:
            print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        if 'sharpe_ratio' in metrics:
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        if 'win_rate' in metrics:
            print(f"Win Rate: {metrics['win_rate']:.1f}%")
        
        # Trading stats
        if 'total_trades' in metrics:
            print(f"Total Trades: {metrics['total_trades']}")
        if 'profitable_trades' in metrics:
            print(f"Profitable Trades: {metrics['profitable_trades']}")
        if 'avg_profit_per_trade' in metrics:
            print(f"Avg Profit/Trade: {metrics['avg_profit_per_trade']:.2f}%")
        
        # Risk metrics
        if 'final_equity' in metrics:
            print(f"Final Equity: {metrics['final_equity']:,.2f}")
        if 'total_fees' in metrics:
            print(f"Total Fees: {metrics['total_fees']:.2f}")
        
        print("="*60)
        print(f"Output files saved to: {self.output_dir}")
        print("="*60)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save a DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=True)
        print(f"Saved {filename}: {len(df)} rows")
    
    def get_output_files(self) -> Dict[str, Path]:
        """Get dictionary of output file paths"""
        return {
            "trades": self.trades_file,
            "equity_curve": self.equity_file,
            "debug_events": self.events_file,
            "adaptive_params": self.adaptive_params_file,
            "metrics": self.metrics_file,
            "debug_signals": self.debug_signals_file
        }
    
    def save_all_outputs(self) -> None:
        """
        Save and finalize all output files
        """
        # Check if files exist and print status
        files_status = self.get_output_files()
        
        print(f"Output files saved to: {self.output_dir}")
        for name, filepath in files_status.items():
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"  {name}: {filepath.name} ({size_kb:.1f} KB)")
            else:
                print(f"  {name}: {filepath.name} (not created - no data)")
        
        print("All outputs finalized.")


def setup_console_logging(verbose: bool = True) -> None:
    """
    Setup console logging configuration
    
    Args:
        verbose: Enable verbose logging
    """
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress pandas warnings in verbose mode
    if not verbose:
        logging.getLogger('pandas').setLevel(logging.WARNING)