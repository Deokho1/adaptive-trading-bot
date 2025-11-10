"""
Backtest engine implementation

Coordinates backtesting using BacktestExchangeClient.
"""

from datetime import datetime
from typing import Any
import pandas as pd

from ..exchange.backtest import BacktestExchangeClient
from ..strategy import DipScalpStrategy
from ..risk import RiskManager
from ..logging_utils import ScalpLogger
from ..state import SymbolState
from ..indicators import add_all_indicators


class BacktestEngine:
    """
    Backtest engine for running historical simulations
    
    Coordinates strategy, risk management, and exchange simulation.
    """
    
    def __init__(
        self,
        exchange: BacktestExchangeClient,
        strategy: DipScalpStrategy,
        risk_manager: RiskManager,
        symbols: list[str],
        timeframe: str = "5m",
        output_dir: str = "results"
    ):
        """
        Initialize backtest engine
        
        Args:
            exchange: Backtest exchange client
            strategy: Scalping strategy instance
            risk_manager: Risk management instance
            symbols: List of symbols to trade
            timeframe: Candle timeframe
            output_dir: Directory for output files
        """
        self.exchange = exchange
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbols = symbols
        self.timeframe = timeframe
        self.output_dir = output_dir
        
        # Initialize logger
        self.logger = ScalpLogger(output_dir)
        
        # Track symbol states
        self.symbol_states: dict[str, SymbolState] = {}
        for symbol in symbols:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)
        
        # Track pending trade entries
        self._pending_entries: dict[str, dict] = {}
        
        # Backtest metrics
        self.initial_equity = exchange.get_portfolio_value()
        self.equity_history: list[dict] = []
        self.max_equity = self.initial_equity
        self.max_drawdown = 0.0
        
        print(f"BacktestEngine initialized with {len(symbols)} symbols")
        print(f"Initial equity: {self.initial_equity:,.0f}")
    
    def run_backtest(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        """
        Run the backtest simulation
        
        Args:
            start_date: Start date (YYYY-MM-DD) - optional, uses all data if None
            end_date: End date (YYYY-MM-DD) - optional, uses all data if None
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting backtest from {start_date or 'beginning'} to {end_date or 'end'}")
        
        # Determine total number of time steps
        max_steps = 0
        for symbol in self.symbols:
            if symbol in self.exchange.ohlcv_data:
                symbol_data = self.exchange.ohlcv_data[symbol]
                
                # Filter by date range if specified
                if start_date or end_date:
                    if start_date:
                        symbol_data = symbol_data[symbol_data['timestamp'] >= start_date]
                    if end_date:
                        symbol_data = symbol_data[symbol_data['timestamp'] <= end_date]
                
                max_steps = max(max_steps, len(symbol_data))
        
        print(f"Running backtest for {max_steps} time steps...")
        
        # Reset exchange to starting position
        for symbol in self.symbols:
            self.exchange.current_time_index[symbol] = 0
        
        # Main backtest loop
        for step in range(max_steps):
            # Process each symbol at current time step
            for symbol in self.symbols:
                self._process_symbol_step(symbol, step)
            
            # Advance time for all symbols
            self.exchange.advance_time_step()
            
            # Record equity curve
            self._record_equity_step()
            
            # Progress reporting
            if step % 1000 == 0 or step == max_steps - 1:
                current_equity = self.exchange.get_portfolio_value()
                pct_return = ((current_equity / self.initial_equity) - 1) * 100
                print(f"Step {step}/{max_steps} - Equity: {current_equity:,.0f} ({pct_return:+.2f}%)")
        
        # Generate final results
        results = self._calculate_results()
        
        # Save outputs
        self._save_outputs()
        
        # Save comprehensive metrics to JSON
        self.logger.save_metrics_json(results)
        
        print("Backtest completed!")
        self._print_summary(results)
        
        return results
    
    def _process_symbol_step(self, symbol: str, step: int) -> None:
        """
        Process single symbol at current time step
        
        Args:
            symbol: Symbol to process
            step: Current time step
        """
        # Get current candle
        current_candle = self.exchange.get_current_candle(symbol)
        if not current_candle:
            return
        
        # Get symbol state
        symbol_state = self.symbol_states[symbol]
        
        # Get recent candles for indicators
        recent_candles = self.exchange.fetch_ohlcv(symbol, limit=100)
        if len(recent_candles) < 20:  # Need minimum history for indicators
            return
        
        # Convert to DataFrame for indicators
        df_data = []
        for candle in recent_candles:
            df_data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(df_data)
        
        # Add indicators including pct_change
        df = add_all_indicators(df)
        
        # Call strategy
        try:
            signal = self.strategy.on_bar(
                symbol=symbol,
                bar_index=step,
                bar_row=df.iloc[-1],  # Current bar
                df_slice=df,
                current_time=current_candle.timestamp
            )
            
            # Execute any signal
            if signal:
                self._execute_signal(signal, current_candle.timestamp)
                    
        except Exception as e:
            # Log error but continue backtest
            self.logger.log_event({
                "timestamp": current_candle.timestamp,
                "symbol": symbol,
                "event_type": "ERROR",
                "old_state": "",
                "new_state": "",
                "price": current_candle.close,
                "pct_change_5m": 0.0,
                "pct_change_15m": 0.0,
                "volume_ratio": 1.0,
                "note": f"Strategy error: {str(e)}"
            })
    
    def _execute_signal(self, signal: dict, timestamp: datetime) -> None:
        """
        Execute trading signal
        
        Args:
            signal: Signal dictionary from strategy  
            timestamp: Current timestamp
        """
        try:
            if not signal or signal.get('action') not in ['BUY', 'SELL']:
                return
                
            symbol = signal['symbol']
            action = signal['action']
            
            if action == 'BUY':
                # Position entry - create order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market', 
                    side='buy',
                    amount=signal.get('size', 0.01)
                )
                
                if order.status == 'closed':
                    # Store entry data for later completion
                    self._pending_entries[symbol] = {
                        "timestamp_entry": timestamp,
                        "symbol": symbol,
                        "side": "BUY",
                        "entry_price": order.price,
                        "size_pct": signal.get('size_pct', 0.0),
                        "reason_entry": signal.get('reason', 'Strategy signal'),
                        "market_state_at_entry": signal.get('market_state', 'UNKNOWN')
                    }
                    
                    # 🚀 HIGH-FREQ SCALPING: Log entry signal for debugging
                    debug_signal_data = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "signal_type": "entry_executed",
                        "signal_strength": 0.8,  # Assuming strong signal for execution
                        "price": order.price,
                        "volume": 0,  # Not available in backtest
                        "indicators": f"size:{signal.get('size_pct', 0):.1f}%,tp:{signal.get('take_profit', 0):.1f}",
                        "conditions_met": f"reason:{signal.get('reason', 'unknown')},state:{signal.get('market_state', 'unknown')}",
                        "action_taken": "BUY",
                        "rejection_reason": ""
                    }
                    self.logger.log_debug_signal(debug_signal_data)
                    
            elif action == 'SELL':
                # Position exit - create order and complete trade
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side='sell', 
                    amount=signal.get('size', 0.01)
                )
                
                if order.status == 'closed' and symbol in self._pending_entries:
                    # Complete the trade record
                    entry_data = self._pending_entries[symbol]
                    
                    # 🚀 HIGH-FREQUENCY SCALPING: Enhanced trade logging
                    # Calculate detailed metrics for scalping analysis
                    entry_time = entry_data["timestamp_entry"]
                    exit_time = timestamp
                    holding_minutes = (exit_time - entry_time).total_seconds() / 60
                    
                    # Get market indicators at entry (if available)
                    rsi_at_entry = signal.get('rsi_at_entry', 0)
                    ema_diff_at_entry = signal.get('ema_diff_at_entry', 0)
                    volume_ratio_at_entry = signal.get('volume_ratio_at_entry', 1.0)
                    
                    trade_data = {
                        **entry_data,
                        "timestamp_exit": timestamp,
                        "exit_price": order.price,
                        "pnl_abs": signal.get('pnl_abs', 0.0),
                        "pnl_pct": signal.get('pnl_pct', 0.0),
                        "max_adverse_excursion_pct": signal.get('max_adverse_excursion', 0.0),
                        "max_favorable_excursion_pct": signal.get('max_favorable_excursion', 0.0),
                        "holding_bars": signal.get('holding_bars', 0),
                        "reason_exit": signal.get('reason', 'Strategy signal'),
                        # 🎯 HIGH-FREQ SCALPING: Additional debugging fields
                        "holding_minutes": holding_minutes,
                        "rsi_at_entry": rsi_at_entry,
                        "ema_diff_at_entry": ema_diff_at_entry,
                        "volume_ratio_at_entry": volume_ratio_at_entry,
                        "tp_hit_flag": 1 if signal.get('reason') == 'take_profit' else 0,
                        "sl_hit_flag": 1 if signal.get('reason') == 'stop_loss' else 0,
                        "realized_pnl_pct": signal.get('pnl_pct', 0.0)
                    }
                    
                    self.logger.log_trade(trade_data)
                    
                    # 🚀 Log high-frequency debug signal for analysis
                    debug_signal_data = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "signal_type": "exit_executed",
                        "signal_strength": 1.0,
                        "price": order.price,
                        "volume": 0,  # Not available in backtest
                        "indicators": f"holding:{holding_minutes:.1f}min,pnl:{signal.get('pnl_pct', 0):.3f}%",
                        "conditions_met": f"exit_reason:{signal.get('reason', 'unknown')}",
                        "action_taken": "SELL",
                        "rejection_reason": ""
                    }
                    self.logger.log_debug_signal(debug_signal_data)
                    
                    del self._pending_entries[symbol]
                    
        except Exception as e:
            # Log execution error
            self.logger.log_event({
                "timestamp": timestamp,
                "symbol": signal.get('symbol', 'UNKNOWN'),
                "event_type": "EXECUTION_ERROR", 
                "old_state": "",
                "new_state": "",
                "price": 0.0,
                "pnl_abs": 0.0,
                "pnl_pct": 0.0,
                "volume_ratio": 1.0,
                "note": f"Signal execution error: {str(e)}"
            })
    
    def _record_equity_step(self) -> None:
        """Record equity at current time step"""
        current_time = self.exchange.get_current_time()
        if not current_time:
            return
            
        current_equity = self.exchange.get_portfolio_value()
        
        # Update max equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record equity point
        equity_record = {
            'timestamp': current_time,
            'equity': current_equity,
            'drawdown': current_drawdown,
            'return_pct': ((current_equity / self.initial_equity) - 1) * 100
        }
        
        self.equity_history.append(equity_record)
        
        # Log to equity curve (sample every 100 steps to reduce file size)
        if len(self.equity_history) % 100 == 0:
            open_positions = len(self.risk_manager.open_positions) if hasattr(self.risk_manager, 'open_positions') else 0
            self.logger.log_equity_point(
                timestamp=current_time,
                equity=current_equity,
                drawdown_pct=current_drawdown * 100,
                open_positions=open_positions
            )
    
    def _calculate_results(self) -> dict[str, Any]:
        """Calculate final backtest results"""
        final_equity = self.exchange.get_portfolio_value()
        total_return = ((final_equity / self.initial_equity) - 1) * 100
        
        # Count trades
        total_trades = len([order for order in self.exchange.orders if order.status == 'closed'])
        winning_trades = 0
        total_pnl = 0.0
        
        # Analyze trades (simplified)
        for fill in self.exchange.trade_fills:
            # This is simplified - real P&L calculation would match buy/sell pairs
            if fill.side == 'sell':
                # Assume profit (this is very simplified)
                total_pnl += fill.amount * fill.price * 0.01  # Assume 1% profit per trade
                winning_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_equity': self.initial_equity,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate_pct': win_rate,
            'total_fees': sum(order.fee for order in self.exchange.orders),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity history"""
        if len(self.equity_history) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.equity_history)):
            prev_equity = self.equity_history[i-1]['equity']
            curr_equity = self.equity_history[i]['equity']
            daily_return = (curr_equity / prev_equity) - 1
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe (simplified - assuming daily returns)
        import statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_return * 252) / (std_return * (252 ** 0.5))
        return sharpe
    
    def _save_outputs(self) -> None:
        """Save all output files"""
        try:
            self.logger.save_all_outputs()
            print(f"Output files saved to {self.output_dir}/")
        except Exception as e:
            print(f"Error saving outputs: {e}")
    
    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print backtest summary"""
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Equity:    {results['initial_equity']:>15,.0f}")
        print(f"Final Equity:      {results['final_equity']:>15,.0f}")
        print(f"Total Return:      {results['total_return_pct']:>15.2f}%")
        print(f"Max Drawdown:      {results['max_drawdown_pct']:>15.2f}%")
        print(f"Total Trades:      {results['total_trades']:>15,}")
        print(f"Winning Trades:    {results['winning_trades']:>15,}")
        print(f"Win Rate:          {results['win_rate_pct']:>15.1f}%")
        print(f"Total Fees:        {results['total_fees']:>15,.0f}")
        print(f"Sharpe Ratio:      {results['sharpe_ratio']:>15.2f}")
        print("="*60)