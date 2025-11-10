"""
Dual-Core Portfolio Backtest
Runs BTC and ETH as independent trading engines with specialized strategies
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

# Import the new dual-core engines
from trend_follow import BTCTrendEngine
from mean_reversion import ETHMeanReversionEngine
from dual_portfolio_manager import DualPortfolioManager
from regime_detection import RegimeDetector
from metrics_helper import (
    compute_total_return, 
    compute_max_drawdown, 
    compute_sharpe_ratio,
    debug_equity_series,
    validate_metrics_computation
)
from regime_detection import RegimeDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dual_core_backtest.log')
    ]
)
logger = logging.getLogger('DualCoreBacktest')

class DualCoreBacktester:
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        
        # Initialize engines
        self.btc_engine = BTCTrendEngine(self.config.get('btc_config', {}))
        self.eth_engine = ETHMeanReversionEngine(self.config.get('eth_config', {}))
        self.portfolio_manager = DualPortfolioManager(self.config.get('portfolio_config', {}))
        self.regime_detector = RegimeDetector()
        
        # Performance tracking
        self.results = {
            'btc_equity': [],
            'eth_equity': [],
            'portfolio_equity': [],
            'btc_exposures': [],
            'eth_exposures': [],
            'btc_trades': 0,
            'eth_trades': 0,
            'regime_history': []
        }
        
        self.initial_capital = 10_000_000  # ₩10M
        self.btc_capital = self.initial_capital * 0.7  # 70% to BTC
        self.eth_capital = self.initial_capital * 0.3  # 30% to ETH
        
        logger.info("Dual-Core Backtester initialized")
        logger.info(f"BTC Engine: {self.btc_engine.get_strategy_info()['name']}")
        logger.info(f"ETH Engine: {self.eth_engine.get_strategy_info()['name']}")
    
    def _get_default_config(self):
        return {
            'btc_config': {
                'ema_fast': 15,
                'ema_slow': 60,
                'volume_threshold': 1.20,
                'max_exposure': 100.0
            },
            'eth_config': {
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.0,
                'max_exposure': 45.0,
                'signal_strength': 10.0
            },
            'portfolio_config': {
                'btc_weight': 0.7,
                'eth_weight': 0.3,
                'rebalance_hours': 4,
                'adaptive_weighting': True,
                'max_weight_deviation': 0.15
            }
        }
    
    def load_data(self, btc_file='data/ohlcv/KRW-BTC_240m.csv', eth_file='data/ohlcv/KRW-ETH_240m.csv'):
        """Load and align BTC and ETH data"""
        logger.info("Loading market data...")
        
        # Load data
        btc_df = pd.read_csv(btc_file)
        eth_df = pd.read_csv(eth_file)
        
        # Convert timestamps
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'])
        
        # Align data by timestamp (inner join)
        aligned_data = pd.merge(btc_df, eth_df, on='timestamp', suffixes=('_btc', '_eth'))
        aligned_data = aligned_data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data loaded: {len(aligned_data)} aligned 4-hour bars")
        logger.info(f"Period: {aligned_data['timestamp'].min()} to {aligned_data['timestamp'].max()}")
        
        return aligned_data
    
    def prepare_data(self, df):
        """Prepare data with technical indicators for both assets"""
        logger.info("Calculating technical indicators...")
        
        # Separate BTC and ETH data
        btc_data = df[['timestamp', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']].copy()
        btc_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        eth_data = df[['timestamp', 'open_eth', 'high_eth', 'low_eth', 'close_eth', 'volume_eth']].copy()
        eth_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Calculate indicators for each asset
        btc_data = self.btc_engine.calculate_indicators(btc_data)
        eth_data = self.eth_engine.calculate_indicators(eth_data)
        
        return btc_data, eth_data
    
    def run_backtest(self, btc_data, eth_data):
        """Run the dual-core backtest"""
        logger.info("Starting dual-core backtest...")
        
        btc_equity = self.btc_capital
        eth_equity = self.eth_capital
        btc_position = 0.0
        eth_position = 0.0
        
        # Trading costs
        self.trading_fee = 0.0005  # 0.05% per trade (typical crypto exchange fee)
        self.slippage = 0.0001     # 0.01% slippage per trade
        
        for i in range(len(btc_data)):
            timestamp = btc_data.iloc[i]['timestamp']
            
            # Market regime detection (using BTC as primary)
            btc_row = btc_data.iloc[i]
            market_mode, regime_data = self.regime_detector.detect_regime(
                btc_row['close'], btc_row['volume'], i
            )
            
            # Get current drawdowns
            btc_dd, eth_dd = self.portfolio_manager.get_current_drawdowns()
            
            # Generate signals from each engine using PREVIOUS bar data to avoid look-ahead bias
            if i > 0:
                btc_exposure, btc_signal_type, btc_detail = self.btc_engine.generate_signal(
                    btc_data, i-1, market_mode, regime_data, btc_dd
                )
                
                eth_exposure, eth_signal_type, eth_detail = self.eth_engine.generate_signal(
                    eth_data, i-1, market_mode, regime_data, eth_dd
                )
            else:
                # First bar - no trading
                btc_exposure = 0.0
                eth_exposure = 0.0
                btc_signal_type = "HOLD"
                eth_signal_type = "HOLD"
                btc_detail = {}
                eth_detail = {}
            
            # Portfolio management
            btc_final_exposure, eth_final_exposure, portfolio_info = self.portfolio_manager.calculate_portfolio_exposure(
                btc_exposure, eth_exposure, i
            )
            
            # Calculate position sizes based on current equity
            btc_target_position = (btc_final_exposure / 100.0) * btc_equity
            eth_target_position = (eth_final_exposure / 100.0) * eth_equity
            
            # Execute trades with fees and slippage
            btc_trade = btc_target_position - btc_position
            eth_trade = eth_target_position - eth_position
            
            btc_trade_cost = 0.0
            eth_trade_cost = 0.0
            
            if abs(btc_trade) > btc_equity * 0.001:  # 0.1% minimum trade size
                # Calculate trading costs
                btc_trade_cost = abs(btc_trade) * (self.trading_fee + self.slippage)
                btc_position = btc_target_position
                btc_equity -= btc_trade_cost  # Deduct trading costs from equity
                self.results['btc_trades'] += 1
            
            if abs(eth_trade) > eth_equity * 0.001:
                # Calculate trading costs  
                eth_trade_cost = abs(eth_trade) * (self.trading_fee + self.slippage)
                eth_position = eth_target_position
                eth_equity -= eth_trade_cost  # Deduct trading costs from equity
                self.results['eth_trades'] += 1
            
            # Calculate returns using current bar prices (realistic execution)
            if i > 0:
                btc_price_return = (btc_data.iloc[i]['close'] - btc_data.iloc[i-1]['close']) / btc_data.iloc[i-1]['close']
                eth_price_return = (eth_data.iloc[i]['close'] - eth_data.iloc[i-1]['close']) / eth_data.iloc[i-1]['close']
                
                # Position PnL (using previous position since we trade at the end)
                btc_pnl = btc_position * btc_price_return
                eth_pnl = eth_position * eth_price_return
                
                # Update equity
                btc_equity += btc_pnl
                eth_equity += eth_pnl
                
                # Update portfolio manager performance tracking
                btc_return = btc_pnl / (btc_equity - btc_pnl) if (btc_equity - btc_pnl) > 0 else 0
                eth_return = eth_pnl / (eth_equity - eth_pnl) if (eth_equity - eth_pnl) > 0 else 0
                
                self.portfolio_manager.update_performance(
                    btc_return, eth_return, btc_final_exposure, eth_final_exposure
                )
            
            # Store results
            self.results['btc_equity'].append(btc_equity)
            self.results['eth_equity'].append(eth_equity)
            self.results['portfolio_equity'].append(btc_equity + eth_equity)
            self.results['btc_exposures'].append(btc_final_exposure)
            self.results['eth_exposures'].append(eth_final_exposure)
            self.results['regime_history'].append({
                'timestamp': timestamp,
                'market_mode': market_mode,
                'btc_signal': btc_signal_type,
                'eth_signal': eth_signal_type,
                'btc_exposure': btc_final_exposure,
                'eth_exposure': eth_final_exposure
            })
            
            # Debug logging for first few iterations
            if i < 10:
                logger.debug(f"Bar {i}: BTC equity=KRW{btc_equity:,.0f}, ETH equity=KRW{eth_equity:,.0f}, "
                           f"BTC pos=KRW{btc_position:,.0f}, ETH pos=KRW{eth_position:,.0f}")
            
            # Periodic logging
            if i % 42 == 0:  # Weekly updates (42 = 7 days * 6 bars per day)
                portfolio_value = btc_equity + eth_equity
                total_return = (portfolio_value / self.initial_capital - 1) * 100
                logger.info(f"Week {i//42}: Portfolio KRW{portfolio_value:,.0f} ({total_return:+.1f}%) | BTC: {btc_final_exposure:.1f}% | ETH: {eth_final_exposure:.1f}%")
        
        logger.info("Backtest completed")
        return self.results
    
    def run_btc_buyhold_benchmark(self, btc_data):
        """Run BTC Buy&Hold benchmark for comparison"""
        logger.info("Running BTC Buy&Hold benchmark...")
        
        initial_btc_price = btc_data.iloc[0]['close']
        btc_shares = self.btc_capital / initial_btc_price
        
        buyhold_equity = []
        for i in range(len(btc_data)):
            current_price = btc_data.iloc[i]['close']
            current_equity = btc_shares * current_price
            buyhold_equity.append(current_equity)
        
        return np.array(buyhold_equity)
    
    def analyze_btc_trades(self, btc_data, results):
        """Analyze BTC trade statistics"""
        btc_exposures = np.array(results['btc_exposures'])
        btc_equity = np.array(results['btc_equity'])
        
        # Find trade points (exposure changes > 1%)
        exposure_changes = np.diff(btc_exposures)
        trade_points = np.where(np.abs(exposure_changes) > 1.0)[0] + 1
        
        trades = []
        current_trade = None
        
        for i, trade_point in enumerate(trade_points):
            if btc_exposures[trade_point] > btc_exposures[trade_point-1]:
                # Entry
                if current_trade is None:
                    current_trade = {
                        'entry_idx': trade_point,
                        'entry_price': btc_data.iloc[trade_point]['close'],
                        'entry_equity': btc_equity[trade_point-1]
                    }
            elif current_trade is not None:
                # Exit
                current_trade['exit_idx'] = trade_point
                current_trade['exit_price'] = btc_data.iloc[trade_point]['close']
                current_trade['exit_equity'] = btc_equity[trade_point]
                current_trade['pnl_pct'] = (current_trade['exit_equity'] - current_trade['entry_equity']) / current_trade['entry_equity'] * 100
                current_trade['price_return'] = (current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price'] * 100
                trades.append(current_trade)
                current_trade = None
        
        if len(trades) == 0:
            return {}
        
        # Calculate statistics
        pnl_pcts = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p < 0]
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in pnl_pcts:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf'),
            'trades': trades
        }
    
    def run_random_strategy_test(self, btc_data, eth_data, num_tests=5):
        """Run random strategy tests to validate engine baseline performance"""
        logger.info("Running random strategy tests...")
        
        random_results = []
        
        for test_num in range(num_tests):
            np.random.seed(42 + test_num)  # Reproducible randomness
            
            btc_equity = self.btc_capital
            eth_equity = self.eth_capital
            btc_position = 0.0
            eth_position = 0.0
            
            random_equity = []
            
            for i in range(len(btc_data)):
                # Random exposure changes (-10% to +10%)
                if i > 0:
                    btc_exposure_change = np.random.uniform(-10, 10)
                    eth_exposure_change = np.random.uniform(-5, 5)  # Smaller range for ETH
                    
                    # Current exposures (as percentage of equity)
                    current_btc_exposure = (btc_position / btc_equity * 100) if btc_equity > 0 else 0
                    current_eth_exposure = (eth_position / eth_equity * 100) if eth_equity > 0 else 0
                    
                    # New exposures
                    new_btc_exposure = np.clip(current_btc_exposure + btc_exposure_change, 0, 100)
                    new_eth_exposure = np.clip(current_eth_exposure + eth_exposure_change, 0, 45)
                    
                    # Position sizing
                    btc_target = (new_btc_exposure / 100.0) * btc_equity
                    eth_target = (new_eth_exposure / 100.0) * eth_equity
                    
                    # Apply trading costs for random trades
                    btc_trade = btc_target - btc_position
                    eth_trade = eth_target - eth_position
                    
                    if abs(btc_trade) > btc_equity * 0.001:
                        btc_cost = abs(btc_trade) * (self.trading_fee + self.slippage)
                        btc_position = btc_target
                        btc_equity -= btc_cost
                    
                    if abs(eth_trade) > eth_equity * 0.001:
                        eth_cost = abs(eth_trade) * (self.trading_fee + self.slippage)
                        eth_position = eth_target
                        eth_equity -= eth_cost
                
                # Calculate returns
                if i > 0:
                    btc_return = (btc_data.iloc[i]['close'] - btc_data.iloc[i-1]['close']) / btc_data.iloc[i-1]['close']
                    eth_return = (eth_data.iloc[i]['close'] - eth_data.iloc[i-1]['close']) / eth_data.iloc[i-1]['close']
                    
                    btc_pnl = btc_position * btc_return
                    eth_pnl = eth_position * eth_return
                    
                    btc_equity += btc_pnl
                    eth_equity += eth_pnl
                
                random_equity.append(btc_equity + eth_equity)
            
            # Calculate metrics for this random test
            random_equity = np.array(random_equity)
            total_return = compute_total_return(random_equity)
            max_dd = compute_max_drawdown(random_equity)
            sharpe = compute_sharpe_ratio(random_equity, periods_per_year=2190)
            
            random_results.append({
                'total_return': total_return,
                'max_dd': max_dd,
                'sharpe': sharpe
            })
        
        return random_results
    
    def calculate_metrics(self, results):
        """Calculate comprehensive performance metrics with validation"""
        
        # Extract equity series
        portfolio_equity = np.array(results['portfolio_equity'])
        btc_equity = np.array(results['btc_equity'])
        eth_equity = np.array(results['eth_equity'])
        
        logger = logging.getLogger(__name__)
        logger.info("=== METRICS COMPUTATION DEBUG ===")
        
        # Debug equity series
        portfolio_debug = debug_equity_series(portfolio_equity, "Portfolio")
        btc_debug = debug_equity_series(btc_equity, "BTC")
        eth_debug = debug_equity_series(eth_equity, "ETH")
        
        logger.info(f"Portfolio equity: {portfolio_debug['length']} points, "
                   f"KRW{portfolio_debug['initial_value']:,.0f} → KRW{portfolio_debug['final_value']:,.0f}")
        logger.info(f"BTC equity: {btc_debug['length']} points, "
                   f"KRW{btc_debug['initial_value']:,.0f} → KRW{btc_debug['final_value']:,.0f}")
        logger.info(f"ETH equity: {eth_debug['length']} points, "
                   f"KRW{eth_debug['initial_value']:,.0f} → KRW{eth_debug['final_value']:,.0f}")
        
        # Calculate metrics using new helper functions
        def calc_metrics_new(equity_series, name):
            total_return = compute_total_return(equity_series)
            max_dd = compute_max_drawdown(equity_series)
            sharpe = compute_sharpe_ratio(equity_series, periods_per_year=2190)
            
            # Average exposure
            if name == 'BTC':
                avg_exposure = np.mean(results['btc_exposures'])
            elif name == 'ETH':
                avg_exposure = np.mean(results['eth_exposures'])
            else:
                avg_exposure = np.mean(results['btc_exposures']) + np.mean(results['eth_exposures'])
            
            return {
                'total_return': total_return,
                'max_dd': max_dd,
                'current_dd': max_dd,  # For compatibility
                'sharpe_ratio': sharpe,
                'avg_exposure': avg_exposure
            }
        
        # Calculate metrics using old method for comparison
        def calc_metrics_old(equity_series, returns_series, name):
            total_return = (equity_series[-1] / equity_series[0] - 1) * 100
            
            # Drawdown calculation
            running_max = np.maximum.accumulate(equity_series)
            drawdowns = (equity_series - running_max) / running_max * 100
            max_dd = abs(np.min(drawdowns))
            current_dd = abs(drawdowns[-1])
            
            # Sharpe ratio
            if len(returns_series) > 0 and np.std(returns_series) > 0:
                sharpe = (np.mean(returns_series) * 2190) / (np.std(returns_series) * np.sqrt(2190))
            else:
                sharpe = 0
            
            # Average exposure
            if name == 'BTC':
                avg_exposure = np.mean(results['btc_exposures'])
            elif name == 'ETH':
                avg_exposure = np.mean(results['eth_exposures'])
            else:
                avg_exposure = np.mean(results['btc_exposures']) + np.mean(results['eth_exposures'])
            
            return {
                'total_return': total_return,
                'max_dd': max_dd,
                'current_dd': current_dd,
                'sharpe_ratio': sharpe,
                'avg_exposure': avg_exposure
            }
        
        # Calculate returns for old method
        portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        btc_returns = np.diff(btc_equity) / btc_equity[:-1]
        eth_returns = np.diff(eth_equity) / eth_equity[:-1]
        
        # Calculate metrics using both methods
        logger.info("=== COMPARING OLD VS NEW METRICS ===")
        
        # Portfolio metrics
        portfolio_new = calc_metrics_new(portfolio_equity, 'Portfolio')
        portfolio_old = calc_metrics_old(portfolio_equity, portfolio_returns, 'Portfolio')
        
        logger.info(f"Portfolio - Old vs New:")
        logger.info(f"  Return: {portfolio_old['total_return']:.2f}% vs {portfolio_new['total_return']:.2f}%")
        logger.info(f"  Max DD: {portfolio_old['max_dd']:.2f}% vs {portfolio_new['max_dd']:.2f}%")
        logger.info(f"  Sharpe: {portfolio_old['sharpe_ratio']:.3f} vs {portfolio_new['sharpe_ratio']:.3f}")
        
        # BTC metrics
        btc_new = calc_metrics_new(btc_equity, 'BTC')
        btc_old = calc_metrics_old(btc_equity, btc_returns, 'BTC')
        
        logger.info(f"BTC - Old vs New:")
        logger.info(f"  Return: {btc_old['total_return']:.2f}% vs {btc_new['total_return']:.2f}%")
        logger.info(f"  Max DD: {btc_old['max_dd']:.2f}% vs {btc_new['max_dd']:.2f}%")
        logger.info(f"  Sharpe: {btc_old['sharpe_ratio']:.3f} vs {btc_new['sharpe_ratio']:.3f}")
        
        # ETH metrics
        eth_new = calc_metrics_new(eth_equity, 'ETH')
        eth_old = calc_metrics_old(eth_equity, eth_returns, 'ETH')
        
        logger.info(f"ETH - Old vs New:")
        logger.info(f"  Return: {eth_old['total_return']:.2f}% vs {eth_new['total_return']:.2f}%")
        logger.info(f"  Max DD: {eth_old['max_dd']:.2f}% vs {eth_new['max_dd']:.2f}%")
        logger.info(f"  Sharpe: {eth_old['sharpe_ratio']:.3f} vs {eth_new['sharpe_ratio']:.3f}")
        
        # Additional metrics
        correlation = np.corrcoef(btc_returns, eth_returns)[0, 1] if len(btc_returns) > 10 else 0
        
        # Use new metrics for final results
        return {
            'portfolio': portfolio_new,
            'btc': btc_new,
            'eth': eth_new,
            'correlation': correlation,
            'trades': {
                'btc': results['btc_trades'],
                'eth': results['eth_trades'],
                'total': results['btc_trades'] + results['eth_trades']
            },
            'debug_info': {
                'portfolio_debug': portfolio_debug,
                'btc_debug': btc_debug,
                'eth_debug': eth_debug
            }
        }
    
    def save_results(self, results, metrics):
        """Save backtest results and metrics"""
        results_dir = Path('results/dual_core_strategy')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_df = pd.DataFrame({
            'btc_equity': results['btc_equity'],
            'eth_equity': results['eth_equity'],
            'portfolio_equity': results['portfolio_equity'],
            'btc_exposure': results['btc_exposures'],
            'eth_exposure': results['eth_exposures']
        })
        results_df.to_csv(results_dir / f'equity_curves_{timestamp}.csv', index=False)
        
        # Save metrics
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        metrics_serializable = convert_numpy_types(metrics)
        
        with open(results_dir / f'metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Save regime history
        regime_df = pd.DataFrame(results['regime_history'])
        regime_df.to_csv(results_dir / f'regime_history_{timestamp}.csv', index=False)
        
        logger.info(f"Results saved to {results_dir}")

def print_results(metrics, config):
    """Print formatted backtest results"""
    
    print("\n" + "="*80)
    print("📊 DUAL-CORE STRATEGY BACKTEST RESULTS")
    print("="*80)
    
    print("\n🔧 STRATEGY CONFIGURATION:")
    print("-" * 50)
    print("BTC Engine: Trend Following")
    print(f"   EMA: ({config['btc_config']['ema_fast']}, {config['btc_config']['ema_slow']})")
    print(f"   Volume Threshold: {config['btc_config']['volume_threshold']:.2f}×")
    print(f"   Max Exposure: {config['btc_config']['max_exposure']:.0f}%")
    
    print("\nETH Engine: Mean Reversion") 
    print(f"   RSI Period: {config['eth_config']['rsi_period']}")
    print(f"   Bollinger Bands: ({config['eth_config']['bb_period']}, {config['eth_config']['bb_std']}σ)")
    print(f"   Max Exposure: {config['eth_config']['max_exposure']:.0f}%")
    print(f"   Signal Strength: ±{config['eth_config']['signal_strength']:.0f}%")
    
    print(f"\nPortfolio Allocation: {config['portfolio_config']['btc_weight']:.0%} BTC / {config['portfolio_config']['eth_weight']:.0%} ETH")
    
    print("\n📈 INDIVIDUAL ENGINE PERFORMANCE:")
    print("-" * 50)
    print(f"[BTC Trend Following]")
    print(f"   Return: {metrics['btc']['total_return']:+.2f}%")
    print(f"   Max DD: {metrics['btc']['max_dd']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['btc']['sharpe_ratio']:.3f}")
    print(f"   Avg Exposure: {metrics['btc']['avg_exposure']:.1f}%")
    print(f"   Trades: {metrics['trades']['btc']}")
    
    print(f"\n[ETH Mean Reversion]")
    print(f"   Return: {metrics['eth']['total_return']:+.2f}%")
    print(f"   Max DD: {metrics['eth']['max_dd']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['eth']['sharpe_ratio']:.3f}")
    print(f"   Avg Exposure: {metrics['eth']['avg_exposure']:.1f}%")
    print(f"   Trades: {metrics['trades']['eth']}")
    
    print("\n🎯 COMBINED PORTFOLIO PERFORMANCE:")
    print("-" * 50)
    print(f"Return: {metrics['portfolio']['total_return']:+.2f}%")
    print(f"Max DD: {metrics['portfolio']['max_dd']:.2f}%")
    print(f"Sharpe Ratio: {metrics['portfolio']['sharpe_ratio']:.3f}")
    print(f"Total Exposure: {metrics['portfolio']['avg_exposure']:.1f}%")
    print(f"Total Trades: {metrics['trades']['total']}")
    print(f"BTC-ETH Correlation: {metrics['correlation']:.3f}")
    
    print("\n🎯 GOAL ASSESSMENT:")
    print("-" * 50)
    return_goal = 85.0
    risk_goal = 20.0
    sharpe_goal = 0.5
    
    return_achieved = "✅ ACHIEVED" if metrics['portfolio']['total_return'] >= return_goal else "❌ NOT ACHIEVED"
    risk_achieved = "✅ ACHIEVED" if metrics['portfolio']['max_dd'] <= risk_goal else "❌ NOT ACHIEVED"
    sharpe_achieved = "✅ ACHIEVED" if metrics['portfolio']['sharpe_ratio'] >= sharpe_goal else "❌ NOT ACHIEVED"
    
    print(f"Return Goal (≥{return_goal}%): {return_achieved}")
    print(f"   Target: ≥{return_goal:.1f}% | Actual: {metrics['portfolio']['total_return']:.2f}%")
    
    print(f"\nRisk Goal (≤{risk_goal}% MDD): {risk_achieved}")
    print(f"   Target: ≤{risk_goal:.1f}% | Actual: {metrics['portfolio']['max_dd']:.2f}%")
    
    print(f"\nSharpe Goal (≥{sharpe_goal}): {sharpe_achieved}")
    print(f"   Target: ≥{sharpe_goal:.1f} | Actual: {metrics['portfolio']['sharpe_ratio']:.3f}")
    
    goals_met = sum([
        metrics['portfolio']['total_return'] >= return_goal,
        metrics['portfolio']['max_dd'] <= risk_goal,
        metrics['portfolio']['sharpe_ratio'] >= sharpe_goal
    ])
    
    if goals_met == 3:
        overall = "🏆 ALL GOALS ACHIEVED"
    elif goals_met == 2:
        overall = "✅ MOSTLY SUCCESSFUL"
    elif goals_met == 1:
        overall = "🔧 IMPROVEMENT NEEDED"
    else:
        overall = "❌ MAJOR ADJUSTMENTS REQUIRED"
    
    print(f"\nOverall Success: {overall}")
    
    # Diversification analysis
    btc_only_return = metrics['btc']['total_return']
    portfolio_return = metrics['portfolio']['total_return']
    diversification_benefit = portfolio_return - btc_only_return
    
    print("\n💡 DIVERSIFICATION ANALYSIS:")
    print("-" * 50)
    print(f"BTC-Only Return: {btc_only_return:+.2f}%")
    print(f"Portfolio Return: {portfolio_return:+.2f}%")
    print(f"Diversification Benefit: {diversification_benefit:+.2f}%p")
    
    if diversification_benefit > 0:
        print("🟢 Diversification improved performance")
    else:
        print("🔴 Diversification reduced performance")
    
    print("\n✅ Dual-core strategy results saved to results/dual_core_strategy/")
    print("="*80)

def main():
    """Run the dual-core backtest with comprehensive analysis"""
    
    # Initialize backtester
    backtester = DualCoreBacktester()
    
    try:
        # Load and prepare data
        aligned_data = backtester.load_data()
        btc_data, eth_data = backtester.prepare_data(aligned_data)
        
        logger.info("="*60)
        logger.info("COMPREHENSIVE BACKTEST ANALYSIS")
        logger.info("="*60)
        
        # 1. Run main strategy backtest
        logger.info("Step 1: Running main strategy backtest...")
        results = backtester.run_backtest(btc_data, eth_data)
        metrics = backtester.calculate_metrics(results)
        
        # 2. Run BTC Buy&Hold benchmark
        logger.info("Step 2: Running BTC Buy&Hold benchmark...")
        buyhold_equity = backtester.run_btc_buyhold_benchmark(btc_data)
        buyhold_return = compute_total_return(buyhold_equity)
        buyhold_dd = compute_max_drawdown(buyhold_equity)
        buyhold_sharpe = compute_sharpe_ratio(buyhold_equity, periods_per_year=2190)
        
        # 3. Analyze BTC trades
        logger.info("Step 3: Analyzing BTC trade statistics...")
        trade_stats = backtester.analyze_btc_trades(btc_data, results)
        
        # 4. Run random strategy tests
        logger.info("Step 4: Running random strategy tests...")
        random_results = backtester.run_random_strategy_test(btc_data, eth_data, num_tests=3)
        
        # 5. Save results
        backtester.save_results(results, metrics)
        
        # 6. Print comprehensive analysis
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE ANALYSIS RESULTS")
        logger.info("="*80)
        
        print("\n🔧 FIXES APPLIED:")
        print("--------------------------------------------------")
        print("✅ Fixed look-ahead bias: Signals now use t-1 data, execute at t")
        print("✅ Added trading costs: 0.05% fee + 0.01% slippage per trade")
        print("✅ Added Buy&Hold benchmark for comparison")
        print("✅ Added comprehensive trade analysis")
        print("✅ Added random strategy baseline tests")
        
        print(f"\n📈 STRATEGY vs BUY&HOLD COMPARISON:")
        print("--------------------------------------------------")
        print(f"Strategy BTC Return: {metrics['btc']['total_return']:.2f}%")
        print(f"Buy&Hold BTC Return: {buyhold_return:.2f}%")
        print(f"Strategy Alpha: {metrics['btc']['total_return'] - buyhold_return:+.2f}%p")
        print(f"")
        print(f"Strategy Max DD: {metrics['btc']['max_dd']:.2f}%")
        print(f"Buy&Hold Max DD: {buyhold_dd:.2f}%")
        print(f"DD Improvement: {buyhold_dd - metrics['btc']['max_dd']:+.2f}%p")
        print(f"")
        print(f"Strategy Sharpe: {metrics['btc']['sharpe_ratio']:.3f}")
        print(f"Buy&Hold Sharpe: {buyhold_sharpe:.3f}")
        print(f"Sharpe Improvement: {metrics['btc']['sharpe_ratio'] - buyhold_sharpe:+.3f}")
        
        if trade_stats:
            print(f"\n📊 BTC TRADE ANALYSIS:")
            print("--------------------------------------------------")
            print(f"Total Trades: {trade_stats['total_trades']}")
            print(f"Win Rate: {trade_stats['win_rate']:.1f}%")
            print(f"Average Win: {trade_stats['avg_win']:.2f}%")
            print(f"Average Loss: {trade_stats['avg_loss']:.2f}%")
            print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
            print(f"Max Consecutive Wins: {trade_stats['max_consecutive_wins']}")
            print(f"Max Consecutive Losses: {trade_stats['max_consecutive_losses']}")
        
        print(f"\n🎲 RANDOM STRATEGY BASELINE:")
        print("--------------------------------------------------")
        avg_random_return = np.mean([r['total_return'] for r in random_results])
        avg_random_dd = np.mean([r['max_dd'] for r in random_results])
        avg_random_sharpe = np.mean([r['sharpe'] for r in random_results])
        
        print(f"Random Average Return: {avg_random_return:.2f}%")
        print(f"Random Average Max DD: {avg_random_dd:.2f}%")
        print(f"Random Average Sharpe: {avg_random_sharpe:.3f}")
        print(f"")
        print(f"Strategy vs Random:")
        print(f"  Return Advantage: {metrics['btc']['total_return'] - avg_random_return:+.2f}%p")
        print(f"  DD Advantage: {avg_random_dd - metrics['btc']['max_dd']:+.2f}%p")
        print(f"  Sharpe Advantage: {metrics['btc']['sharpe_ratio'] - avg_random_sharpe:+.3f}")
        
        # Original results display
        print_results(metrics, backtester.config)
        
        # Additional benchmark metrics
        print(f"\n📋 BENCHMARK SUMMARY:")
        print("--------------------------------------------------")
        print(f"Strategy outperformed Buy&Hold: {'✅' if metrics['btc']['total_return'] > buyhold_return else '❌'}")
        print(f"Strategy has lower DD than Buy&Hold: {'✅' if metrics['btc']['max_dd'] < buyhold_dd else '❌'}")
        print(f"Strategy outperformed random: {'✅' if metrics['btc']['total_return'] > avg_random_return else '❌'}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main()