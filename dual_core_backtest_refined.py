"""
Refined Dual-Core Portfolio Backtest
BTC EMA (20, 80) + SOL 15% weight with enhanced parameters
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

# Import the dual-core engines
from trend_follow import BTCTrendEngine
from sol_range_reversion import SOLRangeMeanReversionEngine
from dual_portfolio_manager import DualPortfolioManager
from regime_detection import RegimeDetector
from metrics_helper import (
    compute_total_return, 
    compute_max_drawdown, 
    compute_sharpe_ratio,
    debug_equity_series,
    validate_metrics_computation
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dual_core_refined.log')
    ]
)
logger = logging.getLogger('RefinedDualCore')

class RefinedDualCoreBacktester:
    def __init__(self, debug_mode=False):
        self.config = self._get_refined_config()
        
        # Debug settings
        self.debug_mode = debug_mode
        self.debug_logs = []
        
        # Initialize engines with refined parameters
        self.btc_engine = BTCTrendEngine(self.config.get('btc_config', {}))
        self.sol_engine = SOLRangeMeanReversionEngine(self.config.get('sol_config', {}))
        self.portfolio_manager = DualPortfolioManager(self.config.get('portfolio_config', {}))
        self.regime_detector = RegimeDetector()
        
        # Performance tracking
        self.results = {
            'btc_equity': [],
            'sol_equity': [],
            'portfolio_equity': [],
            'btc_exposures': [],
            'sol_exposures': [],
            'btc_trades': 0,
            'sol_trades': 0,
            'regime_history': [],
            'debug_data': []
        }
        
        self.initial_capital = 10_000_000  # ₩10M
        self.btc_capital = self.initial_capital * 0.85  # 85% to BTC
        self.sol_capital = self.initial_capital * 0.15  # 15% to SOL
        
        logger.info("🚀 Refined Dual-Core Backtester initialized")
        logger.info(f"Debug Mode: {self.debug_mode}")
        logger.info(f"BTC Engine: {self.btc_engine.get_strategy_info()['name']}")
        logger.info(f"SOL Engine: {self.sol_engine.get_strategy_info()['name']}")
        logger.info(f"Portfolio Allocation: 85% BTC / 15% SOL")
    
    def _get_refined_config(self):
        """Refined configuration with improved parameters"""
        return {
            'btc_config': {
                'ema_fast': 20,          # Improved: 15 → 20
                'ema_slow': 80,          # Improved: 60 → 80
                'volume_threshold': 1.15, # Improved: 1.20 → 1.15
                'max_exposure': 100.0,
                'fee_rate': 0.0005,      # New: 0.05% fee
                'slippage_rate': 0.0002  # New: 0.02% slippage
            },
            'sol_config': {
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.0,
                'max_exposure': 45.0,
                'signal_strength': 10.0,
                'fee_rate': 0.0005,      # New: 0.05% fee
                'slippage_rate': 0.0002  # New: 0.02% slippage
            },
            'portfolio_config': {
                'btc_weight': 0.85,      # Improved: 0.8 → 0.85
                'sol_weight': 0.15,      # Improved: 0.2 → 0.15
                'rebalance_hours': 8,    # Improved: 4 → 8 hours
                'adaptive_weighting': True,
                'max_weight_deviation': 0.10  # Tightened: 0.15 → 0.10
            }
        }
    
    def load_data(self, btc_file='data/ohlcv/KRW-BTC_240m.csv', sol_file='data/ohlcv/KRW-SOL_240m.csv'):
        """Load and align BTC and SOL data"""
        logger.info("Loading market data...")
        
        # Load BTC data
        btc_df = pd.read_csv(btc_file)
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
        logger.info(f"BTC data: {len(btc_df)} bars from {btc_df['timestamp'].iloc[0]}")
        
        # Load SOL data
        sol_df = pd.read_csv(sol_file)
        sol_df['timestamp'] = pd.to_datetime(sol_df['timestamp'])
        logger.info(f"SOL data: {len(sol_df)} bars from {sol_df['timestamp'].iloc[0]}")
        
        # Filter BTC data to match SOL start date
        sol_start = sol_df['timestamp'].iloc[0]
        btc_df = btc_df[btc_df['timestamp'] >= sol_start].reset_index(drop=True)
        logger.info(f"BTC data filtered from {sol_start}: {len(btc_df)} bars")
        
        # Resample to 4H intervals for alignment
        btc_df = btc_df.set_index('timestamp').resample('4h').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        sol_df = sol_df.set_index('timestamp').resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        # Find common time range
        common_start = max(btc_df['timestamp'].min(), sol_df['timestamp'].min())
        common_end = min(btc_df['timestamp'].max(), sol_df['timestamp'].max())
        
        btc_df = btc_df[(btc_df['timestamp'] >= common_start) & 
                        (btc_df['timestamp'] <= common_end)].reset_index(drop=True)
        sol_df = sol_df[(sol_df['timestamp'] >= common_start) & 
                        (sol_df['timestamp'] <= common_end)].reset_index(drop=True)
        
        logger.info(f"Data loaded: {len(btc_df)} aligned 4-hour bars")
        logger.info(f"Period: {common_start} to {common_end}")
        
        # Merge data
        data = pd.merge(btc_df.add_suffix('_btc'), sol_df.add_suffix('_sol'), 
                       left_on='timestamp_btc', right_on='timestamp_sol', how='inner')
        data['timestamp'] = data['timestamp_btc']
        data = data.drop(['timestamp_btc', 'timestamp_sol'], axis=1)
        
        return data
    
    def add_technical_indicators(self, btc_data, sol_data):
        """Add technical indicators to both datasets"""
        logger.info("Calculating technical indicators...")
        
        # BTC indicators
        btc_data['ema_fast'] = btc_data['close'].ewm(span=self.config['btc_config']['ema_fast']).mean()
        btc_data['ema_slow'] = btc_data['close'].ewm(span=self.config['btc_config']['ema_slow']).mean()
        btc_data['volume_ma'] = btc_data['volume'].rolling(20).mean()
        
        # SOL indicators  
        sol_data['rsi'] = self._calculate_rsi(sol_data['close'], self.config['sol_config']['rsi_period'])
        
        # Bollinger Bands for SOL
        bb_period = self.config['sol_config']['bb_period']
        bb_std = self.config['sol_config']['bb_std']
        sol_data['bb_middle'] = sol_data['close'].rolling(bb_period).mean()
        bb_std_dev = sol_data['close'].rolling(bb_period).std()
        sol_data['bb_upper'] = sol_data['bb_middle'] + (bb_std_dev * bb_std)
        sol_data['bb_lower'] = sol_data['bb_middle'] - (bb_std_dev * bb_std)
        
        logger.info("Technical indicators calculated")
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run_backtest(self, btc_data, sol_data):
        """Execute refined dual-core backtest"""
        logger.info("Starting refined dual-core backtest...")
        
        # Initialize equity tracking
        btc_equity = [self.btc_capital]
        sol_equity = [self.sol_capital] 
        portfolio_equity = [self.initial_capital]
        btc_exposures = [0.0]
        sol_exposures = [0.0]
        
        btc_position = 0.0
        sol_position = 0.0
        
        last_rebalance = 0
        rebalance_interval = self.config['portfolio_config']['rebalance_hours'] * 6  # Convert to 4H bars
        
        # Track costs
        total_fees = 0.0
        total_slippage = 0.0
        
        for i in range(1, len(btc_data)):
            current_btc = btc_data.iloc[i]
            current_sol = sol_data.iloc[i]
            
            # BTC trend strategy
            btc_signal = 0.0
            if (current_btc['ema_fast'] > current_btc['ema_slow'] and 
                current_btc['volume'] > current_btc['volume_ma'] * self.config['btc_config']['volume_threshold']):
                btc_signal = 1.0
            elif current_btc['ema_fast'] < current_btc['ema_slow']:
                btc_signal = 0.0
            else:
                btc_signal = btc_position  # Hold current position
            
            # SOL range mean reversion strategy
            sol_signal = 0.0
            if not pd.isna(current_sol['rsi']):
                if (current_sol['close'] <= current_sol['bb_lower'] and current_sol['rsi'] < 30):
                    sol_signal = 0.45  # 45% max exposure
                elif (current_sol['close'] >= current_sol['bb_upper'] and current_sol['rsi'] > 70):
                    sol_signal = 0.0
                else:
                    sol_signal = sol_position  # Hold current position
            
            # Calculate transaction costs
            btc_trade_cost = 0.0
            sol_trade_cost = 0.0
            
            # BTC position changes
            if abs(btc_signal - btc_position) > 0.01:
                trade_amount = abs(btc_signal - btc_position) * btc_equity[-1]
                btc_trade_cost = trade_amount * (self.config['btc_config']['fee_rate'] + 
                                               self.config['btc_config']['slippage_rate'])
                total_fees += trade_amount * self.config['btc_config']['fee_rate']
                total_slippage += trade_amount * self.config['btc_config']['slippage_rate']
                self.results['btc_trades'] += 1
            
            # SOL position changes  
            if abs(sol_signal - sol_position) > 0.01:
                trade_amount = abs(sol_signal - sol_position) * sol_equity[-1]
                sol_trade_cost = trade_amount * (self.config['sol_config']['fee_rate'] + 
                                               self.config['sol_config']['slippage_rate'])
                total_fees += trade_amount * self.config['sol_config']['fee_rate']
                total_slippage += trade_amount * self.config['sol_config']['slippage_rate']
                self.results['sol_trades'] += 1
            
            # Update positions
            btc_position = btc_signal
            sol_position = sol_signal
            
            # Calculate returns
            btc_return = (current_btc['close'] / btc_data.iloc[i-1]['close']) - 1
            sol_return = (current_sol['close'] / sol_data.iloc[i-1]['close']) - 1
            
            # Update equity with position-weighted returns and costs
            new_btc_equity = btc_equity[-1] * (1 + btc_return * btc_position) - btc_trade_cost
            new_sol_equity = sol_equity[-1] * (1 + sol_return * sol_position) - sol_trade_cost
            
            btc_equity.append(max(new_btc_equity, 0))
            sol_equity.append(max(new_sol_equity, 0))
            
            # Rebalancing every 8 hours
            if i - last_rebalance >= rebalance_interval:
                total_equity = btc_equity[-1] + sol_equity[-1]
                target_btc = total_equity * self.config['portfolio_config']['btc_weight']
                target_sol = total_equity * self.config['portfolio_config']['sol_weight']
                
                # Rebalancing costs
                rebalance_amount = abs(btc_equity[-1] - target_btc)
                rebalance_cost = rebalance_amount * 0.0001  # 0.01% rebalancing cost
                total_fees += rebalance_cost
                
                btc_equity[-1] = target_btc - rebalance_cost/2
                sol_equity[-1] = target_sol - rebalance_cost/2
                last_rebalance = i
            
            portfolio_equity.append(btc_equity[-1] + sol_equity[-1])
            btc_exposures.append(btc_position)
            sol_exposures.append(sol_position)
            
            # Weekly progress
            if i % (7 * 6) == 0:  # Every 7 days * 6 bars per day (4H bars)
                week = i // (7 * 6)
                portfolio_return = (portfolio_equity[-1] / self.initial_capital - 1) * 100
                logger.info(f"Week {week}: Portfolio KRW{portfolio_equity[-1]:,.0f} (+{portfolio_return:.1f}%) | "
                          f"BTC: {btc_position*100:.1f}% | SOL: {sol_position*100:.1f}%")
        
        logger.info(f"Backtest completed")
        logger.info(f"Total fees paid: KRW{total_fees:,.0f}")
        logger.info(f"Total slippage: KRW{total_slippage:,.0f}")
        
        # Store results
        self.results.update({
            'btc_equity': btc_equity,
            'sol_equity': sol_equity,
            'portfolio_equity': portfolio_equity,
            'btc_exposures': btc_exposures,
            'sol_exposures': sol_exposures,
            'total_fees': total_fees,
            'total_slippage': total_slippage
        })
        
        return self.results
    
    def calculate_metrics(self, results):
        """Calculate performance metrics"""
        portfolio_equity = results['portfolio_equity']
        btc_equity = results['btc_equity']
        sol_equity = results['sol_equity']
        
        # Portfolio metrics
        total_return = (portfolio_equity[-1] / self.initial_capital - 1) * 100
        max_dd = compute_max_drawdown(portfolio_equity)
        sharpe = compute_sharpe_ratio(portfolio_equity)
        
        # Individual metrics
        btc_return = (btc_equity[-1] / self.btc_capital - 1) * 100
        btc_max_dd = compute_max_drawdown(btc_equity)
        btc_sharpe = compute_sharpe_ratio(btc_equity)
        
        sol_return = (sol_equity[-1] / self.sol_capital - 1) * 100
        sol_max_dd = compute_max_drawdown(sol_equity)
        sol_sharpe = compute_sharpe_ratio(sol_equity)
        
        # Trading metrics
        total_trades = results['btc_trades'] + results['sol_trades']
        avg_btc_exposure = np.mean(results['btc_exposures']) * 100
        avg_sol_exposure = np.mean(results['sol_exposures']) * 100
        
        return {
            'portfolio_return': total_return,
            'portfolio_max_dd': max_dd,
            'portfolio_sharpe': sharpe,
            'btc_return': btc_return,
            'btc_max_dd': btc_max_dd,
            'btc_sharpe': btc_sharpe,
            'sol_return': sol_return,
            'sol_max_dd': sol_max_dd,
            'sol_sharpe': sol_sharpe,
            'total_trades': total_trades,
            'btc_trades': results['btc_trades'],
            'sol_trades': results['sol_trades'],
            'avg_btc_exposure': avg_btc_exposure,
            'avg_sol_exposure': avg_sol_exposure,
            'total_fees': results.get('total_fees', 0),
            'total_slippage': results.get('total_slippage', 0)
        }
    
    def print_results(self, metrics):
        """Print formatted results"""
        print("\n" + "="*80)
        print("🚀 REFINED DUAL-CORE STRATEGY RESULTS")
        print("="*80)
        
        print("\n🔧 ENHANCED CONFIGURATION:")
        print("-"*50)
        print("BTC Engine: Trend Following")
        print(f"   EMA: ({self.config['btc_config']['ema_fast']}, {self.config['btc_config']['ema_slow']})")
        print(f"   Volume Threshold: {self.config['btc_config']['volume_threshold']:.2f}×")
        print(f"   Fee Rate: {self.config['btc_config']['fee_rate']:.3f}%")
        print(f"   Slippage: {self.config['btc_config']['slippage_rate']:.3f}%")
        print()
        print("SOL Engine: Range Mean Reversion") 
        print(f"   RSI Period: {self.config['sol_config']['rsi_period']}")
        print(f"   Bollinger Bands: ({self.config['sol_config']['bb_period']}, {self.config['sol_config']['bb_std']:.1f}σ)")
        print(f"   Max Exposure: {self.config['sol_config']['max_exposure']}%")
        print(f"   Fee Rate: {self.config['sol_config']['fee_rate']:.3f}%")
        print(f"   Slippage: {self.config['sol_config']['slippage_rate']:.3f}%")
        print()
        print(f"Portfolio Allocation: {self.config['portfolio_config']['btc_weight']*100:.0f}% BTC / {self.config['portfolio_config']['sol_weight']*100:.0f}% SOL")
        print(f"Rebalance Frequency: Every {self.config['portfolio_config']['rebalance_hours']} hours")
        
        print("\n📈 INDIVIDUAL ENGINE PERFORMANCE:")
        print("-"*50)
        print(f"[BTC Trend Following]")
        print(f"   Return: {metrics['btc_return']:+.2f}%")
        print(f"   Max DD: {metrics['btc_max_dd']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['btc_sharpe']:.3f}")
        print(f"   Avg Exposure: {metrics['avg_btc_exposure']:.1f}%")
        print(f"   Trades: {metrics['btc_trades']}")
        print()
        print(f"[SOL Range Mean Reversion]")
        print(f"   Return: {metrics['sol_return']:+.2f}%")
        print(f"   Max DD: {metrics['sol_max_dd']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sol_sharpe']:.3f}")
        print(f"   Avg Exposure: {metrics['avg_sol_exposure']:.1f}%")
        print(f"   Trades: {metrics['sol_trades']}")
        
        print("\n🎯 COMBINED PORTFOLIO PERFORMANCE:")
        print("-"*50)
        print(f"Return: {metrics['portfolio_return']:+.2f}%")
        print(f"Max DD: {metrics['portfolio_max_dd']:.2f}%")
        print(f"Sharpe Ratio: {metrics['portfolio_sharpe']:.3f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Total Fees: KRW{metrics['total_fees']:,.0f}")
        print(f"Total Slippage: KRW{metrics['total_slippage']:,.0f}")
        
        print("\n🎯 GOAL ASSESSMENT:")
        print("-"*50)
        return_goal = 37.0
        mdd_goal = 14.0
        sharpe_goal = 1.4
        
        return_check = "✅ ACHIEVED" if metrics['portfolio_return'] >= return_goal else "❌ NOT ACHIEVED"
        mdd_check = "✅ ACHIEVED" if metrics['portfolio_max_dd'] <= mdd_goal else "❌ NOT ACHIEVED"
        sharpe_check = "✅ ACHIEVED" if metrics['portfolio_sharpe'] >= sharpe_goal else "❌ NOT ACHIEVED"
        
        print(f"Return Goal (≥{return_goal}%): {return_check}")
        print(f"   Target: ≥{return_goal}% | Actual: {metrics['portfolio_return']:.2f}%")
        print()
        print(f"Risk Goal (≤{mdd_goal}% MDD): {mdd_check}")
        print(f"   Target: ≤{mdd_goal}% | Actual: {metrics['portfolio_max_dd']:.2f}%")
        print()
        print(f"Sharpe Goal (≥{sharpe_goal}): {sharpe_check}")
        print(f"   Target: ≥{sharpe_goal} | Actual: {metrics['portfolio_sharpe']:.3f}")
        
        # Overall assessment
        goals_met = sum([
            metrics['portfolio_return'] >= return_goal,
            metrics['portfolio_max_dd'] <= mdd_goal, 
            metrics['portfolio_sharpe'] >= sharpe_goal
        ])
        
        if goals_met == 3:
            overall = "✅ ALL GOALS ACHIEVED"
        elif goals_met == 2:
            overall = "✅ MOSTLY SUCCESSFUL" 
        else:
            overall = "❌ NEEDS IMPROVEMENT"
            
        print(f"\nOverall Success: {overall}")
        print("="*80)

def main():
    """Run refined dual-core backtest"""
    backtester = RefinedDualCoreBacktester(debug_mode=False)
    
    # Load data
    data = backtester.load_data()
    
    # Prepare BTC and SOL data
    btc_data = data[['timestamp', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']].copy()
    btc_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    sol_data = data[['timestamp', 'open_sol', 'high_sol', 'low_sol', 'close_sol', 'volume_sol']].copy()
    sol_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Add technical indicators
    backtester.add_technical_indicators(btc_data, sol_data)
    
    # Run backtest
    results = backtester.run_backtest(btc_data, sol_data)
    
    # Calculate and display metrics
    metrics = backtester.calculate_metrics(results)
    backtester.print_results(metrics)

if __name__ == "__main__":
    main()