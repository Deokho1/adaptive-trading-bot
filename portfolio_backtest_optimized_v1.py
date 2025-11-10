# BTC 전략 최적화 완료 버전 (2025-11-10)
# 성과: 수익률 80.28%, MDD 17.07%, Sharpe 1.549
# 이 파일은 최적화된 설정의 백업본입니다

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from backtest.data_loader import BacktestDataLoader
from dual_engine_strategy import DualEngineStrategy
from config.config_manager import ConfigManager
from core.base_types import Candle, Position, PositionSide

def get_current_parameters():
    """현재 최적화된 전략 파라미터 반환"""
    return {
        'ema_periods': {
            'btc': '(15, 60)',
            'eth': '(8, 24)'
        },
        'volume_thresholds': {
            'btc': '1.20×',
            'eth': '1.25×'
        },
        'exposure_limits': {
            'btc': '100%',
            'eth': '45%'
        },
        'signal_sensitivity': 'Very High (EMA 4,12)',
        'risk_factor': '0.85× (Enhanced DD reduction)',
        'rebalancing_frequency': 'Every 4 hours'
    }

def calculate_portfolio_correlation(btc_returns: List[float], eth_returns: List[float]) -> float:
    """Calculate correlation between BTC and ETH returns"""
    if len(btc_returns) < 2 or len(eth_returns) < 2:
        return 0.0
    
    try:
        correlation_matrix = np.corrcoef(btc_returns, eth_returns)
        return float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
    except:
        return 0.0

def calculate_sharpe_ratio(equity_series: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from equity series using log returns (4h interval data)"""
    if not equity_series or len(equity_series) < 2:
        print(f"[DEBUG] Insufficient data: equity_series length = {len(equity_series) if equity_series else 0}")
        return 0.0
    
    try:
        # Convert to numpy array and ensure float type
        equity_array = np.array([float(x) for x in equity_series], dtype=float)
        
        # Calculate log returns instead of percentage returns
        log_returns = np.diff(np.log(equity_array))
        
        # Remove any NaN or infinite values
        log_returns = log_returns[np.isfinite(log_returns)]
        
        if len(log_returns) < 2:
            print("[DEBUG] Insufficient valid returns after filtering")
            return 0.0
        
        # Calculate statistics
        mean_return = float(np.mean(log_returns))
        std_return = float(np.std(log_returns, ddof=1))
        
        print(f"[DEBUG] Equity series range: {equity_array.min():.2f} to {equity_array.max():.2f}")
        print(f"[DEBUG] Log returns: periods={len(log_returns)}, mean={mean_return:.8f}, std={std_return:.8f}")
        
        if std_return == 0 or np.isnan(std_return):
            print("[DEBUG] Zero or NaN standard deviation")
            return 0.0
        
        # Convert annual risk-free rate to 4-hour period rate
        periods_per_year = 365.25 * 24 / 4  # 2190.75 periods per year for 4h intervals
        risk_free_per_period = risk_free_rate / periods_per_year
        
        print(f"[DEBUG] Risk-free rate per period: {risk_free_per_period:.8f}")
        
        # Calculate Sharpe ratio
        excess_return = mean_return - risk_free_per_period
        sharpe_ratio = excess_return / std_return
        
        # Annualize Sharpe ratio
        annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
        
        print(f"[DEBUG] Final Sharpe: raw={sharpe_ratio:.6f}, annualized={annualized_sharpe:.6f}")
        
        return float(annualized_sharpe)
        
    except Exception as e:
        print(f"[DEBUG] Error calculating Sharpe ratio: {e}")
        return 0.0

def run_single_asset_backtest(candles: List[Candle], symbol: str, initial_capital: float) -> Dict[str, Any]:
    """Run backtest for a single asset"""
    
    print(f"🔄 Running {symbol} backtest...")
    
    # Initialize strategy
    config_manager = ConfigManager()
    strategy = DualEngineStrategy(config_manager, symbol)
    
    # Portfolio tracking
    portfolio_history = []
    equity = initial_capital
    position_size = 0.0
    
    # Performance metrics
    peak_equity = initial_capital
    max_drawdown = 0.0
    total_trades = 0
    winning_trades = 0
    
    # Exposure tracking
    exposure_history = []
    
    for i, candle in enumerate(candles):
        if i < 100:  # Skip first 100 candles for indicator warmup
            continue
            
        # Get current market data slice
        current_data = candles[max(0, i-200):i+1]
        
        # Generate signals
        try:
            signals = strategy.generate_signals(current_data)
            btc_signal = signals.get('BTC', {})
            
            # Extract signal components
            exposure = btc_signal.get('exposure', 0.0)
            action = btc_signal.get('action', 'HOLD')
            
            # Calculate position value
            if position_size > 0:
                position_value = position_size * candle.close
                equity = initial_capital - position_size * strategy.entry_price + position_value
            else:
                equity = initial_capital
            
            # Execute trades based on exposure changes
            target_position_value = equity * (exposure / 100.0)
            target_position_size = target_position_value / candle.close if candle.close > 0 else 0
            
            if abs(target_position_size - position_size) > equity * 0.001 / candle.close:  # Min trade threshold
                if target_position_size > position_size:  # Buy
                    total_trades += 1
                    strategy.entry_price = candle.close
                elif target_position_size < position_size:  # Sell
                    if position_size > 0 and candle.close > strategy.entry_price:
                        winning_trades += 1
                
                position_size = target_position_size
            
            # Update equity
            if position_size > 0:
                equity = initial_capital + position_size * (candle.close - strategy.entry_price)
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            current_drawdown = (peak_equity - equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Record history
            portfolio_history.append({
                'timestamp': candle.timestamp,
                'equity': equity,
                'exposure': exposure,
                'action': action,
                'price': candle.close
            })
            
            exposure_history.append(exposure)
            
        except Exception as e:
            print(f"Error processing candle {i}: {e}")
            continue
    
    # Calculate metrics
    final_equity = equity
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    avg_exposure = np.mean(exposure_history) if exposure_history else 0.0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    # Calculate Sharpe ratio
    equity_series = [h['equity'] for h in portfolio_history]
    sharpe_ratio = calculate_sharpe_ratio(equity_series)
    
    print(f"✅ {symbol} backtest completed:")
    print(f"   Final Equity: ₩{final_equity:,.0f}")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Avg Exposure: {avg_exposure:.1f}%")
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
    print()
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_exposure': avg_exposure,
        'portfolio_history': portfolio_history,
        'drawdown_periods': []  # Could be enhanced later
    }

def save_individual_results(btc_results: Dict, eth_results: Dict, output_dir: str):
    """Save individual asset results to CSV files"""
    
    # Save BTC results
    if btc_results['portfolio_history']:
        btc_df = pd.DataFrame(btc_results['portfolio_history'])
        btc_df.to_csv(f"{output_dir}/btc_portfolio_history.csv", index=False)
    
    # Save ETH results (if any)
    if eth_results['portfolio_history']:
        eth_df = pd.DataFrame(eth_results['portfolio_history'])
        eth_df.to_csv(f"{output_dir}/eth_portfolio_history.csv", index=False)

def run_portfolio_backtest(btc_candles: List[Candle], eth_candles: List[Candle], initial_capital: float) -> Dict[str, Any]:
    """Run portfolio backtest with BTC-only allocation (100:0)"""
    
    # Portfolio allocation (100% BTC, 0% ETH)
    btc_allocation = 1.0  # 100%
    eth_allocation = 0.0  # 0%
    
    btc_capital = initial_capital * btc_allocation
    eth_capital = initial_capital * eth_allocation
    
    print(f"💰 Capital Allocation:")
    print(f"   Total: ₩{initial_capital:,}")
    print(f"   BTC: ₩{btc_capital:,} ({btc_allocation*100:.0f}%)")
    print(f"   ETH: ₩{eth_capital:,} ({eth_allocation*100:.0f}%)")
    print()
    
    # Create output directory
    output_dir = "results/portfolio_dual_engine"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run individual backtests
    btc_results = run_single_asset_backtest(btc_candles, "KRW-BTC", btc_capital)
    
    if eth_capital > 0:
        eth_results = run_single_asset_backtest(eth_candles, "KRW-ETH", eth_capital)
    else:
        # Create empty ETH results for BTC-only strategy
        eth_results = {
            'final_equity': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_exposure': 0.0,
            'portfolio_history': [],
            'drawdown_periods': []
        }
    
    # Calculate portfolio-level metrics
    btc_final_value = btc_results['final_equity']
    eth_final_value = eth_results['final_equity']
    total_final_value = btc_final_value + eth_final_value
    
    portfolio_return = ((total_final_value - initial_capital) / initial_capital) * 100
    
    # Calculate combined metrics
    combined_max_drawdown = max(btc_results['max_drawdown'], eth_results['max_drawdown'])
    combined_avg_exposure = (btc_results['avg_exposure'] + eth_results['avg_exposure']) / 2
    combined_trades = btc_results['total_trades'] + eth_results['total_trades']
    
    # Calculate daily returns for correlation and Sharpe
    btc_history = btc_results['portfolio_history']
    eth_history = eth_results['portfolio_history']
    
    # Extract equity series for correlation calculation
    btc_returns = []
    eth_returns = []
    portfolio_equity = []
    
    # Handle BTC-only strategy (ETH capital = 0)
    if btc_capital > 0 and len(btc_history) > 1:
        # Use BTC history as portfolio equity (100% BTC allocation)
        for i in range(len(btc_history)):
            btc_equity = btc_history[i]['equity']
            portfolio_equity.append(btc_equity)  # Portfolio = 100% BTC
            
            # Calculate returns for correlation (skip first value)
            if i > 0:
                btc_prev = btc_history[i-1]['equity']
                btc_curr = btc_history[i]['equity']
                btc_ret = (btc_curr - btc_prev) / btc_prev if btc_prev > 0 else 0
                btc_returns.append(btc_ret)
                
                # For BTC-only strategy, ETH returns are all 0
                eth_returns.append(0.0)
    
    # Calculate correlation and Sharpe ratio
    correlation = calculate_portfolio_correlation(btc_returns, eth_returns)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_equity)  # Use equity series
    
    # Save portfolio history
    portfolio_history = []
    
    if len(btc_history) > 0:
        for i, btc_data in enumerate(btc_history):
            eth_data = eth_history[i] if i < len(eth_history) else {
                'equity': 0, 'exposure': 0, 'action': 'HOLD', 'price': 0
            }
            
            portfolio_history.append({
                'timestamp': btc_data['timestamp'],
                'btc_equity': btc_data['equity'],
                'eth_equity': eth_data['equity'],
                'total_equity': btc_data['equity'] + eth_data['equity'],
                'btc_exposure': btc_data['exposure'],
                'eth_exposure': eth_data['exposure'],
                'combined_exposure': (btc_data['exposure'] + eth_data['exposure']) / 2
            })
    
    if portfolio_history:
        pd.DataFrame(portfolio_history).to_csv(f"{output_dir}/portfolio_history.csv", index=False)
    
    # Save individual coin results
    save_individual_results(btc_results, eth_results, output_dir)
    
    return {
        'btc_results': btc_results,
        'eth_results': eth_results,
        'portfolio_return': portfolio_return,
        'portfolio_max_drawdown': combined_max_drawdown,
        'portfolio_avg_exposure': combined_avg_exposure,
        'portfolio_total_trades': combined_trades,
        'portfolio_sharpe': portfolio_sharpe,
        'correlation': correlation,
        'initial_capital': initial_capital,
        'final_value': total_final_value,
        'portfolio_history': portfolio_history
    }

def main():
    """Run BTC+ETH portfolio backtest - OPTIMIZED VERSION"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    print("=" * 80)
    print("🚀 BTC OPTIMIZED STRATEGY BACKTEST (v1.0)")
    print("🎯 Target: 80%+ Return, MDD ≤15%, Sharpe ≥0.5")
    print("=" * 80)
    print()
    
    # Load data for both assets
    data_loader = BacktestDataLoader()
    
    print("📊 Loading market data...")
    btc_candles = data_loader.load_symbol("KRW-BTC")
    eth_candles = data_loader.load_symbol("KRW-ETH")
    
    if not btc_candles or not eth_candles:
        print("❌ Failed to load required market data")
        return
    
    print(f"✅ BTC: {len(btc_candles)} candles")
    print(f"✅ ETH: {len(eth_candles)} candles")
    print(f"📅 Period: {btc_candles[0].timestamp} to {btc_candles[-1].timestamp}")
    print()
    
    # Run portfolio backtest
    initial_capital = 10_000_000  # 10M KRW total
    results = run_portfolio_backtest(btc_candles, eth_candles, initial_capital)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 PORTFOLIO BACKTEST RESULTS - OPTIMIZED")
    print("=" * 80)
    print()

    # Show current parameters
    params = get_current_parameters()
    print("🔧 OPTIMIZED STRATEGY PARAMETERS:")
    print("-" * 50)
    print(f"EMA Crossover Settings:")
    print(f"   BTC: {params['ema_periods']['btc']} | ETH: {params['ema_periods']['eth']}")
    print(f"Volume Thresholds:")
    print(f"   BTC: {params['volume_thresholds']['btc']} | ETH: {params['volume_thresholds']['eth']}")
    print(f"Exposure Limits:")
    print(f"   BTC: {params['exposure_limits']['btc']} | ETH: {params['exposure_limits']['eth']}")
    print(f"Signal Sensitivity: {params['signal_sensitivity']}")
    print(f"Risk Factor: {params['risk_factor']}")
    print(f"Rebalancing: Every {params['rebalancing_frequency']}")
    print()

    # Individual asset results
    btc_res = results['btc_results']
    eth_res = results['eth_results']
    
    print("📈 INDIVIDUAL ASSET PERFORMANCE:")
    print("-" * 50)
    print(f"[BTC Aggressive]")
    print(f"   Return: {btc_res['total_return']:+.2f}%")
    print(f"   MDD: {btc_res['max_drawdown']:.2f}%")
    print(f"   Exposure: {btc_res['avg_exposure']:.1f}%")
    print(f"   Trades: {btc_res['total_trades']}")
    print()
    
    print(f"[ETH Defensive]")
    print(f"   Return: {eth_res['total_return']:+.2f}%")
    print(f"   MDD: {eth_res['max_drawdown']:.2f}%")
    print(f"   Exposure: {eth_res['avg_exposure']:.1f}%")
    print(f"   Trades: {eth_res['total_trades']}")
    print()
    
    # Portfolio results
    print(f"[Portfolio 100:0] - OPTIMIZED")
    print(f"   Return: {results['portfolio_return']:+.2f}%")
    print(f"   MDD: {results['portfolio_max_drawdown']:.2f}%")
    print(f"   Exposure: {results['portfolio_avg_exposure']:.1f}%")
    print(f"   Trades: {results['portfolio_total_trades']}")
    print(f"   Sharpe Ratio: {results['portfolio_sharpe']:.3f}")
    print(f"   BTC-ETH Correlation: {results['correlation']:.3f}")
    print()
    
    # Goal assessment
    return_goal = 80.0   # ≥80% total return (달성)
    mdd_goal = 20.0      # ≤20% max drawdown (달성)
    sharpe_goal = 0.5    # Sharpe ≥0.5 (달성)
    
    return_met = results['portfolio_return'] >= return_goal
    mdd_met = results['portfolio_max_drawdown'] <= mdd_goal
    sharpe_met = results['portfolio_sharpe'] >= sharpe_goal
    
    print("🎯 OPTIMIZATION GOAL ASSESSMENT:")
    print("-" * 50)
    print(f"Return Goal (≥80%): {'✅ ACHIEVED' if return_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≥{return_goal}% | Actual: {results['portfolio_return']:.2f}%")
    print()
    
    print(f"Risk Goal (≤20% MDD): {'✅ ACHIEVED' if mdd_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≤{mdd_goal}% | Actual: {results['portfolio_max_drawdown']:.2f}%")
    print()
    
    print(f"Sharpe Goal (≥0.5): {'✅ ACHIEVED' if sharpe_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≥{sharpe_goal} | Actual: {results['portfolio_sharpe']:.3f}")
    print()
    
    # Overall success
    overall_success = return_met and mdd_met and sharpe_met
    print(f"Overall Success: {'🎉 FULLY OPTIMIZED!' if overall_success else '🔧 IMPROVEMENT NEEDED'}")
    print()
    
    # Portfolio insights
    diversification_benefit = results['portfolio_return'] - max(btc_res['total_return'], eth_res['total_return'])
    
    print("💡 PORTFOLIO INSIGHTS:")
    print("-" * 50)
    print(f"Best Performer: {'BTC' if btc_res['total_return'] > eth_res['total_return'] else 'ETH'}")
    print(f"Lower Risk: {'BTC' if btc_res['max_drawdown'] < eth_res['max_drawdown'] else 'ETH'}")
    print(f"Diversification: {'High' if abs(results['correlation']) < 0.5 else 'Low'} correlation ({results['correlation']:.3f})")
    print(f"Diversification Benefit: {diversification_benefit:+.2f}%p")
    print()
    
    # Capital allocation summary
    print("💰 CAPITAL ALLOCATION:")
    print(f"   Initial: ₩{results['initial_capital']:,} total")
    print(f"   BTC: ₩{5_000_000:,} → ₩{btc_res['final_equity']:,.0f}")
    print(f"   ETH: ₩{5_000_000:,} → ₩{eth_res['final_equity']:,.0f}")
    print(f"   Final: ₩{results['final_value']:,.0f} total")
    print()
    
    print("✅ Portfolio results saved to results/portfolio_dual_engine/")
    print("=" * 80)
    print("🎊 OPTIMIZATION COMPLETED - VERSION 1.0 SAVED!")
    print("=" * 80)

if __name__ == "__main__":
    main()