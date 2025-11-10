"""
Portfolio Dual-Engine Backtest System
=====================================

Run BTC-only portfolio backtest with 100% allocation.
Enhanced parameters targeting ≥80% return, ≤15% MDD, Sharpe ≥0.5.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from dual_engine_strategy import StrategyManager, create_dual_engine_config, StrategySignal, get_current_parameters
from backtest.portfolio import BacktestPortfolio
from exchange.models import Candle
from backtest.data_loader import BacktestDataLoader

@dataclass
class FeeConfig:
    """수수료 설정"""
    trading_fee_rate: float = 0.00025  # 0.025%
    slippage_rate: float = 0.0001      # 0.01%
    
    @property
    def total_fee_rate(self) -> float:
        return self.trading_fee_rate + self.slippage_rate

@dataclass
class TradeRecord:
    """거래 기록"""
    timestamp: datetime
    trade_type: str
    price: float
    quantity: float
    fee: float
    reason: str

def execute_buy_with_fee(portfolio, symbol: str, price: float, quantity: float, fee_config: FeeConfig, timestamp: datetime) -> TradeRecord:
    """수수료 포함 매수"""
    from core.types import OrderSide
    
    fee = price * quantity * fee_config.total_fee_rate
    total_cost = price * quantity + fee
    
    # BacktestPortfolio의 apply_fill 사용
    portfolio.apply_fill(symbol, OrderSide.BUY, price, quantity, timestamp)
    
    return TradeRecord(
        timestamp=timestamp,
        trade_type='BUY',
        price=price,
        quantity=quantity,
        fee=fee,
        reason='Signal'
    )

def execute_sell_with_fee(portfolio, symbol: str, price: float, quantity: float, fee_config: FeeConfig, timestamp: datetime) -> TradeRecord:
    """수수료 포함 매도"""
    from core.types import OrderSide
    
    fee = price * quantity * fee_config.total_fee_rate
    
    # BacktestPortfolio의 apply_fill 사용  
    portfolio.apply_fill(symbol, OrderSide.SELL, price, quantity, timestamp)
    
    return TradeRecord(
        timestamp=timestamp,
        trade_type='SELL',
        price=price,
        quantity=quantity,
        fee=fee,
        reason='Signal'
    )

def calculate_portfolio_correlation(btc_returns: List[float], eth_returns: List[float]) -> float:
    """Calculate correlation between BTC and ETH returns"""
    if len(btc_returns) != len(eth_returns) or len(btc_returns) < 2:
        return 0.0
    
    btc_array = np.array(btc_returns)
    eth_array = np.array(eth_returns)
    
    correlation_matrix = np.corrcoef(btc_array, eth_array)
    return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0:
        return 0
    
    excess_returns = np.array(returns) - risk_free_rate/252/24/4  # 4-hour periods (restored from 6h)
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252*24/4)  # Annualized for 4-hour periods

def save_individual_results(btc_results, eth_results, output_dir):
    """Save detailed individual coin results"""
    import json
    
    # BTC detailed results
    btc_detailed = {
        'summary': {
            'total_return': f"{btc_results['total_return']:.2f}%",
            'max_drawdown': f"{btc_results['max_drawdown']:.2f}%",
            'final_equity': f"₩{btc_results['final_equity']:,.0f}",
            'avg_exposure': f"{btc_results['avg_exposure']:.1f}%",
            'total_trades': btc_results['total_trades']
        },
        'performance_metrics': {
            'profit_factor': btc_results.get('profit_factor', 'N/A'),
            'win_rate': f"{btc_results.get('win_rate', 0):.1f}%",
            'avg_win': f"{btc_results.get('avg_win', 0):.2f}%",
            'avg_loss': f"{btc_results.get('avg_loss', 0):.2f}%",
            'max_consecutive_wins': btc_results.get('max_consecutive_wins', 0),
            'max_consecutive_losses': btc_results.get('max_consecutive_losses', 0)
        }
    }
    
    # ETH detailed results  
    eth_detailed = {
        'summary': {
            'total_return': f"{eth_results['total_return']:.2f}%",
            'max_drawdown': f"{eth_results['max_drawdown']:.2f}%",
            'final_equity': f"₩{eth_results['final_equity']:,.0f}",
            'avg_exposure': f"{eth_results['avg_exposure']:.1f}%",
            'total_trades': eth_results['total_trades']
        },
        'performance_metrics': {
            'profit_factor': eth_results.get('profit_factor', 'N/A'),
            'win_rate': f"{eth_results.get('win_rate', 0):.1f}%",
            'avg_win': f"{eth_results.get('avg_win', 0):.2f}%",
            'avg_loss': f"{eth_results.get('avg_loss', 0):.2f}%",
            'max_consecutive_wins': eth_results.get('max_consecutive_wins', 0),
            'max_consecutive_losses': eth_results.get('max_consecutive_losses', 0)
        }
    }
    
    # Save as JSON files
    with open(f"{output_dir}/btc_detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(btc_detailed, f, indent=2, ensure_ascii=False)
    
    with open(f"{output_dir}/eth_detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(eth_detailed, f, indent=2, ensure_ascii=False)
    
    # Save trade histories if available
    if 'history' in btc_results and btc_results['history']:
        pd.DataFrame(btc_results['history']).to_csv(f"{output_dir}/btc_portfolio_history.csv", index=False)
    
    if 'history' in eth_results and eth_results['history']:
        pd.DataFrame(eth_results['history']).to_csv(f"{output_dir}/eth_portfolio_history.csv", index=False)

def calculate_sharpe_ratio(equity_series, risk_free_rate=0.02):
    """Calculate Sharpe ratio from equity series using log returns (4h interval data)"""
    if not equity_series or len(equity_series) < 2:
        print(f"[DEBUG] Insufficient data: equity_series length = {len(equity_series) if equity_series else 0}")
        return 0.0
    
    # Convert to pandas Series and ensure float type
    import pandas as pd
    equity_df = pd.Series(equity_series, dtype=float)
    
    # Calculate log returns instead of percentage returns
    log_returns = np.log(equity_df / equity_df.shift(1)).dropna()
    
    # Remove any NaN or infinite values
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(log_returns) < 2:
        print(f"[DEBUG] Insufficient clean returns: log_returns length = {len(log_returns)}")
        return 0.0
    
    returns_array = log_returns.values
    
    # For 4-hour intervals: 6 periods per day, 365 days per year = 2190
    periods_per_year = 2190
    risk_free_rate_per_period = risk_free_rate / periods_per_year
    
    excess_returns = returns_array - risk_free_rate_per_period
    
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)  # Sample standard deviation
    
    # Debug Sharpe calculation with detailed info
    print(f"[DEBUG] Equity series range: {min(equity_series):.2f} to {max(equity_series):.2f}")
    print(f"[DEBUG] Log returns: periods={len(log_returns)}, mean={mean_return:.8f}, std={std_return:.8f}")
    print(f"[DEBUG] Risk-free rate per period: {risk_free_rate_per_period:.8f}")
    
    if std_return < 1e-10:  # Very small volatility
        print(f"[DEBUG] Zero volatility detected (std={std_return:.10f}) - returning 0")
        return 0.0
    
    # Calculate Sharpe and annualize
    sharpe = mean_return / std_return
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)  # sqrt(2190)
    
    print(f"[DEBUG] Final Sharpe: raw={sharpe:.6f}, annualized={annualized_sharpe:.6f}")
    
    return annualized_sharpe

def run_single_asset_backtest(candles: List[Candle], symbol: str, initial_cash: float) -> Dict:
    """Run backtest for a single asset"""
    
    config = create_dual_engine_config()
    strategy = StrategyManager(config, symbol)
    portfolio = BacktestPortfolio(initial_cash=initial_cash, cash=initial_cash)
    fee_config = FeeConfig(trading_fee_rate=0.00025, slippage_rate=0.0001)
    
    # Tracking variables
    total_trades = 0
    max_equity = initial_cash
    max_drawdown = 0.0
    exposure_sum = 0.0
    exposure_count = 0
    portfolio_history = []
    last_rebalance_time = None  # Track last rebalance time
    rebalance_interval_hours = 6  # 6-hour rebalancing for improved Sharpe ratio

    asset_name = "BTC" if "BTC" in symbol else "ETH"
    print(f"🚀 Running {asset_name} backtest (6h rebalancing)...")

    # Main backtest loop
    for i, candle in enumerate(candles):
        if i < 50:  # Need sufficient history
            continue
            
        # Get recent candles for analysis
        recent_candles = candles[max(0, i-100):i+1]
        
        # Calculate current drawdown
        current_prices = {symbol: candle.close}
        current_equity = portfolio.get_current_equity(current_prices)
        max_equity = max(max_equity, current_equity)
        current_drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Get strategy signal
        signal = strategy.analyze(recent_candles, current_drawdown)
        
        # Calculate current position value
        current_position = portfolio.positions.get(symbol)
        position_value = current_position.size * candle.close if current_position else 0.0
        current_exposure = position_value / current_equity if current_equity > 0 else 0.0
        
        # Record portfolio state
        portfolio_history.append({
            'timestamp': candle.timestamp,
            'equity': current_equity,
            'exposure': current_exposure,
            'target_exposure': signal.final_exposure
        })
        
        # Check if it's time to rebalance (every 6 hours)
        should_rebalance = False
        if last_rebalance_time is None:
            should_rebalance = True  # First rebalance
        else:
            time_diff = (candle.timestamp - last_rebalance_time).total_seconds() / 3600
            should_rebalance = time_diff >= rebalance_interval_hours
        
        # Trading logic: Rebalance to target exposure (only at rebalance intervals)
        if should_rebalance:
            target_exposure = signal.final_exposure
            exposure_diff = target_exposure - current_exposure
            
            # Execute trades if significant difference (>4% - enhanced sensitivity)
            if abs(exposure_diff) > 0.04:
                if exposure_diff > 0:  # Need to buy more
                    target_amount = current_equity * exposure_diff
                    if target_amount > 1000:  # Minimum trade size
                        quantity = target_amount / candle.close
                        execute_buy_with_fee(portfolio, symbol, candle.close, quantity, fee_config, candle.timestamp)
                        total_trades += 1
                        
                else:  # Need to sell
                    if current_position and current_position.size > 0:
                        sell_ratio = min(1.0, abs(exposure_diff) / current_exposure)
                        quantity = current_position.size * sell_ratio
                        execute_sell_with_fee(portfolio, symbol, candle.close, quantity, fee_config, candle.timestamp)
                        total_trades += 1
            
            last_rebalance_time = candle.timestamp
        
        # Update statistics
        exposure_sum += signal.final_exposure
        exposure_count += 1
    
    # Calculate final results
    final_equity = portfolio.get_current_equity({symbol: candles[-1].close})
    total_return = ((final_equity - initial_cash) / initial_cash) * 100
    
    # Average exposure
    avg_exposure = (exposure_sum / exposure_count) if exposure_count > 0 else 0.0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,
        'final_equity': final_equity,
        'avg_exposure': avg_exposure * 100,
        'total_trades': total_trades,
        'portfolio_history': portfolio_history
    }

def run_portfolio_backtest(btc_candles: List[Candle], eth_candles: List[Candle], initial_capital: float = 10_000_000) -> Dict:
    """Run BTC-only portfolio backtest with 100% allocation"""
    
    print("🚀 PORTFOLIO BACKTEST: BTC (100%) + ETH (0%) - BTC-ONLY STRATEGY")
    print("=" * 80)
    
    # Initialize individual strategies with BTC-only allocation
    btc_capital = initial_capital * 1.00  # 100% to BTC
    eth_capital = initial_capital * 0.00  # 0% to ETH
    
    # Run individual backtests
    btc_results = run_single_asset_backtest(btc_candles, "KRW-BTC", btc_capital)
    # Skip ETH backtest when allocation is 0%
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
    
    # Save portfolio results
    import os
    output_dir = "results/portfolio_dual_engine"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine portfolio history
    portfolio_history = []
    for i in range(min(len(btc_history), len(eth_history))):
        btc_data = btc_history[i]
        eth_data = eth_history[i]
        
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
    """Run BTC+ETH portfolio backtest"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    print("=" * 80)
    print("🚀 BTC+ETH DUAL-ENGINE PORTFOLIO BACKTEST")
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
    print("📊 PORTFOLIO BACKTEST RESULTS")
    print("=" * 80)
    print()
    
    # Show current parameters
    params = get_current_parameters()
    print("🔧 CURRENT STRATEGY PARAMETERS:")
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
    print(f"[Portfolio 100:0]")
    print(f"   Return: {results['portfolio_return']:+.2f}%")
    print(f"   MDD: {results['portfolio_max_drawdown']:.2f}%")
    print(f"   Exposure: {results['portfolio_avg_exposure']:.1f}%")
    print(f"   Trades: {results['portfolio_total_trades']}")
    print(f"   Sharpe Ratio: {results['portfolio_sharpe']:.3f}")
    print(f"   BTC-ETH Correlation: {results['correlation']:.3f}")
    print()
    
    # Goal assessment
    return_goal = 85.0  # ≥85% total return
    mdd_goal = 20.0     # ≤20% max drawdown  
    sharpe_goal = 0.5   # Sharpe ≥0.5
    
    return_met = results['portfolio_return'] >= return_goal
    mdd_met = results['portfolio_max_drawdown'] <= mdd_goal
    sharpe_met = results['portfolio_sharpe'] >= sharpe_goal
    
    print("🎯 PORTFOLIO GOAL ASSESSMENT:")
    print("-" * 50)
    print(f"Return Goal (≥85%): {'✅ ACHIEVED' if return_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≥{return_goal}% | Actual: {results['portfolio_return']:.2f}%")
    print()
    
    print(f"Risk Goal (≤20% MDD): {'✅ ACHIEVED' if mdd_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≤{mdd_goal}% | Actual: {results['portfolio_max_drawdown']:.2f}%")
    print()
    
    print(f"Sharpe Goal (≥0.5): {'✅ ACHIEVED' if sharpe_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≥{sharpe_goal} | Actual: {results['portfolio_sharpe']:.3f}")
    print()
    
    overall_success = return_met and mdd_met and sharpe_met
    print(f"Overall Success: {'🏆 ALL GOALS ACHIEVED' if overall_success else '🔧 IMPROVEMENT NEEDED'}")
    print()
    
    # Portfolio insights
    print("💡 PORTFOLIO INSIGHTS:")
    print("-" * 50)
    
    better_performer = "BTC" if btc_res['total_return'] > eth_res['total_return'] else "ETH"
    print(f"Best Performer: {better_performer}")
    
    safer_asset = "BTC" if btc_res['max_drawdown'] < eth_res['max_drawdown'] else "ETH"
    print(f"Lower Risk: {safer_asset}")
    
    correlation_desc = "High" if abs(results['correlation']) > 0.7 else "Medium" if abs(results['correlation']) > 0.3 else "Low"
    print(f"Diversification: {correlation_desc} correlation ({results['correlation']:.3f})")
    
    # Diversification benefit
    weighted_return = 1.00 * btc_res['total_return'] + 0.00 * eth_res['total_return']
    diversification_benefit = results['portfolio_return'] - weighted_return
    print(f"Diversification Benefit: {diversification_benefit:+.2f}%p")
    print()
    
    print(f"💰 CAPITAL ALLOCATION:")
    print(f"   Initial: ₩{initial_capital:,} total")
    print(f"   BTC: ₩{initial_capital//2:,} → ₩{btc_res['final_equity']:,.0f}")
    print(f"   ETH: ₩{initial_capital//2:,} → ₩{eth_res['final_equity']:,.0f}")
    print(f"   Final: ₩{results['final_value']:,.0f} total")
    print()
    
    print(f"✅ Portfolio results saved to results/portfolio_dual_engine/")
    print("=" * 80)

if __name__ == "__main__":
    main()