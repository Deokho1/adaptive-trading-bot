"""
Dual-Engine Backtest System
===========================

Enhanced backtest system that integrates the dual-engine trading strategy
with comprehensive performance tracking and detailed logging.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from dual_engine_strategy import StrategyManager, create_dual_engine_config, StrategySignal
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

logger = logging.getLogger(__name__)

# ============================================================================
# Enhanced Results Collector for Dual-Engine Strategy
# ============================================================================

class DualEngineResultsCollector:
    """Comprehensive results collection for dual-engine backtest"""
    
    def __init__(self):
        self.portfolio_history = []
        self.trades = []
        self.engine_signals = []
        self.drawdown_events = []
        self.performance_metrics = {}
        
    def record_signal(self, timestamp: datetime, candles: List[Candle], 
                     signal: StrategySignal, current_drawdown: float):
        """Record detailed signal information"""
        current_price = candles[-1].close if candles else 0
        
        self.engine_signals.append({
            'timestamp': timestamp,
            'price': current_price,
            'final_exposure': signal.final_exposure,
            'regime_exposure': signal.regime_exposure,
            'signal_exposure': signal.signal_exposure,
            'active_engines': ','.join(signal.active_engines),
            'reasoning': signal.reasoning,
            'current_drawdown': current_drawdown
        })
        
    def record_portfolio_state(self, timestamp: datetime, portfolio: BacktestPortfolio, 
                              current_price: float, signal: StrategySignal):
        """Record portfolio state with engine information"""
        equity = portfolio.get_current_equity({'KRW-BTC': current_price})
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': portfolio.cash,
            'position_value': equity - portfolio.cash,
            'final_exposure': signal.final_exposure,
            'regime_exposure': signal.regime_exposure,
            'signal_exposure': signal.signal_exposure,
            'active_engines': ','.join(signal.active_engines)
        })
        
    def record_trade(self, trade_record):
        """Record trade execution"""
        if trade_record:
            self.trades.append(trade_record)
            
    def save_results(self, output_dir: str = "results/dual_engine"):
        """Save comprehensive results to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all data
        if self.portfolio_history:
            pd.DataFrame(self.portfolio_history).to_csv(f"{output_dir}/portfolio_history.csv", index=False)
            
        if self.trades:
            pd.DataFrame([vars(trade) for trade in self.trades]).to_csv(f"{output_dir}/trades.csv", index=False)
            
        if self.engine_signals:
            pd.DataFrame(self.engine_signals).to_csv(f"{output_dir}/engine_signals.csv", index=False)
            
        if self.drawdown_events:
            pd.DataFrame(self.drawdown_events).to_csv(f"{output_dir}/drawdown_events.csv", index=False)
            
        print(f"✅ Dual-engine results saved to {output_dir}/")
        print(f"   📊 Portfolio history: {len(self.portfolio_history)} records")
        print(f"   💱 Trades: {len(self.trades)} executions")
        print(f"   🎯 Engine signals: {len(self.engine_signals)} decisions")

# ============================================================================
# Dual-Engine Backtest Runner
# ============================================================================

def run_dual_engine_backtest(candles: List[Candle], 
                            config: Dict = None,
                            symbol: str = "KRW-BTC",
                            initial_cash: float = 10_000_000) -> Dict:
    """
    Run comprehensive backtest with dual-engine strategy
    
    Args:
        candles: Historical price data
        config: Strategy configuration
        symbol: Trading symbol (for asset-specific parameters)
        initial_cash: Starting portfolio value
        
    Returns:
        Dictionary with comprehensive results
    """
    if config is None:
        config = create_dual_engine_config()
    
    # Initialize components with symbol-specific strategy
    strategy = StrategyManager(config, symbol)
    portfolio = BacktestPortfolio(initial_cash=initial_cash, cash=initial_cash)
    fee_config = FeeConfig(trading_fee_rate=0.00025, slippage_rate=0.0001)
    results = DualEngineResultsCollector()
    
    # Tracking variables
    total_trades = 0
    max_equity = initial_cash
    max_drawdown = 0.0
    exposure_sum = 0.0
    exposure_count = 0
    regime_transitions = 0  # Track regime transition protections
    
    # Regime and signal tracking
    regime_stats = {"TREND_UP": 0, "TREND_DOWN": 0, "RANGE": 0, "NEUTRAL": 0}
    signal_stats = {"BUY": 0, "SELL": 0, "HOLD": 0}
    engine_usage = {"regime_only": 0, "dual_engine": 0}
    
    print(f"🚀 Starting dual-engine backtest...")
    print(f"📊 Data: {len(candles)} candles from {candles[0].timestamp} to {candles[-1].timestamp}")
    print(f"💰 Initial capital: ₩{initial_cash:,.0f}")
    print(f"🔧 Strategy: RegimeEngine + SignalEngine")
    print()
    
    # Main backtest loop
    for i, candle in enumerate(candles):
        if i < 50:  # Need sufficient history
            continue
            
        # Get recent candles for analysis
        recent_candles = candles[max(0, i-100):i+1]
        
        # Calculate current drawdown
        current_prices = {'KRW-BTC': candle.close}
        current_equity = portfolio.get_current_equity(current_prices)
        max_equity = max(max_equity, current_equity)
        current_drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Get strategy signal
        signal = strategy.analyze(recent_candles, current_drawdown)
        
        # Record signal and portfolio state
        results.record_signal(candle.timestamp, recent_candles, signal, current_drawdown)
        results.record_portfolio_state(candle.timestamp, portfolio, candle.close, signal)
        
        # Track regime transitions for ETH
        if hasattr(signal, 'regime_transition_brake') and signal.regime_transition_brake:
            regime_transitions += 1
        
        # Calculate current position value
        current_position = portfolio.positions.get(symbol)
        position_value = current_position.size * candle.close if current_position else 0.0
        current_exposure = position_value / current_equity if current_equity > 0 else 0.0
        
        # Trading logic: Rebalance to target exposure
        target_exposure = signal.final_exposure
        exposure_diff = target_exposure - current_exposure
        
        # Execute trades if significant difference (>12% - enhanced from 5%)
        if abs(exposure_diff) > 0.12:
            asset_name = "ETH" if "ETH" in symbol else "BTC"
            
            if exposure_diff > 0:  # Need to buy more
                target_amount = current_equity * exposure_diff
                if target_amount > 1000:  # Minimum trade size
                    quantity = target_amount / candle.close
                    trade_record = execute_buy_with_fee(
                        portfolio, symbol, candle.close, quantity, fee_config, candle.timestamp
                    )
                    results.record_trade(trade_record)
                    total_trades += 1
                    
                    # Log trade with engine information
                    logger.info(f"[DualEngine] BUY {quantity:.4f} {asset_name} at ₩{candle.close:,.0f}")
                    logger.info(f"              Engines: {signal.active_engines} | {signal.reasoning}")
                    if hasattr(signal, 'regime_transition_brake') and signal.regime_transition_brake:
                        logger.info(f"              [REGIME_TRANSITION_PROTECTION] Active")
                    
            else:  # Need to sell
                if current_position and current_position.size > 0:
                    sell_ratio = min(1.0, abs(exposure_diff) / current_exposure)
                    quantity = current_position.size * sell_ratio
                    trade_record = execute_sell_with_fee(
                        portfolio, symbol, candle.close, quantity, fee_config, candle.timestamp
                    )
                    results.record_trade(trade_record)
                    total_trades += 1
                    
                    # Log trade with engine information  
                    logger.info(f"[DualEngine] SELL {quantity:.4f} {asset_name} at ₩{candle.close:,.0f}")
                    logger.info(f"              Engines: {signal.active_engines} | {signal.reasoning}")
                    if hasattr(signal, 'regime_transition_brake') and signal.regime_transition_brake:
                        logger.info(f"              [REGIME_TRANSITION_PROTECTION] Active")
        
        # Update statistics
        exposure_sum += signal.final_exposure
        exposure_count += 1
        
        # Track engine usage
        if len(signal.active_engines) == 1:
            engine_usage["regime_only"] += 1
        else:
            engine_usage["dual_engine"] += 1
    
    # Calculate final results
    final_equity = portfolio.get_current_equity({symbol: candles[-1].close})
    total_return = ((final_equity - initial_cash) / initial_cash) * 100
    
    # Buy & Hold comparison
    buy_hold_return = ((candles[-1].close - candles[0].close) / candles[0].close) * 100
    vs_buyhold = total_return - buy_hold_return
    
    # Average exposure
    avg_exposure = (exposure_sum / exposure_count) if exposure_count > 0 else 0.0
    
    # Get strategy performance stats
    strategy_stats = strategy.get_performance_summary()
    
    # Prepare results
    results_dict = {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'vs_buyhold': vs_buyhold,
        'max_drawdown': max_drawdown * 100,
        'final_equity': final_equity,
        'avg_exposure': avg_exposure * 100,
        'total_trades': total_trades,
        'regime_transitions': regime_transitions,  # New: track transition protections
        'symbol': symbol,
        'asset_type': strategy.asset_type.value,
        'engine_usage': engine_usage,
        'strategy_stats': strategy_stats,
        'results_collector': results
    }
    
    # Save detailed results
    results.save_results()
    
    return results_dict

# ============================================================================
# Portfolio Backtest Functions
# ============================================================================

def calculate_portfolio_correlation(btc_returns: List[float], eth_returns: List[float]) -> float:
    """Calculate correlation between BTC and ETH returns"""
    if len(btc_returns) != len(eth_returns) or len(btc_returns) < 2:
        return 0.0
    
    btc_array = np.array(btc_returns)
    eth_array = np.array(eth_returns)
    
    correlation_matrix = np.corrcoef(btc_array, eth_array)
    return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)

def run_portfolio_backtest(btc_candles: List[Candle], eth_candles: List[Candle],
                          initial_capital: float = 10_000_000) -> Dict:
    """
    Run combined BTC+ETH portfolio backtest with 50:50 allocation
    
    Args:
        btc_candles: BTC historical data
        eth_candles: ETH historical data
        initial_capital: Total starting capital
        
    Returns:
        Combined portfolio results
    """
    print("🚀 PORTFOLIO BACKTEST: BTC (50%) + ETH (50%)")
    print("=" * 80)
    
    # Initialize individual strategies
    capital_per_asset = initial_capital / 2  # 50:50 split
    
    config = create_dual_engine_config()
    
    # Run individual backtests
    print("📊 Running BTC Aggressive Strategy...")
    btc_results = run_dual_engine_backtest(
        btc_candles, config, "KRW-BTC", capital_per_asset
    )
    
    print("\n📊 Running ETH Defensive Strategy...")
    eth_results = run_dual_engine_backtest(
        eth_candles, config, "KRW-ETH", capital_per_asset
    )
    
    # Calculate portfolio-level metrics
    btc_final_value = btc_results['final_equity']
    eth_final_value = eth_results['final_equity']
    total_final_value = btc_final_value + eth_final_value
    
    portfolio_return = ((total_final_value - initial_capital) / initial_capital) * 100
    
    # Calculate combined metrics
    combined_max_drawdown = max(btc_results['max_drawdown'], eth_results['max_drawdown'])
    combined_avg_exposure = (btc_results['avg_exposure'] + eth_results['avg_exposure']) / 2
    combined_trades = btc_results['total_trades'] + eth_results['total_trades']
    
    # Get returns for correlation and Sharpe calculation
    btc_portfolio_history = btc_results['results_collector'].portfolio_history
    eth_portfolio_history = eth_results['results_collector'].portfolio_history
    
    # Calculate daily returns
    btc_returns = []
    eth_returns = []
    portfolio_returns = []
    
    if len(btc_portfolio_history) > 1 and len(eth_portfolio_history) > 1:
        for i in range(1, min(len(btc_portfolio_history), len(eth_portfolio_history))):
            btc_prev = btc_portfolio_history[i-1]['equity']
            btc_curr = btc_portfolio_history[i]['equity']
            btc_ret = (btc_curr - btc_prev) / btc_prev if btc_prev > 0 else 0
            btc_returns.append(btc_ret)
            
            eth_prev = eth_portfolio_history[i-1]['equity']
            eth_curr = eth_portfolio_history[i]['equity']
            eth_ret = (eth_curr - eth_prev) / eth_prev if eth_prev > 0 else 0
            eth_returns.append(eth_ret)
            
            # Portfolio return (50:50 weighted)
            portfolio_ret = 0.5 * btc_ret + 0.5 * eth_ret
            portfolio_returns.append(portfolio_ret)
    
    # Calculate correlation and Sharpe ratio
    correlation = calculate_portfolio_correlation(btc_returns, eth_returns)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    
    # Save portfolio results
    import os
    output_dir = "results/portfolio_dual_engine"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine portfolio history
    portfolio_history = []
    for i in range(min(len(btc_portfolio_history), len(eth_portfolio_history))):
        btc_data = btc_portfolio_history[i]
        eth_data = eth_portfolio_history[i]
        
        portfolio_history.append({
            'timestamp': btc_data['timestamp'],
            'btc_equity': btc_data['equity'],
            'eth_equity': eth_data['equity'],
            'total_equity': btc_data['equity'] + eth_data['equity'],
            'btc_exposure': btc_data['final_exposure'],
            'eth_exposure': eth_data['final_exposure'],
            'combined_exposure': (btc_data['final_exposure'] + eth_data['final_exposure']) / 2
        })
    
    if portfolio_history:
        pd.DataFrame(portfolio_history).to_csv(f"{output_dir}/portfolio_history.csv", index=False)
    
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

# ============================================================================
# Main Execution
# ============================================================================

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
    
    # Individual asset results
    btc_res = results['btc_results']
    eth_res = results['eth_results']
    
    print("� INDIVIDUAL ASSET PERFORMANCE:")
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
    print(f"[Portfolio 50:50]")
    print(f"   Return: {results['portfolio_return']:+.2f}%")
    print(f"   MDD: {results['portfolio_max_drawdown']:.2f}%")
    print(f"   Exposure: {results['portfolio_avg_exposure']:.1f}%")
    print(f"   Trades: {results['portfolio_total_trades']}")
    print(f"   Sharpe Ratio: {results['portfolio_sharpe']:.3f}")
    print(f"   BTC-ETH Correlation: {results['correlation']:.3f}")
    print()
    
    # Goal assessment
    return_goal = 80.0  # ≥80% total return
    mdd_goal = 18.0     # ≤18% max drawdown  
    sharpe_goal = 0.45  # Sharpe ≥0.45
    
    return_met = results['portfolio_return'] >= return_goal
    mdd_met = results['portfolio_max_drawdown'] <= mdd_goal
    sharpe_met = results['portfolio_sharpe'] >= sharpe_goal
    
    print("🎯 PORTFOLIO GOAL ASSESSMENT:")
    print("-" * 50)
    print(f"Return Goal (≥80%): {'✅ ACHIEVED' if return_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≥{return_goal}% | Actual: {results['portfolio_return']:.2f}%")
    print()
    
    print(f"Risk Goal (≤18% MDD): {'✅ ACHIEVED' if mdd_met else '❌ NOT ACHIEVED'}")
    print(f"   Target: ≤{mdd_goal}% | Actual: {results['portfolio_max_drawdown']:.2f}%")
    print()
    
    print(f"Sharpe Goal (≥0.45): {'✅ ACHIEVED' if sharpe_met else '❌ NOT ACHIEVED'}")
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
    weighted_return = 0.5 * btc_res['total_return'] + 0.5 * eth_res['total_return']
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

        asset_type = "ETH" if "ETH" in symbol else "BTC"
        mode = "Defensive" if asset_type == "ETH" else "Aggressive"
        
        print(f"🔧 Testing {symbol} ({asset_type} - {mode} Mode)")
        print("-" * 50)
        
        # Load data
        data_loader = BacktestDataLoader()
        candles = data_loader.load_symbol(symbol)
        
        if not candles:
            print(f"❌ Failed to load {symbol} data")
            continue
            
        print(f"✅ Loaded {len(candles)} {asset_type} candles")
        print(f"📅 Period: {candles[0].timestamp} to {candles[-1].timestamp}")
        print()
        
        # Create enhanced configuration
        config = create_dual_engine_config()
        print("🔧 Strategy Configuration:")
        
        if asset_type == "ETH":
            print(f"   ETH Defensive Mode:")
            print(f"   - RegimeEngine: TREND_UP=55%, NEUTRAL=27.5%, RANGE=47.5%, TREND_DOWN=6%")
            print(f"   - SignalEngine: EMA{config['ema_short_eth']}/{config['ema_long_eth']}, Volume={config['volume_threshold_eth']}x")
            print(f"   - Max additive: {config['max_additive_exposure_eth']:.1%}, Regime transition protection")
        else:
            print(f"   BTC Aggressive Mode:")
            print(f"   - RegimeEngine: TREND_UP=80%, NEUTRAL=35%, RANGE=52.5%, TREND_DOWN=6%")
            print(f"   - SignalEngine: EMA{config['ema_short_btc']}/{config['ema_long_btc']}, Volume={config['volume_threshold_btc']}x")
            print(f"   - Max additive: {config['max_additive_exposure_btc']:.1%}, 3% pullback tolerance")
        print()
        
        # Run backtest
        results = run_dual_engine_backtest(candles, config, symbol)
        all_results[symbol] = results
        
        # Display results
        print("📊 RESULTS:")
        print(f"💰 Performance:")
        print(f"   Total Return: {results['total_return']:+.2f}%")
        print(f"   Buy & Hold: {results['buy_hold_return']:+.2f}%")
        status = "🎉 OUTPERFORMED" if results['vs_buyhold'] > 0 else "📉 Underperformed"
        print(f"   vs Buy&Hold: {status} by {results['vs_buyhold']:+.2f}%p")
        print()
        
        print(f"�️ Risk Management:")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        
        # Asset-specific goal assessment
        if asset_type == "ETH":
            dd_goal = 25.0  # ETH target: 20-25% MDD
            dd_status = "✅ TARGET MET" if results['max_drawdown'] <= dd_goal else "❌ EXCEEDED TARGET"
            print(f"   ETH Drawdown Goal (≤25%): {dd_status}")
        else:
            dd_goal = 20.0  # BTC target: ≤20% MDD
            return_goal_min, return_goal_max = 130.0, 160.0  # BTC target: 130-160% return
            dd_status = "✅ TARGET MET" if results['max_drawdown'] <= dd_goal else "❌ EXCEEDED TARGET"
            return_status = "✅ TARGET MET" if return_goal_min <= results['total_return'] <= return_goal_max else "❌ MISSED TARGET"
            print(f"   BTC Drawdown Goal (≤20%): {dd_status}")
            print(f"   BTC Return Goal (130-160%): {return_status}")
        print()
        
        print(f"⚙️ Strategy Utilization:")
        print(f"   Average Exposure: {results['avg_exposure']:.1f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Regime Transitions: {results.get('regime_transitions', 0)}")
        
        if results['strategy_stats']:
            stats = results['strategy_stats']
            print(f"   Signal Activation Rate: {stats.get('signal_activation_rate', 0):.1%}")
            if 'transition_protection_rate' in stats:
                print(f"   Transition Protection Rate: {stats['transition_protection_rate']:.1%}")
        print()
        print("=" * 80)
        print()
    
    # Summary comparison
    if len(all_results) > 1:
        print("📊 ASSET COMPARISON SUMMARY:")
        print("-" * 50)
        for symbol, results in all_results.items():
            asset_type = "ETH" if "ETH" in symbol else "BTC"
            mode = "Defensive" if asset_type == "ETH" else "Aggressive"
            print(f"{asset_type} ({mode}): {results['total_return']:+.1f}% return, {results['max_drawdown']:.1f}% MDD")
        print()
        
        # Goal achievement summary
        eth_results = all_results.get("KRW-ETH")
        btc_results = all_results.get("KRW-BTC")
        
        if eth_results:
            eth_goal = eth_results['max_drawdown'] <= 25.0
            print(f"ETH Goal (reduce MDD to 20-25%): {'✅ ACHIEVED' if eth_goal else '❌ NOT ACHIEVED'}")
            
        if btc_results:
            btc_dd_goal = btc_results['max_drawdown'] <= 20.0
            btc_return_goal = 130.0 <= btc_results['total_return'] <= 160.0
            btc_goal = btc_dd_goal and btc_return_goal
            print(f"BTC Goal (130-160% return, ≤20% MDD): {'✅ ACHIEVED' if btc_goal else '❌ NOT ACHIEVED'}")
        print()
    
    print(f"💰 Performance:")
    print(f"   Total Return: {results['total_return']:+.2f}%")
    print(f"   Buy & Hold: {results['buy_hold_return']:+.2f}%")
    status = "🎉 OUTPERFORMED" if results['vs_buyhold'] > 0 else "📉 Underperformed"
    print(f"   vs Buy&Hold: {status} by {results['vs_buyhold']:+.2f}%p")
    print()
    
    print(f"🛡️ Risk Management:")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    dd_status = "✅ TARGET MET" if results['max_drawdown'] < 20 else "❌ EXCEEDED TARGET"
    print(f"   Drawdown Goal (<20%): {dd_status}")
    print()
    
    print(f"⚙️ Strategy Utilization:")
    print(f"   Average Exposure: {results['avg_exposure']:.1f}%")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Regime Only: {results['engine_usage']['regime_only']} periods")
    print(f"   Dual Engine: {results['engine_usage']['dual_engine']} periods")
    
    if results['strategy_stats']:
        stats = results['strategy_stats']
        print(f"   Signal Activation Rate: {stats.get('signal_activation_rate', 0):.1%}")
    print()
    
    # Goal assessment
    goal_met = results['max_drawdown'] < 20 and results['vs_buyhold'] > 0
    print(f"🎯 Goal Achievement:")
    print(f"   Enhance profits while keeping DD < 20%: {'✅ SUCCESS' if goal_met else '❌ NEEDS IMPROVEMENT'}")
    
    if goal_met:
        print(f"   🏆 Strategy successfully enhanced returns by {results['vs_buyhold']:+.1f}%p")
        print(f"   🛡️ While maintaining drawdown at {results['max_drawdown']:.1f}% (within 20% target)")
    else:
        print(f"   💡 Recommendations for improvement:")
        if results['max_drawdown'] >= 20:
            print(f"      - Reduce max exposure or strengthen risk controls")
        if results['vs_buyhold'] <= 0:
            print(f"      - Optimize signal engine parameters for better timing")

if __name__ == "__main__":
    main()