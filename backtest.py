"""
Minimal backtest implementation using the new framework architecture.

This demonstrates how to use the core modules together for a simple backtest loop.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import csv

from core.strategy_core import StrategyConfig, DecisionEngine, MarketData, TradingDecision
from core.exchange_api_backtest import BacktestExchangeAPI


class MinimalBacktestRunner:
    """Minimal backtest runner for framework validation"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize minimal backtest runner
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.equity_history = []
        self.trade_history = []
        
    def generate_sample_data(self, days: int = 1) -> pd.DataFrame:
        """
        Generate sample OHLCV data for testing
        
        Args:
            days: Number of days of data to generate
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        print(f"📊 Generating {days} days of sample OHLCV data...")
        
        # Generate timestamps (1-minute intervals)
        start_time = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, periods=days*1440, freq='1min')
        
        # Generate realistic price data with random walk
        np.random.seed(42)
        base_price = 50000.0
        price_changes = np.random.normal(0, 0.002, len(timestamps))  # 0.2% volatility
        
        # Apply some trend
        trend = np.linspace(0, 0.001, len(timestamps))  # Small upward trend
        price_changes = price_changes + trend
        
        # Calculate cumulative prices
        prices = [base_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Generate OHLCV data
        ohlcv_data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            # Simple OHLC generation
            volatility = abs(np.random.normal(0, 0.001))
            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close_price
            volume = np.random.uniform(100, 1000)
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        print(f"✅ Generated {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    def run_backtest(self, symbol: str = "BTCUSDT", days: int = 1) -> None:
        """
        Run minimal backtest
        
        Args:
            symbol: Trading symbol
            days: Number of days to backtest
        """
        print("=" * 60)
        print("🤖 MINIMAL BACKTEST RUNNER")
        print("=" * 60)
        print(f"Symbol: {symbol}")
        print(f"Duration: {days} days")
        print("Strategy: Mock DecisionEngine (buy every 10 candles)")
        print("-" * 60)
        
        # 1. Generate sample data
        ohlcv_data = self.generate_sample_data(days)
        
        # 2. Initialize components
        config = StrategyConfig(
            symbol=symbol,
            buy_interval=10,  # Buy every 10 candles
            position_size_usd=1000.0
        )
        
        decision_engine = DecisionEngine(config)
        
        exchange = BacktestExchangeAPI(
            ohlcv_data={symbol: ohlcv_data},
            initial_balance=10000.0,
            fee_rate=0.001,  # 0.1%
            slippage_rate=0.0005  # 0.05%
        )
        
        print(f"🏦 Initial Balance: ${exchange.get_balance().total:,.0f}")
        print(f"📈 Processing {len(ohlcv_data)} candles...")
        print("-" * 60)
        
        # 3. Main backtest loop
        for idx, row in ohlcv_data.iterrows():
            # Set current time in exchange
            current_time = row['timestamp']
            exchange.set_current_time(current_time)
            
            # Create market data
            market_data = MarketData(
                timestamp=current_time,
                open=row['open'],
                high=row['high'], 
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # Get trading decision
            decision = decision_engine.make_decision(market_data)
            
            # Get current position and equity
            position = exchange.get_position(symbol)
            current_equity = exchange.get_equity()
            
            # Record equity
            self.equity_history.append({
                'timestamp': current_time,
                'equity': current_equity,
                'price': row['close']
            })
            
            # Execute trades
            if decision.action in ["BUY", "SELL"]:
                try:
                    order = exchange.place_order(
                        symbol=symbol,
                        side=decision.action,
                        size=decision.size_usd,
                        order_type="MARKET"
                    )
                    
                    # Record trade
                    self.trade_history.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'side': decision.action,
                        'size_usd': decision.size_usd,
                        'price': order.filled_price,
                        'reason': decision.reason,
                        'order_id': order.id
                    })
                    
                    # Console log for trade actions
                    position_status = "LONG" if position.size > 0 else "FLAT"
                    print(f"📅 {current_time.strftime('%H:%M:%S')} | "
                          f"💰 ${current_equity:7.0f} | "
                          f"📊 ${row['close']:8.2f} | "
                          f"📍 {position_status:4s} | "
                          f"🔥 {decision.action} ${decision.size_usd:,.0f} ({decision.reason})")
                    
                except Exception as e:
                    print(f"❌ Trade error at {current_time}: {e}")
            
            # Periodic status updates (every 100 candles)
            if idx > 0 and idx % 100 == 0:
                position_status = "LONG" if position.size > 0 else "FLAT"
                print(f"📅 {current_time.strftime('%H:%M:%S')} | "
                      f"💰 ${current_equity:7.0f} | "
                      f"📊 ${row['close']:8.2f} | "
                      f"📍 {position_status:4s} | "
                      f"⏭️  Processing...")
        
        # 4. Final results
        final_equity = exchange.get_equity()
        initial_equity = 10000.0
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        
        print("-" * 60)
        print("📊 BACKTEST RESULTS")
        print("-" * 60)
        print(f"Initial Equity: ${initial_equity:,.0f}")
        print(f"Final Equity:   ${final_equity:,.0f}")
        print(f"Total Return:   {total_return:+.2f}%")
        print(f"Total Trades:   {len(self.trade_history)}")
        
        if self.trade_history:
            buy_trades = len([t for t in self.trade_history if t['side'] == 'BUY'])
            sell_trades = len([t for t in self.trade_history if t['side'] == 'SELL'])
            print(f"Buy Orders:     {buy_trades}")
            print(f"Sell Orders:    {sell_trades}")
        
        print("-" * 60)
        
        # 5. Save results
        self.save_results()
        
        print(f"💾 Results saved to {self.output_dir}/")
        print("✅ Backtest completed successfully!")
        
    def save_results(self) -> None:
        """Save backtest results to CSV files"""
        
        # Save equity curve
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_file = self.output_dir / "equity_curve.csv"
            equity_df.to_csv(equity_file, index=False)
            print(f"📈 Equity curve saved: {equity_file}")
        
        # Save trades
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_file = self.output_dir / "trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💼 Trades saved: {trades_file}")
        
        # Save summary
        summary_file = self.output_dir / "backtest_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            if self.equity_history:
                initial_equity = self.equity_history[0]['equity']
                final_equity = self.equity_history[-1]['equity']
                total_return = ((final_equity - initial_equity) / initial_equity) * 100
                
                writer.writerow(['Initial Equity', f"${initial_equity:.0f}"])
                writer.writerow(['Final Equity', f"${final_equity:.0f}"])
                writer.writerow(['Total Return %', f"{total_return:.2f}%"])
                writer.writerow(['Total Trades', len(self.trade_history)])
                
                if self.trade_history:
                    buy_count = len([t for t in self.trade_history if t['side'] == 'BUY'])
                    sell_count = len([t for t in self.trade_history if t['side'] == 'SELL'])
                    writer.writerow(['Buy Orders', buy_count])
                    writer.writerow(['Sell Orders', sell_count])
        
        print(f"📋 Summary saved: {summary_file}")
        
        # Copy results to OneDrive Bot folder
        self.copy_to_onedrive()
    
    def copy_to_onedrive(self):
        """Copy results to OneDrive Bot folder"""
        import shutil
        import os
        
        onedrive_path = r"C:\Users\DH\OneDrive\문서\Bot"
        results_path = "results"
        
        try:
            # Create OneDrive Bot folder if it doesn't exist
            os.makedirs(onedrive_path, exist_ok=True)
            
            # Copy all result files
            for filename in os.listdir(results_path):
                if filename.endswith('.csv'):
                    src = os.path.join(results_path, filename)
                    dst = os.path.join(onedrive_path, filename)
                    shutil.copy2(src, dst)
                    print(f"📁 Copied {filename} to OneDrive Bot folder")
            
            print(f"✅ All results copied to {onedrive_path}")
            
        except Exception as e:
            print(f"⚠️  OneDrive copy failed: {e}")


def main():
    """Run minimal backtest example"""
    runner = MinimalBacktestRunner()
    runner.run_backtest(symbol="BTCUSDT", days=1)


if __name__ == "__main__":
    main()