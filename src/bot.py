"""
Main Trading Bot Module

Orchestrates all components to run the adaptive dual-mode trading bot
"""

import time
import yaml
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Dict, Optional

from src.core import MarketAnalyzer, StrategyManager, RiskManager, ExecutionEngine
from src.utils import RateLimiter, PositionTracker, setup_logger


class AdaptiveTradingBot:
    """Main trading bot that coordinates all components"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logger
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file')
        )
        
        self.logger.info("=" * 60)
        self.logger.info("Adaptive Dual-Mode Crypto Trading Bot Starting")
        self.logger.info("=" * 60)
        
        # Load environment variables
        load_dotenv()
        
        # Get trading settings
        trading_config = self.config.get('trading', {})
        self.ticker = trading_config.get('ticker', 'KRW-BTC')
        self.investment_ratio = trading_config.get('investment_ratio', 0.95)
        self.check_interval = trading_config.get('check_interval', 10)
        
        # Determine dry run mode
        self.dry_run = trading_config.get('dry_run', True)
        if os.getenv('DRY_RUN', 'True').lower() == 'false':
            self.dry_run = False
        
        # Get API keys
        access_key = os.getenv('UPBIT_ACCESS_KEY')
        secret_key = os.getenv('UPBIT_SECRET_KEY')
        
        # Initialize components
        self.logger.info("Initializing components...")
        
        # Market Analyzer
        market_config = self.config.get('strategy', {}).get('market_analysis', {})
        self.market_analyzer = MarketAnalyzer(market_config)
        
        # Strategy Manager
        strategy_config = self.config.get('strategy', {})
        self.strategy_manager = StrategyManager(strategy_config, self.market_analyzer)
        
        # Risk Manager
        risk_config = self.config.get('risk', {})
        self.risk_manager = RiskManager(risk_config)
        
        # Execution Engine
        self.execution_engine = ExecutionEngine(
            self.config,
            access_key=access_key,
            secret_key=secret_key,
            dry_run=self.dry_run
        )
        
        # Rate Limiter
        rate_config = self.config.get('rate_limit', {})
        self.rate_limiter = RateLimiter(
            requests_per_second=rate_config.get('requests_per_second', 8),
            requests_per_minute=rate_config.get('requests_per_minute', 200)
        )
        
        # Position Tracker
        self.position_tracker = PositionTracker()
        
        self.logger.info(f"Bot initialized - Ticker: {self.ticker}, Dry Run: {self.dry_run}")
        self.logger.info("Components: MarketAnalyzer, StrategyManager, RiskManager, ExecutionEngine, RateLimiter")
        
        # Running state
        self.running = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default config
            return {
                'trading': {'ticker': 'KRW-BTC', 'investment_ratio': 0.95, 'check_interval': 10, 'dry_run': True},
                'strategy': {'trend': {'k_value': 0.5}, 'range': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}},
                'risk': {'max_position_size': 0.95, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.10},
                'rate_limit': {'requests_per_second': 8, 'requests_per_minute': 200},
                'logging': {'level': 'INFO'}
            }
    
    def _check_and_execute_trade(self):
        """Main trading logic loop iteration"""
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Get current market data
            df = self.execution_engine.get_ohlcv(self.ticker, interval='day', count=200)
            if df is None or len(df) < 30:
                self.logger.warning("Insufficient market data")
                return
            
            # Get current price
            current_price = self.execution_engine.get_current_price(self.ticker)
            if current_price is None:
                self.logger.warning("Could not get current price")
                return
            
            # Check if we have a position
            has_position = self.position_tracker.has_position()
            
            # Get trading signal
            signal, details = self.strategy_manager.get_trading_signal(df, current_price, has_position)
            
            self.logger.info(f"Current Price: {current_price:.2f} KRW | Strategy: {details.get('strategy', 'N/A')} | Signal: {signal}")
            
            # Handle position management
            if has_position:
                # Check risk management rules
                entry_price = self.position_tracker.get_entry_price()
                exit_reason = self.risk_manager.should_exit_position(entry_price, current_price)
                
                if exit_reason:
                    self._execute_sell(current_price, exit_reason)
                elif signal == 'sell':
                    self._execute_sell(current_price, 'strategy_signal')
                else:
                    # Log position status
                    metrics = self.risk_manager.get_risk_metrics(
                        entry_price, 
                        current_price, 
                        self.position_tracker.get_position_amount()
                    )
                    self.logger.info(f"Position P/L: {metrics['unrealized_pnl_pct']:.2f}% | "
                                   f"Stop Loss: {metrics['stop_loss_price']:.2f} | "
                                   f"Take Profit: {metrics['take_profit_price']:.2f}")
            else:
                # No position - check for buy signal
                if signal == 'buy':
                    self._execute_buy(current_price, details)
        
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}", exc_info=True)
    
    def _execute_buy(self, current_price: float, signal_details: Dict):
        """Execute buy order"""
        try:
            # Get available balance
            krw_balance = self.execution_engine.get_balance("KRW")
            
            if krw_balance < 5000:  # Minimum 5000 KRW for Upbit
                self.logger.warning(f"Insufficient KRW balance: {krw_balance:.2f}")
                return
            
            # Calculate position size
            amount = self.risk_manager.calculate_position_size(
                krw_balance, 
                current_price, 
                self.investment_ratio
            )
            
            # Validate order
            is_valid, message = self.risk_manager.validate_order(
                'buy', amount, current_price, krw_balance
            )
            
            if not is_valid:
                self.logger.warning(f"Buy order validation failed: {message}")
                return
            
            # Execute buy
            result = self.execution_engine.execute_buy(self.ticker, amount, current_price)
            
            if result.get('status') == 'success':
                # Update position tracker
                self.position_tracker.open_position(self.ticker, current_price, amount)
                self.logger.info(f"✓ BUY ORDER EXECUTED: {amount:.8f} at {current_price:.2f} KRW")
            else:
                self.logger.error(f"Buy order failed: {result.get('reason', 'Unknown error')}")
        
        except Exception as e:
            self.logger.error(f"Error executing buy: {e}", exc_info=True)
    
    def _execute_sell(self, current_price: float, reason: str):
        """Execute sell order"""
        try:
            # Get position details
            amount = self.position_tracker.get_position_amount()
            
            if amount <= 0:
                self.logger.warning("No position to sell")
                return
            
            # Validate order
            crypto_ticker = self.ticker.split('-')[1] if '-' in self.ticker else self.ticker
            crypto_balance = self.execution_engine.get_balance(crypto_ticker)
            
            is_valid, message = self.risk_manager.validate_order(
                'sell', amount, current_price, crypto_balance
            )
            
            if not is_valid:
                self.logger.warning(f"Sell order validation failed: {message}")
                return
            
            # Execute sell
            result = self.execution_engine.execute_sell(self.ticker, amount, current_price)
            
            if result.get('status') == 'success':
                # Update position tracker
                trade = self.position_tracker.close_position(current_price, reason)
                self.logger.info(f"✓ SELL ORDER EXECUTED: {amount:.8f} at {current_price:.2f} KRW")
                self.logger.info(f"  Trade P/L: {trade.get('profit', 0):.2f} KRW ({trade.get('profit_pct', 0):.2f}%)")
                self.logger.info(f"  Reason: {reason}")
            else:
                self.logger.error(f"Sell order failed: {result.get('reason', 'Unknown error')}")
        
        except Exception as e:
            self.logger.error(f"Error executing sell: {e}", exc_info=True)
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        self.logger.info("Trading bot started!")
        
        try:
            while self.running:
                self._check_and_execute_trade()
                
                # Wait for next check
                self.logger.debug(f"Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal...")
            self.stop()
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        
        # Print performance summary
        summary = self.position_tracker.get_performance_summary()
        self.logger.info("=" * 60)
        self.logger.info("Performance Summary:")
        self.logger.info(f"  Total Trades: {summary['total_trades']}")
        self.logger.info(f"  Winning Trades: {summary['winning_trades']}")
        self.logger.info(f"  Losing Trades: {summary['losing_trades']}")
        self.logger.info(f"  Win Rate: {summary['win_rate']:.2f}%")
        self.logger.info(f"  Total Profit: {summary['total_profit']:.2f} KRW")
        
        if self.dry_run:
            krw_balance = self.execution_engine.get_balance("KRW")
            self.logger.info(f"  Final Balance: {krw_balance:.2f} KRW")
        
        self.logger.info("=" * 60)
        self.logger.info("Trading bot stopped.")


if __name__ == "__main__":
    bot = AdaptiveTradingBot()
    bot.start()
