"""
ExecutionEngine Module

Handles order execution with Upbit API integration
Supports both real trading and dry-run simulation mode
"""

import pyupbit
from typing import Dict, Optional
import logging
from datetime import datetime


class ExecutionEngine:
    """Manages trade execution on Upbit exchange"""
    
    def __init__(self, config: Dict, access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None, dry_run: bool = True):
        """
        Initialize ExecutionEngine
        
        Args:
            config: Configuration dictionary
            access_key: Upbit API access key
            secret_key: Upbit API secret key
            dry_run: If True, simulate trades without actual execution
        """
        self.config = config
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        # Initialize Upbit connection
        if not dry_run and access_key and secret_key:
            try:
                self.upbit = pyupbit.Upbit(access_key, secret_key)
                self.logger.info("Connected to Upbit API for live trading")
            except Exception as e:
                self.logger.error(f"Failed to connect to Upbit API: {e}")
                self.upbit = None
        else:
            self.upbit = None
            self.logger.info("Running in DRY RUN mode - no actual trades will be executed")
        
        # Simulated balances for dry run
        self.simulated_krw_balance = 1000000  # Start with 1M KRW
        self.simulated_crypto_balance = 0
        self.simulated_avg_buy_price = 0
    
    def get_balance(self, ticker: str = "KRW") -> float:
        """
        Get balance for a specific ticker
        
        Args:
            ticker: Ticker symbol (e.g., "KRW", "BTC")
            
        Returns:
            Balance amount
        """
        if self.dry_run:
            if ticker == "KRW":
                return self.simulated_krw_balance
            else:
                return self.simulated_crypto_balance
        else:
            if self.upbit is None:
                self.logger.error("Upbit connection not initialized")
                return 0
            
            try:
                balance = self.upbit.get_balance(ticker)
                return balance if balance else 0
            except Exception as e:
                self.logger.error(f"Error getting balance for {ticker}: {e}")
                return 0
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for a ticker
        
        Args:
            ticker: Trading pair (e.g., "KRW-BTC")
            
        Returns:
            Current price or None if error
        """
        try:
            price = pyupbit.get_current_price(ticker)
            return price
        except Exception as e:
            self.logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    def get_ohlcv(self, ticker: str, interval: str = "day", count: int = 200) -> Optional:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            ticker: Trading pair (e.g., "KRW-BTC")
            interval: Time interval ('minute1', 'minute3', 'minute5', 'minute10', 
                     'minute15', 'minute30', 'minute60', 'minute240', 'day', 'week', 'month')
            count: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            return df
        except Exception as e:
            self.logger.error(f"Error getting OHLCV data for {ticker}: {e}")
            return None
    
    def execute_buy(self, ticker: str, amount: float, price: float) -> Dict:
        """
        Execute a buy order
        
        Args:
            ticker: Trading pair (e.g., "KRW-BTC")
            amount: Amount to buy
            price: Price per unit
            
        Returns:
            Dictionary with order result
        """
        total_cost = amount * price
        
        if self.dry_run:
            # Simulate buy order
            if total_cost <= self.simulated_krw_balance:
                self.simulated_krw_balance -= total_cost
                old_balance = self.simulated_crypto_balance
                self.simulated_crypto_balance += amount
                
                # Update average buy price
                if old_balance > 0:
                    self.simulated_avg_buy_price = (
                        (self.simulated_avg_buy_price * old_balance + price * amount) / 
                        self.simulated_crypto_balance
                    )
                else:
                    self.simulated_avg_buy_price = price
                
                result = {
                    'status': 'success',
                    'type': 'buy',
                    'ticker': ticker,
                    'amount': amount,
                    'price': price,
                    'total_cost': total_cost,
                    'timestamp': datetime.now(),
                    'dry_run': True
                }
                self.logger.info(f"[DRY RUN] BUY executed: {amount:.8f} at {price:.2f} KRW "
                               f"(Total: {total_cost:.2f} KRW)")
                return result
            else:
                self.logger.error(f"[DRY RUN] Insufficient KRW balance: {self.simulated_krw_balance:.2f}")
                return {'status': 'failed', 'reason': 'Insufficient balance'}
        else:
            # Real buy order
            if self.upbit is None:
                self.logger.error("Upbit connection not initialized")
                return {'status': 'failed', 'reason': 'No Upbit connection'}
            
            try:
                result = self.upbit.buy_market_order(ticker, total_cost)
                self.logger.info(f"BUY executed: {amount:.8f} at {price:.2f} KRW")
                return {
                    'status': 'success',
                    'type': 'buy',
                    'result': result,
                    'dry_run': False
                }
            except Exception as e:
                self.logger.error(f"Error executing buy order: {e}")
                return {'status': 'failed', 'reason': str(e)}
    
    def execute_sell(self, ticker: str, amount: float, price: float) -> Dict:
        """
        Execute a sell order
        
        Args:
            ticker: Trading pair (e.g., "KRW-BTC")
            amount: Amount to sell
            price: Current price per unit
            
        Returns:
            Dictionary with order result
        """
        total_value = amount * price
        
        if self.dry_run:
            # Simulate sell order
            if amount <= self.simulated_crypto_balance:
                self.simulated_crypto_balance -= amount
                self.simulated_krw_balance += total_value
                
                # Calculate profit
                profit = (price - self.simulated_avg_buy_price) * amount
                profit_pct = ((price - self.simulated_avg_buy_price) / 
                             self.simulated_avg_buy_price * 100)
                
                # Reset avg buy price if no position
                if self.simulated_crypto_balance == 0:
                    self.simulated_avg_buy_price = 0
                
                result = {
                    'status': 'success',
                    'type': 'sell',
                    'ticker': ticker,
                    'amount': amount,
                    'price': price,
                    'total_value': total_value,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'timestamp': datetime.now(),
                    'dry_run': True
                }
                self.logger.info(f"[DRY RUN] SELL executed: {amount:.8f} at {price:.2f} KRW "
                               f"(Total: {total_value:.2f} KRW, Profit: {profit:.2f} KRW, {profit_pct:.2f}%)")
                return result
            else:
                self.logger.error(f"[DRY RUN] Insufficient crypto balance: {self.simulated_crypto_balance:.8f}")
                return {'status': 'failed', 'reason': 'Insufficient balance'}
        else:
            # Real sell order
            if self.upbit is None:
                self.logger.error("Upbit connection not initialized")
                return {'status': 'failed', 'reason': 'No Upbit connection'}
            
            try:
                # Extract crypto ticker (e.g., "BTC" from "KRW-BTC")
                crypto_ticker = ticker.split('-')[1] if '-' in ticker else ticker
                result = self.upbit.sell_market_order(ticker, amount)
                self.logger.info(f"SELL executed: {amount:.8f} at {price:.2f} KRW")
                return {
                    'status': 'success',
                    'type': 'sell',
                    'result': result,
                    'dry_run': False
                }
            except Exception as e:
                self.logger.error(f"Error executing sell order: {e}")
                return {'status': 'failed', 'reason': str(e)}
    
    def get_avg_buy_price(self, ticker: str) -> float:
        """
        Get average buy price for a ticker
        
        Args:
            ticker: Crypto ticker (e.g., "BTC")
            
        Returns:
            Average buy price
        """
        if self.dry_run:
            return self.simulated_avg_buy_price
        else:
            if self.upbit is None:
                return 0
            
            try:
                return self.upbit.get_avg_buy_price(ticker)
            except Exception as e:
                self.logger.error(f"Error getting avg buy price: {e}")
                return 0
