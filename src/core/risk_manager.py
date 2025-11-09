"""
RiskManager Module

Manages trading risks including:
- Position sizing
- Stop loss
- Take profit
- Maximum position limits
"""

from typing import Dict, Optional
import logging


class RiskManager:
    """Manages risk for trading operations"""
    
    def __init__(self, config: Dict):
        """
        Initialize RiskManager
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.max_position_size = config.get('max_position_size', 0.95)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', 0.10)
    
    def calculate_position_size(self, available_krw: float, current_price: float, 
                                investment_ratio: float = None) -> float:
        """
        Calculate position size based on available capital
        
        Args:
            available_krw: Available KRW balance
            current_price: Current price of the asset
            investment_ratio: Ratio of available capital to invest (default from config)
            
        Returns:
            Amount of crypto to buy
        """
        if investment_ratio is None:
            investment_ratio = self.max_position_size
        
        # Calculate investment amount
        investment_amount = available_krw * investment_ratio
        
        # Calculate position size
        position_size = investment_amount / current_price
        
        self.logger.info(f"Position size calculated: {position_size:.8f} units "
                        f"({investment_amount:.2f} KRW at {current_price:.2f})")
        
        return position_size
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """
        Check if stop loss should be triggered
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current market price
            
        Returns:
            True if stop loss should be triggered
        """
        loss_pct = (entry_price - current_price) / entry_price
        
        if loss_pct >= self.stop_loss_pct:
            self.logger.warning(f"Stop loss triggered! Loss: {loss_pct*100:.2f}% "
                              f"(threshold: {self.stop_loss_pct*100:.2f}%)")
            return True
        
        return False
    
    def check_take_profit(self, entry_price: float, current_price: float) -> bool:
        """
        Check if take profit should be triggered
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current market price
            
        Returns:
            True if take profit should be triggered
        """
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct >= self.take_profit_pct:
            self.logger.info(f"Take profit triggered! Profit: {profit_pct*100:.2f}% "
                           f"(threshold: {self.take_profit_pct*100:.2f}%)")
            return True
        
        return False
    
    def should_exit_position(self, entry_price: float, current_price: float) -> Optional[str]:
        """
        Check if position should be exited based on risk management rules
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current market price
            
        Returns:
            Reason for exit ('stop_loss', 'take_profit') or None
        """
        if self.check_stop_loss(entry_price, current_price):
            return 'stop_loss'
        
        if self.check_take_profit(entry_price, current_price):
            return 'take_profit'
        
        return None
    
    def validate_order(self, order_type: str, amount: float, price: float, 
                      available_balance: float) -> Tuple[bool, str]:
        """
        Validate if an order can be placed
        
        Args:
            order_type: 'buy' or 'sell'
            amount: Amount to trade
            price: Price per unit
            available_balance: Available balance (KRW for buy, crypto for sell)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if order_type == 'buy':
            required_krw = amount * price
            if required_krw > available_balance:
                return False, f"Insufficient KRW: required {required_krw:.2f}, available {available_balance:.2f}"
        
        elif order_type == 'sell':
            if amount > available_balance:
                return False, f"Insufficient crypto: required {amount:.8f}, available {available_balance:.8f}"
        
        else:
            return False, f"Invalid order type: {order_type}"
        
        return True, "Order validated"
    
    def get_risk_metrics(self, entry_price: float, current_price: float, 
                         position_size: float) -> Dict:
        """
        Calculate current risk metrics for a position
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current market price
            position_size: Size of the position
            
        Returns:
            Dictionary with risk metrics
        """
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        distance_to_stop_loss = ((entry_price * (1 - self.stop_loss_pct)) - current_price) / current_price
        distance_to_take_profit = ((entry_price * (1 + self.take_profit_pct)) - current_price) / current_price
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'distance_to_stop_loss': distance_to_stop_loss,
            'distance_to_take_profit': distance_to_take_profit,
            'stop_loss_price': entry_price * (1 - self.stop_loss_pct),
            'take_profit_price': entry_price * (1 + self.take_profit_pct)
        }


from typing import Tuple
