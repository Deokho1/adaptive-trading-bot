"""
Configuration for scalp bot

Defines all parameters for the dip scalping strategy.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ScalpConfig:
    """Configuration for the scalp trading bot"""
    
    # Timeframe settings
    candle_interval: str = "5m"
    
    # Spike detection thresholds (negative for down, positive for up)
    spike_down_threshold_5m: float = -1.5  # percent
    spike_down_threshold_15m: float = -3.0  # percent  
    spike_up_threshold_5m: float = 1.5      # percent
    spike_up_threshold_15m: float = 3.0     # percent
    
    # Cooldown after spike detection
    cooldown_bars_after_spike: int = 2
    
    # Entry conditions
    entry_rebound_min_from_low_pct: float = 0.3  # minimum rebound from local low
    min_volume_spike_ratio: float = 1.0          # volume must be this multiple of avg
    
    # Risk management
    take_profit_pct: float = 1.0         # target profit percentage
    stop_loss_pct: float = 0.7           # stop loss percentage  
    max_holding_bars: int = 12           # max bars to hold (1 hour on 5m)
    per_trade_risk_pct: float = 1.0      # risk per trade as % of portfolio
    
    # Position limits
    max_position_per_symbol: int = 1
    max_total_exposure_pct: float = 40.0
    daily_loss_limit_pct: float = 3.0
    
    # Trading costs
    fee_rate: float = 0.0007           # 0.07% fee
    slippage_rate: float = 0.0003      # 0.03% slippage
    
    # Symbols to trade
    symbols: List[str] = None
    
    def __post_init__(self):
        """Set default symbols if none provided"""
        if self.symbols is None:
            self.symbols = ["BTC", "SOL"]
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
            
        if self.per_trade_risk_pct <= 0 or self.per_trade_risk_pct > 10:
            raise ValueError("per_trade_risk_pct must be between 0 and 10")
            
        if self.max_total_exposure_pct <= 0 or self.max_total_exposure_pct > 100:
            raise ValueError("max_total_exposure_pct must be between 0 and 100")
            
        return True


# Default configuration instance
DEFAULT_CONFIG = ScalpConfig()