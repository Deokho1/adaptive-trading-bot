"""
Core strategy implementation for scalping trading bot.

[ABSOLUTE RULE] 
이 파일 내의 로직/숫자/조건은 사람(사용자)이 직접 설계합니다.
자동화 도구는 함수 시그니처와 구조만 제공하고, 
실제 진입/청산 조건, 지표 계산, 파라미터는 건드리지 않습니다.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd


@dataclass
class StrategyConfig:
    """전략 설정 파라미터 - 사용자가 직접 값을 설정"""
    
    # Mock strategy parameters for minimal implementation
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    buy_interval: int = 10  # Buy every 10 candles (for testing)
    position_size_usd: float = 1000.0  # Fixed position size for testing
    

@dataclass
class MarketData:
    """시장 데이터 구조체"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """거래 신호 구조체"""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    signal_type: str  # "dip_buy", "breakout", "exit" etc.
    strength: float  # 0.0 ~ 1.0
    price: float
    indicators: Dict[str, float]  # RSI, EMA 등 지표 값들
    reason: str  # 신호 발생 이유


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # "LONG", "SHORT"
    size: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    

@dataclass
class TradingDecision:
    """거래 결정 구조체"""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    size_usd: float  # Position size in USD
    reason: str  # Decision reason
    price: float  # Current market price


class DecisionEngine:
    """
    Simple decision engine for minimal backtest implementation
    
    This is a mock implementation for testing the framework.
    Real strategy logic should be implemented by the user.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize decision engine
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.candle_count = 0
        self.has_position = False
        self.last_buy_price = 0.0
        
    def make_decision(self, bar: MarketData) -> TradingDecision:
        """
        Make trading decision based on current market data
        
        Args:
            bar: Current market data
            
        Returns:
            TradingDecision: Trading decision
        """
        self.candle_count += 1
        current_price = bar.close
        
        # Simple mock logic: Buy every N candles, sell when profit > 1%
        action = "HOLD"
        size_usd = 0.0
        reason = "no_signal"
        
        if not self.has_position:
            # Buy logic: every N candles
            if self.candle_count % self.config.buy_interval == 0:
                action = "BUY"
                size_usd = self.config.position_size_usd
                reason = f"buy_every_{self.config.buy_interval}_candles"
                self.has_position = True
                self.last_buy_price = current_price
        else:
            # Sell logic: when profit > 1% or loss > -0.5%
            pnl_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
            
            if pnl_pct > 1.0:
                action = "SELL"
                size_usd = self.config.position_size_usd
                reason = f"take_profit_at_{pnl_pct:.2f}%"
                self.has_position = False
            elif pnl_pct < -0.5:
                action = "SELL"
                size_usd = self.config.position_size_usd
                reason = f"stop_loss_at_{pnl_pct:.2f}%"
                self.has_position = False
        
        return TradingDecision(
            timestamp=bar.timestamp,
            symbol=self.config.symbol,
            action=action,
            size_usd=size_usd,
            reason=reason,
            price=current_price
        )