# 전략 로직 (진입·청산 조건)
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
        
    def make_decision(
        self, 
        market_data: MarketData,
        current_position: Optional[Dict] = None
    ) -> TradingDecision:
        """
        Make trading decision based on current market data and position
        
        Args:
            market_data: Current market data (OHLCV)
            current_position: Current position info (None if no position)
                - If provided: {quantity, entry_price, entry_time}
            
        Returns:
            TradingDecision: Trading decision (BUY, SELL, or HOLD)
        """
        current_price = market_data.close
        
        if current_position is None:
            # === 포지션 없음 → 진입 판단 ===
            # TODO: 사용자가 직접 진입 조건 구현
            # 예시:
            # if 진입_조건_만족:
            #     action = "BUY"
            #     size_usd = 진입_금액
            #     reason = "진입_이유"
            # else:
            #     action = "HOLD"
            #     size_usd = 0.0
            #     reason = "진입_조건_불만족"
            
            action = "HOLD"
            size_usd = 0.0
            reason = "진입_조건_미구현"
            
        else:
            # === 포지션 있음 → 청산 판단 (추가 매수 없음) ===
            entry_price = current_position['entry_price']
            entry_time = current_position['entry_time']
            
            # 손익률 계산
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 보유 시간 계산
            hold_time = market_data.timestamp - entry_time
            
            # TODO: 사용자가 직접 청산 조건 구현
            # 예시:
            # if pnl_pct > take_profit_threshold:
            #     action = "SELL"
            #     reason = f"take_profit_{pnl_pct:.2f}%"
            # elif pnl_pct < stop_loss_threshold:
            #     action = "SELL"
            #     reason = f"stop_loss_{pnl_pct:.2f}%"
            # elif hold_time > max_hold_time:
            #     action = "SELL"
            #     reason = f"max_hold_time_exceeded"
            # else:
            #     action = "HOLD"
            #     reason = "청산_조건_불만족"
            
            action = "HOLD"
            size_usd = 0.0
            reason = "청산_조건_미구현"
        
        return TradingDecision(
            timestamp=market_data.timestamp,
            symbol=self.config.symbol,
            action=action,
            size_usd=size_usd,
            reason=reason,
            price=current_price
        )
