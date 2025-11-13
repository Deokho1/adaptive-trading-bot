"""
전략 로직 (진입·청산 조건)
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

from core.market_watcher import MarketWatcher
from core.signal_engine import SignalEngine

# config에서 전략 파라미터 읽기
try:
    import config as app_config
    STRATEGY_CONFIG = app_config.CONFIG.get("strategy", {})
except:
    STRATEGY_CONFIG = {}


@dataclass
class StrategyConfig:
    """전략 설정 파라미터 - config.py에서 관리"""
    
    # 기본 설정
    symbol: str = "KRW-BTC"
    timeframe: str = "1m"
    
    # RSI 파라미터 (config.py에서 읽어옴, 없으면 기본값 사용)
    rsi_period: int = None
    rsi_oversold: float = None
    rsi_exit: float = None
    
    # 진입 조건 (config.py에서 읽어옴, 없으면 기본값 사용)
    entry_volume_ratio: float = None
    entry_position_size_ratio: float = None
    
    # 청산 조건 (config.py에서 읽어옴, 없으면 기본값 사용)
    take_profit_pct: float = None
    stop_loss_pct: float = None
    max_hold_time_minutes: int = None
    
    def __post_init__(self):
        """config.py에서 값을 읽어와서 설정 (없으면 기본값 사용)"""
        # RSI 파라미터
        self.rsi_period = self.rsi_period if self.rsi_period is not None else STRATEGY_CONFIG.get("rsi_period", 14)
        self.rsi_oversold = self.rsi_oversold if self.rsi_oversold is not None else STRATEGY_CONFIG.get("rsi_oversold", 30.0)
        self.rsi_exit = self.rsi_exit if self.rsi_exit is not None else STRATEGY_CONFIG.get("rsi_exit", 50.0)
        
        # 진입 조건
        self.entry_volume_ratio = self.entry_volume_ratio if self.entry_volume_ratio is not None else STRATEGY_CONFIG.get("entry_volume_ratio", 1.2)
        self.entry_position_size_ratio = self.entry_position_size_ratio if self.entry_position_size_ratio is not None else STRATEGY_CONFIG.get("entry_position_size_ratio", 0.1)
        
        # 청산 조건
        self.take_profit_pct = self.take_profit_pct if self.take_profit_pct is not None else STRATEGY_CONFIG.get("take_profit_pct", 0.3)
        self.stop_loss_pct = self.stop_loss_pct if self.stop_loss_pct is not None else STRATEGY_CONFIG.get("stop_loss_pct", -0.2)
        self.max_hold_time_minutes = self.max_hold_time_minutes if self.max_hold_time_minutes is not None else STRATEGY_CONFIG.get("max_hold_time_minutes", 5)
    

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
    RSI 기반 단타 전략 의사결정 엔진
    
    백테스트와 라이브 모드에서 공통으로 사용 가능합니다.
    
    역할:
    1. MarketWatcher로 지표 계산
    2. SignalEngine으로 신호 생성
    3. 포지션 상태별 결정 (손절/익절/시간 체크)
    4. TradingDecision 반환
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize decision engine
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        
        # 모듈 초기화
        self.market_watcher = MarketWatcher(config)
        self.signal_engine = SignalEngine(config)
        
    def initialize_with_history(self, historical_candles: List[MarketData]):
        """
        라이브 모드 시작 시 과거 데이터로 초기화
        
        Args:
            historical_candles: 과거 캔들 데이터 리스트 (시간순)
        """
        self.market_watcher.initialize_with_history(historical_candles)
        print(f"[OK] DecisionEngine initialized")
    
    def make_decision(
        self, 
        market_data: MarketData,
        current_position: Optional[Dict] = None,
        available_balance: Optional[float] = None
    ) -> TradingDecision:
        """
        Make trading decision based on current market data and position
        
        Args:
            market_data: Current market data (OHLCV)
            current_position: Current position info (None if no position)
                - If provided: {quantity, entry_price, entry_time}
            available_balance: 사용 가능한 잔고 (KRW) - 진입 금액 계산용
        
        Returns:
            TradingDecision: Trading decision (BUY, SELL, or HOLD)
        """
        # 1. 시장 데이터 업데이트
        self.market_watcher.update(market_data)
        
        current_price = market_data.close
        
        # 2. 지표 계산 가능 여부 확인
        if not self.market_watcher.is_ready:
            return TradingDecision(
                timestamp=market_data.timestamp,
                symbol=self.config.symbol,
                action="HOLD",
                size_usd=0.0,
                reason="RSI_계산_데이터_부족",
                price=current_price
            )
        
        # 3. 지표 계산
        indicators = self.market_watcher.get_indicators(current_volume=market_data.volume)
        
        if current_position is None:
            # === 포지션 없음 → 진입 판단 ===
            
            # 4. 신호 생성
            signal = self.signal_engine.generate_signal(
                market_data=market_data,
                indicators=indicators,
                current_position=None
            )
            
            if signal and signal.action == "BUY":
                # 진입 금액 계산
                if available_balance is None:
                    size_usd = 1000000  # 기본값: 100만원
                else:
                    size_usd = available_balance * self.config.entry_position_size_ratio
                
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="BUY",
                    size_usd=size_usd,
                    reason=signal.reason,
                    price=current_price
                )
            else:
                # 진입 신호 없음
                rsi = indicators.get('rsi', 0)
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="HOLD",
                    size_usd=0.0,
                    reason=f"RSI_진입_조건_불만족_RSI={rsi:.1f}" if rsi else "RSI_계산_실패",
                    price=current_price
                )
        
        else:
            # === 포지션 있음 → 청산 판단 (추가 매수 없음) ===
            entry_price = current_position['entry_price']
            entry_time = current_position['entry_time']
            
            # 손익률 계산
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 보유 시간 계산 (분 단위)
            # 타임존 일치 처리 (둘 다 naive로 변환)
            from datetime import datetime
            market_ts = market_data.timestamp
            entry_ts = entry_time
            
            # 타임존 제거
            if hasattr(market_ts, 'tzinfo') and market_ts.tzinfo is not None:
                if isinstance(market_ts, datetime):
                    market_ts = market_ts.replace(tzinfo=None)
            if hasattr(entry_ts, 'tzinfo') and entry_ts.tzinfo is not None:
                if isinstance(entry_ts, datetime):
                    entry_ts = entry_ts.replace(tzinfo=None)
            
            hold_time = market_ts - entry_ts
            hold_time_minutes = hold_time.total_seconds() / 60
            
            # 청산 조건 체크 (우선순위 순)
            
            # 1순위: 손절
            if pnl_pct <= self.config.stop_loss_pct:
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="SELL",
                    size_usd=0.0,  # 전체 청산
                    reason=f"손절_{pnl_pct:.2f}%",
                    price=current_price
                )
            
            # 2순위: 익절
            if pnl_pct >= self.config.take_profit_pct:
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="SELL",
                    size_usd=0.0,  # 전체 청산
                    reason=f"익절_{pnl_pct:.2f}%",
                    price=current_price
                )
            
            # 3순위: 최대 보유 시간 초과
            if hold_time_minutes >= self.config.max_hold_time_minutes:
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="SELL",
                    size_usd=0.0,  # 전체 청산
                    reason=f"최대_보유시간_초과_{hold_time_minutes:.1f}분",
                    price=current_price
                )
            
            # 4순위: RSI 중립선 회복 (SignalEngine 사용)
            signal = self.signal_engine.generate_signal(
                market_data=market_data,
                indicators=indicators,
                current_position=current_position
            )
            
            if signal and signal.action == "SELL":
                return TradingDecision(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="SELL",
                    size_usd=0.0,  # 전체 청산
                    reason=signal.reason,
                    price=current_price
                )
            
            # 청산 조건 불만족 → 보유
            rsi = indicators.get('rsi')
            rsi_str = f"{rsi:.1f}" if rsi else "N/A"
            return TradingDecision(
                timestamp=market_data.timestamp,
                symbol=self.config.symbol,
                action="HOLD",
                size_usd=0.0,
                reason=f"보유중_PnL={pnl_pct:.2f}%_RSI={rsi_str}_Hold={hold_time_minutes:.1f}분",
                price=current_price
            )
