"""
시장 감시 모듈
Market data monitoring and indicator calculation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from core.strategy_core import StrategyConfig, MarketData


class MarketWatcher:
    """
    시장 데이터 모니터링 및 지표 계산
    
    역할:
    1. 시장 데이터 히스토리 관리 (순환 버퍼)
    2. 기술적 지표 계산 (RSI, 거래량 비율 등)
    3. 지표 계산 가능 여부 확인
    """
    
    def __init__(self, config):
        """
        Initialize market watcher
        
        Args:
            config: Strategy configuration (StrategyConfig)
        """
        # 런타임 import (순환 import 방지)
        from core.strategy_core import StrategyConfig
        self.config = config
        
        # 과거 데이터 저장 (순환 버퍼)
        self.price_history: List[float] = []  # 종가
        self.volume_history: List[float] = []  # 거래량
        self.timestamp_history: List[datetime] = []  # 타임스탬프
        
        # 필요한 최대 히스토리 길이
        self.max_history = max(30, config.rsi_period + 10)
        
        # 초기화 완료 여부 (RSI 계산 가능한지)
        self.is_ready = False
    
    def update(self, market_data):
        """
        새로운 캔들 데이터 추가 (백테스트/라이브 공통)
        
        Args:
            market_data: 새로운 캔들 데이터 (MarketData)
        """
        # 히스토리에 추가
        self.price_history.append(market_data.close)
        self.volume_history.append(market_data.volume)
        self.timestamp_history.append(market_data.timestamp)
        
        # 최대 개수 유지 (순환 버퍼)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            self.timestamp_history.pop(0)
        
        # RSI 계산 가능 여부 확인
        if len(self.price_history) >= self.config.rsi_period + 1:
            self.is_ready = True
    
    def get_rsi(self, period: int = None) -> Optional[float]:
        """
        RSI 계산
        
        Args:
            period: RSI 계산 기간 (None이면 config 사용)
        
        Returns:
            RSI 값 (0~100) 또는 None (데이터 부족)
        """
        if period is None:
            period = self.config.rsi_period
        
        if len(self.price_history) < period + 1:
            return None
        
        # 가격 변화량 계산
        changes = []
        start_idx = len(self.price_history) - period - 1
        
        for i in range(start_idx + 1, len(self.price_history)):
            change = self.price_history[i] - self.price_history[i-1]
            changes.append(change)
        
        # 상승분/하락분 분리
        ups = [c for c in changes if c > 0]
        downs = [-c for c in changes if c < 0]
        
        # 평균 계산
        avg_up = sum(ups) / period if len(ups) > 0 else 0
        avg_down = sum(downs) / period if len(downs) > 0 else 0
        
        if avg_down == 0:
            return 100.0  # 계속 상승만 한 경우
        
        # RS 및 RSI 계산
        rs = avg_up / avg_down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_volume_ratio(self, current_volume: float, lookback: int = 10) -> Optional[float]:
        """
        현재 거래량 / 평균 거래량 비율 계산
        
        Args:
            current_volume: 현재 거래량
            lookback: 평균 계산 기간
        
        Returns:
            거래량 비율 또는 None (데이터 부족)
        """
        if len(self.volume_history) < lookback:
            return None
        
        avg_volume = sum(self.volume_history[-lookback:]) / lookback
        if avg_volume == 0:
            return None
        
        return current_volume / avg_volume
    
    def get_indicators(self, current_volume: float = None) -> Dict[str, float]:
        """
        모든 지표 반환
        
        Args:
            current_volume: 현재 거래량 (거래량 비율 계산용)
        
        Returns:
            지표 딕셔너리 {rsi, volume_ratio, ...}
        """
        indicators = {}
        
        rsi = self.get_rsi()
        if rsi is not None:
            indicators['rsi'] = rsi
        
        if current_volume is not None:
            volume_ratio = self.get_volume_ratio(current_volume)
            if volume_ratio is not None:
                indicators['volume_ratio'] = volume_ratio
        
        return indicators
    
    
    def initialize_with_history(self, historical_candles):
        """
        라이브 모드 시작 시 과거 데이터로 초기화
        
        Args:
            historical_candles: 과거 캔들 데이터 리스트 (시간순, List[MarketData])
        """
        # 기존 히스토리 초기화
        self.price_history = []
        self.volume_history = []
        self.timestamp_history = []
        self.is_ready = False
        
        # 과거 데이터 추가
        for candle in historical_candles:
            self.update(candle)
        
        print(f"[OK] MarketWatcher initialized with {len(self.price_history)} candles")
