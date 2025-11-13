"""
시그널 엔진
Signal generation and filtering based on market indicators.
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from core.strategy_core import StrategyConfig, MarketData, Signal


class SignalEngine:
    """
    거래 신호 생성 및 필터링
    
    역할:
    1. 지표를 기반으로 거래 신호 생성
    2. 신호 필터링 (거래량 필터 등)
    3. Signal 객체 생성
    """
    
    def __init__(self, config):
        """
        Initialize signal engine
        
        Args:
            config: Strategy configuration (StrategyConfig)
        """
        self.config = config
    
    def generate_signal(
        self,
        market_data,
        indicators: Dict[str, float],
        current_position: Optional[Dict] = None
    ):
        """
        지표를 기반으로 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            indicators: 지표 딕셔너리 (rsi, volume_ratio 등)
            current_position: 현재 포지션 정보 (None이면 포지션 없음)
        
        Returns:
            Signal 객체 또는 None (신호 없음)
        """
        # 런타임 import (순환 import 방지)
        from core.strategy_core import Signal
        
        current_price = market_data.close
        rsi = indicators.get('rsi')
        volume_ratio = indicators.get('volume_ratio')
        
        if current_position is None:
            # === 포지션 없음 → 진입 신호 생성 ===
            
            # RSI 확인
            if rsi is None:
                return None
            
            # 진입 조건: RSI < 과매도 기준
            if rsi < self.config.rsi_oversold:
                # 거래량 필터 (선택적)
                if self.config.entry_volume_ratio > 0:
                    if volume_ratio is None or volume_ratio < self.config.entry_volume_ratio:
                        # 거래량 부족으로 신호 없음
                        return None
                
                # BUY 신호 생성
                signal_strength = self._calculate_buy_signal_strength(rsi, volume_ratio)
                
                return Signal(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="BUY",
                    signal_type="rsi_oversold",
                    strength=signal_strength,
                    price=current_price,
                    indicators=indicators,
                    reason=f"RSI_과매도_진입_RSI={rsi:.1f}"
                )
            
            # 진입 조건 불만족
            return None
        
        else:
            # === 포지션 있음 → 청산 신호 생성 ===
            
            # RSI 중립선 회복 확인
            if rsi is not None and rsi >= self.config.rsi_exit:
                signal_strength = self._calculate_sell_signal_strength(rsi)
                
                return Signal(
                    timestamp=market_data.timestamp,
                    symbol=self.config.symbol,
                    action="SELL",
                    signal_type="rsi_exit",
                    strength=signal_strength,
                    price=current_price,
                    indicators=indicators,
                    reason=f"RSI_중립선_회복_RSI={rsi:.1f}"
                )
            
            # 청산 신호 없음
            return None
    
    def _calculate_buy_signal_strength(self, rsi: float, volume_ratio: Optional[float] = None) -> float:
        """
        BUY 신호 강도 계산 (0.0 ~ 1.0)
        
        Args:
            rsi: RSI 값
            volume_ratio: 거래량 비율
        
        Returns:
            신호 강도 (0.0 ~ 1.0)
        """
        # RSI가 낮을수록 강한 신호
        # RSI 30 기준으로 0.0~30.0 범위를 0.0~1.0으로 매핑
        rsi_strength = max(0.0, min(1.0, (30.0 - rsi) / 30.0))
        
        # 거래량 비율이 높을수록 강한 신호 (선택적)
        if volume_ratio is not None and volume_ratio > 1.0:
            volume_strength = min(1.0, (volume_ratio - 1.0) / 2.0)  # 1.0~3.0 → 0.0~1.0
            # RSI와 거래량의 가중 평균
            return (rsi_strength * 0.7 + volume_strength * 0.3)
        
        return rsi_strength
    
    def _calculate_sell_signal_strength(self, rsi: float) -> float:
        """
        SELL 신호 강도 계산 (0.0 ~ 1.0)
        
        Args:
            rsi: RSI 값
        
        Returns:
            신호 강도 (0.0 ~ 1.0)
        """
        # RSI가 중립선(50)을 넘을수록 강한 신호
        # RSI 50 기준으로 50.0~100.0 범위를 0.0~1.0으로 매핑
        if rsi >= 50.0:
            return min(1.0, (rsi - 50.0) / 50.0)
        return 0.0
