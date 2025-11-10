"""
Backtest exchange API implementation for historical data simulation.

과거 OHLCV 데이터를 사용하여 가상 체결을 수행하는 백테스트 전용 모듈입니다.
실제 거래소 API와 동일한 인터페이스를 제공하여 전략 코드의 재사용성을 보장합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


@dataclass
class Order:
    """주문 정보"""
    id: str
    symbol: str
    side: str  # "BUY", "SELL"
    size: float
    price: Optional[float]  # None for market orders
    order_type: str  # "MARKET", "LIMIT"
    status: str  # "PENDING", "FILLED", "CANCELLED"
    timestamp: datetime
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # "LONG", "SHORT", "NONE"
    size: float
    entry_price: float
    unrealized_pnl: float
    

@dataclass
class Balance:
    """계좌 잔고"""
    currency: str
    total: float
    available: float
    locked: float


class BaseExchangeAPI(ABC):
    """거래소 API 공통 인터페이스"""
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """최신 가격 조회"""
        pass
        
    @abstractmethod  
    def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        pass
        
    @abstractmethod
    def place_order(self, symbol: str, side: str, size: float, 
                   order_type: str = "MARKET", price: Optional[float] = None) -> Order:
        """주문 생성"""
        pass
        
    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """포지션 조회"""
        pass
        
    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> Balance:
        """잔고 조회"""
        pass


class BacktestExchangeAPI(BaseExchangeAPI):
    """
    백테스트용 가상 거래소 API
    
    과거 데이터를 사용하여 가상 체결을 시뮬레이션합니다.
    """
    
    def __init__(self, 
                 ohlcv_data: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0,
                 fee_rate: float = 0.0007,  # 0.07%
                 slippage_rate: float = 0.0003):  # 0.03%
        """
        백테스트 거래소 초기화
        
        Args:
            ohlcv_data: {symbol: DataFrame} 형태의 OHLCV 데이터
            initial_balance: 초기 잔고 (USDT)
            fee_rate: 거래 수수료율
            slippage_rate: 슬리피지율
        """
        self.ohlcv_data = ohlcv_data
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        
        # 계좌 상태
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_id_counter = 1
        
        # 현재 시뮬레이션 시간과 데이터 인덱스
        self.current_time: Optional[datetime] = None
        self.data_indices: Dict[str, int] = {symbol: 0 for symbol in ohlcv_data.keys()}
        
        # 거래 기록
        self.trade_history: List[Dict] = []
        
    def set_current_time(self, timestamp: datetime) -> None:
        """
        현재 시뮬레이션 시간 설정
        
        Args:
            timestamp: 설정할 시간
        """
        self.current_time = timestamp
        
        # 각 심볼별로 해당 시간의 데이터 인덱스 찾기
        for symbol, df in self.ohlcv_data.items():
            # timestamp 컬럼이 있는지 확인
            if 'timestamp' in df.columns:
                # 현재 시간 이전의 마지막 데이터 인덱스 찾기
                mask = df['timestamp'] <= timestamp
                if mask.any():
                    self.data_indices[symbol] = mask.idxmax()
                else:
                    self.data_indices[symbol] = 0
            
    def get_latest_price(self, symbol: str) -> float:
        """
        최신 가격 조회 (현재 시뮬레이션 시간 기준)
        
        Args:
            symbol: 심볼 (예: "BTCUSDT")
            
        Returns:
            float: 현재 가격
        """
        if symbol not in self.ohlcv_data:
            raise ValueError(f"Symbol {symbol} not found in data")
            
        df = self.ohlcv_data[symbol]
        current_idx = self.data_indices[symbol]
        
        if current_idx >= len(df):
            current_idx = len(df) - 1
            
        return float(df.iloc[current_idx]['close'])
        
    def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        OHLCV 데이터 조회 (현재 시뮬레이션 시간까지)
        
        Args:
            symbol: 심볼
            interval: 시간 간격 (백테스트에서는 무시됨)
            limit: 조회할 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        if symbol not in self.ohlcv_data:
            raise ValueError(f"Symbol {symbol} not found in data")
            
        df = self.ohlcv_data[symbol]
        current_idx = self.data_indices[symbol]
        
        # 현재 시간까지의 데이터에서 limit개 반환
        start_idx = max(0, current_idx - limit + 1)
        end_idx = current_idx + 1
        
        return df.iloc[start_idx:end_idx].copy()
        
    def place_order(self, symbol: str, side: str, size: float,
                   order_type: str = "MARKET", price: Optional[float] = None) -> Order:
        """
        주문 생성 및 즉시 체결 시뮬레이션
        
        Args:
            symbol: 심볼
            side: "BUY" 또는 "SELL"
            size: 주문 크기 (USDT 단위)
            order_type: 주문 타입 (백테스트에서는 "MARKET"만 지원)
            price: 가격 (MARKET 주문에서는 무시됨)
            
        Returns:
            Order: 생성된 주문 정보
        """
        if not self.current_time:
            raise ValueError("Current time not set. Call set_current_time() first.")
            
        order_id = f"BT_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        # 현재 가격 조회
        current_price = self.get_latest_price(symbol)
        
        # 슬리피지 적용
        if side == "BUY":
            fill_price = current_price * (1 + self.slippage_rate)
        else:
            fill_price = current_price * (1 - self.slippage_rate)
            
        # 주문 생성
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            status="FILLED",
            timestamp=self.current_time,
            filled_price=fill_price,
            filled_time=self.current_time
        )
        
        # 체결 처리
        self._process_fill(order)
        
        self.orders.append(order)
        return order
        
    def _process_fill(self, order: Order) -> None:
        """주문 체결 처리"""
        symbol = order.symbol
        
        if order.side == "BUY":
            # 롱 포지션 진입/추가
            fill_value = order.size  # USDT 단위
            fill_quantity = fill_value / order.filled_price  # 코인 수량
            fee = fill_value * self.fee_rate
            
            if symbol in self.positions:
                # 기존 포지션에 추가
                pos = self.positions[symbol]
                total_value = (pos.size * pos.entry_price) + fill_value
                total_quantity = pos.size + fill_quantity
                pos.entry_price = total_value / total_quantity
                pos.size = total_quantity
            else:
                # 새 포지션 생성
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side="LONG",
                    size=fill_quantity,
                    entry_price=order.filled_price,
                    unrealized_pnl=0.0
                )
                
            self.balance -= (fill_value + fee)
            
        else:  # SELL
            # 롱 포지션 청산
            if symbol in self.positions:
                pos = self.positions[symbol]
                sell_quantity = order.size / order.filled_price
                
                if sell_quantity >= pos.size:
                    # 전체 청산
                    realized_pnl = (order.filled_price - pos.entry_price) * pos.size
                    fee = order.size * self.fee_rate
                    
                    self.balance += (order.size - fee + realized_pnl)
                    del self.positions[symbol]
                else:
                    # 부분 청산
                    realized_pnl = (order.filled_price - pos.entry_price) * sell_quantity
                    fee = order.size * self.fee_rate
                    
                    self.balance += (order.size - fee + realized_pnl)
                    pos.size -= sell_quantity
                    
        # 거래 기록 추가
        self.trade_history.append({
            'timestamp': order.filled_time,
            'symbol': order.symbol,
            'side': order.side,
            'size': order.size,
            'price': order.filled_price,
            'fee': order.size * self.fee_rate
        })
        
    def get_position(self, symbol: str) -> Position:
        """
        포지션 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            Position: 포지션 정보 (없으면 사이즈 0인 포지션)
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            # 현재 가격으로 미실현 손익 업데이트
            current_price = self.get_latest_price(symbol)
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            return pos
        else:
            return Position(
                symbol=symbol,
                side="NONE",
                size=0.0,
                entry_price=0.0,
                unrealized_pnl=0.0
            )
            
    def get_balance(self, currency: str = "USDT") -> Balance:
        """
        잔고 조회
        
        Args:
            currency: 통화 (백테스트에서는 USDT만 지원)
            
        Returns:
            Balance: 잔고 정보
        """
        return Balance(
            currency=currency,
            total=self.balance,
            available=self.balance,
            locked=0.0
        )
        
    def get_equity(self) -> float:
        """
        총 자산 (잔고 + 포지션 평가액) 계산
        
        Returns:
            float: 총 자산 (USDT)
        """
        total_equity = self.balance
        
        # 모든 포지션의 평가액 추가
        for symbol, position in self.positions.items():
            if position.size > 0:
                current_price = self.get_latest_price(symbol)
                position_value = position.size * current_price
                total_equity += position_value
                
        return total_equity
        
    def get_trade_history(self) -> List[Dict]:
        """거래 기록 반환"""
        return self.trade_history.copy()