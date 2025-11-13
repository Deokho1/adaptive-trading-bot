"""
백테스트용 가상 거래소 API

실제 거래소와 동일한 인터페이스를 제공하여 전략 코드의 재사용성을 보장합니다.
"""

from typing import Dict, List, Optional
from datetime import datetime


class ExchangeAPIBacktest:
    """
    백테스트용 가상 거래소 API
    
    역할:
    1. 가상 잔고 관리
    2. 가상 주문 실행 및 체결 시뮬레이션
    3. 포지션 추적
    4. 수수료 및 슬리피지 적용
    """
    
    def __init__(
        self,
        initial_capital: float = 10000000.0,  # 초기 자본 (KRW)
        fee_rate: float = 0.0005,  # 수수료율 (0.05%)
        slippage_rate: float = 0.0003  # 슬리피지율 (0.03%)
    ):
        """
        초기화
        
        Args:
            initial_capital: 초기 자본 (KRW)
            fee_rate: 거래 수수료율
            slippage_rate: 슬리피지율
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        
        # 계좌 상태
        self.balance = initial_capital  # 현금 잔고 (KRW)
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, entry_price, entry_time}}
        self.orders: List[Dict] = []  # 주문 내역
        self.trade_history: List[Dict] = []  # 거래 내역
        
        # 통계
        self.total_trades = 0
        self.total_fees = 0.0
    
    def get_balance(self) -> float:
        """
        잔고 조회
        
        Returns:
            현금 잔고 (KRW)
        """
        return self.balance
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        모든 포지션 조회
        
        Returns:
            {symbol: {quantity, entry_price, entry_time}} 형태의 딕셔너리
        """
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        특정 심볼의 포지션 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            포지션 정보 또는 None
        """
        return self.positions.get(symbol)
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity_krw: float,  # 주문 금액 (KRW)
        price: float  # 현재 가격
    ) -> Optional[Dict]:
        """
        주문 실행 (시장가 주문 시뮬레이션)
        
        Args:
            symbol: 심볼
            side: "BUY" 또는 "SELL"
            quantity_krw: 주문 금액 (KRW) - BUY일 때 사용
            price: 현재 가격
            
        Returns:
            체결된 주문 정보 또는 None (실패 시)
        """
        # 슬리피지 적용
        if side == "BUY":
            fill_price = price * (1 + self.slippage_rate)
        else:  # SELL
            fill_price = price * (1 - self.slippage_rate)
        
        if side == "BUY":
            # 매수 주문
            if self.balance < quantity_krw:
                print(f"   [WARN] Insufficient balance: {self.balance:.0f} < {quantity_krw:.0f}")
                return None
            
            # 수수료 계산
            fee = quantity_krw * self.fee_rate
            total_cost = quantity_krw + fee
            
            if self.balance < total_cost:
                print(f"   [WARN] Insufficient balance for fee: {self.balance:.0f} < {total_cost:.0f}")
                return None
            
            # 체결 처리
            quantity = (quantity_krw - fee) / fill_price  # 수수료 제외한 금액으로 수량 계산
            
            # 잔고 차감
            self.balance -= total_cost
            
            # 포지션 업데이트 (평균 단가 계산)
            if symbol in self.positions:
                # 기존 포지션에 추가 - 가중평균으로 평균 단가 계산
                existing = self.positions[symbol]
                existing_quantity = existing['quantity']
                existing_cost_basis = existing['entry_price'] * existing_quantity
                new_cost_basis = fill_price * quantity
                
                total_quantity = existing_quantity + quantity
                total_cost_basis = existing_cost_basis + new_cost_basis
                avg_entry_price = total_cost_basis / total_quantity
                
                self.positions[symbol] = {
                    'quantity': total_quantity,
                    'entry_price': avg_entry_price,
                    'entry_time': existing['entry_time']  # 첫 진입 시간 유지
                }
            else:
                # 새 포지션 생성
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': fill_price,
                    'entry_time': datetime.now()
                }
            
            # 거래 기록
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': fill_price,
                'value': quantity_krw,
                'fee': fee
            }
            self.trade_history.append(trade)
            self.total_trades += 1
            self.total_fees += fee
            
            return {
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': fill_price,
                'value': quantity_krw,
                'fee': fee
            }
        
        else:  # SELL
            # 매도 주문 (전체 청산)
            if symbol not in self.positions:
                print(f"   [WARN] No position to sell: {symbol}")
                return None
            
            position = self.positions[symbol]
            quantity = position['quantity']
            
            if quantity <= 0:
                print(f"   [WARN] Invalid position quantity: {quantity}")
                return None
            
            # 매도 금액 계산
            sell_value = quantity * fill_price
            fee = sell_value * self.fee_rate
            net_proceeds = sell_value - fee
            
            # 잔고 추가
            self.balance += net_proceeds
            
            # 손익 계산
            entry_price = position['entry_price']
            pnl = (fill_price - entry_price) * quantity
            pnl_pct = ((fill_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            # 포지션 제거
            del self.positions[symbol]
            
            # 거래 기록
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': fill_price,
                'value': sell_value,
                'fee': fee,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            self.trade_history.append(trade)
            self.total_trades += 1
            self.total_fees += fee
            
            return {
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': fill_price,
                'value': sell_value,
                'fee': fee,
                'entry_price': entry_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
    
    def calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """
        현재 자산 계산 (잔고 + 미실현 손익)
        
        Args:
            current_prices: {symbol: current_price} 형태의 딕셔너리
            
        Returns:
            현재 자산 (KRW)
        """
        equity = self.balance
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # 미실현 손익 계산
                unrealized_pnl = (current_price - entry_price) * quantity
                equity += unrealized_pnl
        
        return equity