# 백테스트용 거래소 API
class ExchangeAPIBacktest:
    """백테스트용 가상 거래소 API"""
    
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.balance = 1000000  # 초기 자금 100만원
    
    def place_order(self, symbol, side, quantity, price):
        """주문 실행 (백테스트)"""
        pass
    
    def get_balance(self):
        """잔고 조회"""
        return self.balance
    
    def get_positions(self):
        """포지션 조회"""
        return self.positions