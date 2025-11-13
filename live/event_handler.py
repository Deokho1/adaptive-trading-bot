# 체결 이벤트·에러·리밸런스 처리
class EventHandler:
    """이벤트 핸들링 시스템"""
    
    def __init__(self):
        self.event_queue = []
    
    def on_order_filled(self, order_event):
        """체결 이벤트 처리"""
        print(f"주문 체결: {order_event}")
    
    def on_error(self, error_event):
        """에러 이벤트 처리"""
        print(f"에러 발생: {error_event}")
    
    def on_rebalance(self, rebalance_event):
        """리밸런스 이벤트 처리"""
        print(f"리밸런스: {rebalance_event}")