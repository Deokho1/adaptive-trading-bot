# 웹소켓 매니저
class WebSocketManager:
    """실시간 데이터 스트림 관리"""
    
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
    
    def connect(self, url, symbol):
        """웹소켓 연결"""
        print(f"웹소켓 연결: {url} - {symbol}")
    
    def subscribe(self, symbol, callback):
        """데이터 구독"""
        self.callbacks[symbol] = callback
    
    def disconnect(self, symbol):
        """연결 해제"""
        if symbol in self.connections:
            del self.connections[symbol]