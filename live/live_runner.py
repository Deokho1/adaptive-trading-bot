# 🚀 실시간 거래 루프
class LiveRunner:
    """실시간 거래 시스템의 메인 루프"""
    
    def __init__(self):
        self.is_running = False
    
    def start(self):
        """실시간 거래 시작"""
        print("실시간 거래 시스템 시작")
        self.is_running = True
    
    def stop(self):
        """실시간 거래 중지"""
        print("실시간 거래 시스템 중지")
        self.is_running = False