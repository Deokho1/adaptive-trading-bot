# 실거래 메인 실행 파일
from live.live_runner import LiveRunner
from live.risk_monitor import RiskMonitor
from live.event_handler import EventHandler
from core.market_watcher import MarketWatcher
from core.signal_engine import SignalEngine
from core.trade_executor import TradeExecutor

def main():
    """실거래 실행"""
    print("실거래 시스템 시작")
    
    # 실거래 시스템 실행
    live_runner = LiveRunner()
    risk_monitor = RiskMonitor()
    event_handler = EventHandler()
    
    # 시스템 시작
    # live_runner.start()

if __name__ == "__main__":
    main()