# 백테스트 메인 실행 파일
from backtest.backtest_runner import BacktestRunner
from core.market_watcher import MarketWatcher
from core.signal_engine import SignalEngine
from core.trade_executor import TradeExecutor

def main():
    """백테스트 실행"""
    print("백테스트 시작")
    
    # 백테스트 실행
    backtest_runner = BacktestRunner()
    # backtest_runner.run()

if __name__ == "__main__":
    main()