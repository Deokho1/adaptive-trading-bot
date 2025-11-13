"""
백테스트 실행기

백테스트 전체 프로세스를 관리하는 메인 엔진입니다.
"""

from typing import Dict, Any
import json
from pathlib import Path

from backtest.data_loader import BacktestDataLoader
from api.exchange_api_backtest import ExchangeAPIBacktest
import config


class BacktestRunner:
    """
    백테스트 실행기
    
    역할:
    1. 백테스트 설정 읽기
    2. DataLoader에 데이터 요청
    3. 백테스트 실행 (향후 구현)
    """
    
    def __init__(self, config_path: str = "backtest_config.json"):
        """
        초기화
        
        Args:
            config_path: 백테스트 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.data_loader = BacktestDataLoader()
        self.exchange: ExchangeAPIBacktest = None  # 가상 거래소 API
        
    def load_config(self) -> Dict[str, Any]:
        """
        백테스트 설정 파일 읽기
        
        Returns:
            설정 딕셔너리
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"[OK] Config loaded: {self.config_path}")
        print(f"   Symbol: {self.config.get('symbol')}")
        print(f"   Interval: {self.config.get('interval')}")
        print(f"   Days: {self.config.get('days')} days")
        
        return self.config
    
    def run(self):
        """
        백테스트 실행
        """
        # 1. 설정 로드
        config = self.load_config()
        
        # 2. DataLoader에 데이터 요청
        print("\n[INFO] Loading data...")
        df = self.data_loader.load_data_for_backtest(
            symbol=config['symbol'],
            interval=config['interval'],
            days=config['days'],
            exchange=config.get('exchange', 'upbit')
        )
        
        print(f"[OK] Data loaded: {len(df)} candles")
        
        # 3. 가상 거래소 API 초기화
        print("\n[INFO] Initializing virtual exchange...")
        backtest_config = config.CONFIG.get("backtest", {})
        
        initial_capital = backtest_config.get("initial_capital", 10000000)
        fee_rate = backtest_config.get("fee_rate", 0.0005)
        slippage_rate = backtest_config.get("slippage_rate", 0.0003)
        
        self.exchange = ExchangeAPIBacktest(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate
        )
        
        print(f"[OK] Virtual exchange initialized")
        print(f"   Initial capital: {initial_capital:,.0f} KRW")
        print(f"   Fee rate: {fee_rate*100:.3f}%")
        print(f"   Slippage rate: {slippage_rate*100:.3f}%")
        
        # TODO: 실제 백테스트 로직 구현
        print("\n[INFO] Backtest ready")
        print("   (Backtest logic will be implemented later)")
