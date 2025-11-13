"""
백테스트 실행기

백테스트 전체 프로세스를 관리하는 메인 엔진입니다.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import pandas as pd

from backtest.data_loader import BacktestDataLoader
from api.exchange_api_backtest import ExchangeAPIBacktest
from core.strategy_core import DecisionEngine, StrategyConfig, MarketData, TradingDecision
from reports.metrics import Metrics
from reports.trade_reporter import TradeReporter
from reports.visualization import Visualization
import config as app_config


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
        self.decision_engine: DecisionEngine = None  # 전략 엔진
        
        # 결과 저장
        self.equity_curve: List[Dict] = []  # 자산 곡선
        self.trades: List[Dict] = []  # 거래 내역
        
        # Reports 모듈
        self.metrics = Metrics()
        self.trade_reporter = TradeReporter(output_dir="results")
        self.visualization = Visualization(output_dir="results")
        
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
        backtest_config = app_config.CONFIG.get("backtest", {})
        
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
        
        # 4. 전략 엔진 초기화
        print("\n[INFO] Initializing strategy engine...")
        strategy_config = StrategyConfig(
            symbol=config['symbol'],
            timeframe=config['interval']
        )
        self.decision_engine = DecisionEngine(strategy_config)
        print(f"[OK] Strategy engine initialized")
        print(f"   Symbol: {strategy_config.symbol}")
        print(f"   Timeframe: {strategy_config.timeframe}")
        
        # 5. 백테스트 루프 실행
        print("\n[INFO] Starting backtest loop...")
        self._run_backtest_loop(df, config['symbol'])
        
        # 6. 마지막 포지션 정리
        self._close_all_positions(df, config['symbol'])
        
        # 7. 결과 리포트 생성
        self._generate_report(initial_capital)
    
    def _run_backtest_loop(self, df: pd.DataFrame, symbol: str):
        """
        백테스트 루프 실행
        
        Args:
            df: 캔들 데이터 DataFrame
            symbol: 거래 심볼
        """
        total_candles = len(df)
        
        # 초기 자산 기록
        initial_equity = self.exchange.get_balance()
        self.equity_curve.append({
            'timestamp': df.iloc[0]['timestamp'],
            'equity': initial_equity,
            'balance': initial_equity,
            'price': df.iloc[0]['close']
        })
        
        # 각 캔들마다 처리
        for idx, row in df.iterrows():
            # 진행률 표시 (10% 단위)
            if idx % max(1, total_candles // 10) == 0:
                progress = (idx / total_candles) * 100
                print(f"   진행률: {progress:.1f}% ({idx}/{total_candles})")
            
            # 1. MarketData 생성
            market_data = MarketData(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # 2. 현재 포지션 확인
            position = self.exchange.get_position(symbol)
            
            # 3. 전략 결정
            available_balance = self.exchange.get_balance()
            decision = self.decision_engine.make_decision(
                market_data=market_data,
                current_position=position,
                available_balance=available_balance
            )
            
            # 4. 거래 실행
            if decision.action == "BUY":
                # 포지션 없을 때만 매수 가능 (추가 매수 금지)
                if position is None:
                    order_result = self.exchange.place_order(
                        symbol=symbol,
                        side="BUY",
                        quantity_krw=decision.size_usd,  # KRW 금액
                        price=decision.price
                    )
                    if order_result:
                        self.trades.append({
                            'timestamp': decision.timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': order_result['price'],
                            'quantity': order_result['quantity'],
                            'value': order_result['value'],
                            'fee': order_result['fee'],
                            'reason': decision.reason
                        })
            
            elif decision.action == "SELL":
                # 포지션 있을 때만 매도 가능
                if position is not None:
                    order_result = self.exchange.place_order(
                        symbol=symbol,
                        side="SELL",
                        quantity_krw=0,  # 0 = 전체 청산
                        price=decision.price
                    )
                    if order_result:
                        self.trades.append({
                            'timestamp': decision.timestamp,
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': order_result['price'],
                            'quantity': order_result['quantity'],
                            'value': order_result['value'],
                            'fee': order_result['fee'],
                            'entry_price': order_result.get('entry_price'),
                            'pnl': order_result.get('pnl', 0),
                            'pnl_pct': order_result.get('pnl_pct', 0),
                            'reason': decision.reason
                        })
            
            # 5. 자산 곡선 업데이트
            current_price = market_data.close
            equity = self.exchange.calculate_equity({symbol: current_price})
            
            self.equity_curve.append({
                'timestamp': market_data.timestamp,
                'equity': equity,
                'balance': self.exchange.get_balance(),
                'price': current_price
            })
        
        print(f"[OK] Backtest loop completed: {total_candles} candles processed")
    
    def _close_all_positions(self, df: pd.DataFrame, symbol: str):
        """
        백테스트 종료 시 남은 포지션 강제 청산
        
        Args:
            df: 캔들 데이터 DataFrame
            symbol: 거래 심볼
        """
        position = self.exchange.get_position(symbol)
        if position:
            last_price = df.iloc[-1]['close']
            print(f"\n[INFO] Closing remaining position at end of backtest...")
            
            order_result = self.exchange.place_order(
                symbol=symbol,
                side="SELL",
                quantity_krw=0,  # 전체 청산
                price=last_price
            )
            
            if order_result:
                self.trades.append({
                    'timestamp': df.iloc[-1]['timestamp'],
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': order_result['price'],
                    'quantity': order_result['quantity'],
                    'value': order_result['value'],
                    'fee': order_result['fee'],
                    'entry_price': order_result.get('entry_price'),
                    'pnl': order_result.get('pnl', 0),
                    'pnl_pct': order_result.get('pnl_pct', 0),
                    'reason': 'backtest_end_force_close'
                })
                print(f"[OK] Position closed")
    
    def _generate_report(self, initial_capital: float):
        """
        백테스트 결과 리포트 생성 (덮어쓰기 방식)
        
        Args:
            initial_capital: 초기 자본
        """
        if not self.equity_curve:
            print("\n[WARN] No equity curve data")
            return
        
        # 1. 지표 계산
        metrics_result = self.metrics.calculate_all_metrics(
            self.equity_curve,
            self.trades,
            initial_capital
        )
        
        # 2. 텍스트 리포트 출력
        summary = self.trade_reporter.generate_trade_summary(
            self.trades,
            metrics_result
        )
        print(summary)
        
        # 3. CSV 파일 저장 (덮어쓰기)
        csv_paths = self.trade_reporter.export_to_csv(
            self.trades,
            self.equity_curve,
            metrics_result
        )
        print(f"\n[OK] CSV files saved:")
        for name, path in csv_paths.items():
            print(f"   {name}: {path}")
        
        # 4. 엑셀 내보내기 (덮어쓰기)
        excel_path = self.trade_reporter.export_to_excel(
            self.trades,
            self.equity_curve,
            metrics_result,
            filename="backtest_report.xlsx"
        )
        if excel_path:
            print(f"[OK] Excel report: {excel_path}")
        
        # 5. 차트 생성 (덮어쓰기)
        chart_paths = self.visualization.plot_all(
            self.equity_curve,
            self.trades
        )
        if chart_paths:
            print(f"[OK] Charts saved:")
            for name, path in chart_paths.items():
                print(f"   {name}: {path}")
