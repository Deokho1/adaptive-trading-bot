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
        self.decision_engine: DecisionEngine = None  # 전략 엔진
        
        # 결과 저장
        self.equity_curve: List[Dict] = []  # 자산 곡선
        self.trades: List[Dict] = []  # 거래 내역
        
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
        
        # 7. 결과 출력
        self._print_results(initial_capital)
    
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
            decision = self.decision_engine.make_decision(
                market_data=market_data,
                current_position=position
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
    
    def _print_results(self, initial_capital: float):
        """
        백테스트 결과 출력
        
        Args:
            initial_capital: 초기 자본
        """
        if not self.equity_curve:
            print("\n[WARN] No equity curve data")
            return
        
        final_equity = self.equity_curve[-1]['equity']
        final_balance = self.exchange.get_balance()
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # 거래 통계
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        total_trades = len(buy_trades) + len(sell_trades)
        
        # 수익 거래 통계
        profitable_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        win_rate = (len(profitable_trades) / len(sell_trades) * 100) if sell_trades else 0
        
        # 총 수수료
        total_fees = sum(t.get('fee', 0) for t in self.trades)
        
        # 평균 보유 시간 (단타 특화)
        hold_times = []
        for sell_trade in sell_trades:
            entry_price = sell_trade.get('entry_price')
            if entry_price:
                # 매도 거래의 entry_price로 매수 거래 찾기
                matching_buy = next(
                    (b for b in buy_trades 
                     if abs(b['price'] - entry_price) < entry_price * 0.01 and 
                     b['timestamp'] < sell_trade['timestamp']),
                    None
                )
                if matching_buy:
                    hold_time = sell_trade['timestamp'] - matching_buy['timestamp']
                    hold_times.append(hold_time.total_seconds() / 60)  # 분 단위
        
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        print("\n" + "="*60)
        print("📊 백테스트 결과")
        print("="*60)
        print(f"초기 자본: {initial_capital:,.0f} KRW")
        print(f"최종 자산: {final_equity:,.0f} KRW")
        print(f"최종 잔고: {final_balance:,.0f} KRW")
        print(f"총 수익률: {total_return:+.2f}%")
        print(f"\n거래 통계:")
        print(f"  총 거래 수: {total_trades}회 (매수: {len(buy_trades)}, 매도: {len(sell_trades)})")
        print(f"  승률: {win_rate:.1f}% ({len(profitable_trades)}승 / {len(losing_trades)}패)")
        print(f"  총 수수료: {total_fees:,.0f} KRW")
        if avg_hold_time > 0:
            print(f"  평균 보유 시간: {avg_hold_time:.1f}분")
        print("="*60)
