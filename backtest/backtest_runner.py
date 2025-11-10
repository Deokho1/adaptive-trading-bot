"""
Backtest execution framework.

백테스트 실행, 결과 수집, 성과 분석을 수행하는 메인 백테스트 엔진입니다.
전략과 거래소 API 사이의 브릿지 역할을 하며, 결과를 다양한 형태로 출력합니다.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from pathlib import Path

from ..core.strategy_core import ScalpingStrategy, StrategyConfig, MarketData, Signal
from ..core.exchange_api_backtest import BacktestExchangeAPI, Order, Position


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    fee_rate: float = 0.0007  # 0.07%
    slippage_rate: float = 0.0003  # 0.03%
    output_dir: str = "results"


@dataclass
class BacktestResults:
    """백테스트 결과"""
    config: BacktestConfig
    strategy_config: StrategyConfig
    
    # 성과 지표
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    
    # 수익성 지표
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    
    # 리스크 지표
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 세부 데이터
    trades: List[Dict]
    equity_curve: List[Dict]
    signals: List[Dict]
    
    # 실행 정보
    execution_time_seconds: float
    data_points_processed: int


class BacktestRunner:
    """
    백테스트 실행기
    
    전략과 거래소 API를 연결하여 백테스트를 실행하고 결과를 생성합니다.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        백테스트 러너 초기화
        
        Args:
            output_dir: 결과 출력 디렉터리 (기본값: results)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 결과 저장용
        self.trades_log: List[Dict] = []
        self.equity_log: List[Dict] = []
        self.signals_log: List[Dict] = []
        self.debug_events: List[Dict] = []
        
    def run_backtest(self, 
                    strategy_config: StrategyConfig,
                    backtest_config: BacktestConfig,
                    ohlcv_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        백테스트 실행
        
        Args:
            strategy_config: 전략 설정
            backtest_config: 백테스트 설정  
            ohlcv_data: OHLCV 데이터 {symbol: DataFrame}
            
        Returns:
            BacktestResults: 백테스트 결과
        """
        print(f"🚀 백테스트 시작: {backtest_config.start_date} ~ {backtest_config.end_date}")
        start_time = datetime.now()
        
        # 1. 거래소 API 초기화
        exchange = BacktestExchangeAPI(
            ohlcv_data=ohlcv_data,
            initial_balance=backtest_config.initial_balance,
            fee_rate=backtest_config.fee_rate,
            slippage_rate=backtest_config.slippage_rate
        )
        
        # 2. 전략 초기화
        strategy = ScalpingStrategy(strategy_config)
        
        # 3. 데이터 준비
        symbol = strategy_config.symbol
        if symbol not in ohlcv_data:
            raise ValueError(f"Symbol {symbol} not found in OHLCV data")
            
        df = ohlcv_data[symbol].copy()
        
        # 날짜 필터링
        if 'timestamp' in df.columns:
            df = df[
                (df['timestamp'] >= backtest_config.start_date) & 
                (df['timestamp'] <= backtest_config.end_date)
            ].reset_index(drop=True)
        
        print(f"📊 처리할 데이터: {len(df)} 개 봉")
        
        # 4. 백테스트 루프 실행
        data_points = 0
        for idx, row in df.iterrows():
            try:
                # 현재 시간 설정
                current_time = row['timestamp'] if 'timestamp' in row else datetime.now()
                exchange.set_current_time(current_time)
                
                # 시장 데이터 생성
                market_data = MarketData(
                    timestamp=current_time,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                
                # 전략 신호 생성
                signal = strategy.on_bar(market_data)
                
                # 신호 로깅
                self._log_signal(signal)
                
                # 신호에 따른 주문 실행
                if signal.action in ["BUY", "SELL"]:
                    self._execute_signal(exchange, signal)
                
                # 자산 곡선 기록
                self._log_equity(current_time, exchange.get_equity())
                
                data_points += 1
                
                # 진행상황 출력 (1000개마다)
                if data_points % 1000 == 0:
                    progress = (data_points / len(df)) * 100
                    current_equity = exchange.get_equity()
                    print(f"📈 진행률: {progress:.1f}% | 현재 자산: {current_equity:,.0f}")
                
            except Exception as e:
                # 에러 로깅
                self.debug_events.append({
                    'timestamp': current_time,
                    'event': 'ERROR',
                    'message': str(e),
                    'bar_index': idx
                })
                continue
        
        # 5. 백테스트 완료 후 결과 계산
        execution_time = (datetime.now() - start_time).total_seconds()
        results = self._calculate_results(
            strategy_config, backtest_config, exchange, 
            execution_time, data_points
        )
        
        # 6. 결과 저장
        self._save_results(results)
        
        print(f"✅ 백테스트 완료! 실행시간: {execution_time:.1f}초")
        print(f"📁 결과 저장됨: {self.output_dir}")
        
        return results
        
    def _execute_signal(self, exchange: BacktestExchangeAPI, signal: Signal) -> None:
        """신호에 따른 주문 실행"""
        try:
            # TODO: 포지션 크기 계산 로직은 전략에서 가져와야 함
            # 현재는 고정값 사용 (실제로는 전략에서 계산)
            order_size = 1000.0  # USDT 단위
            
            order = exchange.place_order(
                symbol=signal.symbol,
                side=signal.action,
                size=order_size,
                order_type="MARKET"
            )
            
            # 거래 로그 추가
            self._log_trade(order, signal)
            
        except Exception as e:
            self.debug_events.append({
                'timestamp': signal.timestamp,
                'event': 'ORDER_ERROR',
                'message': str(e),
                'signal': asdict(signal)
            })
    
    def _log_signal(self, signal: Signal) -> None:
        """신호 로깅"""
        self.signals_log.append({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'price': signal.price,
            'indicators': signal.indicators,
            'reason': signal.reason
        })
        
    def _log_trade(self, order: Order, signal: Signal) -> None:
        """거래 로깅"""
        self.trades_log.append({
            'timestamp': order.filled_time,
            'symbol': order.symbol,
            'side': order.side,
            'size': order.size,
            'price': order.filled_price,
            'order_id': order.id,
            'signal_type': signal.signal_type,
            'signal_reason': signal.reason
        })
        
    def _log_equity(self, timestamp: datetime, equity: float) -> None:
        """자산 곡선 로깅"""
        self.equity_log.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
    def _calculate_results(self, 
                          strategy_config: StrategyConfig,
                          backtest_config: BacktestConfig,
                          exchange: BacktestExchangeAPI,
                          execution_time: float,
                          data_points: int) -> BacktestResults:
        """백테스트 결과 계산"""
        
        initial_equity = backtest_config.initial_balance
        final_equity = exchange.get_equity()
        
        # 기본 수익률
        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
        
        # 거래 통계 계산
        trades = exchange.get_trade_history()
        total_trades = len([t for t in trades if t['side'] == 'BUY'])  # 매수 기준 거래 횟수
        
        # 손익 계산 (간단한 버전)
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        # TODO: 실제 거래별 손익 계산 로직 필요
        # 현재는 추정값
        if total_trades > 0:
            winning_trades = int(total_trades * 0.5)  # 임시값
            losing_trades = total_trades - winning_trades
            win_rate_pct = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        else:
            win_rate_pct = 0
        
        # 최대 낙폭 계산
        max_drawdown_pct = self._calculate_max_drawdown()
        
        # 리스크 지표 계산 (간단한 버전)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return BacktestResults(
            config=backtest_config,
            strategy_config=strategy_config,
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=0.0,  # TODO: 계산 필요
            average_win=0.0,    # TODO: 계산 필요
            average_loss=0.0,   # TODO: 계산 필요
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0.0,  # TODO: 계산 필요
            calmar_ratio=0.0,   # TODO: 계산 필요
            trades=self.trades_log,
            equity_curve=self.equity_log,
            signals=self.signals_log,
            execution_time_seconds=execution_time,
            data_points_processed=data_points
        )
        
    def _calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        if not self.equity_log:
            return 0.0
            
        equities = [log['equity'] for log in self.equity_log]
        peak = equities[0]
        max_dd = 0.0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
                
        return max_dd
        
    def _calculate_sharpe_ratio(self) -> float:
        """샤프 비율 계산 (간단한 버전)"""
        if len(self.equity_log) < 2:
            return 0.0
            
        # 일일 수익률 계산
        daily_returns = []
        for i in range(1, len(self.equity_log)):
            prev_equity = self.equity_log[i-1]['equity']
            curr_equity = self.equity_log[i]['equity']
            daily_return = (curr_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
            
        import statistics
        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
        
        if std_return == 0:
            return 0.0
            
        # 무위험 수익률 0으로 가정
        return mean_return / std_return * (252 ** 0.5)  # 연율화
        
    def _save_results(self, results: BacktestResults) -> None:
        """결과를 파일로 저장"""
        
        # 1. 메인 결과 JSON
        results_dict = asdict(results)
        # datetime 객체를 문자열로 변환
        results_dict = self._convert_datetime_to_str(results_dict)
        
        with open(self.output_dir / "backtest_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        # 2. 거래 내역 CSV
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(self.output_dir / "trades.csv", index=False)
        
        # 3. 자산 곡선 CSV
        if self.equity_log:
            equity_df = pd.DataFrame(self.equity_log)
            equity_df.to_csv(self.output_dir / "equity_curve.csv", index=False)
        
        # 4. 신호 로그 CSV
        if self.signals_log:
            signals_df = pd.DataFrame(self.signals_log)
            signals_df.to_csv(self.output_dir / "signals.csv", index=False)
        
        # 5. 디버그 이벤트 CSV
        if self.debug_events:
            debug_df = pd.DataFrame(self.debug_events)
            debug_df.to_csv(self.output_dir / "debug_events.csv", index=False)
        
        print(f"📁 결과 파일 저장:")
        print(f"   - backtest_results.json (메인 결과)")
        print(f"   - trades.csv (거래 내역)")  
        print(f"   - equity_curve.csv (자산 곡선)")
        print(f"   - signals.csv (신호 로그)")
        print(f"   - debug_events.csv (디버그 이벤트)")
        
        # 6. OneDrive 자동 백업
        self._backup_to_onedrive()
        
    def _backup_to_onedrive(self) -> None:
        """OneDrive로 결과 자동 백업"""
        onedrive_path = r"C:\Users\DH\OneDrive\문서\Bot"
        
        if not os.path.exists(onedrive_path):
            print("⚠️ OneDrive backup path not found")
            return
            
        try:
            import shutil
            for file_path in self.output_dir.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, onedrive_path)
            print(f"📁 Results automatically backed up to: {onedrive_path}")
        except Exception as e:
            print(f"⚠️ Backup failed: {e}")
        
    def _convert_datetime_to_str(self, obj: Any) -> Any:
        """datetime 객체를 문자열로 재귀 변환"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_datetime_to_str(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_str(item) for item in obj]
        else:
            return obj


def create_sample_config() -> Tuple[StrategyConfig, BacktestConfig]:
    """샘플 설정 생성 (테스트용)"""
    
    strategy_config = StrategyConfig(
        symbol="BTCUSDT",
        timeframe="1m"
        # TODO: 실제 파라미터는 사용자가 설정
    )
    
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        initial_balance=10000.0,
        fee_rate=0.0007,
        slippage_rate=0.0003,
        output_dir="backtest_results"
    )
    
    return strategy_config, backtest_config