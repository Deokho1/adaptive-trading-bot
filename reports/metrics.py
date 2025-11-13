"""
성능 지표 계산
Performance metrics calculation for backtest results.
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta


class Metrics:
    """백테스트 및 실거래 성능 지표"""
    
    def __init__(self):
        pass
    
    def calculate_sharpe_ratio(
        self, 
        returns: List[float], 
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns: 일일 수익률 리스트
            risk_free_rate: 무위험 수익률 (연율, 기본값: 0.0)
            periods_per_year: 연간 거래일 수 (기본값: 252일)
        
        Returns:
            샤프 비율
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # 연율화
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return float(sharpe)
    
    def calculate_max_drawdown(self, equity_curve: List[Dict]) -> Dict:
        """
        최대 낙폭 계산
        
        Args:
            equity_curve: 자산 곡선 리스트 [{'timestamp': ..., 'equity': ...}, ...]
        
        Returns:
            {
                'max_drawdown_pct': 최대 낙폭 (%),
                'max_drawdown_duration': 최대 낙폭 지속 기간 (일),
                'drawdown_curve': 낙폭 곡선 리스트
            }
        """
        if not equity_curve or len(equity_curve) < 2:
            return {
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration': 0.0,
                'drawdown_curve': []
            }
        
        equities = [e['equity'] for e in equity_curve]
        timestamps = [e['timestamp'] for e in equity_curve]
        
        # 최고점 추적
        peak = equities[0]
        max_drawdown_pct = 0.0
        max_drawdown_start = None
        max_drawdown_end = None
        drawdown_curve = []
        
        for i, equity in enumerate(equities):
            if equity > peak:
                peak = equity
            
            drawdown_pct = ((equity - peak) / peak) * 100 if peak > 0 else 0.0
            drawdown_curve.append({
                'timestamp': timestamps[i],
                'drawdown_pct': drawdown_pct
            })
            
            if drawdown_pct < max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
                if max_drawdown_start is None:
                    max_drawdown_start = timestamps[i]
                max_drawdown_end = timestamps[i]
        
        # 최대 낙폭 지속 기간 계산
        max_drawdown_duration = 0.0
        if max_drawdown_start and max_drawdown_end:
            duration = max_drawdown_end - max_drawdown_start
            max_drawdown_duration = duration.total_seconds() / (24 * 3600)  # 일 단위
        
        return {
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration': max_drawdown_duration,
            'drawdown_curve': drawdown_curve
        }
    
    def calculate_win_rate(self, trades: List[Dict]) -> Dict:
        """
        승률 및 거래 통계
        
        Args:
            trades: 거래 내역 리스트 (SELL 거래만 분석)
        
        Returns:
            {
                'win_rate': 승률 (%),
                'total_trades': 총 거래 수,
                'wins': 승리 거래 수,
                'losses': 손실 거래 수,
                'avg_win': 평균 승리 금액,
                'avg_loss': 평균 손실 금액,
                'profit_factor': 수익 팩터 (총 수익 / 총 손실)
            }
        """
        # SELL 거래만 필터링 (실제 손익 발생)
        sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'pnl' in t]
        
        if not sell_trades:
            return {
                'win_rate': 0.0,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        profitable_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        wins = len(profitable_trades)
        losses = len(losing_trades)
        total_trades = len(sell_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0.0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        
        total_profit = sum([t['pnl'] for t in profitable_trades])
        total_loss = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor)
        }
    
    def calculate_all_metrics(
        self,
        equity_curve: List[Dict],
        trades: List[Dict],
        initial_capital: float
    ) -> Dict:
        """
        모든 지표 계산
        
        Args:
            equity_curve: 자산 곡선 리스트
            trades: 거래 내역 리스트
            initial_capital: 초기 자본
        
        Returns:
            모든 성능 지표 딕셔너리
        """
        if not equity_curve:
            return {}
        
        final_equity = equity_curve[-1]['equity']
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # 일일 수익률 계산
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]['equity']
            curr_equity = equity_curve[i]['equity']
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)
        
        # 지표 계산
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        drawdown_info = self.calculate_max_drawdown(equity_curve)
        trade_stats = self.calculate_win_rate(trades)
        
        # 총 수수료
        total_fees = sum([t.get('fee', 0) for t in trades])
        
        # 평균 보유 시간 계산
        hold_times = []
        buy_trades = [t for t in trades if t.get('action') == 'BUY']
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        
        for sell_trade in sell_trades:
            entry_price = sell_trade.get('entry_price')
            if entry_price:
                # 매도 거래의 entry_price로 매수 거래 찾기
                matching_buy = next(
                    (b for b in buy_trades 
                     if abs(b.get('price', 0) - entry_price) < entry_price * 0.01 and 
                     b.get('timestamp') < sell_trade.get('timestamp')),
                    None
                )
                if matching_buy:
                    hold_time = sell_trade['timestamp'] - matching_buy['timestamp']
                    hold_times.append(hold_time.total_seconds() / 60)  # 분 단위
        
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0
        
        # 총 손익
        total_pnl = sum([t.get('pnl', 0) for t in sell_trades])
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': drawdown_info['max_drawdown_pct'],
            'max_drawdown_duration_days': drawdown_info['max_drawdown_duration'],
            'win_rate_pct': trade_stats['win_rate'],
            'total_trades': trade_stats['total_trades'],
            'wins': trade_stats['wins'],
            'losses': trade_stats['losses'],
            'avg_win': trade_stats['avg_win'],
            'avg_loss': trade_stats['avg_loss'],
            'profit_factor': trade_stats['profit_factor'],
            'total_fees': total_fees,
            'total_pnl': total_pnl,
            'avg_hold_time_minutes': float(avg_hold_time)
        }
