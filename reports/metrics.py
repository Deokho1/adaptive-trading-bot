"""
Metrics calculation and performance analysis utilities.

백테스트 결과의 성과 지표 계산, 리스크 분석, 통계 생성 등을 담당합니다.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import math


def calculate_performance_metrics(equity_curve: List[Dict],
                                trades: List[Dict]) -> Dict[str, float]:
    """
    성과 지표 계산
    
    Args:
        equity_curve: 자산 곡선 데이터
        trades: 거래 내역
        
    Returns:
        Dict: 계산된 성과 지표들
    """
    if not equity_curve or len(equity_curve) < 2:
        return {}
    
    # 기본 데이터 추출
    initial_equity = equity_curve[0]['equity']
    final_equity = equity_curve[-1]['equity']
    
    # 기본 수익률
    total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
    
    # 최대 낙폭
    max_drawdown_pct = calculate_max_drawdown(equity_curve)
    
    # 거래 통계
    trade_stats = calculate_trade_statistics(trades)
    
    # 리스크 조정 수익률
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    sortino_ratio = calculate_sortino_ratio(equity_curve)
    calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
    
    return {
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        **trade_stats
    }


def calculate_max_drawdown(equity_curve: List[Dict]) -> float:
    """최대 낙폭 계산"""
    if not equity_curve:
        return 0.0
        
    equities = [point['equity'] for point in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    
    for equity in equities:
        if equity > peak:
            peak = equity
        
        drawdown = (peak - equity) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
            
    return max_dd


def calculate_trade_statistics(trades: List[Dict]) -> Dict[str, float]:
    """거래 통계 계산"""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'average_win': 0,
            'average_loss': 0
        }
    
    # TODO: 실제 거래별 손익 계산 로직 구현 필요
    # 현재는 기본 구조만 제공
    
    total_trades = len(trades)
    # TODO: 매수/매도 쌍으로 실제 거래 횟수 계산
    # TODO: 거래별 손익 계산
    
    return {
        'total_trades': total_trades,
        'winning_trades': 0,  # TODO
        'losing_trades': 0,   # TODO
        'win_rate_pct': 0,    # TODO
        'profit_factor': 0,   # TODO
        'average_win': 0,     # TODO
        'average_loss': 0     # TODO
    }


def calculate_sharpe_ratio(equity_curve: List[Dict], 
                          risk_free_rate: float = 0.0) -> float:
    """샤프 비율 계산"""
    if len(equity_curve) < 2:
        return 0.0
    
    # 일일 수익률 계산
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i-1]['equity']
        curr = equity_curve[i]['equity']
        daily_return = (curr - prev) / prev
        returns.append(daily_return)
    
    if not returns:
        return 0.0
    
    # 평균 수익률과 변동성
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return 0.0
    
    # 샤프 비율 (연율화)
    excess_return = mean_return - risk_free_rate
    return (excess_return / std_dev) * math.sqrt(252)  # 252 거래일


def calculate_sortino_ratio(equity_curve: List[Dict],
                           risk_free_rate: float = 0.0) -> float:
    """소르티노 비율 계산 (하방 변동성만 고려)"""
    if len(equity_curve) < 2:
        return 0.0
    
    # 일일 수익률 계산
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i-1]['equity']
        curr = equity_curve[i]['equity']
        daily_return = (curr - prev) / prev
        returns.append(daily_return)
    
    if not returns:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    
    # 하방 편차 계산 (음수 수익률만)
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_deviation = math.sqrt(downside_variance)
    
    if downside_deviation == 0:
        return 0.0
    
    # 소르티노 비율
    excess_return = mean_return - risk_free_rate
    return (excess_return / downside_deviation) * math.sqrt(252)


def generate_performance_report(results_dict: Dict) -> Dict[str, str]:
    """
    성과 리포트 생성
    
    Args:
        results_dict: 백테스트 결과 딕셔너리
        
    Returns:
        Dict: 포맷된 리포트 문자열들
    """
    report = {}
    
    # 기본 정보
    report['header'] = f"""
=== 백테스트 성과 리포트 ===
기간: {results_dict.get('config', {}).get('start_date', '')} ~ {results_dict.get('config', {}).get('end_date', '')}
심볼: {results_dict.get('strategy_config', {}).get('symbol', '')}
초기 자본: {results_dict.get('initial_equity', 0):,.0f}
"""
    
    # 수익률 섹션
    total_return = results_dict.get('total_return_pct', 0)
    final_equity = results_dict.get('final_equity', 0)
    
    report['returns'] = f"""
=== 수익률 분석 ===
최종 자산: {final_equity:,.0f}
총 수익률: {total_return:.2f}%
최대 낙폭: {results_dict.get('max_drawdown_pct', 0):.2f}%
"""
    
    # 거래 통계
    total_trades = results_dict.get('total_trades', 0)
    win_rate = results_dict.get('win_rate_pct', 0)
    
    report['trades'] = f"""
=== 거래 통계 ===
총 거래 수: {total_trades}
승률: {win_rate:.1f}%
승리 거래: {results_dict.get('winning_trades', 0)}
손실 거래: {results_dict.get('losing_trades', 0)}
"""
    
    # 리스크 지표
    sharpe = results_dict.get('sharpe_ratio', 0)
    sortino = results_dict.get('sortino_ratio', 0)
    
    report['risk'] = f"""
=== 리스크 분석 ===
샤프 비율: {sharpe:.2f}
소르티노 비율: {sortino:.2f}
칼마 비율: {results_dict.get('calmar_ratio', 0):.2f}
"""
    
    return report