"""
Analysis module for scalp bot

# ANALYSIS: 분석 모듈 초기화
Contains trade analysis and reporting utilities.
"""

from .trade_analyzer import TradeAnalyzer, analyze_backtest_results

__all__ = ['TradeAnalyzer', 'analyze_backtest_results']