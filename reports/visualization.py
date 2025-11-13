"""
시각화
Chart and graph generation for backtest results.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


class Visualization:
    """차트 및 그래프 생성"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualization
        
        Args:
            output_dir: 결과 저장 폴더 (기본값: "results")
                        매번 덮어쓰기 방식으로 저장
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정 (Windows)
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
        except:
            try:
                plt.rcParams['font.family'] = 'AppleGothic'  # macOS
            except:
                pass  # 기본 폰트 사용
        
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    def plot_equity_curve(
        self,
        equity_curve: List[Dict],
        trades: List[Dict] = None,
        filename: str = "equity_curve.png"
    ) -> str:
        """
        자산 곡선 그래프 (덮어쓰기)
        
        Args:
            equity_curve: 자산 곡선 리스트
            trades: 거래 내역 리스트 (선택적, 거래 포인트 표시용)
            filename: 파일명 (기본값: "equity_curve.png")
        
        Returns:
            파일 경로
        """
        if not equity_curve:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 자산 곡선 데이터 준비
        timestamps = [e['timestamp'] for e in equity_curve]
        equities = [e['equity'] for e in equity_curve]
        
        # 자산 곡선 그리기
        ax.plot(timestamps, equities, linewidth=1.5, color='#2E86AB', label='자산')
        
        # 초기 자본선
        if equity_curve:
            initial_equity = equity_curve[0]['equity']
            ax.axhline(y=initial_equity, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='초기 자본')
        
        # 거래 포인트 표시
        if trades:
            buy_trades = [t for t in trades if t.get('action') == 'BUY']
            sell_trades = [t for t in trades if t.get('action') == 'SELL']
            
            # 매수 포인트
            if buy_trades:
                buy_times = [t['timestamp'] for t in buy_trades]
                buy_equities = []
                for buy_time in buy_times:
                    # 해당 시간의 자산 값 찾기
                    closest_idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - buy_time).total_seconds()))
                    buy_equities.append(equities[closest_idx])
                ax.scatter(buy_times, buy_equities, color='green', marker='^', s=50, label='매수', zorder=5)
            
            # 매도 포인트
            if sell_trades:
                sell_times = [t['timestamp'] for t in sell_trades]
                sell_equities = []
                for sell_time in sell_times:
                    closest_idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - sell_time).total_seconds()))
                    sell_equities.append(equities[closest_idx])
                ax.scatter(sell_times, sell_equities, color='red', marker='v', s=50, label='매도', zorder=5)
        
        ax.set_xlabel('시간')
        ax.set_ylabel('자산 (KRW)')
        ax.set_title('자산 곡선')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_drawdown(
        self,
        equity_curve: List[Dict],
        filename: str = "drawdown.png"
    ) -> str:
        """
        낙폭 차트 (덮어쓰기)
        
        Args:
            equity_curve: 자산 곡선 리스트
            filename: 파일명 (기본값: "drawdown.png")
        
        Returns:
            파일 경로
        """
        if not equity_curve or len(equity_curve) < 2:
            return ""
        
        # 낙폭 계산
        equities = [e['equity'] for e in equity_curve]
        timestamps = [e['timestamp'] for e in equity_curve]
        
        peak = equities[0]
        drawdowns = []
        
        for equity in equities:
            if equity > peak:
                peak = equity
            drawdown_pct = ((equity - peak) / peak) * 100 if peak > 0 else 0.0
            drawdowns.append(drawdown_pct)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(timestamps, drawdowns, 0, color='red', alpha=0.3, label='낙폭')
        ax.plot(timestamps, drawdowns, linewidth=1, color='red')
        
        ax.set_xlabel('시간')
        ax.set_ylabel('낙폭 (%)')
        ax.set_title('낙폭 차트')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_trade_distribution(
        self,
        trades: List[Dict],
        filename: str = "trade_distribution.png"
    ) -> str:
        """
        거래 분포 차트 (덮어쓰기)
        
        Args:
            trades: 거래 내역 리스트
            filename: 파일명 (기본값: "trade_distribution.png")
        
        Returns:
            파일 경로
        """
        # SELL 거래만 필터링 (실제 손익 발생)
        sell_trades = [t for t in trades if t.get('action') == 'SELL' and 'pnl' in t]
        
        if not sell_trades:
            return ""
        
        pnls = [t.get('pnl', 0) for t in sell_trades]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. PnL 히스토그램
        ax1 = axes[0]
        ax1.hist(pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, label='손익분기점')
        ax1.set_xlabel('손익 (KRW)')
        ax1.set_ylabel('빈도')
        ax1.set_title('거래 손익 분포')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 시간대별 거래 분포
        ax2 = axes[1]
        trade_times = [t['timestamp'] for t in sell_trades]
        trade_hours = [t.hour for t in trade_times]
        
        hour_counts = {}
        for hour in trade_hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        hours = sorted(hour_counts.keys())
        counts = [hour_counts[h] for h in hours]
        
        ax2.bar(hours, counts, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('시간 (시)')
        ax2.set_ylabel('거래 수')
        ax2.set_title('시간대별 거래 분포')
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_all(
        self,
        equity_curve: List[Dict],
        trades: List[Dict]
    ) -> Dict[str, str]:
        """
        모든 차트 생성 (덮어쓰기)
        
        Args:
            equity_curve: 자산 곡선 리스트
            trades: 거래 내역 리스트
        
        Returns:
            {
                'equity_curve': 'results/equity_curve.png',
                'drawdown': 'results/drawdown.png',
                'trade_distribution': 'results/trade_distribution.png'
            }
        """
        result = {}
        
        equity_path = self.plot_equity_curve(equity_curve, trades)
        if equity_path:
            result['equity_curve'] = equity_path
        
        drawdown_path = self.plot_drawdown(equity_curve)
        if drawdown_path:
            result['drawdown'] = drawdown_path
        
        distribution_path = self.plot_trade_distribution(trades)
        if distribution_path:
            result['trade_distribution'] = distribution_path
        
        return result
