"""
백테스트 결과 시각화 모듈.

이 모듈은 백테스트 결과를 그래프로 표시하는 기능을 제공합니다.
자산 곡선, 현금 잔고, 드로우다운 등을 시각화할 수 있습니다.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd

logger = logging.getLogger("bot")


def format_krw(x, pos):
    """KRW 금액을 천단위 콤마로 포맷팅"""
    return f'{x:,.0f}'


class BacktestPlotter:
    """
    백테스트 결과 시각화 클래스.
    
    포트폴리오 데이터를 받아서 자산 곡선, 현금 잔고, 드로우다운 등을
    그래프로 표시하고 저장하는 기능을 제공합니다.
    """
    
    def __init__(self, output_dir: Path = Path("results"), title: str = "백테스트 결과 - 적응형 듀얼모드 봇"):
        self.output_dir = output_dir
        self.title = title

    def plot_backtest_results(
        self, 
        df: pd.DataFrame, 
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> Optional[Path]:
        """
        DataFrame으로부터 백테스트 결과를 시각화합니다.

        Args:
            df: 백테스트 결과 DataFrame (timestamp, total_value, cash, portfolio_value 컬럼 필요)
            save_path: 저장할 파일 경로 (None이면 자동 생성)
            show: True면 matplotlib 창으로 표시

        Returns:
            저장된 파일 경로 (save_path가 지정된 경우)
        """
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 최고점 대비 드로우다운 계산
        df = df.copy()
        df['peak'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['peak']) / df['peak'] * 100
        
        # 기본 통계 계산
        initial_value = df['total_value'].iloc[0]
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        max_drawdown = df['drawdown'].min()
        
        # 3개 서브플롯 생성
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(self.title, fontsize=16, y=0.98)
        
        # 1. 자산 곡선 및 현금 잔고
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['total_value'], 'b-', linewidth=2, label='총 자산')
        ax1.plot(df['timestamp'], df['cash'], 'g-', linewidth=1, alpha=0.7, label='현금')
        ax1.plot(df['timestamp'], df['portfolio_value'], 'r-', linewidth=1, alpha=0.7, label='포트폴리오')
        
        ax1.set_ylabel('자산 (KRW)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_krw))
        
        # 시작/종료 값 표시
        ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5)
        ax1.text(df['timestamp'].iloc[0], initial_value * 1.02, f'시작: {initial_value:,.0f}', 
                fontsize=10, alpha=0.8)
        ax1.text(df['timestamp'].iloc[-1], final_value * 1.02, f'종료: {final_value:,.0f}', 
                fontsize=10, alpha=0.8)
        
        # 2. 수익률 (%)
        ax2 = axes[1]
        returns = (df['total_value'] / initial_value - 1) * 100
        ax2.plot(df['timestamp'], returns, 'purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('누적 수익률 (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 수익률 색상 변경
        ax2.fill_between(df['timestamp'], 0, returns, 
                        where=(returns >= 0), color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(df['timestamp'], 0, returns, 
                        where=(returns < 0), color='red', alpha=0.3, interpolate=True)
        
        # 3. 드로우다운
        ax3 = axes[2]
        ax3.fill_between(df['timestamp'], 0, df['drawdown'], color='red', alpha=0.3)
        ax3.plot(df['timestamp'], df['drawdown'], 'red', linewidth=1)
        ax3.set_ylabel('드로우다운 (%)', fontsize=12)
        ax3.set_xlabel('시간', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # x축 날짜 포맷 설정
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        # 통계 정보 텍스트 박스
        stats_text = f"""백테스트 통계:
시작 자산: {initial_value:,.0f} KRW
최종 자산: {final_value:,.0f} KRW
총 수익률: {total_return:+.2f}%
최대 드로우다운: {max_drawdown:.2f}%
백테스트 기간: {len(df)}개 데이터포인트"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"백테스트 차트가 저장되었습니다: {save_path}")
        
        if show:
            plt.show()
        elif save_path:
            plt.close()
        
        return save_path