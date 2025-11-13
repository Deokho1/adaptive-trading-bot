"""
거래 리포터
Trade reporting and export functionality.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import csv


class TradeReporter:
    """거래 결과 리포팅"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize trade reporter
        
        Args:
            output_dir: 결과 저장 폴더 (기본값: "results")
                        매번 덮어쓰기 방식으로 저장
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_trade_summary(
        self,
        trades: List[Dict],
        metrics: Dict
    ) -> str:
        """
        거래 요약 텍스트 생성
        
        Args:
            trades: 거래 내역 리스트
            metrics: 성능 지표 딕셔너리
        
        Returns:
            요약 텍스트
        """
        lines = []
        lines.append("\n" + "="*60)
        lines.append("[백테스트 결과]")
        lines.append("="*60)
        
        # 기본 정보
        lines.append(f"초기 자본: {metrics.get('initial_capital', 0):,.0f} KRW")
        lines.append(f"최종 자산: {metrics.get('final_equity', 0):,.0f} KRW")
        lines.append(f"총 수익률: {metrics.get('total_return_pct', 0):+.2f}%")
        
        # 거래 통계
        lines.append(f"\n거래 통계:")
        lines.append(f"  총 거래 수: {metrics.get('total_trades', 0)}회")
        lines.append(f"  승률: {metrics.get('win_rate_pct', 0):.1f}% ({metrics.get('wins', 0)}승 / {metrics.get('losses', 0)}패)")
        lines.append(f"  평균 승리: {metrics.get('avg_win', 0):,.0f} KRW")
        lines.append(f"  평균 손실: {metrics.get('avg_loss', 0):,.0f} KRW")
        lines.append(f"  수익 팩터: {metrics.get('profit_factor', 0):.2f}")
        
        # 성능 지표
        lines.append(f"\n성능 지표:")
        lines.append(f"  샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}")
        lines.append(f"  최대 낙폭: {metrics.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"  최대 낙폭 지속: {metrics.get('max_drawdown_duration_days', 0):.1f}일")
        
        # 기타
        lines.append(f"\n기타:")
        lines.append(f"  총 수수료: {metrics.get('total_fees', 0):,.0f} KRW")
        lines.append(f"  총 손익: {metrics.get('total_pnl', 0):,.0f} KRW")
        if metrics.get('avg_hold_time_minutes', 0) > 0:
            lines.append(f"  평균 보유 시간: {metrics.get('avg_hold_time_minutes', 0):.1f}분")
        
        lines.append("="*60)
        
        return "\n".join(lines)
    
    def export_to_csv(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        metrics: Dict
    ) -> Dict[str, str]:
        """
        CSV 파일로 내보내기 (덮어쓰기)
        
        Args:
            trades: 거래 내역 리스트
            equity_curve: 자산 곡선 리스트
            metrics: 성능 지표 딕셔너리
        
        Returns:
            {
                'trades': 'results/trades.csv',
                'equity_curve': 'results/equity_curve.csv',
                'summary': 'results/backtest_summary.csv'
            }
        """
        # trades.csv
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = self.output_dir / "trades.csv"
            trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
        else:
            trades_path = self.output_dir / "trades.csv"
            pd.DataFrame().to_csv(trades_path, index=False)
        
        # equity_curve.csv
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_path = self.output_dir / "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False, encoding='utf-8-sig')
        else:
            equity_path = self.output_dir / "equity_curve.csv"
            pd.DataFrame().to_csv(equity_path, index=False)
        
        # backtest_summary.csv
        summary_path = self.output_dir / "backtest_summary.csv"
        summary_data = [
            {'Metric': k, 'Value': v} 
            for k, v in metrics.items()
        ]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        return {
            'trades': str(trades_path),
            'equity_curve': str(equity_path),
            'summary': str(summary_path)
        }
    
    def export_to_excel(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        metrics: Dict,
        filename: str = "backtest_report.xlsx"
    ) -> str:
        """
        엑셀로 내보내기 (덮어쓰기)
        
        Args:
            trades: 거래 내역 리스트
            equity_curve: 자산 곡선 리스트
            metrics: 성능 지표 딕셔너리
            filename: 파일명 (기본값: "backtest_report.xlsx")
        
        Returns:
            파일 경로
        """
        try:
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet1: 거래 내역
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df.to_excel(writer, sheet_name='거래 내역', index=False)
                else:
                    pd.DataFrame().to_excel(writer, sheet_name='거래 내역', index=False)
                
                # Sheet2: 자산 곡선
                if equity_curve:
                    equity_df = pd.DataFrame(equity_curve)
                    equity_df.to_excel(writer, sheet_name='자산 곡선', index=False)
                else:
                    pd.DataFrame().to_excel(writer, sheet_name='자산 곡선', index=False)
                
                # Sheet3: 성능 지표
                metrics_data = [
                    {'Metric': k, 'Value': v} 
                    for k, v in metrics.items()
                ]
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='성능 지표', index=False)
            
            return str(filepath)
        
        except ImportError:
            # openpyxl이 없으면 CSV만 사용
            print("[WARN] openpyxl not installed, skipping Excel export")
            return ""
    
    def generate_report(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        metrics: Dict,
        config: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        전체 리포트 생성 (텍스트 + CSV + Excel)
        
        Args:
            trades: 거래 내역 리스트
            equity_curve: 자산 곡선 리스트
            metrics: 성능 지표 딕셔너리
            config: 백테스트 설정 (선택적)
        
        Returns:
            생성된 파일 경로 딕셔너리
        """
        # CSV 파일 저장
        csv_paths = self.export_to_csv(trades, equity_curve, metrics)
        
        # Excel 파일 저장
        excel_path = self.export_to_excel(trades, equity_curve, metrics)
        
        result = csv_paths.copy()
        if excel_path:
            result['excel'] = excel_path
        
        return result
