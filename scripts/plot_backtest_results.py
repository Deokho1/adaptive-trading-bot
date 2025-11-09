#!/usr/bin/env python3
"""
백테스트 결과 시각화 스크립트

사용법:
    python scripts/plot_backtest_results.py [CSV_파일_경로]
    
예시:
    python scripts/plot_backtest_results.py results/backtest_history_20241217_143052.csv
    
CSV 파일을 지정하지 않으면 results/ 디렉토리에서 가장 최신 파일을 자동 선택합니다.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization.backtest_plotter import BacktestPlotter


def find_latest_backtest_csv(results_dir: Path) -> Path:
    """results 디렉토리에서 가장 최신 백테스트 CSV 파일을 찾습니다."""
    csv_files = list(results_dir.glob("backtest_history_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"백테스트 결과 파일이 {results_dir}에서 발견되지 않았습니다.")
    
    # 파일명의 타임스탬프로 정렬하여 가장 최신 파일 반환
    csv_files.sort(key=lambda x: x.name, reverse=True)
    return csv_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="백테스트 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="시각화할 백테스트 결과 CSV 파일 경로"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="출력 이미지 파일 경로 (기본값: 자동 생성)"
    )
    
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="차트를 화면에 표시 (저장하지 않음)"
    )
    
    args = parser.parse_args()
    
    # CSV 파일 경로 결정
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"오류: 파일을 찾을 수 없습니다: {csv_path}")
            sys.exit(1)
    else:
        try:
            results_dir = project_root / "results"
            csv_path = find_latest_backtest_csv(results_dir)
            print(f"최신 백테스트 결과 파일 선택: {csv_path.name}")
        except FileNotFoundError as e:
            print(f"오류: {e}")
            sys.exit(1)
    
    # CSV 파일 읽기
    try:
        print(f"백테스트 결과 로딩 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 컬럼명 매핑 (CSV의 실제 컬럼명에 맞춰 변환)
        column_mapping = {
            'ts': 'timestamp',
            'equity': 'total_value', 
            'positions_value': 'portfolio_value'
        }
        
        # 컬럼명 변경
        df = df.rename(columns=column_mapping)
        
        # 타임스탬프 컬럼을 datetime으로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"데이터 로드 완료: {len(df)}개 레코드")
        print(f"백테스트 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
    except Exception as e:
        print(f"오류: CSV 파일 읽기 실패: {e}")
        sys.exit(1)
    
    # 플로터 생성 및 시각화
    try:
        plotter = BacktestPlotter()
        
        if args.show:
            # 화면에 표시
            print("차트를 생성하고 표시 중...")
            plotter.plot_backtest_results(df, show=True)
            
        else:
            # 파일로 저장
            if args.output:
                output_path = Path(args.output)
            else:
                # 자동 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = project_root / "results"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"backtest_plot_{timestamp}.png"
            
            print(f"차트를 저장하는 중: {output_path}")
            plotter.plot_backtest_results(df, save_path=output_path)
            print(f"차트가 저장되었습니다: {output_path}")
            
    except Exception as e:
        print(f"오류: 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()