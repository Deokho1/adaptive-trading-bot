#!/usr/bin/env python
"""
Adaptive Trading Bot CLI Tool

Usage:
  python cli.py collect <symbol> <interval> <days>    # 데이터 수집
  python cli.py list                                  # 수집된 데이터 확인
  python cli.py analyze <file>                        # 데이터 분석
  python cli.py test                                  # 파이프라인 테스트
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Add project path
sys.path.append(str(Path(__file__).parent))

def parse_date(date_str):
    """날짜 문자열을 datetime 객체로 변환"""
    if not date_str:
        return None
    
    try:
        # YYYY-MM-DD 형식
        if len(date_str) == 10:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        # YYYY-MM-DD HH:MM 형식
        elif len(date_str) == 16:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        # YYYY-MM-DD HH:MM:SS 형식  
        elif len(date_str) == 19:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Invalid date format")
            
        # UTC로 변환
        return pytz.UTC.localize(dt)
        
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS")

def collect_data(symbol, interval, days=None, start_date=None, end_date=None):
    """데이터 수집 + 자동 검증 (날짜 범위 지원)"""
    
    # 날짜 설정 로직
    if start_date or end_date:
        # 특정 날짜 범위 지정
        if start_date and end_date:
            start_dt = parse_date(start_date)
            end_dt = parse_date(end_date)
            period_desc = f"{start_date} ~ {end_date}"
        elif start_date:
            start_dt = parse_date(start_date)
            end_dt = datetime.now(pytz.UTC)
            period_desc = f"{start_date} ~ now"
        else:  # end_date만 지정
            end_dt = parse_date(end_date)
            start_dt = end_dt - timedelta(days=days or 7)
            period_desc = f"{days or 7} days ~ {end_date}"
    else:
        # 기존 방식: 현재부터 N일 전
        end_dt = datetime.now(pytz.UTC)
        start_dt = end_dt - timedelta(days=days or 1)
        period_desc = f"{days or 1} days back"
    
    print(f"Collecting {symbol} {interval} data for {period_desc}...")
    
    try:
        from data_tools.build_datasets import DatasetBuilder
        from data_tools.verify_integrity import DataIntegrityVerifier
        
        builder = DatasetBuilder()
        
        # 1단계: 데이터 수집 + 저장
        result = builder.build_single_dataset(
            exchange="upbit",
            symbol=symbol,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt,
            save_formats=["csv"]
        )
        
        if result["status"] == "completed":
            print(f"✅ Success: {result['candles_collected']} candles collected")
            print(f"📁 Files created: {len(result['files_created'])}")
            
            # 2단계: 자동 검증
            if result['files_created']:
                print("🔍 Starting data verification...")
                verifier = DataIntegrityVerifier()
                
                for file_path in result['files_created']:
                    # CSV 파일만 검증 (메타데이터 JSON 제외)
                    if not file_path.endswith('.csv'):
                        continue
                        
                    file_name = Path(file_path).name
                    verification_result = verifier.verify_single_file(f"backtest_data/processed/{file_name}")
                    
                    status = verification_result.get('status', 'unknown')
                    issues_count = len(verification_result.get('issues', []))
                    
                    if status == 'healthy':
                        print(f"✅ Verification: {file_name} - HEALTHY ({issues_count} issues)")
                    else:
                        print(f"⚠️ Verification: {file_name} - {status.upper()} ({issues_count} issues)")
                        
                print("🎯 Data pipeline completed: collect → save → verify ✅")
            
        else:
            print(f"❌ Failed: {result.get('errors', [])}")
            
    except Exception as e:
        print(f"Error: {e}")

def list_data():
    """수집된 데이터 목록"""
    print("📊 Available backtest data:")
    
    try:
        from backtest.data_loader import BacktestDataLoader
        
        loader = BacktestDataLoader()
        available = loader.list_available_data()
        
        for file_type, files in available.items():
            print(f"\n{file_type.upper()} files:")
            for file in files:
                print(f"  - {file}")
                
        if not available:
            print("  No data files found. Use 'collect' command first.")
            
    except Exception as e:
        print(f"Error: {e}")

def analyze_data(filename):
    """데이터 분석"""
    print(f"📈 Analyzing {filename}...")
    
    try:
        from backtest.data_loader import BacktestDataLoader
        
        loader = BacktestDataLoader()
        df = loader.load_candles_from_file(filename)
        
        print(f"\n📊 Basic Info:")
        print(f"  • Size: {df.shape}")
        print(f"  • Period: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"  • Symbol: {df['symbol'].iloc[0]}")
        
        print(f"\n💰 Price Info:")
        print(f"  • Start: {df.iloc[0]['close']:,.0f}원")
        print(f"  • End: {df.iloc[-1]['close']:,.0f}원")
        print(f"  • High: {df['high'].max():,.0f}원")
        print(f"  • Low: {df['low'].min():,.0f}원")
        
        change = ((df.iloc[-1]['close'] - df.iloc[0]['open']) / df.iloc[0]['open']) * 100
        print(f"  • Change: {change:+.2f}%")
        
        print(f"\n📊 Volume:")
        print(f"  • Total: {df['volume'].sum():.2f} {df['symbol'].iloc[0].split('-')[1]}")
        print(f"  • Average: {df['volume'].mean():.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_pipeline():
    """파이프라인 테스트"""
    print("🧪 Testing pipeline...")
    
    try:
        from data_tools.test_pipeline import quick_pipeline_check
        quick_pipeline_check()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Trading Bot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 사용법 (현재부터 N일 전)
  python cli.py collect KRW-BTC 1h 3        # 3일 전부터 현재까지
  
  # 특정 날짜 범위 지정  
  python cli.py collect KRW-BTC 1h --start 2025-11-01 --end 2025-11-10
  python cli.py collect KRW-ETH 1d --start "2025-11-01 09:00"
  
  # 기타 명령어
  python cli.py list                        # 데이터 파일 목록
  python cli.py analyze data.csv            # 데이터 분석
  python cli.py test                        # 파이프라인 테스트
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect market data')
    collect_parser.add_argument('symbol', help='Symbol (e.g., KRW-BTC)')
    collect_parser.add_argument('interval', help='Interval (e.g., 1h, 1d)')
    collect_parser.add_argument('days', type=int, nargs='?', default=1, 
                                help='Number of days (optional if using --start/--end)')
    
    # 날짜 범위 옵션 추가
    collect_parser.add_argument('--start', type=str, 
                                help='Start date (YYYY-MM-DD or "YYYY-MM-DD HH:MM")')
    collect_parser.add_argument('--end', type=str,
                                help='End date (YYYY-MM-DD or "YYYY-MM-DD HH:MM")')
    
    # List command
    subparsers.add_parser('list', help='List available data files')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data file')
    analyze_parser.add_argument('filename', help='Data file name')
    
    # Test command
    subparsers.add_parser('test', help='Test pipeline')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        collect_data(args.symbol, args.interval, args.days, args.start, args.end)
    elif args.command == 'list':
        list_data()
    elif args.command == 'analyze':
        analyze_data(args.filename)
    elif args.command == 'test':
        test_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()