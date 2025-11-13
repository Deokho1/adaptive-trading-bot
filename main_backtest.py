# 백테스트 메인 실행 파일
import sys
import os

# Windows 콘솔 인코딩 설정 (UTF-8)
if sys.platform == 'win32':
    # 콘솔 코드 페이지를 UTF-8로 설정
    os.system('chcp 65001 >nul 2>&1')
    # 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from backtest.backtest_runner import BacktestRunner

def main():
    """백테스트 실행"""
    print("="*60)
    print("Adaptive Trading Bot - Backtest")
    print("="*60)
    
    try:
        # 백테스트 실행
        runner = BacktestRunner()
        runner.run()
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print("\n[INFO] Solution:")
        print("   1. Check if backtest_config.json exists")
        print("   2. Check if config file format is correct")
    except Exception as e:
        print(f"\n[ERROR] Backtest execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()