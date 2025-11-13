# 백테스트 메인 실행 파일
import sys
import os
import io

# Windows 콘솔 인코딩 설정 (UTF-8) - 최우선 실행
if sys.platform == 'win32':
    # 환경 변수 먼저 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # 콘솔 코드 페이지를 UTF-8로 설정
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    
    # stdout/stderr 인코딩 강제 설정
    try:
        if hasattr(sys.stdout, 'buffer'):
            if sys.stdout.encoding != 'utf-8' or sys.stdout.encoding is None:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, 
                    encoding='utf-8', 
                    errors='replace', 
                    line_buffering=True
                )
        if hasattr(sys.stderr, 'buffer'):
            if sys.stderr.encoding != 'utf-8' or sys.stderr.encoding is None:
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, 
                    encoding='utf-8', 
                    errors='replace', 
                    line_buffering=True
                )
    except Exception:
        # 인코딩 설정 실패 시 무시 (기본 인코딩 사용)
        pass

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