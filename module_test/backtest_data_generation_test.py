"""
🧪 데이터 파이프라인 통합 테스트

fetch → build → verify → backtest 전체 흐름을 검증하는 통합 테스트입니다.

실제 API를 사용하여 end-to-end 테스트를 진행하므로 
개발 및 QA 환경에서 파이프라인 검증 목적으로 사용하세요.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

# 우리 모듈들
from data_tools.fetch_market_data import UpbitDataFetcher
from data_tools.build_datasets import DatasetBuilder, quick_build_upbit_dataset  
from data_tools.verify_integrity import DataIntegrityVerifier, quick_verify_all
from backtest.data_loader import BacktestDataLoader


class PipelineIntegrationTest:
    """
    🔗 데이터 파이프라인 통합 테스트
    
    실제 워크플로우를 시뮬레이션하여 전체 시스템이 올바르게 작동하는지 검증합니다.
    """
    
    def __init__(self, test_data_dir: str = "backtest_data_test"):
        self.test_dir = Path(test_data_dir)
        self.test_symbols = ["KRW-BTC"]  # 테스트용 심볼
        self.test_intervals = ["1h"]      # 테스트용 간격
        self.test_days = 7               # 최근 7일
        
        self.results = {
            "start_time": datetime.now(),
            "stages": {},
            "final_status": "unknown"
        }
    
    def run_full_pipeline_test(self) -> dict:
        """🚀 전체 파이프라인 테스트 실행"""
        print("🧪 데이터 파이프라인 통합 테스트 시작")
        print("=" * 60)
        
        try:
            # 1단계: Fetch 테스트
            print("\n1️⃣ FETCH 단계 테스트")
            self._test_fetch_stage()
            
            # 2단계: Build 테스트  
            print("\n2️⃣ BUILD 단계 테스트")
            self._test_build_stage()
            
            # 3단계: Verify 테스트
            print("\n3️⃣ VERIFY 단계 테스트")
            self._test_verify_stage()
            
            # 4단계: Backtest Load 테스트
            print("\n4️⃣ BACKTEST LOAD 단계 테스트")  
            self._test_backtest_load_stage()
            
            # 전체 결과 평가
            print("\n📊 통합 테스트 결과")
            self._evaluate_results()
            
        except Exception as e:
            print(f"❌ 통합 테스트 실패: {e}")
            self.results["final_status"] = "failed"
            self.results["error"] = str(e)
        
        self.results["end_time"] = datetime.now()
        self.results["total_duration"] = (
            self.results["end_time"] - self.results["start_time"]
        ).total_seconds()
        
        return self.results
    
    def _test_fetch_stage(self):
        """1단계: API 데이터 수집 테스트"""
        stage_result = {"start_time": datetime.now(), "status": "unknown"}
        
        try:
            fetcher = UpbitDataFetcher()
            
            # 시장 목록 조회 테스트
            print("   📝 시장 목록 조회...")
            markets = fetcher.get_market_list()
            assert len(markets) > 0, "시장 목록이 비어있음"
            print(f"   ✅ {len(markets)}개 시장 조회 완료")
            
            # 캔들 데이터 수집 테스트
            print("   📊 캔들 데이터 수집...")
            candles = fetcher.fetch_candles("KRW-BTC", "1h", count=24)  # 24시간
            assert len(candles) > 0, "캔들 데이터 수집 실패"
            print(f"   ✅ {len(candles)}개 캔들 수집 완료")
            
            # 스키마 검증
            from data_tools.schema import candles_to_dataframe, ensure_candle_schema
            df = candles_to_dataframe(candles)
            ensure_candle_schema(df)
            print("   ✅ 스키마 검증 통과")
            
            stage_result["status"] = "passed"
            stage_result["candles_collected"] = len(candles)
            
        except Exception as e:
            stage_result["status"] = "failed"
            stage_result["error"] = str(e)
            raise
        
        finally:
            stage_result["end_time"] = datetime.now()
            self.results["stages"]["fetch"] = stage_result
    
    def _test_build_stage(self):
        """2단계: 데이터셋 구축 테스트"""
        stage_result = {"start_time": datetime.now(), "status": "unknown"}
        
        try:
            # 테스트용 디렉터리 설정
            builder = DatasetBuilder(str(self.test_dir))
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.test_days)
            
            print(f"   📦 데이터셋 구축 중... ({start_date.date()} ~ {end_date.date()})")
            
            # 단일 데이터셋 구축
            build_result = builder.build_single_dataset(
                exchange="upbit",
                symbol="KRW-BTC",
                interval="1h", 
                start_date=start_date,
                end_date=end_date,
                save_formats=["parquet"]
            )
            
            assert build_result["status"] == "completed", f"구축 실패: {build_result.get('errors', [])}"
            assert build_result["candles_collected"] > 0, "수집된 캔들이 없음"
            assert len(build_result["files_created"]) > 0, "생성된 파일이 없음"
            
            print(f"   ✅ {build_result['candles_collected']}개 캔들로 {len(build_result['files_created'])}개 파일 생성")
            
            stage_result["status"] = "passed"
            stage_result["build_result"] = build_result
            
        except Exception as e:
            stage_result["status"] = "failed"
            stage_result["error"] = str(e)
            raise
        
        finally:
            stage_result["end_time"] = datetime.now()
            self.results["stages"]["build"] = stage_result
    
    def _test_verify_stage(self):
        """3단계: 데이터 검증 테스트"""
        stage_result = {"start_time": datetime.now(), "status": "unknown"}
        
        try:
            verifier = DataIntegrityVerifier(str(self.test_dir))
            
            print("   🔍 데이터 무결성 검증 중...")
            
            # 생성된 파일들 검증
            verification_summary = verifier.verify_multiple_files("*.parquet", detailed_check=True)
            
            assert verification_summary["files_checked"] > 0, "검증할 파일이 없음"
            
            # 최소한 warning 이하여야 함 (critical/error는 실패로 간주)
            critical_files = verification_summary["status_counts"].get("critical", 0)
            error_files = verification_summary["status_counts"].get("error", 0)
            
            if critical_files > 0 or error_files > 0:
                raise AssertionError(f"데이터 품질 문제: critical={critical_files}, error={error_files}")
            
            print(f"   ✅ {verification_summary['files_checked']}개 파일 검증 완료")
            
            healthy = verification_summary["status_counts"].get("healthy", 0)
            warning = verification_summary["status_counts"].get("warning", 0)
            print(f"   📊 상태: healthy={healthy}, warning={warning}")
            
            stage_result["status"] = "passed"
            stage_result["verification_summary"] = verification_summary
            
        except Exception as e:
            stage_result["status"] = "failed"
            stage_result["error"] = str(e)
            raise
        
        finally:
            stage_result["end_time"] = datetime.now()
            self.results["stages"]["verify"] = stage_result
    
    def _test_backtest_load_stage(self):
        """4단계: 백테스트 로더 테스트"""
        stage_result = {"start_time": datetime.now(), "status": "unknown"}
        
        try:
            loader = BacktestDataLoader(str(self.test_dir / "processed"))
            
            print("   📂 백테스트 데이터 로딩...")
            
            # 사용 가능한 파일 확인
            available = loader.list_available_data()
            assert len(available) > 0, "로드할 데이터 파일이 없음"
            print(f"   📝 사용 가능한 파일: {available}")
            
            # 파일 로딩 테스트
            parquet_files = available.get("parquet", [])
            if parquet_files:
                test_file = parquet_files[0]
                print(f"   📊 테스트 파일 로딩: {test_file}")
                
                df = loader.load_candles_from_file(test_file)
                assert not df.empty, "로드된 DataFrame이 비어있음"
                assert 'timestamp' in df.columns, "필수 컬럼 누락"
                
                print(f"   ✅ {len(df)}행 데이터 로딩 완료")
                
                # 품질 리포트 생성 테스트
                quality_report = loader.get_data_quality_report(test_file)
                assert quality_report["status"] in ["clean", "issues_found"], "품질 리포트 생성 실패"
                print(f"   📋 품질 리포트: {quality_report['status']}")
                
                stage_result["loaded_rows"] = len(df)
                stage_result["quality_status"] = quality_report["status"]
            
            stage_result["status"] = "passed"
            stage_result["available_files"] = available
            
        except Exception as e:
            stage_result["status"] = "failed"
            stage_result["error"] = str(e)
            raise
        
        finally:
            stage_result["end_time"] = datetime.now()
            self.results["stages"]["backtest_load"] = stage_result
    
    def _evaluate_results(self):
        """전체 결과 평가 및 요약"""
        all_passed = all(
            stage["status"] == "passed" 
            for stage in self.results["stages"].values()
        )
        
        if all_passed:
            self.results["final_status"] = "passed"
            print("🎉 모든 단계 통과! 파이프라인이 정상 작동합니다.")
        else:
            self.results["final_status"] = "failed"
            print("❌ 일부 단계에서 실패가 발생했습니다.")
        
        # 단계별 요약
        print("\n📋 단계별 결과:")
        for stage_name, stage_result in self.results["stages"].items():
            status_emoji = "✅" if stage_result["status"] == "passed" else "❌"
            duration = (stage_result["end_time"] - stage_result["start_time"]).total_seconds()
            print(f"   {status_emoji} {stage_name.upper()}: {stage_result['status']} ({duration:.1f}s)")
            
            if stage_result["status"] == "failed":
                print(f"      오류: {stage_result.get('error', 'Unknown error')}")
    
    def cleanup_test_data(self):
        """테스트 데이터 정리"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"🧹 테스트 데이터 정리 완료: {self.test_dir}")


def run_pipeline_test(cleanup_after: bool = True) -> dict:
    """
    🚀 파이프라인 통합 테스트 실행
    
    Args:
        cleanup_after: 테스트 후 임시 데이터 삭제 여부
        
    Returns:
        테스트 결과 딕셔너리
    """
    tester = PipelineIntegrationTest()
    
    try:
        results = tester.run_full_pipeline_test()
        return results
    finally:
        if cleanup_after:
            tester.cleanup_test_data()


def quick_pipeline_check():
    """빠른 파이프라인 상태 체크"""
    print("⚡ 빠른 파이프라인 체크")
    
    try:
        # 1. API 연결 체크
        from data_tools.fetch_market_data import UpbitDataFetcher
        fetcher = UpbitDataFetcher()
        markets = fetcher.get_market_list()
        print(f"   ✅ API 연결: {len(markets)}개 마켓 조회")
        
        # 2. 기존 데이터 체크
        from data_tools.verify_integrity import health_check
        health_check()
        
        print("   ✅ 빠른 체크 완료")
        
    except Exception as e:
        print(f"   ❌ 체크 실패: {e}")


if __name__ == "__main__":
    print("🧪 파이프라인 통합 테스트")
    print("=" * 50)
    
    # 사용자 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 빠른 체크만
        quick_pipeline_check()
    else:
        # 전체 통합 테스트
        print("⚠️  이 테스트는 실제 API를 사용하며 몇 분 소요될 수 있습니다.")
        
        user_input = input("계속 진행하시겠습니까? (y/N): ")
        if user_input.lower() == 'y':
            results = run_pipeline_test(cleanup_after=True)
            
            print(f"\n📊 최종 결과: {results['final_status']}")
            print(f"⏱️  총 소요시간: {results['total_duration']:.1f}초")
        else:
            print("테스트 취소됨")