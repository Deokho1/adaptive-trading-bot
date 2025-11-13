"""
🔍 데이터 무결성 검증기

build_datasets.py가 만든 파일들이 정상적인 캔들 시계열인지 확인하는 QA 단계입니다.

주요 검사 항목:
- 타임스탬프 연속성: 캔들이 일정한 간격으로 이어져 있는지
- 결측값: open/high/low/close/volume 중 NaN 있는지  
- 중복: 동일 타임스탬프 중복 캔들 존재 여부
- 이상치: 거래량 또는 가격 급등락 (통계적 기준)
- 스키마 일관성: ensure_candle_schema()로 구조 검증

흐름: fetch_market_data.py → build_datasets.py → verify_integrity.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import shutil
import logging

from .schema import ensure_candle_schema, validate_candle_data, dataframe_to_candles


class DataIntegrityVerifier:
    """
    🕵️ 데이터 무결성 검증기
    
    데이터 품질을 체계적으로 검사하고 문제가 있는 파일을 분리 관리합니다.
    """
    
    def __init__(self, base_data_dir: str = "backtest_data"):
        self.base_dir = Path(base_data_dir)
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        self._ensure_directories()
        self._setup_logging()
        
    def _ensure_directories(self):
        """필요한 디렉터리 생성"""
        pass  # 기본 폴더들은 build_datasets.py에서 생성됨
            
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def verify_single_file(
        self, 
        file_path,  # Union[str, Path] 
        detailed_check: bool = True
    ) -> Dict[str, Any]:
        """
        📊 단일 파일 무결성 검증
        
        Args:
            file_path: 검증할 파일 경로 (str 또는 Path)
            detailed_check: 상세 검사 여부 (시간이 오래 걸림)
            
        Returns:
            검증 결과 딕셔너리
        """
        # Path 객체로 변환
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        self.logger.info(f"🔍 검증 시작: {file_path.name}")
        
        verification_result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "verification_timestamp": datetime.now().isoformat(),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "status": "unknown",
            "issues": [],
            "metrics": {},
            "recommendations": []
        }
        
        try:
            # 1. 파일 로딩 및 기본 스키마 검증
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.pkl':
                df = pd.read_pickle(file_path)
            else:
                verification_result["status"] = "unsupported_format"
                verification_result["issues"].append(f"Unsupported file format: {file_path.suffix}")
                return verification_result
            
            verification_result["metrics"]["total_rows"] = len(df)
            verification_result["metrics"]["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            if df.empty:
                verification_result["status"] = "empty_file"
                verification_result["issues"].append("File is empty")
                return verification_result
            
            # 2. 스키마 검증
            try:
                df = ensure_candle_schema(df, strict=False)
                verification_result["metrics"]["schema_compliance"] = "passed"
            except Exception as e:
                verification_result["issues"].append(f"Schema validation failed: {e}")
                verification_result["metrics"]["schema_compliance"] = "failed"
            
            # 3. 기본 데이터 품질 검사
            self._check_missing_values(df, verification_result)
            self._check_duplicates(df, verification_result)
            self._check_timestamp_consistency(df, verification_result)
            self._check_price_validity(df, verification_result)
            
            # 4. 상세 검사 (선택사항)
            if detailed_check:
                self._check_outliers(df, verification_result)
                self._check_volume_patterns(df, verification_result)
                self._check_ohlc_relationships(df, verification_result)
            
            # 5. 전체 상태 판정
            if not verification_result["issues"]:
                verification_result["status"] = "healthy"
            elif len(verification_result["issues"]) <= 2:
                verification_result["status"] = "warning"
                verification_result["recommendations"].append("Minor issues detected, monitoring recommended")
            else:
                verification_result["status"] = "critical"
                verification_result["recommendations"].append("Multiple issues detected, file review required")
            
            # 6. 메트릭 완성
            if 'timestamp' in df.columns:
                verification_result["metrics"]["date_range"] = {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat(),
                    "duration_days": (df['timestamp'].max() - df['timestamp'].min()).days
                }
            
            self.logger.info(f"   ✅ 검증 완료: {verification_result['status']} ({len(verification_result['issues'])}개 이슈)")
            
        except Exception as e:
            verification_result["status"] = "error"
            verification_result["issues"].append(f"Verification failed: {str(e)}")
            self.logger.error(f"   ❌ 검증 실패: {e}")
        
        return verification_result
    
    def _check_missing_values(self, df: pd.DataFrame, result: Dict):
        """결측값 검사"""
        missing_info = {}
        critical_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_info[col] = missing_count
                    result["issues"].append(f"Missing values in {col}: {missing_count} rows")
        
        result["metrics"]["missing_values"] = missing_info
        
        # 심각도 판정
        total_missing = sum(missing_info.values())
        missing_ratio = total_missing / len(df) if len(df) > 0 else 0
        
        if missing_ratio > 0.1:  # 10% 이상 결측
            result["recommendations"].append("High missing value ratio detected - data quality review needed")
    
    def _check_duplicates(self, df: pd.DataFrame, result: Dict):
        """중복 검사"""
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            duplicates = df.duplicated(['timestamp', 'symbol']).sum()
        elif 'timestamp' in df.columns:
            duplicates = df.duplicated(['timestamp']).sum()
        else:
            duplicates = df.duplicated().sum()
        
        result["metrics"]["duplicate_rows"] = duplicates
        
        if duplicates > 0:
            result["issues"].append(f"Duplicate rows detected: {duplicates}")
            result["recommendations"].append("Remove duplicate entries before analysis")
    
    def _check_timestamp_consistency(self, df: pd.DataFrame, result: Dict):
        """타임스탬프 연속성 검사"""
        if 'timestamp' not in df.columns:
            return
        
        df_sorted = df.sort_values('timestamp')
        
        # 파일명에서 간격 정보 추출 시도
        filename = result["file_name"]
        interval = None
        for candidate in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]:
            if candidate in filename:
                interval = candidate
                break
        
        if not interval:
            result["metrics"]["timestamp_check"] = "interval_unknown"
            return
        
        # 예상 간격 (초 단위)
        interval_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
            "1d": 86400
        }
        
        expected_delta = interval_seconds.get(interval, 0)
        
        if expected_delta > 0 and len(df_sorted) > 1:
            # 실제 시간 간격들 계산
            actual_deltas = df_sorted['timestamp'].diff().dt.total_seconds().dropna()
            
            # 정상 범위 (±10% 허용)
            tolerance = expected_delta * 0.1
            normal_deltas = actual_deltas[
                (actual_deltas >= expected_delta - tolerance) & 
                (actual_deltas <= expected_delta + tolerance)
            ]
            
            consistency_ratio = len(normal_deltas) / len(actual_deltas)
            result["metrics"]["timestamp_consistency"] = {
                "expected_interval_seconds": expected_delta,
                "consistency_ratio": consistency_ratio,
                "irregular_gaps": len(actual_deltas) - len(normal_deltas)
            }
            
            if consistency_ratio < 0.9:  # 90% 미만 일관성
                result["issues"].append(f"Irregular timestamp intervals detected (consistency: {consistency_ratio:.1%})")
                result["recommendations"].append("Check for missing candles or data collection issues")
    
    def _check_price_validity(self, df: pd.DataFrame, result: Dict):
        """가격 데이터 유효성 검사"""
        price_issues = []
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col not in df.columns:
                continue
                
            # 0 이하 값 체크
            zero_or_negative = (df[col] <= 0).sum()
            if zero_or_negative > 0:
                price_issues.append(f"{col}: {zero_or_negative} zero/negative values")
        
        # OHLC 관계 검증 (샘플링)
        if all(col in df.columns for col in price_columns):
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            
            invalid_ohlc = sample_df[
                ~((sample_df['low'] <= sample_df['open']) & (sample_df['open'] <= sample_df['high']) &
                  (sample_df['low'] <= sample_df['close']) & (sample_df['close'] <= sample_df['high']))
            ]
            
            if not invalid_ohlc.empty:
                price_issues.append(f"Invalid OHLC relationships: {len(invalid_ohlc)} rows")
        
        result["metrics"]["price_issues"] = price_issues
        
        for issue in price_issues:
            result["issues"].append(f"Price validation: {issue}")
    
    def _check_outliers(self, df: pd.DataFrame, result: Dict):
        """이상값 검사 (상세 검사)"""
        outlier_info = {}
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            values = df[col].dropna()
            if len(values) < 10:  # 데이터가 너무 적으면 스킵
                continue
            
            # IQR 기반 이상값 탐지
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            if not outliers.empty:
                outlier_info[col] = {
                    "count": len(outliers),
                    "ratio": len(outliers) / len(values),
                    "extreme_values": {
                        "min": float(outliers.min()),
                        "max": float(outliers.max())
                    }
                }
        
        result["metrics"]["outliers"] = outlier_info
        
        # 심각한 이상값이 많으면 경고
        for col, info in outlier_info.items():
            if info["ratio"] > 0.05:  # 5% 이상 이상값
                result["issues"].append(f"High outlier ratio in {col}: {info['ratio']:.1%}")
    
    def _check_volume_patterns(self, df: pd.DataFrame, result: Dict):
        """거래량 패턴 검사"""
        if 'volume' not in df.columns:
            return
        
        volume = df['volume'].dropna()
        if len(volume) < 10:
            return
        
        # 거래량 0인 캔들 비율
        zero_volume_ratio = (volume == 0).sum() / len(volume)
        
        # 평균 대비 극단적 거래량 비율
        mean_volume = volume.mean()
        extreme_volume_ratio = (volume > mean_volume * 10).sum() / len(volume)
        
        volume_metrics = {
            "zero_volume_ratio": zero_volume_ratio,
            "extreme_volume_ratio": extreme_volume_ratio,
            "average_volume": float(mean_volume)
        }
        
        result["metrics"]["volume_patterns"] = volume_metrics
        
        if zero_volume_ratio > 0.1:  # 10% 이상 거래량 0
            result["issues"].append(f"High zero-volume ratio: {zero_volume_ratio:.1%}")
        
        if extreme_volume_ratio > 0.02:  # 2% 이상 극단적 거래량
            result["issues"].append(f"Frequent volume spikes detected: {extreme_volume_ratio:.1%}")
    
    def _check_ohlc_relationships(self, df: pd.DataFrame, result: Dict):
        """OHLC 관계 상세 검사"""
        price_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in price_columns):
            return
        
        # 다양한 OHLC 관계 검증
        relationship_checks = {
            "high_is_max": ((df['high'] >= df['open']) & 
                          (df['high'] >= df['low']) & 
                          (df['high'] >= df['close'])).all(),
            "low_is_min": ((df['low'] <= df['open']) & 
                         (df['low'] <= df['high']) & 
                         (df['low'] <= df['close'])).all(),
            "reasonable_ranges": (df['high'] - df['low']).quantile(0.95) < df['close'].mean() * 0.2  # 95% 캔들이 평균가 20% 범위 내
        }
        
        result["metrics"]["ohlc_relationships"] = relationship_checks
        
        for check_name, passed in relationship_checks.items():
            if not passed:
                result["issues"].append(f"OHLC relationship check failed: {check_name}")
    
    def verify_multiple_files(
        self,
        file_pattern: str = "*.parquet",
        detailed_check: bool = True,
    ) -> Dict[str, Any]:
        """
        📦 여러 파일 일괄 검증
        
        Args:
            file_pattern: 검증할 파일 패턴 (예: "*.parquet", "*btc*.csv")
            detailed_check: 상세 검사 여부
            
        Returns:
            전체 검증 결과 요약
        """
        self.logger.info(f"🚀 일괄 검증 시작: {file_pattern}")
        
        files_to_verify = list(self.processed_dir.glob(file_pattern))
        
        if not files_to_verify:
            self.logger.warning(f"검증할 파일이 없습니다: {file_pattern}")
            return {"status": "no_files", "files_checked": 0}
        
        verification_results = []
        summary = {
            "verification_timestamp": datetime.now().isoformat(),
            "total_files": len(files_to_verify),
            "files_checked": 0,
            "status_counts": {"healthy": 0, "warning": 0, "critical": 0, "error": 0},
            "common_issues": {}
        }
        
        for file_path in files_to_verify:
            result = self.verify_single_file(file_path, detailed_check)
            verification_results.append(result)
            
            summary["files_checked"] += 1
            status = result.get("status", "error")
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            
            # 공통 이슈 집계
            for issue in result.get("issues", []):
                issue_type = issue.split(":")[0] if ":" in issue else issue
                summary["common_issues"][issue_type] = summary["common_issues"].get(issue_type, 0) + 1
        
        # 요약 출력
        self.logger.info(f"✅ 검증 완료: {summary['files_checked']}개 파일")
        for status, count in summary["status_counts"].items():
            if count > 0:
                self.logger.info(f"   • {status}: {count}개")
        
        return summary
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """📈 데이터 상태 대시보드"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "file_counts": {
                "processed": len(list(self.processed_dir.glob("*")))
            },
            "recent_verifications": []
        }
        
        return dashboard


# 편의 함수들
def quick_verify_all(detailed: bool = False) -> Dict:
    """모든 processed 파일 빠른 검증"""
    verifier = DataIntegrityVerifier()
    return verifier.verify_multiple_files("*", detailed_check=detailed)


def health_check() -> None:
    """데이터 상태 간단 체크"""
    verifier = DataIntegrityVerifier()
    dashboard = verifier.get_health_dashboard()
    
    print("📊 데이터 상태 체크")
    print(f"  • Processed 파일: {dashboard['file_counts']['processed']}개")
    
    if dashboard["recent_verifications"]:
        latest = dashboard["recent_verifications"][0]
        print(f"  • 최근 검증: {latest['files_checked']}개 파일")
        for status, count in latest["status_counts"].items():
            if count > 0:
                print(f"    - {status}: {count}개")


if __name__ == "__main__":
    # 사용 예시
    print("🔍 DataIntegrityVerifier 테스트")
    
    verifier = DataIntegrityVerifier()
    
    # 전체 상태 체크
    health_check()
    
    # 샘플 검증 (파일이 있다면)
    processed_files = list(verifier.processed_dir.glob("*.parquet"))
    if processed_files:
        print(f"\n🧪 샘플 파일 검증: {processed_files[0].name}")
        result = verifier.verify_single_file(processed_files[0])
        print(f"결과: {result['status']}")
        if result["issues"]:
            print("이슈:")
            for issue in result["issues"][:3]:  # 최대 3개만 출력
                print(f"  - {issue}")
    else:
        print("\n⚠️ 검증할 파일이 없습니다. build_datasets.py를 먼저 실행해보세요.")