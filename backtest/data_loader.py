"""
📂 백테스트용 데이터 로더

processed/ 폴더의 데이터를 읽어서 스키마 검증 후 백테스트 엔진에 전달합니다.
모든 데이터는 표준 스키마를 거쳐야 백테스트에서 사용할 수 있습니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
import glob
import pickle

from data_tools.schema import (
    Candle, ensure_candle_schema, validate_candle_data, 
    candles_to_dataframe, dataframe_to_candles, REQUIRED_CANDLE_COLUMNS
)


class BacktestDataLoader:
    """
    🔄 백테스트용 데이터 로더
    
    역할:
    1. processed/ 폴더에서 데이터 파일 읽기
    2. 표준 스키마 검증 및 강제 적용  
    3. 백테스트 엔진이 원하는 형태로 데이터 제공
    4. 메모리 효율적인 배치 로딩 지원
    """
    
    def __init__(self, data_dir: str = "backtest_data/processed"):
        self.data_dir = Path(data_dir)
        self.cache = {}  # 메모리 캐시
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """데이터 디렉터리 생성"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """사용 가능한 데이터 파일 목록"""
        files = {}
        
        # CSV 파일들
        csv_files = list(self.data_dir.glob("*.csv"))
        if csv_files:
            files['csv'] = [f.name for f in csv_files]
        
        # Parquet 파일들  
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if parquet_files:
            files['parquet'] = [f.name for f in parquet_files]
        
        # Pickle 파일들
        pickle_files = list(self.data_dir.glob("*.pkl"))
        if pickle_files:
            files['pickle'] = [f.name for f in pickle_files]
        
        return files
    
    def load_candles_from_file(
        self, 
        filename: str,
        symbol_filter: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        📊 파일에서 캔들 데이터 로드 및 스키마 검증
        
        Args:
            filename: 데이터 파일명
            symbol_filter: 특정 심볼만 필터링
            date_range: (시작일, 종료일) 범위 필터링  
            validate: 스키마 검증 여부
            
        Returns:
            검증된 캔들 DataFrame
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # 캐시 확인
        cache_key = f"{filename}_{symbol_filter}_{date_range}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # 파일 형식에 따른 로딩
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif filename.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        if df.empty:
            return df
        
        # 스키마 검증 및 강제
        if validate:
            try:
                df = ensure_candle_schema(df, strict=True)
            except Exception as e:
                raise ValueError(f"Schema validation failed for {filename}: {e}")
        
        # 필터링 적용
        if symbol_filter:
            df = df[df['symbol'] == symbol_filter]
        
        if date_range:
            start_date, end_date = date_range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # 시간순 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 캐시 저장 (메모리 제한 고려)
        if len(self.cache) < 10:  # 최대 10개 파일 캐시
            self.cache[cache_key] = df.copy()
        
        return df
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """
        📈 여러 심볼의 데이터를 동시에 로드
        
        Returns:
            {symbol: DataFrame} 형태의 딕셔너리
        """
        result = {}
        
        # 사용 가능한 파일들 확인
        available_files = self.list_available_data()
        all_files = []
        for file_type, files in available_files.items():
            all_files.extend(files)
        
        for symbol in symbols:
            symbol_data = None
            
            # 심볼별 전용 파일 찾기
            symbol_files = [f for f in all_files if symbol.replace('-', '_') in f]
            
            if symbol_files:
                # 가장 적절한 파일 선택 (간격 매칭)
                best_file = None
                for file in symbol_files:
                    if interval in file:
                        best_file = file
                        break
                
                if not best_file:
                    best_file = symbol_files[0]  # 첫 번째 파일 사용
                
                try:
                    symbol_data = self.load_candles_from_file(
                        best_file,
                        symbol_filter=symbol,
                        date_range=(start_date, end_date)
                    )
                except Exception as e:
                    print(f"Warning: Failed to load {symbol} from {best_file}: {e}")
            
            if symbol_data is None or symbol_data.empty:
                # 통합 파일에서 찾기
                for file in all_files:
                    try:
                        symbol_data = self.load_candles_from_file(
                            file,
                            symbol_filter=symbol,
                            date_range=(start_date, end_date)
                        )
                        if not symbol_data.empty:
                            break
                    except Exception:
                        continue
            
            result[symbol] = symbol_data if symbol_data is not None else pd.DataFrame()
        
        return result
    
    def create_batch_iterator(
        self,
        filename: str,
        batch_size_days: int = 30,
        overlap_days: int = 1,
        **kwargs
    ):
        """
        🔄 메모리 효율적인 배치 이터레이터
        
        큰 데이터셋을 작은 배치로 나누어 처리할 수 있게 해줍니다.
        """
        # 전체 데이터의 날짜 범위 확인
        df_sample = self.load_candles_from_file(filename, **kwargs)
        if df_sample.empty:
            return
        
        start_date = df_sample['timestamp'].min()
        end_date = df_sample['timestamp'].max()
        
        current_start = start_date
        batch_delta = timedelta(days=batch_size_days)
        overlap_delta = timedelta(days=overlap_days)
        
        while current_start < end_date:
            current_end = min(current_start + batch_delta, end_date)
            
            # 배치 로드
            batch_df = self.load_candles_from_file(
                filename,
                date_range=(current_start, current_end),
                **kwargs
            )
            
            if not batch_df.empty:
                yield batch_df
            
            # 다음 배치 시작점 (오버랩 고려)
            current_start = current_end - overlap_delta
    
    def get_data_quality_report(self, filename: str) -> Dict:
        """
        📋 데이터 품질 리포트 생성
        """
        try:
            df = self.load_candles_from_file(filename, validate=False)
            
            if df.empty:
                return {"status": "empty", "message": "No data in file"}
            
            # 기본 통계
            report = {
                "file_info": {
                    "filename": filename,
                    "total_rows": len(df),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "date_range": {
                    "start": df['timestamp'].min() if 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max() if 'timestamp' in df.columns else None,
                    "days": None
                },
                "symbols": list(df['symbol'].unique()) if 'symbol' in df.columns else [],
                "schema_compliance": {
                    "missing_columns": [],
                    "extra_columns": [],
                    "type_issues": []
                },
                "data_quality": {
                    "missing_values": {},
                    "duplicate_rows": 0,
                    "price_anomalies": []
                }
            }
            
            # 날짜 범위 계산
            if report["date_range"]["start"] and report["date_range"]["end"]:
                report["date_range"]["days"] = (
                    report["date_range"]["end"] - report["date_range"]["start"]
                ).days
            
            # 스키마 준수 체크
            required_cols = set(REQUIRED_CANDLE_COLUMNS)
            actual_cols = set(df.columns)
            
            report["schema_compliance"]["missing_columns"] = list(required_cols - actual_cols)
            report["schema_compliance"]["extra_columns"] = list(actual_cols - required_cols)
            
            # 결측값 체크
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    report["data_quality"]["missing_values"][col] = missing_count
            
            # 중복행 체크
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                report["data_quality"]["duplicate_rows"] = df.duplicated(['timestamp', 'symbol']).sum()
            
            # 가격 이상값 체크 (간단한 버전)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    # 0 이하 값
                    zero_or_negative = (df[col] <= 0).sum()
                    if zero_or_negative > 0:
                        report["data_quality"]["price_anomalies"].append(
                            f"{col}: {zero_or_negative} zero/negative values"
                        )
                    
                    # 극단값 (간단한 체크)
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    outliers = ((df[col] > q99 * 10) | (df[col] < q01 / 10)).sum()
                    if outliers > 0:
                        report["data_quality"]["price_anomalies"].append(
                            f"{col}: {outliers} potential outliers"
                        )
            
            # 전체 상태 요약
            issues = []
            issues.extend(report["schema_compliance"]["missing_columns"])
            issues.extend(list(report["data_quality"]["missing_values"].keys()))
            if report["data_quality"]["duplicate_rows"] > 0:
                issues.append("duplicates")
            if report["data_quality"]["price_anomalies"]:
                issues.append("price_anomalies")
            
            report["status"] = "clean" if not issues else "issues_found"
            report["issues_summary"] = issues
            
            return report
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to analyze file: {str(e)}"
            }
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "parquet",
        validate: bool = True
    ):
        """
        💾 처리된 데이터 저장 (스키마 검증 포함)
        """
        if validate:
            df = ensure_candle_schema(df)
        
        file_path = self.data_dir / filename
        
        if format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "pickle":
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(df)} rows to {file_path}")


# 편의 함수들
def quick_load(filename: str, symbol: str = None) -> pd.DataFrame:
    """빠른 데이터 로딩"""
    loader = BacktestDataLoader()
    return loader.load_candles_from_file(filename, symbol_filter=symbol)


def data_summary(filename: str) -> None:
    """데이터 파일 요약 출력"""
    loader = BacktestDataLoader()
    report = loader.get_data_quality_report(filename)
    
    print(f"\n📊 Data Summary: {filename}")
    print("-" * 50)
    print(f"Status: {report['status']}")
    print(f"Total rows: {report['file_info']['total_rows']:,}")
    print(f"Memory usage: {report['file_info']['memory_usage_mb']:.1f} MB")
    
    if report['date_range']['start']:
        print(f"Date range: {report['date_range']['start']} to {report['date_range']['end']}")
        print(f"Duration: {report['date_range']['days']} days")
    
    print(f"Symbols: {len(report['symbols'])} ({', '.join(report['symbols'][:5])}{'...' if len(report['symbols']) > 5 else ''})")
    
    if report['issues_summary']:
        print(f"⚠️  Issues found: {', '.join(report['issues_summary'])}")
    else:
        print("✅ No issues detected")


if __name__ == "__main__":
    # 간단한 테스트
    loader = BacktestDataLoader()
    available = loader.list_available_data()
    print("Available data files:", available)