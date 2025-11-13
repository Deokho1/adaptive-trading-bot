"""
📦 백테스트용 데이터셋 구축기

fetcher로 수집한 데이터들을 파일로 정리하고 저장하는 배치 스크립트입니다.

주요 역할:
1. 여러 심볼 / 여러 타임프레임 한꺼번에 생성
2. processed 폴더에 파일 저장  
3. 저장 포맷 변환 (CSV, Parquet, Pickle)
4. 메타데이터 로그 작성

흐름: fetch_market_data.py → build_datasets.py → verify_integrity.py
"""

import json
import pandas as pd
import pytz
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import os
import time

from .fetch_market_data import UpbitDataFetcher, MarketDataFetcher
from .schema import candles_to_dataframe, ensure_candle_schema


class DatasetBuilder:
    """
    🏗️ 백테스트용 데이터셋 구축기
    
    여러 심볼과 타임프레임을 조합하여 체계적으로 데이터를 수집하고 저장합니다.
    """
    
    def __init__(self, base_data_dir: str = "backtest_data"):
        self.base_dir = Path(base_data_dir)
        self.processed_dir = self.base_dir / "processed" 
        self.metadata_dir = self.base_dir / "metadata"
        
        self._ensure_directories()
        self.fetcher = MarketDataFetcher()
        
    def _ensure_directories(self):
        """필요한 디렉터리들 생성"""
        for dir_path in [self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _normalize_symbol_for_filename(self, symbol: str) -> str:
        """심볼명을 파일명에 적합하게 변환"""
        return symbol.replace('-', '_').replace('/', '_').lower()
    
    def _generate_filename(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime,
        end_date: datetime,
        file_type: str = "processed"
    ) -> str:
        """파일명 생성 (간격 정보 포함)"""
        norm_symbol = self._normalize_symbol_for_filename(symbol)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        return f"{norm_symbol}_{interval}_{start_str}_{end_str}_{file_type}"
    
    def build_single_dataset(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        save_formats: List[str] = ["parquet", "csv"]
    ) -> Dict:
        """
        🎯 단일 심볼/간격 데이터셋 구축
        
        Args:
            exchange: 거래소명 ("upbit" 등)
            symbol: 마켓 코드 ("KRW-BTC" 등)
            interval: 캔들 간격 ("1m", "5m", "1h" 등)
            start_date: 수집 시작일
            end_date: 수집 종료일
            save_formats: 저장할 포맷 리스트
            
        Returns:
            구축 결과 정보
        """
        print(f"Dataset build starting: {symbol} {interval} ({start_date.date()} ~ {end_date.date()})")
        
        build_info = {
            "symbol": symbol,
            "exchange": exchange, 
            "interval": interval,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "build_timestamp": datetime.now().isoformat(),
            "status": "started",
            "candles_collected": 0,
            "files_created": [],
            "errors": []
        }
        
        try:
            # 1. 날짜 타임존 처리 
            if start_date and start_date.tzinfo is None:
                start_date = pytz.UTC.localize(start_date)
            if end_date and end_date.tzinfo is None:
                end_date = pytz.UTC.localize(end_date)
                
            # 2. 데이터 수집
            fetcher = self.fetcher.get_fetcher(exchange)
            
            if hasattr(fetcher, 'fetch_candles_bulk'):
                candles = fetcher.fetch_candles_bulk(symbol, interval, start_date, end_date)
            else:
                # bulk 지원 안 하면 일반 fetch로 대체
                candles = fetcher.fetch_candles(symbol, interval, count=200)
            
            build_info["candles_collected"] = len(candles)
            
            if not candles:
                build_info["status"] = "no_data"
                build_info["errors"].append("No candles received from API")
                return build_info
            
            # 2. DataFrame 변환 및 검증
            df = candles_to_dataframe(candles)
            df = ensure_candle_schema(df)
            
            print(f"   OK {len(df)} candles collected")
            
            # 3. 가공 데이터 저장 (여러 포맷)
            processed_filename = self._generate_filename(symbol, interval, start_date, end_date, "processed")
            
            for fmt in save_formats:
                if fmt == "parquet":
                    file_path = self.processed_dir / f"{processed_filename}.parquet"
                    df.to_parquet(file_path, index=False)
                elif fmt == "csv":
                    file_path = self.processed_dir / f"{processed_filename}.csv"
                    df.to_csv(file_path, index=False)
                elif fmt == "pickle":
                    file_path = self.processed_dir / f"{processed_filename}.pkl"
                    df.to_pickle(file_path)
                else:
                    build_info["errors"].append(f"Unsupported format: {fmt}")
                    continue
                
                build_info["files_created"].append(str(file_path))
                print(f"   Processed data saved: {file_path.name}")
            
            # 5. 메타데이터 로그 작성
            metadata = {
                **build_info,
                "data_quality": {
                    "date_range_actual": {
                        "start": df['timestamp'].min().isoformat(),
                        "end": df['timestamp'].max().isoformat()
                    },
                    "missing_candles": self._detect_missing_candles(df, interval),
                    "price_stats": {
                        "min_price": float(df['close'].min()),
                        "max_price": float(df['close'].max()),
                        "avg_price": float(df['close'].mean())
                    },
                    "volume_stats": {
                        "total_volume": float(df['volume'].sum()),
                        "avg_volume": float(df['volume'].mean())
                    }
                }
            }
            
            metadata_path = self.metadata_dir / f"{processed_filename}_meta.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            build_info["files_created"].append(str(metadata_path))
            build_info["status"] = "completed"
            
            print(f"   BUILD COMPLETED! {len(build_info['files_created'])} files created")
            
        except Exception as e:
            build_info["status"] = "error"
            build_info["errors"].append(str(e))
            print(f"   ERROR occurred: {e}")
        
        return build_info
    
    def build_multiple_datasets(
        self,
        pairs_config: List[Dict],
        batch_delay: float = 1.0
    ) -> List[Dict]:
        """
        📦 여러 데이터셋 일괄 구축
        
        Args:
            pairs_config: 구축할 데이터셋 설정 리스트
                예: [{"exchange": "upbit", "symbol": "KRW-BTC", "interval": "1h", ...}, ...]
            batch_delay: 각 구축 사이 딜레이 (초)
            
        Returns:
            각 구축 작업의 결과 리스트
        """
        print(f"🚀 일괄 데이터셋 구축 시작 - 총 {len(pairs_config)}개 작업")
        
        results = []
        total_start_time = time.time()
        
        for i, config in enumerate(pairs_config, 1):
            print(f"\n📋 작업 {i}/{len(pairs_config)}")
            
            result = self.build_single_dataset(**config)
            results.append(result)
            
            # 다음 작업 전 딜레이 (API 부하 방지)
            if i < len(pairs_config):
                print(f"   ⏳ {batch_delay}초 대기...")
                time.sleep(batch_delay)
        
        # 전체 요약
        total_time = time.time() - total_start_time
        successful = len([r for r in results if r["status"] == "completed"])
        failed = len(results) - successful
        
        print(f"\n📊 일괄 구축 완료!")
        print(f"   • 총 소요시간: {total_time:.1f}초")
        print(f"   • 성공: {successful}개, 실패: {failed}개")
        
        if failed > 0:
            print("   ❌ 실패한 작업들:")
            for result in results:
                if result["status"] != "completed":
                    print(f"     - {result['symbol']} {result['interval']}: {result.get('errors', [])}")
        
        # 전체 요약 메타데이터 저장
        summary_path = self.metadata_dir / f"build_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary = {
            "batch_info": {
                "total_jobs": len(pairs_config),
                "successful": successful,
                "failed": failed,
                "total_time_seconds": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "job_results": results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   📄 요약 리포트 저장: {summary_path.name}")
        
        return results
    
    def _detect_missing_candles(self, df: pd.DataFrame, interval: str) -> int:
        """캔들 누락 개수 추정"""
        if len(df) < 2:
            return 0
        
        # 간격별 예상 시간 차이 (분 단위)
        interval_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440, "1w": 10080
        }
        
        if interval not in interval_minutes:
            return 0  # 알 수 없는 간격
        
        expected_minutes = interval_minutes[interval]
        df_sorted = df.sort_values('timestamp')
        
        start_time = df_sorted['timestamp'].iloc[0]
        end_time = df_sorted['timestamp'].iloc[-1]
        
        # 예상 캔들 수
        total_minutes = (end_time - start_time).total_seconds() / 60
        expected_candles = int(total_minutes / expected_minutes) + 1
        
        # 실제 캔들 수와 비교
        actual_candles = len(df)
        missing = max(0, expected_candles - actual_candles)
        
        return missing
    
    def get_build_status(self) -> Dict:
        """구축된 데이터셋 현황 조회"""
        status = {
            "directories": {
                "processed_parquet": len(list(self.processed_dir.glob("*.parquet"))), 
                "processed_csv": len(list(self.processed_dir.glob("*.csv"))),
                "metadata": len(list(self.metadata_dir.glob("*.json")))
            },
            "recent_builds": []
        }
        
        # 최근 구축 작업들 (메타데이터 기준)
        metadata_files = sorted(
            self.metadata_dir.glob("*_meta.json"), 
            key=lambda x: x.stat().st_mtime, 
            reverse=True
        )[:10]
        
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    status["recent_builds"].append({
                        "symbol": meta.get("symbol"),
                        "interval": meta.get("interval"),
                        "candles": meta.get("candles_collected", 0),
                        "status": meta.get("status"),
                        "timestamp": meta.get("build_timestamp")
                    })
            except Exception:
                continue
        
        return status


# 편의 함수들
def quick_build_upbit_dataset(
    symbols: List[str],
    intervals: List[str],
    days_back: int = 30,
    **kwargs
) -> List[Dict]:
    """
    ⚡ Upbit 데이터셋 빠른 구축
    
    Args:
        symbols: 심볼 리스트 (예: ["KRW-BTC", "KRW-ETH"])
        intervals: 간격 리스트 (예: ["1h", "1d"])
        days_back: 며칠 전부터 수집할지
    """
    builder = DatasetBuilder()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    configs = []
    for symbol in symbols:
        for interval in intervals:
            configs.append({
                "exchange": "upbit",
                "symbol": symbol,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                **kwargs
            })
    
    return builder.build_multiple_datasets(configs)


if __name__ == "__main__":
    # 사용 예시
    print("🔧 DatasetBuilder 테스트")
    
    # 간단한 구축 테스트
    builder = DatasetBuilder()
    
    # 단일 데이터셋 구축 (최근 7일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    result = builder.build_single_dataset(
        exchange="upbit",
        symbol="KRW-BTC", 
        interval="1h",
        start_date=start_date,
        end_date=end_date,
        save_formats=["parquet"]
    )
    
    print(f"\n구축 결과: {result['status']}")
    if result["files_created"]:
        print("생성된 파일들:")
        for file in result["files_created"]:
            print(f"  - {file}")
    
    # 현재 상태 확인
    status = builder.get_build_status()
    print(f"\n📊 현재 데이터셋 상태:")
    for dir_name, count in status["directories"].items():
        print(f"  {dir_name}: {count}개 파일")