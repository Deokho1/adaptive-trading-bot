"""
SOL 데이터 수집 스크립트
2년치 4시간 단위 OHLCV 데이터를 수집하여 백테스트용 CSV로 저장
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SOLDataFetcher:
    def __init__(self):
        self.base_url = "https://api.upbit.com/v1"
        self.symbol = "KRW-SOL"
        self.interval = "240"  # 4시간
        self.output_file = "data/ohlcv/KRW-SOL_240m.csv"
        
    def fetch_candles(self, count=200, to_time=None):
        """업비트 API에서 캔들 데이터 수집"""
        url = f"{self.base_url}/candles/minutes/{self.interval}"
        
        params = {
            "market": self.symbol,
            "count": count
        }
        
        if to_time:
            params["to"] = to_time
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API 요청 실패: {e}")
            return None
            
    def collect_historical_data(self, start_date="2023-11-07", days=730):  # BTC와 동일한 시작일
        """2년치 SOL 데이터 수집 (BTC 시작일부터)"""
        logger.info(f"SOL 데이터 수집 시작: {start_date}부터 {days}일치")
        
        all_data = []
        
        # 시작일 설정
        from datetime import datetime
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # 최대 수집 가능한 캔들 수 (업비트 제한)
        max_candles = 200
        collected_candles = 0
        
        while collected_candles < days * 6:  # 4시간 간격이므로 하루 6개
            # 한 번에 최대 200개씩 수집
            batch_size = min(max_candles, (days * 6) - collected_candles)
            
            logger.info(f"배치 수집: {collected_candles}/{days * 6} ({collected_candles/(days * 6)*100:.1f}%)")
            
            candles = self.fetch_candles(count=batch_size, to_time=current_time)
            
            if not candles:
                logger.error("데이터 수집 실패")
                break
                
            # 시작일 이전 데이터는 제외
            valid_candles = []
            for candle in candles:
                candle_time = datetime.strptime(candle['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S")
                if candle_time >= start_datetime:
                    valid_candles.append(candle)
            
            # 데이터 처리
            for candle in valid_candles:
                all_data.append({
                    'timestamp': candle['candle_date_time_kst'],
                    'open': float(candle['opening_price']),
                    'high': float(candle['high_price']),
                    'low': float(candle['low_price']),
                    'close': float(candle['trade_price']),
                    'volume': float(candle['candle_acc_trade_volume'])
                })
            
            collected_candles += len(candles)
            
            # 다음 배치를 위한 시간 설정 (가장 오래된 캔들의 시간)
            if candles:
                current_time = candles[-1]['candle_date_time_kst']
                
                # 시작일보다 이전까지 갔으면 중단
                oldest_time = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S")
                if oldest_time < start_datetime:
                    logger.info(f"목표 시작일({start_date}) 도달. 수집 완료")
                    break
            
            # API 제한 준수 (초당 10회)
            time.sleep(0.1)
            
        logger.info(f"데이터 수집 완료: {len(all_data)}개 캔들")
        return all_data
    
    def save_to_csv(self, data):
        """CSV 파일로 저장"""
        df = pd.DataFrame(data)
        
        # 타임스탬프 변환 및 정렬
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # 디렉토리 생성
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        df.to_csv(output_path, index=False)
        
        logger.info(f"SOL 데이터 저장 완료: {output_path}")
        logger.info(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        logger.info(f"총 {len(df)}개 데이터 포인트")
        
        # 데이터 품질 검증
        self.validate_data(df)
        
        return df
    
    def validate_data(self, df):
        """데이터 품질 검증"""
        logger.info("=== 데이터 품질 검증 ===")
        
        # 기본 통계
        logger.info(f"가격 범위: {df['close'].min():,.0f} ~ {df['close'].max():,.0f} KRW")
        logger.info(f"평균 거래량: {df['volume'].mean():.2f}")
        logger.info(f"누락 데이터: {df.isnull().sum().sum()}개")
        
        # 시간 간격 검증 (4시간 = 240분)
        time_diffs = df['timestamp'].diff().dt.total_seconds() / 60
        expected_interval = 240
        
        correct_intervals = (time_diffs == expected_interval).sum()
        total_intervals = len(time_diffs) - 1
        
        logger.info(f"시간 간격 정확도: {correct_intervals}/{total_intervals} ({correct_intervals/total_intervals*100:.1f}%)")
        
        # 가격 이상치 검증
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.3).sum()  # 30% 이상 변동
        
        logger.info(f"극단적 가격 변동 (>30%): {extreme_changes}개")
        
        if extreme_changes > 0:
            logger.warning("극단적 가격 변동이 감지되었습니다. 데이터 확인 필요")
            
    def run(self):
        """메인 실행 함수"""
        logger.info("🔗 SOL 데이터 수집 시작")
        
        # 데이터 수집
        data = self.collect_historical_data()
        
        if not data:
            logger.error("데이터 수집 실패")
            return None
            
        # CSV 저장
        df = self.save_to_csv(data)
        
        logger.info("✅ SOL 데이터 준비 완료!")
        return df

if __name__ == "__main__":
    fetcher = SOLDataFetcher()
    sol_data = fetcher.run()
    
    if sol_data is not None:
        print(f"\n📊 SOL 데이터 미리보기:")
        print(sol_data.head(10))
        print(f"\n📈 최신 SOL 가격: {sol_data['close'].iloc[-1]:,.0f} KRW")