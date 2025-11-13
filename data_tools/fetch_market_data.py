"""
🔄 Upbit API 연동 및 표준 스키마 변환

Upbit 거래소에서 데이터를 가져와서 우리 표준 스키마(Candle)로 변환합니다.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time
import pytz

from .schema import Candle, ensure_candle_schema, normalize_symbol


class UpbitDataFetcher:
    """
    📈 Upbit 거래소 데이터 수집기
    
    Upbit API에서 받은 데이터를 우리 표준 스키마로 변환합니다.
    """
    
    BASE_URL = "https://api.upbit.com/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 0.5  # 0.5초 딜레이로 안전하게 (초당 2회)
    
    def _rate_limit(self):
        """API 호출 제한 준수"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_market_list(self) -> List[str]:
        """사용 가능한 마켓 목록 조회"""
        self._rate_limit()
        
        response = self.session.get(f"{self.BASE_URL}/market/all")
        response.raise_for_status()
        
        markets = response.json()
        return [market['market'] for market in markets if market['market'].startswith('KRW-')]
    
    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1m", 
        count: int = 200,
        to: Optional[datetime] = None
    ) -> List[Candle]:
        """
        🕯️ 캔들 데이터 수집 및 표준 스키마 변환
        
        Args:
            symbol: 마켓 코드 (예: "KRW-BTC")
            interval: 캔들 간격 ("1m", "5m", "1h", "1d" 등)
            count: 가져올 캔들 수 (최대 200)
            to: 마지막 캔들 시간 (None이면 최신)
            
        Returns:
            표준 Candle 객체 리스트
        """
        self._rate_limit()
        
        # 간격별 API 엔드포인트 매핑
        interval_endpoints = {
            "1m": "minutes/1", "3m": "minutes/3", "5m": "minutes/5",
            "15m": "minutes/15", "30m": "minutes/30", "1h": "minutes/60", 
            "4h": "minutes/240", "1d": "days", "1w": "weeks", "1M": "months"
        }
        
        if interval not in interval_endpoints:
            raise ValueError(f"Unsupported interval: {interval}")
        
        endpoint = interval_endpoints[interval]
        url = f"{self.BASE_URL}/candles/{endpoint}"
        
        params = {
            "market": symbol,
            "count": min(count, 200)  # Upbit 최대 200개 제한
        }
        
        if to:
            # Upbit은 UTC 기준 Z 포맷 선호 (호환성 향상)
            params["to"] = to.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        raw_data = response.json()
        
        # 표준 Candle 객체로 변환
        candles = []
        for item in raw_data:
            # Upbit 시간은 UTC 기준이지만 timezone 정보 없음
            timestamp = datetime.fromisoformat(item['candle_date_time_utc']).replace(tzinfo=pytz.UTC)
            
            candle = Candle(
                timestamp=timestamp,
                open=float(item['opening_price']),
                high=float(item['high_price']),
                low=float(item['low_price']),
                close=float(item['trade_price']),
                volume=float(item['candle_acc_trade_volume']),
                quote_volume=float(item['candle_acc_trade_price']),
                symbol=normalize_symbol(symbol, "upbit"),
                exchange="upbit",
                interval=interval
            )
            candles.append(candle)
        
        # 시간순 정렬 (오래된 것부터)
        candles.sort(key=lambda x: x.timestamp)
        
        return candles
    
    def fetch_candles_bulk(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Candle]:
        """
        📅 기간별 대량 캔들 데이터 수집
        
        200개 제한을 우회하여 긴 기간 데이터를 수집합니다.
        진행상황 표시 및 Rate Limit 안전 처리 포함.
        """
        all_candles = []
        current_end = end_date
        batch_count = 0
        
        # 예상 요청 수 계산 (대략적)
        time_diff = end_date - start_date
        if interval == "1m":
            expected_candles = int(time_diff.total_seconds() / 60)
        elif interval == "1h":
            expected_candles = int(time_diff.total_seconds() / 3600)
        elif interval == "1d":
            expected_candles = time_diff.days
        else:
            expected_candles = 1000  # 기본값
            
        expected_batches = max(1, expected_candles // 200)
        
        print(f"   Fetching {symbol} {interval} data...")
        print(f"   Expected ~{expected_candles} candles in {expected_batches} batches")
        
        # 명확한 루프 조건으로 변경
        while current_end >= start_date:
            batch_count += 1
            print(f"   Batch {batch_count}/{expected_batches} - fetching...", end=" ")
            
            try:
                batch = self.fetch_candles(
                    symbol=symbol,
                    interval=interval, 
                    count=200,
                    to=current_end
                )
                
                # API에서 데이터가 없으면 종료
                if not batch:
                    print("No more data")
                    break
                
                print(f"OK ({len(batch)} candles)")
                    
                # 시작 날짜보다 이전 데이터 필터링
                valid_batch = [c for c in batch if c.timestamp >= start_date]
                all_candles.extend(valid_batch)
                
                # 다음 배치를 위해 시간 업데이트
                oldest_timestamp = batch[0].timestamp
                current_end = oldest_timestamp - timedelta(seconds=1)
                
                # 가장 오래된 캔들이 시작일보다 이전이면 충분히 수집함
                if oldest_timestamp <= start_date:
                    print(f"   Reached start date, stopping")
                    break
                    
                # 너무 많은 요청 방지 (안전장치)
                if batch_count > 100:
                    print(f"   Reached batch limit, stopping")
                    break
                    
            except Exception as e:
                print(f"ERROR: {e}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print("   Rate limit hit, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                else:
                    raise
        
        # 중복 제거 및 시간순 정렬
        unique_candles = {}
        for candle in all_candles:
            key = (candle.timestamp, candle.symbol)
            if key not in unique_candles:
                unique_candles[key] = candle
        
        result = list(unique_candles.values())
        result.sort(key=lambda x: x.timestamp)
        
        # 스키마 검증을 통해 데이터 품질 보장
        from .schema import candles_to_dataframe, ensure_candle_schema
        if result:
            df = candles_to_dataframe(result)
            ensure_candle_schema(df)  # 검증만 하고 결과는 Candle 리스트로 반환
        
        return result
    
    def fetch_orderbook(self, symbol: str) -> Dict[str, Any]:
        """현재 오더북 조회"""
        self._rate_limit()
        
        response = self.session.get(
            f"{self.BASE_URL}/orderbook",
            params={"markets": symbol}
        )
        response.raise_for_status()
        
        return response.json()[0]
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """현재가 정보 조회"""
        self._rate_limit()
        
        response = self.session.get(
            f"{self.BASE_URL}/ticker",
            params={"markets": symbol}
        )
        response.raise_for_status()
        
        return response.json()[0]


# 🌐 멀티 거래소 지원을 위한 팩토리
class MarketDataFetcher:
    """거래소별 데이터 수집기 통합 인터페이스"""
    
    def __init__(self):
        self.fetchers = {
            "upbit": UpbitDataFetcher()
        }
    
    def get_fetcher(self, exchange: str):
        """거래소별 전용 수집기 반환"""
        if exchange not in self.fetchers:
            raise ValueError(f"Unsupported exchange: {exchange}")
        return self.fetchers[exchange]
    
    def fetch_candles(self, exchange: str, symbol: str, **kwargs) -> List[Candle]:
        """통합 캔들 데이터 수집"""
        fetcher = self.get_fetcher(exchange)
        return fetcher.fetch_candles(symbol, **kwargs)


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 간단한 테스트
    upbit = UpbitDataFetcher()
    
    try:
        candles = upbit.fetch_candles("KRW-BTC", "1h", count=10)
        
        print(f"수집된 캔들 수: {len(candles)}")
        if candles:
            print(f"최신 캔들: {candles[-1]}")
            
            # DataFrame 변환 테스트
            from .schema import candles_to_dataframe
            df = candles_to_dataframe(candles)
            print(f"\nDataFrame shape: {df.shape}")
            print(df.head())
            
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        print("API 호출 문제이거나 네트워크 연결을 확인해주세요.")