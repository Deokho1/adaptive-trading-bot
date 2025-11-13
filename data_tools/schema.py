"""
🏛️ 적응형 트레이딩 봇 데이터 스키마 공식 선언문

이 파일은 우리 프로젝트의 "데이터 헌법"입니다.
어떤 거래소에서 데이터를 받든, 우리 시스템 안에서는 모두 이 형태로 통일합니다.

핵심 원칙:
"입구에서만 변환하고, 안쪽은 편하게 쓴다"
- 입구: fetch_market_data, exchange_api_* → 거래소별 데이터를 우리 스키마로 변환
- 안쪽: 백테스트, 전략 로직, 리포트 → 오직 이 스키마만 바라봄
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class Candle:
    """
    📊 표준 캔들(OHLCV) 데이터 구조
    
    모든 거래소 데이터는 이 형태로 정규화됩니다.
    """
    timestamp: datetime      # 캔들 시작 시간 (UTC 기준, timezone-aware)
    open: float             # 구간 첫 체결가
    high: float             # 구간 최고가  
    low: float              # 구간 최저가
    close: float            # 구간 마지막 체결가
    volume: float           # 코인 수량 기준 거래량 (BTC 개수 등)
    quote_volume: float     # 원화/USDT 기준 거래금액
    symbol: str             # 마켓 코드 (예: "BTC-KRW", "BTCUSDT")
    exchange: str           # 거래소 이름 (예: "upbit", "binance")
    interval: str           # 캔들 주기 (예: "1m", "5m", "1h", "1d")
    
    def __post_init__(self):
        """데이터 검증 및 타입 변환"""
        # 가격 데이터는 반드시 양수
        for price_field in ['open', 'high', 'low', 'close']:
            value = getattr(self, price_field)
            if value <= 0:
                raise ValueError(f"{price_field} must be positive, got {value}")
        
        # OHLC 논리적 관계 검증
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(f"Invalid OHLC relationship: O={self.open}, H={self.high}, L={self.low}, C={self.close}")
        
        # 거래량은 0 이상
        if self.volume < 0 or self.quote_volume < 0:
            raise ValueError("Volume must be non-negative")


@dataclass 
class OrderBook:
    """
    📋 오더북 데이터 구조
    """
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[tuple]       # [(가격, 수량), ...] 내림차순 정렬
    asks: List[tuple]       # [(가격, 수량), ...] 오름차순 정렬
    
    @property
    def best_bid(self) -> Optional[float]:
        """최고 매수가"""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """최저 매도가"""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """호가 스프레드"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


# 🎯 표준 DataFrame 스키마 정의
CANDLE_SCHEMA = {
    'timestamp': 'datetime64[ns, UTC]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'quote_volume': 'float64',
    'symbol': 'str',
    'exchange': 'str',
    'interval': 'str'
}

REQUIRED_CANDLE_COLUMNS = list(CANDLE_SCHEMA.keys())


def ensure_candle_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    🔍 DataFrame이 표준 캔들 스키마를 준수하는지 검증 및 강제
    
    Args:
        df: 검증할 DataFrame
        strict: True시 모든 컬럼 필수, False시 부분적 허용
        
    Returns:
        스키마에 맞춰 정리된 DataFrame
        
    Raises:
        ValueError: 스키마 위반시
    """
    if df.empty:
        return df
    
    # 필수 컬럼 존재 확인
    missing_cols = set(REQUIRED_CANDLE_COLUMNS) - set(df.columns)
    if missing_cols and strict:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 타입 변환 시도
    df_copy = df.copy()
    
    for col, dtype in CANDLE_SCHEMA.items():
        if col in df_copy.columns:
            try:
                if dtype.startswith('datetime'):
                    df_copy[col] = pd.to_datetime(df_copy[col], utc=True)
                elif dtype == 'float64':
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                elif dtype == 'str':
                    df_copy[col] = df_copy[col].astype(str)
            except Exception as e:
                raise ValueError(f"Failed to convert column {col} to {dtype}: {e}")
    
    # 가격 데이터 범위 검증
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df_copy.columns:
            if (df_copy[col] <= 0).any():
                raise ValueError(f"Column {col} contains non-positive values")
    
    # OHLC 논리적 관계 검증 (샘플링)
    if all(col in df_copy.columns for col in price_columns):
        sample_size = min(1000, len(df_copy))  # 성능을 위해 샘플링
        sample = df_copy.sample(n=sample_size) if len(df_copy) > sample_size else df_copy
        
        invalid_rows = sample[
            ~((sample['low'] <= sample['open']) & (sample['open'] <= sample['high']) &
              (sample['low'] <= sample['close']) & (sample['close'] <= sample['high']))
        ]
        
        if not invalid_rows.empty:
            raise ValueError(f"Found {len(invalid_rows)} rows with invalid OHLC relationships")
    
    # 컬럼 순서 정리
    available_cols = [col for col in REQUIRED_CANDLE_COLUMNS if col in df_copy.columns]
    return df_copy[available_cols]


def validate_candle_data(candles: List[Candle]) -> Dict[str, Any]:
    """
    📊 캔들 데이터 품질 리포트 생성
    
    Returns:
        검증 결과 및 통계 정보
    """
    if not candles:
        return {"status": "empty", "message": "No candle data provided"}
    
    report = {
        "total_candles": len(candles),
        "date_range": {
            "start": min(c.timestamp for c in candles),
            "end": max(c.timestamp for c in candles)
        },
        "symbols": list(set(c.symbol for c in candles)),
        "exchanges": list(set(c.exchange for c in candles)),
        "intervals": list(set(c.interval for c in candles)),
        "issues": []
    }
    
    # 시간 순서 확인
    timestamps = [c.timestamp for c in candles]
    if timestamps != sorted(timestamps):
        report["issues"].append("Candles are not in chronological order")
    
    # 중복 확인
    timestamp_symbol_pairs = [(c.timestamp, c.symbol) for c in candles]
    if len(timestamp_symbol_pairs) != len(set(timestamp_symbol_pairs)):
        report["issues"].append("Duplicate timestamp-symbol combinations found")
    
    # 가격 이상값 확인
    prices = []
    for c in candles:
        prices.extend([c.open, c.high, c.low, c.close])
    
    if prices:
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        outlier_threshold = q3 + 3 * iqr
        outliers = [p for p in prices if p > outlier_threshold]
        
        if outliers:
            report["issues"].append(f"Found {len(outliers)} potential price outliers")
    
    report["status"] = "clean" if not report["issues"] else "issues_found"
    return report


def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    """캔들 리스트를 DataFrame으로 변환"""
    if not candles:
        return pd.DataFrame(columns=REQUIRED_CANDLE_COLUMNS)
    
    data = []
    for candle in candles:
        data.append({
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low, 
            'close': candle.close,
            'volume': candle.volume,
            'quote_volume': candle.quote_volume,
            'symbol': candle.symbol,
            'exchange': candle.exchange,
            'interval': candle.interval
        })
    
    return pd.DataFrame(data)


def dataframe_to_candles(df: pd.DataFrame) -> List[Candle]:
    """DataFrame을 캔들 리스트로 변환"""
    df = ensure_candle_schema(df)
    
    candles = []
    for _, row in df.iterrows():
        candles.append(Candle(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            quote_volume=row['quote_volume'],
            symbol=row['symbol'],
            exchange=row['exchange'],
            interval=row['interval']
        ))
    
    return candles


# 🏷️ 거래소별 심볼 정규화 맵핑
SYMBOL_NORMALIZATION = {
    "upbit": {
        "KRW-BTC": "BTC-KRW",
        "KRW-ETH": "ETH-KRW",
        # 필요시 추가...
    },
    "binance": {
        "BTCUSDT": "BTC-USDT", 
        "ETHUSDT": "ETH-USDT",
        # 필요시 추가...
    }
}

def normalize_symbol(raw_symbol: str, exchange: str) -> str:
    """거래소별 심볼을 표준 형태로 정규화"""
    mapping = SYMBOL_NORMALIZATION.get(exchange, {})
    return mapping.get(raw_symbol, raw_symbol)