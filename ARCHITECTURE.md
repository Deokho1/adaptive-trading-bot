# 🏗️ Adaptive Trading Bot - 아키텍처 분석 문서

## 📋 목차
1. [전체 아키텍처 개요](#전체-아키텍처-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [핵심 모듈 역할](#핵심-모듈-역할)
4. [파일간 의존성](#파일간-의존성)
5. [데이터 흐름](#데이터-흐름)
6. [실행 모드](#실행-모드)

---

## 전체 아키텍처 개요

### 아키텍처 패턴
- **계층형 아키텍처 (Layered Architecture)**
- **표준 스키마 기반 데이터 파이프라인**
- **백테스트/라이브 분리 설계**

### 핵심 설계 원칙
1. **표준 스키마 우선**: 모든 거래소 데이터는 `Candle` 스키마로 통일
2. **입구에서만 변환**: 거래소별 변환은 `data_tools`에서만 수행
3. **내부는 편하게**: 코어 로직은 표준 스키마만 사용
4. **백테스트/라이브 분리**: 동일한 전략 로직을 양쪽에서 재사용

---

## 프로젝트 구조

```
adaptive-trading-bot/
├── api/                          # 거래소 API 추상화
│   ├── exchange_api_backtest.py  # 백테스트용 가상 거래소
│   ├── exchange_api_live.py      # 실거래용 거래소 API (KIS 등)
│   └── websocket_manager.py      # WebSocket 실시간 데이터
│
├── core/                         # 핵심 비즈니스 로직
│   ├── strategy_core.py          # 전략 로직 (진입/청산 조건)
│   ├── market_watcher.py         # 시장 감시 모듈
│   ├── signal_engine.py          # 신호 생성 엔진
│   ├── trade_executor.py         # 거래 실행 모듈
│   └── adaptive_memory.py       # 적응형 메모리 (학습/최적화)
│
├── data_tools/                   # 데이터 수집 및 처리 파이프라인
│   ├── fetch_market_data.py      # 거래소 API 연동 (Upbit 등)
│   ├── build_datasets.py         # 데이터셋 구축 및 저장
│   ├── verify_integrity.py       # 데이터 무결성 검증
│   └── schema.py                 # 표준 데이터 스키마 정의
│
├── backtest/                     # 백테스트 엔진
│   ├── data_loader.py            # 백테스트 데이터 로더
│   ├── backtest_runner.py        # 백테스트 실행기
│   └── result_analyzer.py        # 결과 분석기
│
├── live/                         # 실시간 거래 시스템
│   ├── live_runner.py            # 실시간 거래 루프
│   ├── risk_monitor.py           # 리스크 모니터링
│   └── event_handler.py          # 이벤트 핸들링
│
├── reports/                      # 리포트 및 분석
│   ├── metrics.py                # 성능 지표 계산
│   ├── trade_reporter.py         # 거래 리포트
│   └── visualization.py          # 시각화
│
├── backtest_data/                # 백테스트 데이터 저장소
│   ├── processed/                # 처리된 데이터 (CSV, Parquet)
│   └── metadata/                 # 메타데이터 (JSON)
│
├── main_backtest.py              # 백테스트 메인 진입점
├── main_live.py                  # 실거래 메인 진입점
├── cli.py                        # CLI 도구 (데이터 수집/관리)
└── config.py                     # 설정 파일
```

---

## 핵심 모듈 역할

### 1. 데이터 계층 (Data Layer)

#### `data_tools/schema.py` ⭐ **핵심 스키마 정의**
- **역할**: 프로젝트 전체의 데이터 표준 정의
- **주요 클래스**:
  - `Candle`: 표준 OHLCV 캔들 구조체
  - `OrderBook`: 오더북 데이터 구조
- **주요 함수**:
  - `ensure_candle_schema()`: DataFrame 스키마 검증 및 강제
  - `candles_to_dataframe()`: Candle 리스트 → DataFrame 변환
  - `dataframe_to_candles()`: DataFrame → Candle 리스트 변환
- **의존성**: 없음 (최하위 계층)

#### `data_tools/fetch_market_data.py` 📥 **데이터 수집**
- **역할**: 거래소 API에서 데이터 수집 및 표준 스키마 변환
- **주요 클래스**:
  - `UpbitDataFetcher`: Upbit API 연동
  - `MarketDataFetcher`: 멀티 거래소 팩토리
- **주요 기능**:
  - `fetch_candles()`: 캔들 데이터 수집 (최대 200개)
  - `fetch_candles_bulk()`: 대량 데이터 수집 (200개 제한 우회)
  - Rate Limit 자동 처리
- **의존성**: `schema.py`

#### `data_tools/build_datasets.py` 🏗️ **데이터셋 구축**
- **역할**: 수집한 데이터를 파일로 저장 및 메타데이터 생성
- **주요 클래스**: `DatasetBuilder`
- **주요 기능**:
  - `build_single_dataset()`: 단일 심볼/간격 데이터셋 구축
  - `build_multiple_datasets()`: 일괄 데이터셋 구축
  - CSV, Parquet, Pickle 포맷 지원
  - 메타데이터 자동 생성
- **의존성**: `fetch_market_data.py`, `schema.py`

#### `data_tools/verify_integrity.py` 🔍 **데이터 검증**
- **역할**: 저장된 데이터의 무결성 검증
- **주요 클래스**: `DataIntegrityVerifier`
- **검증 항목**:
  - 타임스탬프 연속성
  - 결측값 체크
  - 중복 캔들 검사
  - 가격 이상치 탐지
  - 스키마 일관성
- **의존성**: `schema.py`

#### `backtest/data_loader.py` 📂 **백테스트 데이터 로더**
- **역할**: 저장된 데이터를 백테스트 엔진에 제공
- **주요 클래스**: `BacktestDataLoader`
- **주요 기능**:
  - `load_candles_from_file()`: 파일에서 데이터 로드
  - `load_data_for_backtest()`: 백테스트용 데이터 로드 (자동 생성 포함)
    - 기존 데이터 파일 자동 탐지
    - 없으면 `DatasetBuilder`로 자동 생성
    - 생성 후 자동 로드
  - `load_multiple_symbols()`: 여러 심볼 동시 로드
  - `create_batch_iterator()`: 메모리 효율적 배치 처리
  - `_find_data_file()`: 기존 데이터 파일 찾기
  - 스키마 검증 및 캐싱
- **의존성**: `data_tools/schema.py`, `data_tools/build_datasets.py`

---

### 2. 코어 비즈니스 로직 (Core Business Logic)

#### `core/strategy_core.py` 🎯 **전략 로직**
- **역할**: 진입/청산 조건 정의 (사용자가 직접 설계)
- **주요 클래스**:
  - `StrategyConfig`: 전략 설정 파라미터
  - `MarketData`: 시장 데이터 구조체
  - `Signal`: 거래 신호 구조체
  - `Position`: 포지션 정보
  - `TradingDecision`: 거래 결정
  - `DecisionEngine`: 의사결정 엔진 (현재 Mock 구현)
- **현재 상태**: 기본 구조만 정의됨, 실제 전략 로직은 미구현
- **의존성**: `pandas`, `datetime`

#### `core/market_watcher.py` 👁️ **시장 감시**
- **역할**: 시장 데이터 모니터링 및 분석
- **현재 상태**: 스켈레톤만 존재
- **의존성**: (미정)

#### `core/signal_engine.py` ⚡ **신호 생성**
- **역할**: 거래 신호 생성 및 필터링
- **현재 상태**: 스켈레톤만 존재
- **의존성**: (미정)

#### `core/trade_executor.py` 💼 **거래 실행**
- **역할**: 실제 주문 실행 및 체결 관리
- **현재 상태**: 스켈레톤만 존재
- **의존성**: (미정)

#### `core/adaptive_memory.py` 🧠 **적응형 메모리**
- **역할**: 학습 및 최적화 (향후 확장)
- **현재 상태**: 스켈레톤만 존재
- **의존성**: (미정)

---

### 3. 실행 계층 (Execution Layer)

#### `api/exchange_api_backtest.py` 🧪 **백테스트용 거래소**
- **역할**: 백테스트 환경에서 가상 거래소 시뮬레이션
- **주요 클래스**: `ExchangeAPIBacktest`
- **기능**:
  - 가상 잔고 관리
  - 가상 주문 실행
  - 포지션 추적
- **의존성**: 없음

#### `api/exchange_api_live.py` 💰 **실거래용 거래소**
- **역할**: 실제 거래소 API 연동 (KIS 등)
- **주요 클래스**: `ExchangeAPILive`
- **기능**:
  - API 인증
  - 실제 주문 실행
  - 잔고/포지션 조회
- **현재 상태**: 스켈레톤만 존재
- **의존성**: (API 키 필요)

#### `backtest/backtest_runner.py` 🏃 **백테스트 실행기**
- **역할**: 백테스트 전체 프로세스 관리
- **주요 기능**:
  - `load_config()`: 백테스트 설정 파일 읽기 (`backtest_config.json`)
  - `run()`: 백테스트 실행 메인 루프
  - `BacktestDataLoader`를 통해 데이터 자동 로드/생성
- **현재 상태**: 기본 구조 구현 완료 (데이터 로드까지)
- **의존성**: `backtest/data_loader.py`

#### `live/live_runner.py` 🚀 **실시간 거래 루프**
- **역할**: 실시간 거래 시스템의 메인 루프
- **주요 클래스**: `LiveRunner`
- **기능**: 실시간 거래 시작/중지
- **의존성**: `live/risk_monitor.py`, `live/event_handler.py`, `core/*`

#### `live/risk_monitor.py` ⚠️ **리스크 모니터링**
- **역할**: 실시간 리스크 감시 및 제한
- **주요 클래스**: `RiskMonitor`
- **기능**:
  - 포지션 리스크 체크
  - 포트폴리오 리스크 체크
  - 최대 낙폭/레버리지 제한
- **의존성**: 없음

#### `live/event_handler.py` 📢 **이벤트 핸들링**
- **역할**: 체결/에러/리밸런스 이벤트 처리
- **주요 클래스**: `EventHandler`
- **기능**:
  - 주문 체결 이벤트
  - 에러 이벤트
  - 리밸런스 이벤트
- **의존성**: 없음

---

### 4. 리포트 계층 (Reporting Layer)

#### `reports/metrics.py` 📊 **성능 지표**
- **역할**: 백테스트/실거래 성능 지표 계산
- **주요 클래스**: `Metrics`
- **기능**:
  - 샤프 비율 계산
  - 최대 낙폭 계산
  - 승률 계산
- **현재 상태**: 스켈레톤만 존재
- **의존성**: 없음

#### `backtest/result_analyzer.py` 📈 **결과 분석**
- **역할**: 백테스트 결과 분석
- **현재 상태**: 스켈레톤만 존재
- **의존성**: `reports/metrics.py`

---

### 5. 진입점 (Entry Points)

#### `cli.py` 🖥️ **CLI 도구**
- **역할**: 데이터 수집 및 관리 명령줄 도구
- **주요 명령어**:
  - `collect`: 데이터 수집
  - `list`: 수집된 데이터 목록
  - `analyze`: 데이터 분석
  - `test`: 파이프라인 테스트
- **의존성**: `data_tools/*`, `backtest/data_loader.py`

#### `main_backtest.py` 🧪 **백테스트 진입점**
- **역할**: 백테스트 실행 진입점
- **의존성**: `backtest/backtest_runner.py`, `core/*`

#### `main_live.py` 💰 **실거래 진입점**
- **역할**: 실거래 실행 진입점
- **의존성**: `live/*`, `core/*`

---

## 파일간 의존성

### 의존성 그래프

```
┌─────────────────────────────────────────────────────────┐
│                    진입점 (Entry Points)                 │
│  main_backtest.py  main_live.py  cli.py                  │
└────────────┬──────────────────┬──────────────────────────┘
             │                  │
             ▼                  ▼
┌──────────────────────┐  ┌──────────────────────┐
│   백테스트 계층       │  │    라이브 계층        │
│  backtest_runner.py   │  │   live_runner.py     │
│  data_loader.py       │  │   risk_monitor.py    │
│  result_analyzer.py   │  │   event_handler.py   │
└──────────┬───────────┘  └──────────┬───────────┘
            │                        │
            └────────────┬───────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    코어 비즈니스 로직                     │
│  strategy_core.py  market_watcher.py  signal_engine.py  │
│  trade_executor.py  adaptive_memory.py                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    API 추상화 계층                       │
│  exchange_api_backtest.py  exchange_api_live.py         │
│  websocket_manager.py                                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    데이터 파이프라인                      │
│  fetch_market_data.py → build_datasets.py                │
│  → verify_integrity.py → data_loader.py                  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    표준 스키마 (Foundation)                │
│                    schema.py ⭐                           │
└─────────────────────────────────────────────────────────┘
```

### 계층별 의존성 규칙

1. **최하위 계층**: `schema.py` - 의존성 없음
2. **데이터 계층**: `schema.py`에만 의존
3. **API 계층**: 독립적 (거래소별 구현)
4. **코어 계층**: API 계층과 데이터 계층 사용
5. **실행 계층**: 코어 계층 사용
6. **진입점**: 실행 계층 사용

---

## 데이터 흐름

### 1. 데이터 수집 파이프라인

```
[Upbit API]
    │
    ▼
[fetch_market_data.py]
    │ (Candle 객체 리스트)
    ▼
[build_datasets.py]
    │ (DataFrame 변환)
    │ (스키마 검증)
    │ (파일 저장: CSV/Parquet)
    ▼
[backtest_data/processed/]
    │
    ▼
[verify_integrity.py]
    │ (데이터 품질 검증)
    ▼
[✅ 검증 완료 데이터]
```

### 2. 백테스트 실행 흐름

```
[main_backtest.py]
    │
    ▼
[backtest_runner.py]
    │
    ├─→ [backtest_config.json] 읽기
    │    (심볼, 간격, 기간 설정)
    │
    ├─→ [data_loader.py]
    │    ├─→ 기존 데이터 파일 찾기
    │    │    └─ 있으면 → 로드 ✅
    │    │
    │    └─ 없으면 → [DatasetBuilder] → 데이터 생성
    │                  └─ 생성 후 로드 ✅
    │
    ├─→ [strategy_core.py] (향후 구현)
    │    (전략 로직 실행)
    │
    ├─→ [exchange_api_backtest.py] (향후 구현)
    │    (가상 주문 실행)
    │
    └─→ [result_analyzer.py] (향후 구현)
         (결과 분석)
```

### 3. 실거래 실행 흐름

```
[main_live.py]
    │
    ▼
[live_runner.py]
    │
    ├─→ [market_watcher.py] → [fetch_market_data.py]
    │                        (실시간 데이터 수집)
    │
    ├─→ [signal_engine.py] → [strategy_core.py]
    │                       (신호 생성)
    │
    ├─→ [risk_monitor.py]
    │    (리스크 체크)
    │
    ├─→ [trade_executor.py] → [exchange_api_live.py]
    │                        (실제 주문 실행)
    │
    └─→ [event_handler.py]
         (이벤트 처리)
```

---

## 실행 모드

### 1. 데이터 수집 모드 (CLI)

```bash
# 데이터 수집
python cli.py collect KRW-BTC 1h 7

# 특정 날짜 범위
python cli.py collect KRW-BTC 1h --start 2025-11-01 --end 2025-11-10

# 데이터 목록 확인
python cli.py list

# 데이터 분석
python cli.py analyze data.csv

# 파이프라인 테스트
python cli.py test
```

### 2. 백테스트 모드

```bash
python main_backtest.py
```

**흐름**:
1. `BacktestRunner` 초기화
2. `backtest_config.json` 설정 파일 읽기
3. `BacktestDataLoader.load_data_for_backtest()`로 데이터 로드
   - 기존 데이터 있으면 → 바로 로드
   - 없으면 → 자동 생성 후 로드
4. `DecisionEngine`으로 전략 실행 (향후 구현)
5. `ExchangeAPIBacktest`로 가상 거래 (향후 구현)
6. `ResultAnalyzer`로 결과 분석 (향후 구현)

### 3. 실거래 모드

```bash
python main_live.py
```

**흐름**:
1. `LiveRunner` 초기화
2. `MarketWatcher`로 실시간 데이터 수집
3. `SignalEngine`으로 신호 생성
4. `RiskMonitor`로 리스크 체크
5. `TradeExecutor`로 실제 주문 실행
6. `EventHandler`로 이벤트 처리

---

## 현재 구현 상태

### ✅ 완성된 모듈
- **데이터 파이프라인**: 완전 구현
  - `schema.py`: 완성
  - `fetch_market_data.py`: 완성
  - `build_datasets.py`: 완성
  - `verify_integrity.py`: 완성
  - `data_loader.py`: 완성 (자동 데이터 생성 포함)
- **CLI 도구**: 완성
- **백테스트 러너**: 기본 구조 완성
  - `backtest_runner.py`: 설정 읽기, 데이터 로드까지 완성
  - `backtest_config.json`: 백테스트 설정 파일

### 🚧 부분 구현
- **코어 모듈**: 스켈레톤만 존재
  - `strategy_core.py`: Mock 구현만 존재
  - `market_watcher.py`: 스켈레톤
  - `signal_engine.py`: 스켈레톤
  - `trade_executor.py`: 스켈레톤
- **백테스트 실행기**: 데이터 로드까지 완성, 전략 실행은 미구현
- **라이브 실행기**: 기본 구조만

### ❌ 미구현
- 실제 전략 로직 (사용자 설계 필요)
- 실거래 API 연동 (KIS 등)
- WebSocket 실시간 데이터
- 리포트 및 시각화
- 적응형 메모리/학습 기능

---

## 향후 개발 우선순위

1. **전략 로직 구현** (`core/strategy_core.py`)
   - 설계 문서의 Trend/Range 전략 구현
   - MarketAnalyzer 구현
   - StrategyManager 구현

2. **백테스트 엔진 완성** (`backtest/backtest_runner.py`)
   - 전략 실행 루프
   - 포지션 관리
   - 성능 지표 계산

3. **실거래 시스템 완성** (`live/*`)
   - 실시간 데이터 수집
   - 주문 실행 및 체결 관리
   - 리스크 모니터링 강화

4. **리포트 및 분석** (`reports/*`)
   - 성능 지표 계산
   - 시각화
   - 거래 리포트

---

## 주요 설계 특징

### 1. 표준 스키마 우선 설계
- 모든 거래소 데이터를 `Candle` 스키마로 통일
- 입구(`data_tools`)에서만 변환 수행
- 내부 로직은 표준 스키마만 사용

### 2. 백테스트/라이브 분리
- 동일한 전략 로직 재사용
- API 추상화로 환경 분리
- 테스트 가능한 구조

### 3. 데이터 품질 보장
- 3단계 검증: 스키마 → 무결성 → 로딩
- 메타데이터 자동 생성
- 데이터 품질 리포트

### 4. 확장 가능한 구조
- 멀티 거래소 지원 준비
- 전략 모듈화
- 리포트 시스템 분리

---

## 기술 스택

- **언어**: Python 3.x
- **주요 라이브러리**:
  - `pandas`: 데이터 처리
  - `numpy`: 수치 계산
  - `pyupbit`: Upbit API
  - `requests`: HTTP 요청
- **데이터 포맷**: CSV, Parquet, Pickle
- **설정**: `config.py`, `.env` (환경 변수)

---

## 참고 문서

- `design document`: 상세 설계 문서
- `README.md`: 프로젝트 개요
- `docs/CLI_USAGE.md`: CLI 사용법
- `COPILOT_COMMANDS.md`: 개발 명령어 모음

