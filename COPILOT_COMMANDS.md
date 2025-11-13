
---

## 0. backtestdata 폴더의 역할 한 줄 요약

> **`backtestdata/` = “과거 시세를 기준으로 전략을 시뮬레이션하기 위한, 고정된 입력·출력 데이터 저장소”**

* `data/` 쪽이 “수집·정제용”이라면
* `backtestdata/`는 **“실제 전략 돌려보는 실험실”**에 가깝다고 보면 됨.

---

## 1. 전체 구조 재정리

```bash
backtestdata/
├── market_info/      # 심볼·수수료·단위 등 거래소/마켓 메타데이터
├── candles/          # 캔들(OHLCV) 시계열, 백테스트용 “진짜 가격”
├── features/         # 지표/파생 피처 (RSI, EMA 등)
├── signals/          # 전략이 만든 매수/매도 신호
├── positions/        # 시뮬레이션 중 포지션/잔고 타임라인
├── results/          # 전략별 성능 요약, 에쿼티커브 등ㄴ
└── config/           # 백테스트 환경/전략 파라미터 설정
```

아래부터는 **폴더별로:**

1. 어떤 파일이 있고
2. 각 파일에 **어떤 컬럼/정보가 들어있고**
3. 백테스트 코드에서 그걸 **어떻게 사용하는지**까지 적어볼게.

---

## 2. `market_info/` — 마켓/심볼 정보

### 2-1. upbit_markets.json

* **예상 경로**

  ```bash
  backtestdata/market_info/upbit_markets.json
  ```

* **예상 구조 (예시)**

  ```json
  [
    {
      "market": "KRW-BTC",
      "base": "BTC",
      "quote": "KRW",
      "min_order_value": 5000,
      "price_tick": 1000,
      "size_step": 0.0001,
      "taker_fee": 0.0005,
      "maker_fee": 0.0005,
      "active": true
    }
  ]
  ```

* **들어있는 정보**

  * 어떤 심볼이 있는지 (`KRW-BTC`, `KRW-ETH` 등)
  * 최소 주문 금액 (`min_order_value`)
  * 가격 단위(틱 사이즈), 수량 단위 (`price_tick`, `size_step`)
  * 매수/매도 수수료 (`taker_fee`, `maker_fee`)
  * 현재 거래 가능한지 (`active`)

* **언제/어떻게 생성되나**

  * Upbit `/market/all`, `/candles` 등 메타 데이터 한번 긁어와서,
  * 정리해서 한 번 만들어 두고, **자주 안 바뀜** (가끔만 업데이트).

* **백테스트에서 어떻게 쓰냐**

  * **주문 가능 여부** 확인 (`active`가 false면 거래 무시)
  * **최소 주문 금액/수량 체크**

    * 예: 3000원짜리 주문은 `min_order_value = 5000`이면 invalid → 주문 스킵
  * **틱 단위 맞추기**

    * 예: 전략이 94,550,123원 이런 가격을 내면 → `price_tick=1000` 맞춰 94,551,000 → 94,551,000은 안맞으니 94,551,000을 94,555,000 or 94,551,000이 아니라 94,555,000 이런 식으로 조정
  * **수수료 계산**

    * 체결 금액 × `taker_fee` 로 수수료 차감

---

## 3. `candles/` — 백테스트용 캔들 데이터

### 3-1. 파일 구조

```bash
backtestdata/candles/
├── BTC_KRW_1m.parquet
├── BTC_KRW_1h.parquet
├── ETH_KRW_1m.parquet
└── ...
```

* 파일명 컨벤션(예시)

  * `{SYMBOL}_{TIMEFRAME}.parquet`
  * ex) `BTC_KRW_1m.parquet`, `BTC_KRW_1h.parquet`

### 3-2. 컬럼 구조 (예시)

```text
timestamp        # UTC 또는 KST 기준, 백테스트 기준 시간
open             # 시가
high             # 고가
low              # 저가
close            # 종가
volume           # 거래량 (base 자산, 예: BTC 수량)
quote_volume     # 거래대금 (quote 자산, 예: KRW)
trade_count      # (있다면) 해당 봉 동안 거래 횟수
```

* **특징**

  * 우리가 `Candle` 표준 스키마로 맞춰 놓은 형태 그대로.
  * **backtestdata의 candles는 "동결된 스냅샷"**
    → 계속 덮어쓰기보다는, 특정 날짜 기준으로 고정시켜두는 게 재현성에 좋음.

### 3-3. 사용 방식

* **전략 로직의 기본 입력**

  * 예: `rsi_strategy.py` 안에서:

    ```python
    candles = load_candles("BTC_KRW", "1h")
    # candles: DataFrame[timestamp, open, high, low, close, volume, ...]
    ```

* **features, signals가 이걸 기준으로 align**

  * features, signals 파일들도 `timestamp` 기준으로 **candles와 join/merge**함.

* **슬리피지, 체결 가격의 기준**

  * 매수/매도 체결 가격을 `open` 또는 `close` 등에 붙여서 시뮬레이션.

---

## 4. `features/` — 기술지표·파생 피처

### 4-1. 파일 구조

```bash
backtestdata/features/
├── BTC_KRW_1h_features.parquet
├── ETH_KRW_1h_features.parquet
└── ...
```

* 파일명 예시

  * `{SYMBOL}_{TIMEFRAME}_features.parquet`

### 4-2. 컬럼 구조 (예시)

```text
timestamp        # candles와 동일한 기준
rsi_14           # 14-period RSI
ema_12           # 12-period EMA
ema_26           # 26-period EMA
macd             # macd (ema_12 - ema_26)
signal_line      # macd 시그널
bb_upper         # 볼린저 상단
bb_middle        # 볼린저 중단
bb_lower         # 볼린저 하단
...
```

* **어떤 정보?**

  * 전략에서 반복 계산하면 느린 지표들을 **미리 계산해 둔 결과**.
  * candle 기반으로 계산하지만 **candle과 분리 저장**해서 재사용 가능.

* **언제 생성되나**

  * 별도의 피처 생성 스크립트 예:

    ```bash
    python build_features.py --symbol BTC_KRW --timeframe 1h
    ```

* **백테스트 사용**

  * 전략 코드 안에서:

    ```python
    candles = load_candles("BTC_KRW", "1h")
    feats   = load_features("BTC_KRW", "1h")

    df = candles.merge(feats, on="timestamp", how="left")
    # df에 rsi_14, macd 등 포함된 상태로 시그널 계산
    ```

  * → 이렇게 하면 전략은 “지표 계산 로직” 말고, **“언제 매수/매도 할지 조건만”** 집중해서 짜면 됨.

---

## 5. `signals/` — 전략이 만든 매수/매도 신호

### 5-1. 파일 구조

```bash
backtestdata/signals/
├── rsi_strategy_BTC_KRW_1h.csv
├── breakout_strategy_BTC_KRW_1h.csv
└── ...
```

* 파일명 컨벤션

  * `{strategy_name}_{SYMBOL}_{TIMEFRAME}.csv`

### 5-2. 컬럼 구조 (예시)

```text
timestamp          # 신호가 발생한 시점
signal             # -1(매도), 0(홀드), 1(매수) 같은 정수 코드
target_position    # 목표 포지션 비율 (예: 0.0 ~ 1.0)
confidence         # (옵션) 신호의 확신도 0.0 ~ 1.0
meta               # (옵션) 디버깅용 추가 정보 (JSON string 등)
```

* **어떤 정보?**

  * “이 시점에, 이 전략이 어떤 액션을 취하라고 말했는지”
  * 즉, 전략 로직의 **의사결정 기록**.

* **언제 생성되나**

  * 전략 러너에서:

    ```python
    df = join(candles, features)
    df["signal"] = compute_rsi_signal(df)
    save_signals(df[["timestamp", "signal"]], strategy_name, symbol, timeframe)
    ```

* **백테스트 엔진에서 어떻게 쓰냐**

  * 포지션 시뮬레이터가 `signals`를 읽고,

  * 각 `timestamp`에서:

    1. 지금 signal이 1이면 → **매수 주문 생성(or 목표 포지션까지 매수)**
    2. -1이면 → **청산 or 숏 포지션 진입**
    3. 0이면 → **유지**

  * 이때 **시장 제약(수수료, 최소 주문금액, 슬리피지, 체결 방향)**은
    `config/`와 `market_info/`를 참고해서 적용.

---

## 6. `positions/` — 시뮬레이션 중 포지션/잔고 로그

### 6-1. 파일 구조

```bash
backtestdata/positions/
├── rsi_strategy_BTC_KRW_1h_positions.csv
├── rsi_strategy_portfolio_1h_positions.csv
└── ...
```

* 심볼별 포지션 로그 / 포트폴리오 전체 로그 등으로 나눌 수 있음.

### 6-2. 컬럼 구조 (예시: 심볼 단위)

```text
timestamp          # 포지션 스냅샷 시점
symbol             # 예: "KRW-BTC"
position_size      # 보유 수량 (BTC)
avg_entry_price    # 평균 진입 단가
unrealized_pnl     # 미실현 손익
realized_pnl       # 실현 손익 누계
cash_after_trade   # 해당 시점 이후 남은 현금
equity             # 총 자산 (현금 + 보유 자산 시가 - 수수료)
drawdown           # 고점 대비 낙폭
```

* **어떤 정보?**

  * “전략이 이 타임라인 동안 실제로 어떻게 포지션을 들고 있었는지”
  * 매 타임스텝별 **상태(State)**를 다 기록한 것.

* **언제 생성되나**

  * signals + candles + config를 가지고 포지션 시뮬레이션할 때 내부에서:

    ```python
    for t in timeline:
        # signal 읽고 주문 생성 → 체결 → 포지션/현금 업데이트
        log_position(t, position_size, avg_entry_price, pnl, cash, equity)
    ```

* **어떻게 쓰냐**

  * 사후 분석:

    * “어느 구간에서 계단식으로 물렸는지”
    * “MDD가 언제 발생했는지”
    * “레버리지 비율이 어느 시점에 과도했는지”
  * 시각화:

    * 에쿼티 커브를 그릴 때 equity 컬럼을 시계열 그래프로.

---

## 7. `results/` — 전략 성능 요약 & 에쿼티 커브

### 7-1. 파일 구조

```bash
backtestdata/results/
├── rsi_strategy_BTC_KRW_1h_summary.json
├── rsi_strategy_BTC_KRW_1h_equity.parquet
├── rsi_strategy_BTC_KRW_1h_trades.csv   # (선택: 트레이드 단위 로그)
└── ...
```

### 7-2. summary.json 예시

```json
{
  "strategy": "rsi_strategy",
  "symbol": "KRW-BTC",
  "timeframe": "1h",
  "start_date": "2021-01-01",
  "end_date": "2024-01-01",
  "initial_balance": 10000000,
  "final_balance": 18400000,
  "total_return": 0.84,
  "max_drawdown": 0.23,
  "sharpe_ratio": 1.92,
  "win_rate": 0.61,
  "trade_count": 132,
  "avg_hold_time_hours": 37.5
}
```

* **어떤 정보?**

  * 전략의 **성적표**.
  * 수익률, MDD, 샤프, 승률, 트레이드 수, 평균 보유 기간 등.

* **언제 생성되나**

  * 포지션 로그를 모두 만든 뒤,
  * 별도 evaluator가 지표를 계산해서 `summary.json`으로 저장.

* **사용처**

  * 여러 전략/파라미터 조합을 돌려놓고,
    `results/` 폴더만 봐도 “어떤 셋이 성능이 좋은지” 비교 가능.
  * 나중에 dashboard나 notebook에서 summary 파일들만 모아와서 테이블/그래프 만들기.

### 7-3. equity.parquet 예시

* 컬럼:

  ```text
  timestamp   # 시각
  equity      # 총자산
  drawdown    # 낙폭
  ```
* 에쿼티 커브 그리는 용도.

---

## 8. `config/` — 백테스트 환경 & 전략 파라미터

### 8-1. backtest_config.yaml (예시)

```yaml
start_date: "2021-01-01"
end_date: "2024-01-01"
initial_balance: 10000000
base_currency: "KRW"

timeframe: "1h"
symbols:
  - "KRW-BTC"
  - "KRW-ETH"

commission: 0.0005       # 수수료 비율
slippage_bps: 5          # 슬리피지 (basis points)
max_leverage: 1.0
position_sizing: "fixed_fraction"
risk_per_trade: 0.01     # 자산의 1%씩 베팅
strategy: "rsi_strategy"
strategy_params:
  rsi_period: 14
  oversold: 30
  overbought: 70
```

* **어떤 정보?**

  * 이 백테스트 run이 **어떤 조건으로 돌려졌는지** 기록.
* **사용처**

  * 재현성:

    * “그때 그 좋은 성과 나왔던 설정이 뭐였지?” → 이 yaml 그대로 다시 로드해서 동일 환경 재실행.
  * parameter sweep 시:

    * 여러 `strategy_params` 조합으로 파일 여러 개 생성.

---

## 9. 전체 플로우 한 번에 정리

```text
(1) config/backtest_config.yaml 읽기
      ↓
(2) market_info/upbit_markets.json 로드
      ↓
(3) candles/{SYMBOL}_{TIMEFRAME}.parquet 로드
      ↓
(4) features/{SYMBOL}_{TIMEFRAME}_features.parquet 로드
      ↓
(5) 전략 로직: candles+features → signals/{strategy}_{SYMBOL}.csv 생성
      ↓
(6) 백테스트 엔진:
        candles + signals + config + market_info
        → 포지션/잔고 시뮬레이션
        → positions/{strategy}_...csv 생성
      ↓
(7) evaluator:
        positions → results/{strategy}_summary.json, equity.parquet 생성
```

이렇게 보면,

* `candles`, `features` = **입력 데이터 레이어**
* `signals` = **전략 결정 레이어**
* `positions` = **실행/시뮬레이션 레이어**
* `results` = **평가 레이어**
* `market_info`, `config` = **환경 메타데이터**

라고 정리할 수 있어.

---

