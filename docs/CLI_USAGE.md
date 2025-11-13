# 📋 CLI 사용법 가이드

## 🎯 기본 사용법

```bash
python cli.py <명령어> [옵션들...]
```

## 📊 데이터 수집 (collect)

### 기본 문법
```bash
python cli.py collect <심볼> <간격> <일수>
```

### 예시들
```bash
# 비트코인 1시간 캔들 3일치
python cli.py collect KRW-BTC 1h 3

# 이더리움 일봉 7일치  
python cli.py collect KRW-ETH 1d 7

# 에이다 5분봉 1일치
python cli.py collect KRW-ADA 5m 1

# 도지코인 30분봉 2일치
python cli.py collect KRW-DOGE 30m 2
```

### 🔍 지원하는 심볼들
- **주요 코인**: `KRW-BTC`, `KRW-ETH`, `KRW-ADA`, `KRW-DOT`
- **인기 알트**: `KRW-DOGE`, `KRW-SHIB`, `KRW-AVAX`, `KRW-ATOM`
- **전체 목록**: [업비트 마켓](https://api.upbit.com/v1/market/all) 참고

### ⏰ 지원하는 간격들
- **분봉**: `1m`, `3m`, `5m`, `10m`, `15m`, `30m`
- **시간봉**: `1h`, `4h` 
- **일봉**: `1d`
- **주봉**: `1w`
- **월봉**: `1M`

---

## 📋 데이터 목록 보기 (list)

```bash
python cli.py list
```

**출력 예시:**
```
📊 Available backtest data:

CSV files:
  - krw_btc_1h_20251109_20251111_processed.csv
  - krw_eth_1d_20251104_20251111_processed.csv

No parquet files found.
```

---

## 📈 데이터 분석 (analyze)

### 기본 문법
```bash
python cli.py analyze <파일명>
```

### 예시들
```bash
# CSV 파일 분석
python cli.py analyze krw_btc_1h_20251109_20251111_processed.csv

# 다른 파일 분석  
python cli.py analyze krw_eth_1d_20251104_20251111_processed.csv
```

### 📊 분석 결과 예시
```
📈 Analyzing krw_btc_1h_20251109_20251111_processed.csv...

📊 Basic Info:
  • Size: (48, 10)
  • Period: 2025-11-09 05:00:00+00:00 ~ 2025-11-11 04:00:00+00:00
  • Symbol: BTC-KRW

💰 Price Info:
  • Start: 151,784,000원
  • End: 157,344,000원
  • High: 159,150,000원
  • Low: 151,283,000원
  • Change: +3.79%

📊 Volume:
  • Total: 3297.56 KRW
  • Average: 68.70
```

---

## 🧪 시스템 테스트 (test)

```bash
python cli.py test
```

**기능:**
- API 연결 상태 확인
- 데이터 파일 무결성 체크
- 전체 파이프라인 동작 검증

---

## 🚀 고급 사용법

### 📁 배치 수집 (PowerShell)
```bash
# 여러 코인 한번에
@("KRW-BTC", "KRW-ETH", "KRW-ADA") | ForEach-Object { 
    python cli.py collect $_ 1h 3 
}

# 여러 간격 한번에
@("1h", "4h", "1d") | ForEach-Object { 
    python cli.py collect KRW-BTC $_ 2 
}
```

### ⏰ 자동화 스크립트 예시
```bash
# daily_collect.ps1
python cli.py collect KRW-BTC 1d 1
python cli.py collect KRW-ETH 1d 1
python cli.py collect KRW-ADA 1d 1
python cli.py test
```

### 🔍 파일 관리
```bash
# 특정 날짜 데이터 찾기
Get-ChildItem -Path "backtest_data\processed" -Filter "*20251111*"

# 특정 코인 데이터 찾기  
Get-ChildItem -Path "backtest_data\processed" -Filter "*btc*"

# 오래된 파일 정리 (30일 이상)
Get-ChildItem -Path "backtest_data\processed" | Where-Object {$_.CreationTime -lt (Get-Date).AddDays(-30)}
```

---

## 🐍 Python 코드 실행 방식 (고급 사용자용)

CLI 외에 Python 코드로 직접 실행도 가능합니다:

```python
# 빠른 데이터 수집
from data_tools.build_datasets import quick_build_upbit_dataset
quick_build_upbit_dataset(['KRW-BTC'], ['1h'], days_back=3)

# 세부 옵션 조절
from data_tools.build_datasets import DatasetBuilder
builder = DatasetBuilder()
result = builder.build_single_dataset(
    symbol="KRW-BTC",
    interval="1h", 
    days_back=7,
    save_formats=["csv", "parquet"]
)

# 데이터 직접 분석
from backtest.data_loader import BacktestDataLoader
loader = BacktestDataLoader()
df = loader.load_candles_from_file("krw_btc_1h_processed.csv")
print(df.describe())
```

---

## 📂 파일 구조

```
adaptive-trading-bot/
├── cli.py                    # 🎯 메인 CLI 스크립트
├── COPILOT_COMMANDS.md      # 🤖 코파일럿용 명령어
├── data_tools/
│   ├── CLI_USAGE.md         # 📋 이 파일 (CLI 사용법)
│   ├── schema.py            # 📊 데이터 스키마
│   ├── fetch_market_data.py # 🌐 API 수집
│   ├── build_datasets.py    # 🏗️ 데이터셋 구축
│   └── verify_integrity.py  # ✅ 데이터 검증
├── backtest_data/
│   ├── processed/           # 🔄 가공된 데이터 (CSV)
│   ├── raw/                # 📥 원본 데이터  
│   └── metadata/           # 📊 수집 로그
└── backtest/
    └── data_loader.py       # 📖 데이터 로더
```

---

## ❓ 도움말

```bash
# 전체 도움말
python cli.py --help

# 특정 명령어 도움말
python cli.py collect --help
python cli.py analyze --help
```

---

## 🚨 문제해결

### ❌ 자주 발생하는 오류들

1. **"pytz is not defined"**
   ```bash
   pip install pytz
   ```

2. **"No module named 'requests'"**  
   ```bash
   pip install requests pandas
   ```

3. **"파일을 찾을 수 없습니다"**
   - `python cli.py list` 먼저 실행
   - 정확한 파일명 확인

4. **"API 오류"**
   - 인터넷 연결 확인
   - `python cli.py test` 실행

### 💡 팁들

- **빠른 확인**: `python cli.py list` 먼저 실행
- **분석 전**: `python cli.py test`로 상태 체크  
- **배치 작업**: PowerShell 스크립트 활용
- **디버깅**: Python 코드 방식 사용

---

## 🔮 향후 추가될 기능들

- [ ] `python cli.py backtest <strategy>` - 백테스팅 실행
- [ ] `python cli.py strategy <name>` - 전략 생성/수정
- [ ] `python cli.py live <mode>` - 실시간 트레이딩
- [ ] `python cli.py report <period>` - 성과 리포트
- [ ] `python cli.py alert <condition>` - 알림 설정

---

**📝 마지막 업데이트**: 2025-11-11  
**🔧 CLI 버전**: 1.0.0  
**📍 파일 위치**: `data_tools/CLI_USAGE.md`