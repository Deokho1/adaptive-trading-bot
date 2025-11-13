# 설정 파일
CONFIG = {
    "api": {
        "kis": {
            "app_key": "",
            "app_secret": "",
            "access_token": ""
        }
    },
    "trading": {
        "risk_level": 0.02,
        "max_position_size": 0.1
    },
    "backtest": {
        "initial_capital": 10000000,  # 초기 자본 1천만원
        "fee_rate": 0.0005,  # 업비트 수수료율 0.05%
        "slippage_rate": 0.0003  # 비트코인 단타용 슬리피지율 0.03%
    },
    "strategy": {
        # RSI 파라미터
        "rsi_period": 14,  # RSI 계산 기간
        "rsi_oversold": 25.0,  # 과매도 기준 (진입) - RSI가 이 값 이하일 때 진입
        "rsi_exit": 55.0,  # 청산 기준 (중립선 회복) - RSI가 이 값 이상일 때 청산
        
        # 진입 조건
        "entry_volume_ratio": 1.5,  # 거래량 필터 (평균의 N배 이상, 0이면 필터 비활성화)
        "entry_position_size_ratio": 0.1,  # 진입 금액 비율 (자본의 N%)
        
        # 청산 조건
        "take_profit_pct": 0.5,  # 익절: +N%
        "stop_loss_pct": -0.15,  # 손절: -N%
        "max_hold_time_minutes": 5  # 최대 보유 시간 (분)
    }
}