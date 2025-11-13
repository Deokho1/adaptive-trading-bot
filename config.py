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
    }
}