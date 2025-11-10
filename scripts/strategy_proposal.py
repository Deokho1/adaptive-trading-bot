"""
현실적인 단타 전략 설정 제안

데이터 분석과 수수료 분석을 바탕으로 한 최적화된 파라미터
"""

def propose_strategy_settings():
    """전략 설정 제안"""
    
    print("🎯 현실적인 단타 전략 설정 제안")
    print("="*60)
    
    print("📊 분석 결과 요약:")
    print("  • 업비트 총 거래비용: ~0.12%")
    print("  • 데이터 분석: 평균 변동성 0.21%, ±0.8% 스파이크 빈도 적당")
    print("  • SOL이 가장 활발 (변동성 0.27%), BTC가 가장 안정 (0.14%)")
    
    # 3가지 시나리오 제안
    scenarios = {
        "보수적": {
            "name": "보수적 (안정 추구)",
            "target_profit": 0.25,
            "stop_loss": 0.15,
            "spike_threshold": 0.6,
            "description": "낮은 리스크, 높은 승률 목표"
        },
        "균형": {
            "name": "균형형 (권장)",
            "target_profit": 0.35,
            "stop_loss": 0.20,
            "spike_threshold": 0.4,
            "description": "적당한 리스크, 적당한 수익"
        },
        "공격적": {
            "name": "공격적 (고수익 추구)",
            "target_profit": 0.50,
            "stop_loss": 0.25,
            "spike_threshold": 0.3,
            "description": "높은 리스크, 높은 수익 목표"
        }
    }
    
    print("\n🔧 3가지 시나리오 제안:")
    print("-" * 60)
    
    for key, scenario in scenarios.items():
        print(f"\n{scenario['name']}:")
        print(f"  • 목표수익: {scenario['target_profit']}%")
        print(f"  • 손절매: {scenario['stop_loss']}%") 
        print(f"  • 스파이크 임계값: ±{scenario['spike_threshold']}%")
        print(f"  • 특징: {scenario['description']}")
        
        # 수익성 계산
        net_profit = scenario['target_profit'] - 0.12  # 수수료 차감
        risk_reward = scenario['target_profit'] / scenario['stop_loss']
        
        print(f"  • 순수익: {net_profit:.2f}%")
        print(f"  • 손익비: {risk_reward:.1f}:1")

    return scenarios


def calculate_expected_performance(scenarios):
    """각 시나리오별 예상 성과"""
    
    print(f"\n📈 시나리오별 예상 성과 (1000만원 기준)")
    print("="*60)
    
    # 스파이크 빈도 데이터 (실제 분석 결과)
    spike_frequencies = {
        0.3: 0.4,   # 0.3% 스파이크: 하루 0.4회
        0.4: 0.25,  # 0.4% 스파이크: 하루 0.25회  
        0.6: 0.1    # 0.6% 스파이크: 하루 0.1회
    }
    
    capital = 1000  # 1000만원
    
    for name, scenario in scenarios.items():
        print(f"\n{scenario['name']}:")
        
        # 예상 거래 빈도
        threshold = scenario['spike_threshold']
        daily_trades = spike_frequencies.get(threshold, 0.2)
        
        # 승률 추정 (보수적일수록 높음)
        if threshold >= 0.6:
            win_rate = 0.70
        elif threshold >= 0.4:
            win_rate = 0.60
        else:
            win_rate = 0.55
        
        # 1회당 기대수익
        win_profit = scenario['target_profit'] - 0.12
        loss_amount = scenario['stop_loss'] + 0.12
        
        expected_return_per_trade = (win_rate * win_profit) - ((1-win_rate) * loss_amount)
        
        # 일/월 수익
        daily_profit = daily_trades * expected_return_per_trade * capital / 100
        monthly_profit = daily_profit * 22  # 22 거래일
        
        print(f"  • 예상 거래빈도: 일 {daily_trades:.1f}회")
        print(f"  • 예상 승률: {win_rate*100:.0f}%")
        print(f"  • 1회당 기대수익: {expected_return_per_trade:.3f}%")
        print(f"  • 일 수익: {daily_profit/10:.0f}만원")
        print(f"  • 월 수익: {monthly_profit/10:.0f}만원")
        print(f"  • 월 수익률: {monthly_profit/capital:.1f}%")


def recommend_final_setting():
    """최종 권장 설정"""
    
    print(f"\n🚀 최종 권장 설정 (균형형 기반)")
    print("="*60)
    
    settings = {
        "take_profit_pct": 0.35,
        "stop_loss_pct": 0.20,
        "spike_down_threshold_5m": -0.4,
        "spike_up_threshold_5m": 0.4,
        "spike_down_threshold_15m": -1.0,
        "spike_up_threshold_15m": 1.0,
        "min_volume_spike_ratio": 1.5,
        "per_trade_risk_pct": 3.0,
        "max_total_exposure_pct": 20.0,
        "fee_rate": 0.0005,
        "slippage_rate": 0.0007
    }
    
    print("📋 config.py 수정용 코드:")
    print("```python")
    for key, value in settings.items():
        print(f"{key} = {value}")
    print("```")
    
    print(f"\n💡 선택 이유:")
    print("  • 0.35% 목표: 수수료 3배로 안전한 마진")
    print("  • 0.20% 손절: 수수료 1.7배로 타이트한 리스크 관리")
    print("  • 0.4% 스파이크: 적당한 빈도와 수익성의 균형")
    print("  • 볼륨 1.5배: 가짜 신호 필터링")
    print("  • 3% 리스크: 보수적 포지션 사이징")
    
    print(f"\n⚠️  주의사항:")
    print("  • 승률 60% 이상 유지 필수")
    print("  • 하루 5-10회 거래 목표")
    print("  • 손절 타이밍 중요 (감정 개입 금지)")
    print("  • 연속 손실시 일시 중단")
    
    return settings


def main():
    """메인 함수"""
    
    # 1. 시나리오 제안
    scenarios = propose_strategy_settings()
    
    # 2. 성과 예상
    calculate_expected_performance(scenarios)
    
    # 3. 최종 권장 설정
    final_settings = recommend_final_setting()
    
    print(f"\n🎯 다음 단계:")
    print("="*60)
    print("1. 위 설정으로 백테스트 실행")
    print("2. 결과 확인 후 파라미터 미세조정")
    print("3. 여러 기간/심볼로 검증")
    print("4. 실제 소액으로 테스트")
    print("5. 안정성 확인 후 본격 운용")


if __name__ == "__main__":
    main()