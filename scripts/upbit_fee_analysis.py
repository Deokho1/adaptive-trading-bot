"""
업비트 수수료 기반 최소 수익률 계산기

업비트 거래 수수료를 고려한 최소 목표 수익률을 분석합니다.
"""

def calculate_upbit_fees():
    """업비트 수수료 체계 분석"""
    
    print("💰 업비트 거래 수수료 분석")
    print("="*50)
    
    # 업비트 수수료 체계 (2024년 기준)
    maker_fee = 0.0005  # 0.05% (지정가 주문)
    taker_fee = 0.0005  # 0.05% (시장가 주문)
    
    print(f"📊 업비트 수수료:")
    print(f"  • Maker (지정가): {maker_fee*100:.2f}%")
    print(f"  • Taker (시장가): {taker_fee*100:.2f}%")
    
    # 단타봇은 보통 시장가 사용 (빠른 진입/청산)
    buy_fee = taker_fee   # 매수 수수료
    sell_fee = taker_fee  # 매도 수수료
    
    total_fee_rate = buy_fee + sell_fee
    print(f"  • 총 수수료 (왕복): {total_fee_rate*100:.3f}%")
    
    # 슬리피지 추가 고려
    slippage_rate = 0.0002  # 0.02% (보수적 추정)
    total_cost_rate = total_fee_rate + slippage_rate
    
    print(f"  • 슬리피지 추정: {slippage_rate*100:.2f}%")
    print(f"  • 총 거래 비용: {total_cost_rate*100:.3f}%")
    
    return total_cost_rate


def calculate_minimum_profit_targets(total_cost_rate):
    """최소 목표 수익률 계산"""
    
    print(f"\n🎯 최소 목표 수익률 계산")
    print("="*50)
    
    # 손익분기점 + 안전 마진
    breakeven = total_cost_rate
    
    safety_margins = [1.5, 2.0, 2.5, 3.0]
    
    print(f"💀 손익분기점: {breakeven*100:.3f}%")
    print(f"\n📈 권장 목표 수익률:")
    
    recommended_targets = []
    
    for margin in safety_margins:
        target = breakeven * margin
        net_profit = target - total_cost_rate
        
        print(f"  • {margin:.1f}x 안전마진: {target*100:.2f}% (순이익: {net_profit*100:.2f}%)")
        recommended_targets.append(target)
    
    return recommended_targets


def analyze_win_rate_scenarios(total_cost_rate, targets):
    """승률 시나리오별 기대수익 분석"""
    
    print(f"\n🎲 승률별 기대수익 분석")
    print("="*50)
    
    win_rates = [40, 50, 60, 70, 80]  # 승률 %
    
    print(f"{'승률':>4} | {'목표 수익률':>12} | {'기대수익률':>12} | {'평가':>8}")
    print("-" * 50)
    
    for win_rate in win_rates:
        win_rate_decimal = win_rate / 100
        
        for i, target in enumerate(targets):
            # 승리시: +target, 패배시: -total_cost_rate (손절)
            expected_return = (win_rate_decimal * target) - ((1 - win_rate_decimal) * total_cost_rate)
            
            if expected_return > 0:
                evaluation = "🟢 수익"
            elif expected_return > -0.001:
                evaluation = "🟡 균형"
            else:
                evaluation = "🔴 손실"
            
            margin = [1.5, 2.0, 2.5, 3.0][i]
            
            print(f"{win_rate:>3}% | {target*100:>9.2f}% | {expected_return*100:>9.2f}% | {evaluation}")


def calculate_optimal_strategy_params(total_cost_rate):
    """최적 전략 파라미터 제안"""
    
    print(f"\n🔧 최적 전략 파라미터 제안")
    print("="*50)
    
    # 수수료 기반 최소 수익률
    min_profitable = total_cost_rate * 2  # 2배 안전마진
    
    # 업비트 특성 고려
    print(f"📋 업비트 최적화 설정:")
    print(f"  • 최소 목표 수익률: {min_profitable*100:.2f}%")
    print(f"  • 권장 take_profit: {min_profitable*100*1.2:.2f}% (여유 20%)")
    
    # 손절매 설정
    max_loss = total_cost_rate * 1.5  # 수수료 1.5배까지만 손실 허용
    print(f"  • 권장 stop_loss: {max_loss*100:.2f}%")
    
    # 스파이크 임계값 (진입 조건)
    # 목표 수익률보다 큰 움직임을 잡아야 함
    min_spike = min_profitable * 1.5
    print(f"  • 최소 스파이크 크기: {min_spike*100:.2f}%")
    print(f"  • 권장 spike_threshold: {min_spike*100:.1f}%")
    
    return {
        'take_profit_pct': min_profitable * 100 * 1.2,
        'stop_loss_pct': max_loss * 100,
        'spike_threshold': min_spike * 100
    }


def main():
    """메인 분석 함수"""
    
    print("🏦 업비트 수수료 기반 전략 최적화")
    print("="*60)
    
    # 1. 수수료 계산
    total_cost_rate = calculate_upbit_fees()
    
    # 2. 최소 목표 수익률
    targets = calculate_minimum_profit_targets(total_cost_rate)
    
    # 3. 승률 시나리오 분석
    analyze_win_rate_scenarios(total_cost_rate, targets)
    
    # 4. 최적 파라미터 제안
    optimal_params = calculate_optimal_strategy_params(total_cost_rate)
    
    # 5. 최종 권장사항
    print(f"\n🚀 최종 권장사항")
    print("="*50)
    
    print(f"💡 핵심 포인트:")
    print(f"  • 업비트 총 거래비용: ~{total_cost_rate*100:.2f}%")
    print(f"  • 최소 {optimal_params['take_profit_pct']:.1f}% 이상 수익을 노려야 함")
    print(f"  • 손절매는 {optimal_params['stop_loss_pct']:.1f}% 이하로 타이트하게")
    print(f"  • {optimal_params['spike_threshold']:.1f}% 이상 스파이크만 진입")
    
    print(f"\n📊 실제 설정 코드:")
    print(f"```python")
    print(f"take_profit_pct = {optimal_params['take_profit_pct']:.1f}")
    print(f"stop_loss_pct = {optimal_params['stop_loss_pct']:.1f}")
    print(f"spike_down_threshold_5m = -{optimal_params['spike_threshold']:.1f}")
    print(f"spike_up_threshold_5m = {optimal_params['spike_threshold']:.1f}")
    print(f"```")
    
    print(f"\n⚠️  중요한 고려사항:")
    print(f"  • 승률 60% 이상이어야 장기적으로 수익")
    print(f"  • 너무 작은 움직임은 수수료에 먹힘")
    print(f"  • 빠른 진입/청산이 핵심 (시간=돈)")
    print(f"  • 볼륨이 충분한 시점에만 거래")


if __name__ == "__main__":
    main()