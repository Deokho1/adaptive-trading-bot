#!/usr/bin/env python3
"""
실제 수익/손실 분석기

고빈도 스캘핑봇의 실제 수익성을 정확히 계산해보자
"""

import pandas as pd
import numpy as np

def analyze_real_profitability():
    """실제 수익성 정확히 분석"""
    print("💰 실제 수익/손실 분석")
    print("="*60)
    
    # 거래 내역 로드
    df = pd.read_csv("results/trades.csv")
    
    print(f"📊 총 거래 수: {len(df)}건")
    
    # 실제 PnL 계산
    total_pnl_abs = df['pnl_abs'].sum()
    winning_trades = df[df['pnl_abs'] > 0]
    losing_trades = df[df['pnl_abs'] <= 0]
    
    print(f"\n💚 승리 거래: {len(winning_trades)}건")
    print(f"❌ 손실 거래: {len(losing_trades)}건")
    print(f"🎯 실제 승률: {len(winning_trades)/len(df)*100:.1f}%")
    
    # 수익/손실 상세
    total_wins = winning_trades['pnl_abs'].sum()
    total_losses = losing_trades['pnl_abs'].sum()
    
    print(f"\n💰 총 수익 합계: {total_wins:+.2f}")
    print(f"💸 총 손실 합계: {total_losses:+.2f}")
    print(f"🏦 순 손익(PnL): {total_pnl_abs:+.2f}")
    
    # 평균 거래
    avg_win = winning_trades['pnl_abs'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_abs'].mean() if len(losing_trades) > 0 else 0
    
    print(f"\n📈 평균 승리: +{avg_win:.2f}")
    print(f"📉 평균 손실: {avg_loss:.2f}")
    print(f"📊 수익손실비: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
    
    # 수수료 추정
    estimated_fees = len(df) * 0.0005 * 2 * 10000  # 0.05% 양방향 수수료
    print(f"\n💳 추정 수수료: -{estimated_fees:.2f}")
    print(f"🏁 수수료 차감 후: {total_pnl_abs - estimated_fees:+.2f}")
    
    # 초기 자본 대비 수익률
    initial_capital = 10_000_000
    return_pct = (total_pnl_abs / initial_capital) * 100
    
    print(f"\n💼 초기 자본: {initial_capital:,}")
    print(f"📊 수익률: {return_pct:+.4f}%")
    
    # 일별 분해
    df['date'] = pd.to_datetime(df['timestamp_entry']).dt.date
    daily_pnl = df.groupby('date')['pnl_abs'].sum()
    
    print(f"\n📅 일별 손익:")
    for date, pnl in daily_pnl.items():
        print(f"  {date}: {pnl:+.2f}")
    
    # 결론
    print(f"\n🎯 결론:")
    if total_pnl_abs > 0:
        print("✅ 수익 발생!")
        print(f"   총 {total_pnl_abs:.2f} 수익")
    else:
        print("❌ 손실 발생!")
        print(f"   총 {abs(total_pnl_abs):.2f} 손실")
        print(f"   주요 원인: {'수수료가 수익을 초과' if abs(total_pnl_abs) < estimated_fees else '전략 자체의 손실'}")
        
    # 개선점
    print(f"\n💡 개선 방향:")
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        if abs(avg_loss) > avg_win:
            print("   - 손실 크기가 수익보다 큼 → 손절 타이밍 개선 필요")
        if len(losing_trades) > len(winning_trades):
            print("   - 승률이 낮음 → 진입 조건 더 엄격하게")
        if estimated_fees > abs(total_pnl_abs):
            print("   - 수수료가 수익 압박 → 거래 빈도 줄이거나 수익폭 확대")

def main():
    analyze_real_profitability()

if __name__ == "__main__":
    main()