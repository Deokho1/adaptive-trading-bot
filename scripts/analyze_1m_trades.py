#!/usr/bin/env python3
"""
1분봉 거래 상세 분석기

거래가 발생했으니 내용을 분석해보자!
"""

import pandas as pd
import numpy as np

def analyze_1m_trades():
    """1분봉 거래 상세 분석"""
    print("📊 1분봉 거래 분석")
    print("="*60)
    
    # 거래 내역 로드
    trades_file = "scalp_bot/outputs/trades.csv"
    try:
        trades = pd.read_csv(trades_file)
        trades['timestamp_entry'] = pd.to_datetime(trades['timestamp_entry'])
        trades['timestamp_exit'] = pd.to_datetime(trades['timestamp_exit'])
        
        print(f"✅ 총 {len(trades)}건 거래 분석")
        print()
        
        # 기본 통계
        winners = trades[trades['pnl_abs'] > 0]
        losers = trades[trades['pnl_abs'] <= 0]
        
        print(f"📈 승리 거래: {len(winners)}건")
        print(f"📉 손실 거래: {len(losers)}건")
        print(f"🎯 승률: {len(winners)/len(trades)*100:.1f}%")
        print()
        
        # 손익 분석
        total_pnl = trades['pnl_abs'].sum()
        avg_winner = winners['pnl_abs'].mean() if len(winners) > 0 else 0
        avg_loser = losers['pnl_abs'].mean() if len(losers) > 0 else 0
        
        print(f"💰 총 손익: {total_pnl:.1f}")
        print(f"💚 평균 승리: {avg_winner:.1f}")
        print(f"❌ 평균 손실: {avg_loser:.1f}")
        print(f"📊 수익손실비: {abs(avg_winner/avg_loser):.2f}:1" if avg_loser != 0 else "N/A")
        print()
        
        # 거래 상세 내역
        print("🔍 거래 상세 내역:")
        print("-"*60)
        
        for i, row in trades.iterrows():
            pnl_emoji = "💚" if row['pnl_abs'] > 0 else "❌"
            duration = row['holding_bars']
            
            print(f"{pnl_emoji} #{i+1}: {row['timestamp_entry'].strftime('%m/%d %H:%M')}")
            print(f"   진입: {row['entry_price']:.1f} → 청산: {row['exit_price']:.1f}")
            print(f"   손익: {row['pnl_abs']:+.1f} ({row['pnl_pct']:+.2f}%)")
            print(f"   기간: {duration}분 | 청산사유: {row['reason_exit']}")
            print(f"   최대손실: {row['max_adverse_excursion_pct']:.2f}%")
            print()
        
        # 청산 사유 분석
        print("🏁 청산 사유 분석:")
        exit_reasons = trades['reason_exit'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}건 ({count/len(trades)*100:.1f}%)")
        print()
        
        # 거래 시간 분석
        print("⏰ 거래 시간 분석:")
        trades['hour'] = trades['timestamp_entry'].dt.hour
        hour_dist = trades['hour'].value_counts().sort_index()
        
        for hour, count in hour_dist.items():
            print(f"  {hour:02d}시: {count}건")
        print()
        
        # 보유 시간 분석
        print("📏 보유 시간 분석:")
        avg_holding = trades['holding_bars'].mean()
        max_holding = trades['holding_bars'].max()
        min_holding = trades['holding_bars'].min()
        
        print(f"평균 보유: {avg_holding:.1f}분")
        print(f"최대 보유: {max_holding}분")
        print(f"최소 보유: {min_holding}분")
        print()
        
        # 결론 및 개선점
        print("💡 분석 결과:")
        print("-"*40)
        
        if len(losers) > 0:
            quick_losses = losers[losers['holding_bars'] <= 5]
            if len(quick_losses) > 0:
                print(f"⚠️  {len(quick_losses)}건이 5분 이내 손실 (조기 손절)")
        
        if avg_loser < 0 and avg_winner > 0:
            if abs(avg_loser) > avg_winner:
                print("⚠️  평균 손실이 평균 수익보다 큼")
        
        if len(trades) < 10:
            print("⚠️  거래 수가 적어 통계적 유의성 부족")
        
        # 파라미터 개선 제안
        print("\n🔧 파라미터 개선 제안:")
        
        if len(trades[trades['reason_exit'] == 'stop_loss']) > len(trades[trades['reason_exit'] == 'take_profit']):
            print("• 손절이 많음 → 진입 조건 더 엄격하게")
        
        if trades['holding_bars'].mean() < 5:
            print("• 보유시간 너무 짧음 → 최대보유시간 연장 고려")
        
        if total_pnl <= 0:
            print("• 총 손실 → 스파이크 임계값 더 낮추거나 리바운드 조건 완화")
    
    except FileNotFoundError:
        print("❌ 거래 파일 없음: scalp_bot/outputs/trades.csv")
    except Exception as e:
        print(f"❌ 분석 오류: {e}")

def main():
    analyze_1m_trades()

if __name__ == "__main__":
    main()