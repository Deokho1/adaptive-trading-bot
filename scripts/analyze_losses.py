#!/usr/bin/env python3
"""
손실 원인 심층 분석기

왜 이렇게 많이 졌는지 원인을 파헤쳐보자!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_losing_trades():
    """패배 거래 집중 분석"""
    print("🔍 패배 거래 심층 분석")
    print("="*60)
    
    # 거래 데이터 로드
    trades_df = pd.read_csv("scalp_bot/outputs/trades.csv")
    trades_df['timestamp_entry'] = pd.to_datetime(trades_df['timestamp_entry'])
    trades_df['timestamp_exit'] = pd.to_datetime(trades_df['timestamp_exit'])
    
    # 패배 거래만 필터링
    losing_trades = trades_df[trades_df['pnl_pct'] < 0].copy()
    winning_trades = trades_df[trades_df['pnl_pct'] > 0].copy()
    
    print(f"총 거래: {len(trades_df)}개")
    print(f"패배 거래: {len(losing_trades)}개 ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    print(f"승리 거래: {len(winning_trades)}개 ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    
    # 1. 손실 규모 분석
    print(f"\n💸 손실 규모 분석")
    print("-"*40)
    
    loss_ranges = [
        (-0.1, 0.0),
        (-0.2, -0.1), 
        (-0.3, -0.2),
        (-0.5, -0.3),
        (-1.0, -0.5),
        (-2.0, -1.0),
        (-10.0, -2.0)
    ]
    
    for min_loss, max_loss in loss_ranges:
        mask = (losing_trades['pnl_pct'] >= min_loss) & (losing_trades['pnl_pct'] < max_loss)
        count = mask.sum()
        if count > 0:
            avg_loss = losing_trades[mask]['pnl_pct'].mean()
            print(f"{min_loss:.1f}% ~ {max_loss:.1f}%: {count:2d}회 (평균 {avg_loss:.3f}%)")
    
    # 2. 빠른 손절 분석
    print(f"\n⚡ 빠른 손절 분석 (Stop Loss)")
    print("-"*40)
    
    stop_loss_trades = losing_trades[losing_trades['reason_exit'] == 'stop_loss']
    print(f"Stop Loss 거래: {len(stop_loss_trades)}개 / {len(losing_trades)}개")
    
    # 홀딩 기간별 Stop Loss 분석
    for holding_range in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 12)]:
        mask = (stop_loss_trades['holding_bars'] >= holding_range[0]) & (stop_loss_trades['holding_bars'] <= holding_range[1])
        subset = stop_loss_trades[mask]
        if len(subset) > 0:
            avg_loss = subset['pnl_pct'].mean()
            count = len(subset)
            print(f"{holding_range[0]}-{holding_range[1]}바 홀딩: {count:2d}회, 평균 {avg_loss:.3f}% 손실")
    
    # 3. 진입 타이밍 문제 분석
    print(f"\n🎯 진입 타이밍 문제 분석")
    print("-"*40)
    
    # 즉시 손절되는 거래 (0-1바)
    immediate_stops = stop_loss_trades[stop_loss_trades['holding_bars'] <= 1]
    print(f"즉시 손절 (0-1바): {len(immediate_stops)}개")
    if len(immediate_stops) > 0:
        print(f"  평균 손실: {immediate_stops['pnl_pct'].mean():.3f}%")
        print(f"  → 진입 타이밍이 너무 이른 것으로 판단")
    
    # 4. 심볼별 손실 패턴
    print(f"\n🪙 심볼별 손실 패턴")
    print("-"*40)
    
    for symbol in losing_trades['symbol'].unique():
        symbol_losses = losing_trades[losing_trades['symbol'] == symbol]
        symbol_wins = winning_trades[winning_trades['symbol'] == symbol]
        
        loss_count = len(symbol_losses)
        win_count = len(symbol_wins)
        total_symbol_trades = loss_count + win_count
        
        avg_loss = symbol_losses['pnl_pct'].mean()
        avg_win = symbol_wins['pnl_pct'].mean() if len(symbol_wins) > 0 else 0
        
        print(f"{symbol}: {loss_count}패/{win_count}승 (승률 {win_count/total_symbol_trades*100:.1f}%)")
        print(f"  평균 손실: {avg_loss:.3f}%, 평균 승리: {avg_win:.3f}%")
    
    # 5. 시간대별 손실 패턴
    print(f"\n🕐 시간대별 손실 패턴")
    print("-"*40)
    
    losing_trades['hour'] = losing_trades['timestamp_entry'].dt.hour
    hourly_losses = losing_trades.groupby('hour').agg({
        'pnl_pct': ['count', 'mean']
    }).round(3)
    
    # 손실이 많은 시간대 Top 5
    hour_loss_counts = losing_trades['hour'].value_counts().head(5)
    print("손실이 많은 시간대 (Top 5):")
    for hour, count in hour_loss_counts.items():
        avg_loss = losing_trades[losing_trades['hour'] == hour]['pnl_pct'].mean()
        print(f"{hour:2d}시: {count:2d}회 손실, 평균 {avg_loss:.3f}%")

def analyze_stop_loss_trigger():
    """Stop Loss 발동 원인 분석"""
    print(f"\n🛑 Stop Loss 발동 원인 분석")
    print("="*60)
    
    trades_df = pd.read_csv("scalp_bot/outputs/trades.csv")
    stop_loss_trades = trades_df[trades_df['reason_exit'] == 'stop_loss'].copy()
    
    print(f"Stop Loss 거래: {len(stop_loss_trades)}개")
    print(f"현재 Stop Loss 설정: -0.15%")
    
    # Stop Loss 손실 분포
    print(f"\nStop Loss 실제 손실 분포:")
    loss_actual = stop_loss_trades['pnl_pct']
    print(f"평균 손실: {loss_actual.mean():.3f}%")
    print(f"최대 손실: {loss_actual.min():.3f}%")
    print(f"최소 손실: {loss_actual.max():.3f}%")
    print(f"표준편차: {loss_actual.std():.3f}%")
    
    # -0.15% 근처에서 손절된 거래 vs 더 큰 손실
    normal_stops = stop_loss_trades[(stop_loss_trades['pnl_pct'] >= -0.2) & (stop_loss_trades['pnl_pct'] <= -0.1)]
    big_stops = stop_loss_trades[stop_loss_trades['pnl_pct'] < -0.2]
    
    print(f"\n정상 손절 (-0.2% ~ -0.1%): {len(normal_stops)}개")
    print(f"큰 손실 (-0.2% 이하): {len(big_stops)}개")
    
    if len(big_stops) > 0:
        print(f"큰 손실 평균: {big_stops['pnl_pct'].mean():.3f}%")
        print("→ 갭 하락이나 슬리피지로 인한 손실 확대 가능성")

def analyze_market_conditions():
    """시장 조건과 손실의 상관관계"""
    print(f"\n📊 시장 조건 vs 손실 상관관계")
    print("="*60)
    
    # 실제 시장 데이터와 거래 결과 매칭 분석이 필요하지만
    # 현재는 거래 데이터만으로 패턴 분석
    
    trades_df = pd.read_csv("scalp_bot/outputs/trades.csv")
    trades_df['timestamp_entry'] = pd.to_datetime(trades_df['timestamp_entry'])
    
    # 시간대별 승부 패턴
    trades_df['hour'] = trades_df['timestamp_entry'].dt.hour
    
    print("시간대별 승률 분석:")
    for hour in sorted(trades_df['hour'].unique()):
        hour_trades = trades_df[trades_df['hour'] == hour]
        win_rate = (hour_trades['pnl_pct'] > 0).mean() * 100
        avg_pnl = hour_trades['pnl_pct'].mean()
        count = len(hour_trades)
        
        if count >= 3:  # 3회 이상 거래한 시간대만
            status = "🟢" if win_rate > 50 else "🔴"
            print(f"{hour:2d}시: {status} {count:2d}회, 승률 {win_rate:4.1f}%, 평균 {avg_pnl:+.3f}%")

def generate_loss_prevention_strategy():
    """손실 방지 전략 제안"""
    print(f"\n💡 손실 방지 전략 제안")
    print("="*60)
    
    trades_df = pd.read_csv("scalp_bot/outputs/trades.csv")
    trades_df['timestamp_entry'] = pd.to_datetime(trades_df['timestamp_entry'])  # datetime 변환 추가
    losing_trades = trades_df[trades_df['pnl_pct'] < 0]
    stop_loss_trades = losing_trades[losing_trades['reason_exit'] == 'stop_loss']
    
    # 1. Stop Loss 조정 제안
    immediate_stops = stop_loss_trades[stop_loss_trades['holding_bars'] <= 1]
    
    if len(immediate_stops) > len(stop_loss_trades) * 0.3:  # 30% 이상이 즉시 손절
        print("1. ⚠️ 즉시 손절 비율이 높음 (30% 이상)")
        print("   제안: Stop Loss를 -0.15% → -0.10%로 더 타이트하게")
        print("   또는: 진입 조건을 더 엄격하게 (볼륨 필터 강화)")
    
    # 2. 진입 타이밍 개선 제안
    print(f"\n2. 🎯 진입 타이밍 개선 방안")
    print(f"   현재 리바운드 최소 조건: 0.3%")
    print(f"   제안: 0.3% → 0.5%로 상향 (더 확실한 리바운드 대기)")
    
    # 3. 시간대 필터링 제안
    hourly_performance = trades_df.groupby(trades_df['timestamp_entry'].dt.hour)['pnl_pct'].mean()
    bad_hours = hourly_performance[hourly_performance < -0.05].index.tolist()
    
    if bad_hours:
        print(f"\n3. 🚫 거래 금지 시간대 제안")
        print(f"   성과가 나쁜 시간대: {bad_hours}시")
        print(f"   제안: 해당 시간대 거래 중단")
    
    # 4. 심볼 선별 제안
    symbol_performance = trades_df.groupby('symbol')['pnl_pct'].mean()
    worst_symbol = symbol_performance.idxmin()
    worst_performance = symbol_performance[worst_symbol]
    
    if worst_performance < -0.02:  # -0.02% 이하
        print(f"\n4. 🪙 심볼 필터링 제안")
        print(f"   최악 성과 심볼: {worst_symbol} (평균 {worst_performance:.3f}%)")
        print(f"   제안: {worst_symbol} 거래 중단 검토")

def main():
    """메인 분석 실행"""
    print("💸 패배 거래 집중 분석기 - 왜 이렇게 많이 졌나?")
    print("="*60)
    
    if not os.path.exists("scalp_bot/outputs/trades.csv"):
        print("❌ 거래 데이터가 없습니다!")
        return
    
    analyze_losing_trades()
    analyze_stop_loss_trigger() 
    analyze_market_conditions()
    generate_loss_prevention_strategy()
    
    print(f"\n🎯 분석 완료! 위 제안사항을 적용해서 손실을 줄여보세요.")

if __name__ == "__main__":
    main()