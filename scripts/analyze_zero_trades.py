#!/usr/bin/env python3
"""
거래 0건 원인 분석기

왜 거래가 하나도 안 일어났는지 분석해보자!
"""

import pandas as pd
import numpy as np
import os

def analyze_zero_trades():
    """거래 0건 원인 심층 분석"""
    print("🔍 거래 0건 원인 분석")
    print("="*60)
    
    # 1. 데이터 확인
    data_file = "data/binance_BTCUSDT_1m_20251110.csv"
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일 없음: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ 데이터 로드 완료: {len(df):,}개 캔들")
    print(f"📅 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 2. 1분봉 변동률 계산 (5분봉 기준 체크)
    df['pct_change_1m'] = ((df['close'] - df['open']) / df['open'] * 100).round(4)
    
    # 5분 누적 변동률 (5바 기준)
    df['pct_change_5m'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100).round(4)
    
    # 15분 누적 변동률 (15바 기준)
    df['pct_change_15m'] = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15) * 100).round(4)
    
    # 볼륨 비율 (20일 평균)
    df['volume_ma20'] = df['volume'].rolling(20*24*60).mean()  # 20일 평균
    df['volume_ratio'] = (df['volume'] / df['volume_ma20']).round(2)
    
    print(f"\n📊 변동률 통계 (1분봉):")
    print(f"1분 변동률 평균: {df['pct_change_1m'].mean():.4f}%")
    print(f"1분 변동률 표준편차: {df['pct_change_1m'].std():.4f}%")
    print(f"1분 최대 상승: +{df['pct_change_1m'].max():.4f}%")
    print(f"1분 최대 하락: {df['pct_change_1m'].min():.4f}%")
    
    # 3. 스파이크 조건 체크
    print(f"\n🔍 스파이크 조건 분석:")
    print(f"설정: -0.6% 하락, +0.6% 상승 스파이크")
    
    # 1분봉 기준 스파이크
    spikes_down_1m = df[df['pct_change_1m'] <= -0.6]
    spikes_up_1m = df[df['pct_change_1m'] >= 0.6]
    
    print(f"1분봉 -0.6% 이하: {len(spikes_down_1m)}개")
    print(f"1분봉 +0.6% 이상: {len(spikes_up_1m)}개")
    
    # 5분 누적 스파이크
    df_5m = df[df['pct_change_5m'].notna()]
    spikes_down_5m = df_5m[df_5m['pct_change_5m'] <= -0.6]
    spikes_up_5m = df_5m[df_5m['pct_change_5m'] >= 0.6]
    
    print(f"5분 누적 -0.6% 이하: {len(spikes_down_5m)}개")
    print(f"5분 누적 +0.6% 이상: {len(spikes_up_5m)}개")
    
    # 4. 볼륨 조건 체크
    print(f"\n📊 볼륨 조건 분석:")
    print(f"설정: 2.0배 이상 볼륨 스파이크")
    
    df_vol = df[df['volume_ratio'].notna()]
    high_volume = df_vol[df_vol['volume_ratio'] >= 2.0]
    
    print(f"2.0배 이상 볼륨: {len(high_volume)}개 ({len(high_volume)/len(df_vol)*100:.1f}%)")
    
    if len(high_volume) > 0:
        print(f"볼륨 스파이크 평균: {high_volume['volume_ratio'].mean():.1f}배")
        print(f"최대 볼륨: {high_volume['volume_ratio'].max():.1f}배")
    
    # 5. 복합 조건 체크 (스파이크 + 볼륨)
    print(f"\n🎯 복합 조건 분석:")
    
    # 하락 스파이크 + 고볼륨
    combined_down = df_vol[(df_vol['pct_change_1m'] <= -0.6) & (df_vol['volume_ratio'] >= 2.0)]
    print(f"하락 스파이크(-0.6%) + 고볼륨(2.0x): {len(combined_down)}개")
    
    # 더 완화된 조건 테스트
    combined_down_loose = df_vol[(df_vol['pct_change_1m'] <= -0.3) & (df_vol['volume_ratio'] >= 1.5)]
    print(f"완화 조건(-0.3% + 1.5x볼륨): {len(combined_down_loose)}개")
    
    # 6. 날짜별 분석
    if len(df) > 0:
        print(f"\n📅 날짜별 스파이크 분포:")
        df['date'] = df['timestamp'].dt.date
        
        daily_stats = df.groupby('date').agg({
            'pct_change_1m': lambda x: len(x[(x <= -0.6) | (x >= 0.6)]),
            'volume_ratio': lambda x: len(x[x >= 2.0]) if x.notna().any() else 0
        })
        
        for date, row in daily_stats.iterrows():
            spike_count = row['pct_change_1m']
            vol_count = row['volume_ratio']
            print(f"{date}: 스파이크 {spike_count}개, 고볼륨 {vol_count}개")
    
    # 7. 추천 조건
    print(f"\n💡 조건 완화 제안:")
    
    # 다양한 임계값 테스트
    for threshold in [0.3, 0.4, 0.5]:
        down_count = len(df[df['pct_change_1m'] <= -threshold])
        up_count = len(df[df['pct_change_1m'] >= threshold])
        print(f"±{threshold}% 임계값: 하락 {down_count}개, 상승 {up_count}개")
    
    for vol_ratio in [1.2, 1.5, 1.8]:
        vol_count = len(df_vol[df_vol['volume_ratio'] >= vol_ratio])
        print(f"{vol_ratio}x 볼륨: {vol_count}개 ({vol_count/len(df_vol)*100:.1f}%)")
    
    # 8. 실제 거래 조건 시뮬레이션
    print(f"\n🔬 거래 조건 시뮬레이션:")
    
    # 가장 완화된 조건
    relaxed_condition = df_vol[
        (df_vol['pct_change_1m'] <= -0.3) & 
        (df_vol['volume_ratio'] >= 1.2)
    ]
    
    print(f"완화된 조건(-0.3% + 1.2x볼륨): {len(relaxed_condition)}개")
    
    if len(relaxed_condition) > 0:
        print(f"샘플 시간대:")
        sample = relaxed_condition.head(5)[['timestamp', 'pct_change_1m', 'volume_ratio', 'close']]
        for _, row in sample.iterrows():
            print(f"  {row['timestamp']}: {row['pct_change_1m']:+.3f}% (볼륨x{row['volume_ratio']:.1f}) @ {row['close']:.1f}")

def main():
    """메인 실행"""
    print("🕵️ 거래 0건 원인 분석기")
    print("="*60)
    
    analyze_zero_trades()
    
    print(f"\n🎯 분석 완료!")

if __name__ == "__main__":
    main()