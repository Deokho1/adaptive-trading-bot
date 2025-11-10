"""
코인 데이터 스파이크 분석기

실제 바이낸스 데이터에서 가격 변동 패턴과 스파이크 빈도를 분석합니다.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def analyze_price_spikes(df, symbol, timeframe="5m"):
    """
    가격 스파이크 분석
    
    Args:
        df: OHLCV DataFrame
        symbol: 심볼명
        timeframe: 타임프레임
        
    Returns:
        분석 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"{symbol} {timeframe} 스파이크 분석")
    print(f"{'='*60}")
    
    # 기본 통계
    print(f"데이터 기간: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    print(f"총 캔들 수: {len(df):,}개")
    print(f"평균 가격: {df['close'].mean():,.2f}")
    print(f"가격 범위: {df['close'].min():,.2f} ~ {df['close'].max():,.2f}")
    
    # 5분봉 퍼센트 변화 계산
    df['pct_change_5m'] = ((df['close'] - df['open']) / df['open'] * 100).round(3)
    
    # 15분봉 퍼센트 변화 (3개 바 누적)
    df['close_3bars_ago'] = df['close'].shift(3)
    df['pct_change_15m'] = ((df['close'] - df['close_3bars_ago']) / df['close_3bars_ago'] * 100).round(3)
    
    # 볼륨 배율 계산
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = (df['volume'] / df['volume_ma20']).round(2)
    
    # 변동성 통계
    print(f"\n📊 5분봉 변동률 통계:")
    print(f"평균: {df['pct_change_5m'].mean():.3f}%")
    print(f"표준편차: {df['pct_change_5m'].std():.3f}%")
    print(f"최대 상승: +{df['pct_change_5m'].max():.3f}%")
    print(f"최대 하락: {df['pct_change_5m'].min():.3f}%")
    
    # 스파이크 구간별 카운팅
    spike_ranges = [
        (0.0, 0.5, "미세"),
        (0.5, 1.0, "소폭"),
        (1.0, 1.5, "중간"),
        (1.5, 2.0, "큰폭"),
        (2.0, 3.0, "급등"),
        (3.0, 5.0, "폭등"),
        (5.0, float('inf'), "극한")
    ]
    
    print(f"\n🔥 5분봉 상승 스파이크 분포:")
    total_up_spikes = 0
    for min_val, max_val, label in spike_ranges:
        count = len(df[(df['pct_change_5m'] >= min_val) & (df['pct_change_5m'] < max_val)])
        pct = count / len(df) * 100
        total_up_spikes += count
        range_str = f"{min_val}~{max_val}%" if max_val != float('inf') else f"{min_val}%+"
        print(f"{label:>4} ({range_str:>8}): {count:>5,}회 ({pct:>4.1f}%)")
    
    print(f"\n📉 5분봉 하락 스파이크 분포:")
    total_down_spikes = 0
    for min_val, max_val, label in spike_ranges:
        count = len(df[(df['pct_change_5m'] <= -min_val) & (df['pct_change_5m'] > -max_val)])
        pct = count / len(df) * 100
        total_down_spikes += count
        range_str = f"-{max_val}~-{min_val}%" if max_val != float('inf') else f"-{min_val}%-"
        print(f"{label:>4} ({range_str:>8}): {count:>5,}회 ({pct:>4.1f}%)")
    
    # 15분봉 스파이크 (3개 바 누적)
    print(f"\n🔥 15분봉 (3바 누적) 상승 스파이크:")
    df_15m = df[df['pct_change_15m'].notna()]
    for min_val, max_val, label in spike_ranges:
        count = len(df_15m[(df_15m['pct_change_15m'] >= min_val) & (df_15m['pct_change_15m'] < max_val)])
        pct = count / len(df_15m) * 100 if len(df_15m) > 0 else 0
        range_str = f"{min_val}~{max_val}%" if max_val != float('inf') else f"{min_val}%+"
        print(f"{label:>4} ({range_str:>8}): {count:>5,}회 ({pct:>4.1f}%)")
    
    print(f"\n📉 15분봉 (3바 누적) 하락 스파이크:")
    for min_val, max_val, label in spike_ranges:
        count = len(df_15m[(df_15m['pct_change_15m'] <= -min_val) & (df_15m['pct_change_15m'] > -max_val)])
        pct = count / len(df_15m) * 100 if len(df_15m) > 0 else 0
        range_str = f"-{max_val}~-{min_val}%" if max_val != float('inf') else f"-{min_val}%-"
        print(f"{label:>4} ({range_str:>8}): {count:>5,}회 ({pct:>4.1f}%)")
    
    # 볼륨 스파이크 분석
    volume_ranges = [
        (1.0, 1.5, "약간"),
        (1.5, 2.0, "보통"),
        (2.0, 3.0, "높음"),
        (3.0, 5.0, "매우높음"),
        (5.0, float('inf'), "극한")
    ]
    
    print(f"\n📊 볼륨 스파이크 분포 (20일 평균 대비):")
    df_vol = df[df['volume_ratio'].notna()]
    for min_val, max_val, label in volume_ranges:
        count = len(df_vol[(df_vol['volume_ratio'] >= min_val) & (df_vol['volume_ratio'] < max_val)])
        pct = count / len(df_vol) * 100 if len(df_vol) > 0 else 0
        range_str = f"{min_val}~{max_val}배" if max_val != float('inf') else f"{min_val}배+"
        print(f"{label:>6} ({range_str:>8}): {count:>5,}회 ({pct:>4.1f}%)")
    
    # 극한 스파이크 상세 분석
    extreme_up = df[df['pct_change_5m'] >= 2.0].copy()
    extreme_down = df[df['pct_change_5m'] <= -2.0].copy()
    
    if len(extreme_up) > 0:
        print(f"\n🚀 극한 상승 스파이크 TOP 10:")
        top_up = extreme_up.nlargest(10, 'pct_change_5m')[['timestamp', 'pct_change_5m', 'volume_ratio', 'close']]
        for idx, row in top_up.iterrows():
            print(f"  {row['timestamp']}: +{row['pct_change_5m']:>5.2f}% (볼륨x{row['volume_ratio']:>4.1f}) @{row['close']:>8,.0f}")
    
    if len(extreme_down) > 0:
        print(f"\n💥 극한 하락 스파이크 TOP 10:")
        top_down = extreme_down.nsmallest(10, 'pct_change_5m')[['timestamp', 'pct_change_5m', 'volume_ratio', 'close']]
        for idx, row in top_down.iterrows():
            print(f"  {row['timestamp']}: {row['pct_change_5m']:>6.2f}% (볼륨x{row['volume_ratio']:>4.1f}) @{row['close']:>8,.0f}")
    
    # 전략 최적화 제안
    print(f"\n💡 전략 최적화 제안:")
    
    # 1.5% 이상 하락 스파이크 빈도
    big_down_spikes = len(df[df['pct_change_5m'] <= -1.5])
    print(f"• 현재 임계값 -1.5% 이하 하락: {big_down_spikes:,}회 ({big_down_spikes/len(df)*100:.1f}%)")
    
    # 추천 임계값
    for threshold in [0.5, 0.8, 1.0, 1.2]:
        down_count = len(df[df['pct_change_5m'] <= -threshold])
        up_count = len(df[df['pct_change_5m'] >= threshold])
        print(f"• 임계값 ±{threshold}%: 하락 {down_count:,}회 ({down_count/len(df)*100:.1f}%) / 상승 {up_count:,}회 ({up_count/len(df)*100:.1f}%)")
    
    return {
        'symbol': symbol,
        'total_candles': len(df),
        'avg_price': df['close'].mean(),
        'volatility_std': df['pct_change_5m'].std(),
        'max_up_spike': df['pct_change_5m'].max(),
        'max_down_spike': df['pct_change_5m'].min(),
        'big_up_spikes': len(df[df['pct_change_5m'] >= 1.5]),
        'big_down_spikes': len(df[df['pct_change_5m'] <= -1.5]),
    }


def main():
    """메인 분석 함수"""
    print("🔍 코인 데이터 스파이크 분석기")
    print("="*60)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ data 폴더가 없습니다. 먼저 데이터를 다운로드하세요.")
        return
    
    # CSV 파일 찾기
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("❌ CSV 파일이 없습니다.")
        return
    
    print(f"📁 발견된 파일: {len(csv_files)}개")
    for f in csv_files:
        print(f"  - {f}")
    
    results = []
    
    # 각 파일 분석
    for csv_file in csv_files:
        try:
            # 심볼명 추출
            parts = csv_file.replace('.csv', '').split('_')
            symbol = parts[1] if len(parts) >= 2 else csv_file.replace('.csv', '')
            
            # 데이터 로드
            filepath = os.path.join(data_dir, csv_file)
            df = pd.read_csv(filepath)
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 분석 실행
            result = analyze_price_spikes(df, symbol)
            results.append(result)
            
        except Exception as e:
            print(f"❌ {csv_file} 분석 실패: {e}")
    
    # 전체 요약
    if results:
        print(f"\n{'='*60}")
        print("🎯 전체 요약 및 전략 권장사항")
        print(f"{'='*60}")
        
        avg_volatility = np.mean([r['volatility_std'] for r in results])
        total_big_down = sum([r['big_down_spikes'] for r in results])
        total_candles = sum([r['total_candles'] for r in results])
        
        print(f"평균 변동성 (표준편차): {avg_volatility:.3f}%")
        print(f"전체 -1.5% 이하 하락: {total_big_down:,}회 ({total_big_down/total_candles*100:.1f}%)")
        
        print(f"\n🔧 권장 전략 파라미터:")
        if avg_volatility < 1.0:
            print("• 낮은 변동성 → 임계값: ±0.8%")
        elif avg_volatility < 1.5:
            print("• 중간 변동성 → 임계값: ±1.0%")
        else:
            print("• 높은 변동성 → 임계값: ±1.2%")
        
        if total_big_down / total_candles > 0.01:  # 1% 이상
            print("• 충분한 스파이크 빈도 → 공격적 진입 가능")
        else:
            print("• 낮은 스파이크 빈도 → 보수적 진입 권장")


if __name__ == "__main__":
    main()