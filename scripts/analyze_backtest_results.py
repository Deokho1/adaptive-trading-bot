#!/usr/bin/env python3
"""
백테스트 결과 상세 분석기

trades.csv와 equity_curve.csv를 분석하여 전략의 성과를 깊이 있게 평가합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_backtest_data():
    """백테스트 결과 파일들 로드"""
    base_dir = "scalp_bot/outputs"
    
    # 거래 내역 로드
    trades_file = os.path.join(base_dir, "trades.csv")
    equity_file = os.path.join(base_dir, "equity_curve.csv")
    
    if not os.path.exists(trades_file):
        print(f"❌ 거래 파일을 찾을 수 없습니다: {trades_file}")
        return None, None
    
    if not os.path.exists(equity_file):
        print(f"❌ 자산 곡선 파일을 찾을 수 없습니다: {equity_file}")
        return None, None
    
    trades_df = pd.read_csv(trades_file)
    equity_df = pd.read_csv(equity_file)
    
    # 시간 컬럼 변환
    trades_df['timestamp_entry'] = pd.to_datetime(trades_df['timestamp_entry'])
    trades_df['timestamp_exit'] = pd.to_datetime(trades_df['timestamp_exit'])
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    return trades_df, equity_df

def analyze_trade_performance(trades_df):
    """거래 성과 상세 분석"""
    print("📊 거래 성과 분석")
    print("="*60)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
    
    print(f"총 거래 수: {total_trades:,}개")
    print(f"승리 거래: {winning_trades:,}개 ({winning_trades/total_trades*100:.1f}%)")
    print(f"패배 거래: {losing_trades:,}개 ({losing_trades/total_trades*100:.1f}%)")
    
    # PnL 통계
    avg_pnl = trades_df['pnl_pct'].mean()
    median_pnl = trades_df['pnl_pct'].median()
    std_pnl = trades_df['pnl_pct'].std()
    
    print(f"\n💰 수익률 통계:")
    print(f"평균 수익률: {avg_pnl:.3f}%")
    print(f"중위 수익률: {median_pnl:.3f}%") 
    print(f"수익률 표준편차: {std_pnl:.3f}%")
    print(f"최대 수익: +{trades_df['pnl_pct'].max():.3f}%")
    print(f"최대 손실: {trades_df['pnl_pct'].min():.3f}%")
    
    # 승리/패배 거래 분석
    if winning_trades > 0:
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean()
        print(f"\n🎯 승리 거래 평균: +{avg_win:.3f}%")
    
    if losing_trades > 0:
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean()
        print(f"📉 패배 거래 평균: {avg_loss:.3f}%")
        
        # Risk-Reward 비율
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            print(f"🎲 Risk-Reward 비율: 1:{risk_reward:.2f}")

def analyze_holding_periods(trades_df):
    """홀딩 기간 분석"""
    print(f"\n⏱️ 홀딩 기간 분석")
    print("-"*40)
    
    holding_stats = trades_df['holding_bars'].describe()
    print(f"평균 홀딩: {holding_stats['mean']:.1f}바 ({holding_stats['mean']*5:.0f}분)")
    print(f"중위 홀딩: {holding_stats['50%']:.0f}바 ({holding_stats['50%']*5:.0f}분)")
    print(f"최대 홀딩: {holding_stats['max']:.0f}바 ({holding_stats['max']*5:.0f}분)")
    
    # 홀딩 기간별 수익률
    for holding_range in [(0, 2), (3, 5), (6, 8), (9, 12)]:
        mask = (trades_df['holding_bars'] >= holding_range[0]) & (trades_df['holding_bars'] <= holding_range[1])
        subset = trades_df[mask]
        if len(subset) > 0:
            avg_pnl = subset['pnl_pct'].mean()
            count = len(subset)
            print(f"{holding_range[0]}-{holding_range[1]}바 홀딩: {count:3d}회, 평균 {avg_pnl:+.3f}%")

def analyze_by_symbol(trades_df):
    """심볼별 성과 분석"""
    print(f"\n🪙 심볼별 성과 분석")
    print("-"*40)
    
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        count = len(symbol_trades)
        win_rate = (symbol_trades['pnl_pct'] > 0).mean() * 100
        avg_pnl = symbol_trades['pnl_pct'].mean()
        total_pnl = symbol_trades['pnl_abs'].sum()
        
        print(f"{symbol:8s}: {count:3d}회, 승률 {win_rate:4.1f}%, 평균 {avg_pnl:+.3f}%, 총 {total_pnl:+8,.0f}원")

def analyze_exit_reasons(trades_df):
    """청산 이유 분석"""
    print(f"\n🚪 청산 이유 분석")
    print("-"*40)
    
    exit_analysis = trades_df.groupby('reason_exit').agg({
        'pnl_pct': ['count', 'mean'],
        'holding_bars': 'mean'
    }).round(3)
    
    for reason in trades_df['reason_exit'].unique():
        subset = trades_df[trades_df['reason_exit'] == reason]
        count = len(subset)
        avg_pnl = subset['pnl_pct'].mean()
        avg_holding = subset['holding_bars'].mean()
        
        print(f"{reason:12s}: {count:3d}회 ({count/len(trades_df)*100:4.1f}%), 평균 {avg_pnl:+.3f}%, {avg_holding:.1f}바")

def analyze_time_patterns(trades_df):
    """시간대별 패턴 분석"""
    print(f"\n🕐 시간대별 거래 패턴")
    print("-"*40)
    
    # 시간별 거래 분포
    trades_df['hour'] = trades_df['timestamp_entry'].dt.hour
    hourly_stats = trades_df.groupby('hour').agg({
        'pnl_pct': ['count', 'mean']
    }).round(3)
    
    print("시간대별 거래 빈도 (상위 5개):")
    hour_counts = trades_df['hour'].value_counts().head(5)
    for hour, count in hour_counts.items():
        avg_pnl = trades_df[trades_df['hour'] == hour]['pnl_pct'].mean()
        print(f"{hour:2d}시: {count:3d}회, 평균 {avg_pnl:+.3f}%")
    
    # 요일별 분석
    trades_df['weekday'] = trades_df['timestamp_entry'].dt.day_name()
    weekday_stats = trades_df.groupby('weekday')['pnl_pct'].agg(['count', 'mean']).round(3)
    
    print(f"\n📅 요일별 거래 패턴:")
    for weekday in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        if weekday in weekday_stats.index:
            count = weekday_stats.loc[weekday, 'count']
            avg_pnl = weekday_stats.loc[weekday, 'mean']
            print(f"{weekday:9s}: {count:3.0f}회, 평균 {avg_pnl:+.3f}%")

def analyze_streak_patterns(trades_df):
    """연승/연패 패턴 분석"""
    print(f"\n🔥 연승/연패 패턴 분석")
    print("-"*40)
    
    # 승부 결과 계산
    trades_df['win'] = trades_df['pnl_pct'] > 0
    
    # 연승/연패 계산
    streaks = []
    current_streak = 0
    current_type = None
    
    for _, row in trades_df.iterrows():
        if row['win']:
            if current_type == 'win':
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(('lose', current_streak))
                current_streak = 1
                current_type = 'win'
        else:
            if current_type == 'lose':
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(('win', current_streak))
                current_streak = 1
                current_type = 'lose'
    
    # 마지막 streak 추가
    if current_streak > 0:
        streaks.append((current_type, current_streak))
    
    # 연승/연패 통계
    win_streaks = [length for streak_type, length in streaks if streak_type == 'win']
    lose_streaks = [length for streak_type, length in streaks if streak_type == 'lose']
    
    if win_streaks:
        print(f"최대 연승: {max(win_streaks)}회")
        print(f"평균 연승: {np.mean(win_streaks):.1f}회")
    
    if lose_streaks:
        print(f"최대 연패: {max(lose_streaks)}회") 
        print(f"평균 연패: {np.mean(lose_streaks):.1f}회")

def analyze_equity_curve(equity_df):
    """자산 곡선 분석"""
    print(f"\n📈 자산 곡선 분석")
    print("-"*40)
    
    initial_equity = equity_df['equity'].iloc[0]
    final_equity = equity_df['equity'].iloc[-1]
    max_equity = equity_df['equity'].max()
    min_equity = equity_df['equity'].min()
    
    # 최대 낙폭 계산
    running_max = equity_df['equity'].expanding().max()
    drawdown = (equity_df['equity'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    print(f"초기 자산: {initial_equity:>12,.0f}원")
    print(f"최종 자산: {final_equity:>12,.0f}원")
    print(f"총 수익률: {(final_equity/initial_equity-1)*100:>11.3f}%")
    print(f"최대 자산: {max_equity:>12,.0f}원")
    print(f"최저 자산: {min_equity:>12,.0f}원")
    print(f"최대 낙폭: {max_drawdown:>11.3f}%")
    
    # 변동성 계산 (일간 기준)
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    daily_volatility = equity_df['daily_return'].std() * np.sqrt(288) * 100  # 5분봉이므로 288개/일
    
    if daily_volatility > 0:
        print(f"일간 변동성: {daily_volatility:>11.3f}%")

def generate_recommendations(trades_df):
    """개선 방안 제안"""
    print(f"\n💡 전략 개선 방안")
    print("="*60)
    
    # 현재 성과 요약
    win_rate = (trades_df['pnl_pct'] > 0).mean()
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean()
    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print(f"현재 성과: 승률 {win_rate*100:.1f}%, 평균 {avg_pnl:.3f}%")
    
    recommendations = []
    
    # 1. 수익률 개선
    if avg_pnl < 0.05:  # 평균 수익률이 0.05% 미만
        recommendations.append("🎯 Take Profit을 0.35% → 0.45%로 상향 조정 (수수료 대비 개선)")
    
    # 2. 손실 제한
    if avg_loss < -0.25:  # 평균 손실이 -0.25% 초과
        recommendations.append("🛡️ Stop Loss를 0.20% → 0.15%로 하향 조정 (손실 제한 강화)")
    
    # 3. 거래 빈도
    if len(trades_df) > 300:  # 거래가 너무 많음
        recommendations.append("⚡ 스파이크 임계값을 0.6% → 0.8%로 상향 (고품질 신호만)")
    elif len(trades_df) < 100:  # 거래가 너무 적음
        recommendations.append("📈 스파이크 임계값을 0.6% → 0.4%로 하향 (거래 기회 확대)")
    
    # 4. 홀딩 기간
    avg_holding = trades_df['holding_bars'].mean()
    if avg_holding < 2:
        recommendations.append("⏰ 최소 홀딩 기간 2바 설정 (너무 빠른 청산 방지)")
    
    # 5. 심볼별 성과
    symbol_performance = trades_df.groupby('symbol')['pnl_pct'].mean()
    worst_symbol = symbol_performance.idxmin()
    if symbol_performance[worst_symbol] < -0.1:
        recommendations.append(f"🚫 {worst_symbol} 거래 제외 검토 (지속적 손실)")
    
    # 6. 볼륨 필터
    recommendations.append("📊 볼륨 스파이크 필터 추가 (고볼륨 신호만 거래)")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("현재 전략이 잘 최적화되어 있습니다! 🎉")

def main():
    """메인 분석 실행"""
    print("🔍 백테스트 결과 상세 분석기")
    print("="*60)
    
    # 데이터 로드
    trades_df, equity_df = load_backtest_data()
    
    if trades_df is None or equity_df is None:
        print("❌ 분석할 데이터가 없습니다.")
        return
    
    print(f"✅ 데이터 로드 완료: 거래 {len(trades_df)}건, 자산 기록 {len(equity_df)}건\n")
    
    # 각종 분석 실행
    analyze_trade_performance(trades_df)
    analyze_holding_periods(trades_df)
    analyze_by_symbol(trades_df)
    analyze_exit_reasons(trades_df)
    analyze_time_patterns(trades_df)
    analyze_streak_patterns(trades_df)
    analyze_equity_curve(equity_df)
    generate_recommendations(trades_df)
    
    print(f"\n🎯 분석 완료! 위 인사이트를 바탕으로 전략을 개선해보세요.")

if __name__ == "__main__":
    main()