#!/usr/bin/env python3
"""
수수료 반영 백테스트

Upbit 수수료 구조:
- 일반 거래: 0.05% (매수/매도 각각)
- 총 왕복 수수료: 0.1%
- 김프 수수료나 슬리피지 고려하면 더 높을 수 있음
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

from exchange.models import Candle, Position
from market.market_analyzer import MarketAnalyzer
from market.indicators import compute_rsi
from core.types import OrderSide, MarketMode
from backtest.data_loader import BacktestDataLoader
from backtest.portfolio import BacktestPortfolio

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("fee_included_backtest")

from enum import Enum

class SimpleSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class FeeConfig:
    """수수료 설정"""
    trading_fee_rate: float = 0.0005  # 0.05% (Upbit 기본)
    slippage_rate: float = 0.0001     # 0.01% (슬리피지)
    
    @property
    def total_fee_rate(self) -> float:
        """총 수수료율 (매수/매도 시 각각 적용)"""
        return self.trading_fee_rate + self.slippage_rate

@dataclass
class TradeRecord:
    """거래 기록"""
    timestamp: datetime
    trade_type: str  # 'BUY', 'SELL'
    mode: str
    price: float
    quantity: float
    gross_amount: float    # 수수료 전 금액
    fee_amount: float      # 수수료
    net_amount: float      # 실제 거래 금액
    position_size_ratio: float

def generate_adaptive_signal(candles: List[Candle], current_idx: int, market_mode: MarketMode, prev_market_mode: MarketMode = None) -> SimpleSignal:
    """
    "바닥에서 사고 천장에서 팔기" 전략 구현 🎯
    
    Args:
        candles: 캔들 데이터
        current_idx: 현재 인덱스
        market_mode: 현재 마켓 모드
        prev_market_mode: 이전 마켓 모드 (트렌드 전환 감지용)
    
    Returns:
        거래 신호 (BUY/SELL/HOLD)
    """
    if current_idx < 20:
        return SimpleSignal.HOLD
    
    # RSI 계산
    recent_candles = candles[max(0, current_idx-19):current_idx+1]
    closes = [float(c.close) for c in recent_candles]
    
    if len(closes) < 14:
        return SimpleSignal.HOLD
    
    rsi_values = compute_rsi(closes, period=14)
    if not rsi_values or len(rsi_values) < 2:
        return SimpleSignal.HOLD
    
    current_rsi = rsi_values[-1]
    prev_rsi = rsi_values[-2] if len(rsi_values) >= 2 else current_rsi
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    # 🎯 "하락이 끝났다" = 바닥 매수 신호
    # 1) TREND_DOWN → TREND_UP 전환 순간
    if prev_market_mode == MarketMode.TREND_DOWN and market_mode == MarketMode.TREND_UP:
        logger.info(f"[BOTTOM_BUY] 트렌드 전환 감지! TREND_DOWN → TREND_UP (RSI: {current_rsi:.1f})")
        return SimpleSignal.BUY
    
    # 2) RSI 과매도 반전 신호 (하락 끝났다!)
    if current_rsi < 35 and prev_rsi < current_rsi and prev_rsi < 30:
        logger.info(f"[BOTTOM_BUY] RSI 과매도 반전! {prev_rsi:.1f} → {current_rsi:.1f}")
        return SimpleSignal.BUY
    
    # 🎯 "상승이 끝났다" = 천장 매도 신호  
    # 1) TREND_UP → TREND_DOWN 전환 순간
    if prev_market_mode == MarketMode.TREND_UP and market_mode == MarketMode.TREND_DOWN:
        logger.info(f"[TOP_SELL] 트렌드 전환 감지! TREND_UP → TREND_DOWN (RSI: {current_rsi:.1f})")
        return SimpleSignal.SELL
    
    # 2) RSI 과매수 반전 신호 (상승 끝났다!)
    if current_rsi > 65 and prev_rsi > current_rsi and prev_rsi > 70:
        logger.info(f"[TOP_SELL] RSI 과매수 반전! {prev_rsi:.1f} → {current_rsi:.1f}")
        return SimpleSignal.SELL
    
    # 💡 기존 모드별 보조 신호들 (더 관대하게)
    if market_mode == MarketMode.TREND_UP:
        if current_rsi < 30:  # 상승장 조정에서 매수
            return SimpleSignal.BUY
        elif current_rsi > 80:  # 극도 과매수에서 일부 매도
            return SimpleSignal.SELL
    
    elif market_mode == MarketMode.TREND_DOWN:
        if current_rsi < 25:  # 하락장 극도 과매도에서 매수
            return SimpleSignal.BUY
        elif current_rsi > 60:  # 하락장 반등에서 매도
            return SimpleSignal.SELL
    
    elif market_mode == MarketMode.RANGE:
        if current_rsi < 35:  # 박스권 하단 (더 관대)
            return SimpleSignal.BUY
        elif current_rsi > 65:  # 박스권 상단 (더 관대)
            return SimpleSignal.SELL
    
    elif market_mode == MarketMode.NEUTRAL:
        if current_rsi < 30:  # 중립에서도 적극적
            return SimpleSignal.BUY
        elif current_rsi > 70:  # 중립에서도 적극적
            return SimpleSignal.SELL
    
    # 기존 TREND 모드 (레거시 호환)
    elif market_mode == MarketMode.TREND:
        if current_rsi < 30 and prev_candle and current_candle.close > prev_candle.close:
            return SimpleSignal.BUY
        elif current_rsi > 85:
            return SimpleSignal.SELL
    
    return SimpleSignal.HOLD

def get_regime_exposure(mode: MarketMode) -> float:
    """
    체제 기반 노출도 할당 (단순 버전)
    
    Args:
        mode: 현재 마켓 모드
        
    Returns:
        목표 노출도 (0.0 ~ 1.0)
    """
    exposure_map = {
        MarketMode.TREND_UP: 1.0,    # 100% - 완전 투자
        MarketMode.TREND_DOWN: 0.2,  # 20% - 반등 대비 일부 유지
        MarketMode.NEUTRAL: 0.8,     # 80% - 적극적 보수
        MarketMode.RANGE: 0.8,       # 80% - 적극적 중간
        MarketMode.TREND: 0.8        # 80% - 레거시 호환성
    }
    
    target_exposure = exposure_map.get(mode, 0.0)
    logger.info(f"[INFO][bot] Mode={mode.value.upper()} → Target Exposure={target_exposure:.0%}")
    return target_exposure

def get_adaptive_exposure(market_mode: MarketMode, mdd_last_30: float = 0.0) -> float:
    """
    마켓 모드별 적응형 포지션 사이징 (더 적극적으로 수정) 🚀
    
    Args:
        market_mode: 현재 마켓 모드
        mdd_last_30: 최근 30캔들 최대 낙폭 (선택사항)
    
    Returns:
        exposure: 포지션 노출 비율 (0.0 ~ 1.0)
    """
    # 🔥 더 적극적인 노출 비율
    exposure_map = {
        MarketMode.TREND_UP: 0.8,    # 80% - 상승 트렌드 (90%→80% 약간 보수적)
        MarketMode.TREND_DOWN: 0.3,  # 30% - 하락 트렌드 (10%→30% 3배 증가!)
        MarketMode.RANGE: 0.5,       # 50% - 박스권 (20%→50% 2.5배 증가!)
        MarketMode.NEUTRAL: 0.6,     # 60% - 중립 (50%→60% 증가)
        MarketMode.TREND: 0.7        # 70% - 레거시 호환성
    }
    
    base_exposure = exposure_map.get(market_mode, 0.0)
    
    # MDD 기반 리스크 조절 (선택사항)
    if mdd_last_30 < -15.0:  # 최근 30캔들에서 -15% 이상 하락
        base_exposure *= 0.7  # 30% 감소 (기존 50% 감소에서 완화)
        logger.info(f"[RISK] MDD adjustment: {mdd_last_30:.1f}% → exposure reduced to {base_exposure:.1%}")
    
    return base_exposure

def execute_buy_with_fee(
    portfolio: BacktestPortfolio, 
    symbol: str, 
    candle: Candle, 
    target_amount: float, 
    fee_config: FeeConfig,
    market_mode: MarketMode
) -> TradeRecord:
    """수수료 포함 매수 실행"""
    
    # 수수료 계산
    fee_amount = target_amount * fee_config.total_fee_rate
    net_amount = target_amount - fee_amount  # 실제 구매에 사용되는 금액
    quantity = net_amount / candle.close
    
    # 포지션 업데이트
    if symbol not in portfolio.positions:
        portfolio.positions[symbol] = Position(
            symbol=symbol,
            mode=market_mode,
            entry_price=candle.close,
            size=0,
            entry_time=candle.timestamp,
            peak_price=candle.close
        )
    
    # 평균 매수가 계산
    old_size = portfolio.positions[symbol].size
    old_value = old_size * portfolio.positions[symbol].entry_price
    new_size = old_size + quantity
    new_value = old_value + net_amount
    
    portfolio.positions[symbol].entry_price = new_value / new_size if new_size > 0 else candle.close
    portfolio.positions[symbol].size = new_size
    portfolio.positions[symbol].peak_price = max(portfolio.positions[symbol].peak_price, candle.close)
    
    # 현금 차감 (수수료 포함 전체 금액)
    portfolio.cash -= target_amount
    
    return TradeRecord(
        timestamp=candle.timestamp,
        trade_type='BUY',
        mode=market_mode.name,
        price=candle.close,
        quantity=quantity,
        gross_amount=target_amount,
        fee_amount=fee_amount,
        net_amount=net_amount,
        position_size_ratio=target_amount / (portfolio.cash + target_amount)
    )

def execute_sell_with_fee(
    portfolio: BacktestPortfolio,
    symbol: str,
    candle: Candle,
    sell_ratio: float,
    fee_config: FeeConfig,
    market_mode: MarketMode
) -> TradeRecord:
    """수수료 포함 매도 실행"""
    
    if symbol not in portfolio.positions or portfolio.positions[symbol].size <= 0:
        return None
    
    # 매도할 수량
    sell_quantity = portfolio.positions[symbol].size * sell_ratio
    gross_amount = sell_quantity * candle.close
    
    # 수수료 계산
    fee_amount = gross_amount * fee_config.total_fee_rate
    net_amount = gross_amount - fee_amount  # 실제 받는 금액
    
    # 포지션 업데이트
    portfolio.positions[symbol].size -= sell_quantity
    
    # 현금 증가 (수수료 차감 후)
    portfolio.cash += net_amount
    
    return TradeRecord(
        timestamp=candle.timestamp,
        trade_type='SELL',
        mode=market_mode.name,
        price=candle.close,
        quantity=sell_quantity,
        gross_amount=gross_amount,
        fee_amount=fee_amount,
        net_amount=net_amount,
        position_size_ratio=sell_ratio
    )

def backtest_with_fees():
    """수수료 포함 백테스트"""
    
    print("💰 수수료 반영 백테스트!")
    print("="*60)
    print("🏪 Upbit 수수료 구조:")
    print("   거래 수수료: 0.05% (매수/매도 각각)")
    print("   슬리피지: 0.01%")
    print("   총 수수료: 0.06% (편도)")
    print()
    
    try:
        # 수수료 설정
        fee_config = FeeConfig(
            trading_fee_rate=0.0005,  # 0.05%
            slippage_rate=0.0001      # 0.01%
        )
        
        print(f"🔧 수수료 설정:")
        print(f"   거래 수수료: {fee_config.trading_fee_rate*100:.3f}%")
        print(f"   슬리피지: {fee_config.slippage_rate*100:.3f}%")
        print(f"   총 수수료: {fee_config.total_fee_rate*100:.3f}% (편도)")
        print(f"   왕복 수수료: {fee_config.total_fee_rate*2*100:.3f}%")
        print()
        
        # 데이터 로드
        data_loader = BacktestDataLoader()
        candles = data_loader.load_symbol("KRW-BTC")
        
        if not candles:
            print("❌ 데이터를 로드할 수 없습니다.")
            return
        
        print(f"✅ 데이터 로드 완료: {len(candles)}개 캔들")
        print(f"📅 기간: {candles[0].timestamp} ~ {candles[-1].timestamp}")
        
        # MarketAnalyzer 설정 (쿨다운 제거)
        config = {
            "market_analyzer": {
                "adx_period": 14, "atr_period": 14, "bb_period": 20, "bb_stddev": 2.0,
                "adx_trend_enter": 15.0, "adx_trend_exit": 12.0, "atr_trend_min": 0.5,
                "adx_range_enter": 35.0, "adx_range_exit": 40.0,
                "bw_range_enter": 15.0, "bw_range_exit": 18.0, "atr_range_max": 8.0,
                "cooldown_hours": 0, "ma_period": 30, "slope_lookback": 3  # 더 반응적인 트렌드 감지
            }
        }
        
        analyzer = MarketAnalyzer(config)
        portfolio = BacktestPortfolio(initial_cash=10_000_000, cash=10_000_000)
        
        # 통계 추적 (새로운 모드 포함)
        mode_stats = {
            "TREND_UP": 0, 
            "TREND_DOWN": 0, 
            "RANGE": 0, 
            "NEUTRAL": 0,
            "TREND": 0  # 레거시 호환성
        }
        trade_records = []
        total_fees_paid = 0.0
        last_trade_index = -2
        total_exposure_weighted = 0.0
        exposure_samples = 0
        prev_market_mode = None  # 이전 모드 추적용
        
        print("🔄 백테스트 실행 중 (수수료 포함)...")
        
        # 메인 백테스트 루프
        for i, candle in enumerate(candles):
            if i == 0:
                continue
            
            # 마켓 모드 분석 (더 많은 캔들 사용)
            if i >= 60:  # MA50 + 여유분을 위해 60개 캔들 사용
                recent_candles = candles[max(0, i-59):i+1]
                market_mode = analyzer.update_mode(recent_candles, candle.timestamp)
                mode_stats[market_mode.value.upper()] += 1
                
                # 노출도 샘플링 (매 시간마다)
                if i % 24 == 0:  # 24시간마다 샘플링
                    current_exposure = get_adaptive_exposure(market_mode)
                    total_exposure_weighted += current_exposure
                    exposure_samples += 1
            else:
                market_mode = MarketMode.NEUTRAL
                mode_stats["NEUTRAL"] += 1
            
            # 이전 모드 업데이트
            if i >= 60:
                prev_market_mode = market_mode
            
            # 적응적 거래 로직 (수수료 + 적응형 포지션 사이징)
            if i >= 60 and (i - last_trade_index) > 1:  # 60개 캔들 후부터 거래
                signal = generate_adaptive_signal(candles, i, market_mode, prev_market_mode)
                
                if signal == SimpleSignal.BUY and portfolio.cash > 200000:  # 최소 거래금액
                    # 적응형 포지션 사이징
                    exposure = get_adaptive_exposure(market_mode)
                    base_position_size = 0.05  # 기본 5%
                    adjusted_position_size = base_position_size * exposure
                    
                    target_amount = portfolio.cash * adjusted_position_size
                    
                    # 로그 출력
                    logger.info(f"[INFO][bot] Mode={market_mode.value.upper()} → Exposure={exposure:.0%} → Position={adjusted_position_size:.1%}")
                    
                    # 수수료 포함 매수 실행
                    trade_record = execute_buy_with_fee(
                        portfolio, "KRW-BTC", candle, target_amount, fee_config, market_mode
                    )
                    
                    if trade_record:
                        trade_records.append(trade_record)
                        total_fees_paid += trade_record.fee_amount
                        last_trade_index = i
                
                elif signal == SimpleSignal.SELL and "KRW-BTC" in portfolio.positions:
                    if portfolio.positions["KRW-BTC"].size > 0.00001:
                        # 모드별 매도 비율 (방향성 반영)
                        if market_mode in [MarketMode.TREND_UP, MarketMode.TREND]:
                            sell_ratio = 0.1  # 10% - 상승 트렌드에서는 적게 매도
                        elif market_mode == MarketMode.TREND_DOWN:
                            sell_ratio = 0.4  # 40% - 하락 트렌드에서는 적극 매도
                        elif market_mode == MarketMode.RANGE:
                            sell_ratio = 0.3  # 30% - 박스권에서는 중간
                        else:  # NEUTRAL
                            sell_ratio = 0.2  # 20% - 중립에서는 보수적
                        
                        # 수수료 포함 매도 실행
                        trade_record = execute_sell_with_fee(
                            portfolio, "KRW-BTC", candle, sell_ratio, fee_config, market_mode
                        )
                        
                        if trade_record:
                            trade_records.append(trade_record)
                            total_fees_paid += trade_record.fee_amount
                            last_trade_index = i
            
            # 포트폴리오 업데이트 (MDD 추적 포함)
            current_prices = {"KRW-BTC": candle.close}
            portfolio.update_equity(current_prices, candle.timestamp)
        
        # 최종 결과 계산
        initial_cash = 10_000_000
        final_balance = portfolio.cash
        
        # 남은 포지션 청산 (수수료 포함)
        final_position_value = 0
        final_liquidation_fee = 0
        
        if "KRW-BTC" in portfolio.positions and portfolio.positions["KRW-BTC"].size > 0:
            position_size = portfolio.positions["KRW-BTC"].size
            final_price = candles[-1].close
            gross_value = position_size * final_price
            liquidation_fee = gross_value * fee_config.total_fee_rate
            final_position_value = gross_value - liquidation_fee
            final_liquidation_fee = liquidation_fee
            total_fees_paid += liquidation_fee
        
        final_balance += final_position_value
        
        # 수익률 계산
        total_return = ((final_balance - initial_cash) / initial_cash) * 100
        
        # Buy & Hold 계산 (수수료 포함)
        buy_hold_buy_fee = initial_cash * fee_config.total_fee_rate
        buy_hold_net_investment = initial_cash - buy_hold_buy_fee
        buy_hold_btc_amount = buy_hold_net_investment / candles[0].close
        buy_hold_gross_final = buy_hold_btc_amount * candles[-1].close
        buy_hold_sell_fee = buy_hold_gross_final * fee_config.total_fee_rate
        buy_hold_net_final = buy_hold_gross_final - buy_hold_sell_fee
        buy_hold_return = ((buy_hold_net_final - initial_cash) / initial_cash) * 100
        
        # 결과 출력
        print(f"\n{'='*80}")
        print(f"💰 수수료 반영 백테스트 결과")
        print(f"{'='*80}")
        
        # 모드 분포 (PART 3 요구사항)
        total_periods = sum(mode_stats.values())
        print(f"[RESULT] Market Mode Counts:")
        for mode, count in mode_stats.items():
            if count > 0:  # 0이 아닌 모드만 출력
                percentage = (count / total_periods) * 100 if total_periods > 0 else 0
                print(f"{mode}: {percentage:.1f}%")
        
        print(f"\n💰 수수료 상세 분석:")
        print(f"   총 거래 횟수: {len(trade_records)}회")
        print(f"   총 수수료 지출: {total_fees_paid:,.0f}원")
        print(f"   평균 거래당 수수료: {total_fees_paid/len(trade_records):,.0f}원" if trade_records else "N/A")
        print(f"   수수료율 (초기 자본 대비): {(total_fees_paid/initial_cash)*100:.2f}%")
        print(f"   최종 청산 수수료: {final_liquidation_fee:,.0f}원")
        
        # 평균 노출도 계산 (샘플링된 데이터 기반)
        if exposure_samples > 0:
            avg_exposure_sampling = (total_exposure_weighted / exposure_samples * 100)
        else:
            avg_exposure_sampling = 0
        
        # 거래 기반 평균 노출도 (기존 방식)
        trade_exposure_total = 0.0
        trade_exposure_count = 0
        
        for trade in trade_records:
            if trade.trade_type == 'BUY':
                # 모드에 따른 노출도 매핑
                if 'TREND_UP' in trade.mode:
                    exposure = 0.9
                elif 'TREND_DOWN' in trade.mode:
                    exposure = 0.1
                elif 'RANGE' in trade.mode:
                    exposure = 0.2
                elif 'NEUTRAL' in trade.mode:
                    exposure = 0.5
                else:  # TREND (legacy)
                    exposure = 0.7
                trade_exposure_total += exposure
                trade_exposure_count += 1
        
        avg_exposure_trades = (trade_exposure_total / trade_exposure_count * 100) if trade_exposure_count > 0 else 0
        
        # PART 3: 개선된 최종 결과 출력
        portfolio.log_final_results()  # 포트폴리오에서 제공하는 표준화된 출력
        
        print("=" * 60)
        print(f"[RESULT] Average Exposure: {avg_exposure_sampling:.0f}%")
        print("=" * 60)
        
        print(f"\n📈 성능 비교 (수수료 반영):")
        print(f"   🤖 전략 (수수료 포함): +{total_return:.2f}%")
        print(f"   📈 Buy & Hold (수수료 포함): +{buy_hold_return:.2f}%")
        
        vs_buyhold = total_return - buy_hold_return
        status = "🟢 승!" if vs_buyhold > 0 else "🔴 패!"
        print(f"   {status} Buy&Hold 대비: {vs_buyhold:+.2f}%p")
        
        # 수수료 없는 버전과 비교
        no_fee_final = portfolio.cash + (portfolio.positions["KRW-BTC"].size * candles[-1].close if "KRW-BTC" in portfolio.positions else 0)
        no_fee_return = ((no_fee_final - initial_cash) / initial_cash) * 100
        fee_impact = no_fee_return - total_return
        
        print(f"\n🏪 수수료 영향 분석:")
        print(f"   수수료 미반영 수익률: +{no_fee_return:.2f}%")
        print(f"   수수료 반영 수익률: +{total_return:.2f}%")
        print(f"   수수료 영향: -{fee_impact:.2f}%p")
        
        # 거래 빈도 분석
        if trade_records:
            days_trading = (candles[-1].timestamp - candles[0].timestamp).days
            trades_per_month = len(trade_records) / (days_trading / 30)
            print(f"\n📊 거래 빈도 분석:")
            print(f"   거래 기간: {days_trading}일")
            print(f"   월평균 거래: {trades_per_month:.1f}회")
            
            # 모드별 거래 통계 (새로운 모드 포함)
            all_modes = ["TREND_UP", "TREND_DOWN", "TREND", "RANGE", "NEUTRAL"]
            for mode in all_modes:
                mode_trades = [t for t in trade_records if mode in t.mode]
                if mode_trades:
                    total_fees_mode = sum(t.fee_amount for t in mode_trades)
                    print(f"   {mode}: {len(mode_trades)}회, 수수료 {total_fees_mode:,.0f}원")
        
        print(f"\n💡 결론:")
        if vs_buyhold > 0:
            print(f"   ✅ 수수료를 고려해도 Buy&Hold 상회!")
            print(f"   📊 실제 투자 가능한 전략")
        else:
            print(f"   ❌ 수수료 고려 시 Buy&Hold 대비 저조")
            print(f"   🔧 거래 빈도 줄이거나 전략 개선 필요")
        
        if fee_impact > 5:
            print(f"   🚨 과도한 거래로 인한 수수료 부담 ({fee_impact:.1f}%p)")
            print(f"   📉 거래 빈도 최적화 필요")
        
        print(f"{'='*80}")
        print(f"💰 수수료 반영 백테스트 완료!")
        
    except Exception as e:
        logger.error(f"백테스트 중 오류: {e}")
        print(f"❌ 오류 발생: {e}")

def run_regime_based_backtest():
    """
    단순한 체제 기반 할당 백테스트 실행
    """
    print("🎯 체제 기반 할당 전략 (단순 버전)")
    print("=" * 60)
    print("📋 전략 규칙:")
    print("   TREND_UP   → 100% 노출도 (완전 투자)")
    print("   TREND_DOWN → 0% 노출도 (현금 보유)")
    print("   NEUTRAL    → 70% 노출도")
    print("   RANGE      → 50% 노출도")
    print("   리밸런싱 임계값: 5%")
    print()
    
    # 데이터 로드
    data_loader = BacktestDataLoader()
    candles = data_loader.load_symbol("KRW-BTC")
    
    if not candles:
        print("❌ 데이터를 로드할 수 없습니다.")
        return
    
    print(f"✅ 데이터 로드 완료: {len(candles)}개 캔들")
    print(f"📅 기간: {candles[0].timestamp} ~ {candles[-1].timestamp}")
    
    # 설정 및 초기화
    config = {
        "market_analyzer": {
            "adx_period": 14, "atr_period": 14, "bb_period": 20, "bb_stddev": 2.0,
            "adx_trend_enter": 22.0, "adx_trend_exit": 18.0, "atr_trend_min": 1.0,
            "adx_range_enter": 30.0, "adx_range_exit": 35.0,
            "bw_range_enter": 12.0, "bw_range_exit": 15.0, "atr_range_max": 6.0,
            "cooldown_hours": 2, "ma_period": 30, "slope_lookback": 3
        }
    }
    
    analyzer = MarketAnalyzer(config)
    portfolio = BacktestPortfolio(initial_cash=10_000_000, cash=10_000_000)
    symbol = "KRW-BTC"
    
    # 통계 추적
    mode_counts = {"TREND_UP": 0, "TREND_DOWN": 0, "RANGE": 0, "NEUTRAL": 0, "TREND": 0}
    total_rebalances = 0
    exposure_sum = 0.0
    exposure_samples = 0
    max_equity = 10_000_000
    max_drawdown = 0.0
    
    print("🔄 체제 기반 백테스트 실행 중...")
    
    # 메인 백테스트 루프
    for i, candle in enumerate(candles):
        if i == 0:
            continue
            
        # 마켓 모드 분석 (60개 캔들 사용)
        if i >= 60:
            recent_candles = candles[max(0, i-59):i+1]
            market_mode = analyzer.update_mode(recent_candles, candle.timestamp)
            mode_counts[market_mode.value.upper()] += 1
            
            # 목표 노출도 계산
            target_exposure = get_regime_exposure(market_mode)
            
            # 현재 포트폴리오 상태
            current_equity = portfolio.get_current_equity({symbol: candle.close})
            target_position_value = target_exposure * current_equity
            
            # 현재 포지션 가치
            if symbol in portfolio.positions:
                current_position_value = portfolio.positions[symbol].size * candle.close
            else:
                current_position_value = 0.0
            
            current_exposure = current_position_value / current_equity if current_equity > 0 else 0.0
            exposure_diff = abs(current_exposure - target_exposure)
            
            # 12% 이상 차이날 때만 리밸런싱 (보다 관대한 허용오차)
            REBALANCE_TOL = 0.12  # 12%
            if exposure_diff > REBALANCE_TOL:
                if current_position_value < target_position_value:
                    # 매수 필요
                    amount_to_buy = target_position_value - current_position_value
                    if amount_to_buy > portfolio.cash * 0.999:
                        amount_to_buy = portfolio.cash * 0.999
                    
                    if amount_to_buy > 1000:  # 최소 거래 금액
                        shares_to_buy = amount_to_buy / candle.close
                        portfolio.apply_fill(symbol, OrderSide.BUY, candle.close, shares_to_buy, candle.timestamp)
                        logger.info(f"[INFO][bot] Rebalance: Mode={market_mode.value.upper()}, BUY {shares_to_buy:.6f} BTC at ₩{candle.close:,.0f} (exposure: {current_exposure:.1%} → {target_exposure:.1%})")
                        total_rebalances += 1
                
                elif current_position_value > target_position_value:
                    # 매도 필요
                    amount_to_sell = current_position_value - target_position_value
                    shares_to_sell = amount_to_sell / candle.close
                    
                    if shares_to_sell > 0.000001:
                        available_shares = portfolio.positions[symbol].size
                        shares_to_sell = min(shares_to_sell, available_shares)
                        portfolio.apply_fill(symbol, OrderSide.SELL, candle.close, shares_to_sell, candle.timestamp)
                        logger.info(f"[INFO][bot] Rebalance: Mode={market_mode.value.upper()}, SELL {shares_to_sell:.6f} BTC at ₩{candle.close:,.0f} (exposure: {current_exposure:.1%} → {target_exposure:.1%})")
                        total_rebalances += 1
            
            # 노출도 샘플링 (24시간마다)
            if i % 24 == 0:
                updated_equity = portfolio.get_current_equity({symbol: candle.close})
                if symbol in portfolio.positions:
                    position_value = portfolio.positions[symbol].size * candle.close
                    current_exposure_sample = position_value / updated_equity
                else:
                    current_exposure_sample = 0.0
                
                exposure_sum += current_exposure_sample
                exposure_samples += 1
            
            # 최대 드로우다운 추적
            current_equity = portfolio.get_current_equity({symbol: candle.close})
            if current_equity > max_equity:
                max_equity = current_equity
            else:
                drawdown = (current_equity - max_equity) / max_equity
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
        else:
            mode_counts["NEUTRAL"] += 1
    
    # 최종 결과 계산
    final_equity = portfolio.get_current_equity({symbol: candles[-1].close})
    total_return = (final_equity - 10_000_000) / 10_000_000 * 100
    avg_exposure = exposure_sum / exposure_samples if exposure_samples > 0 else 0.0
    
    # Buy & Hold 계산
    btc_start_price = candles[0].close
    btc_end_price = candles[-1].close
    buy_hold_return = (btc_end_price - btc_start_price) / btc_start_price * 100
    
    print()
    print("=" * 80)
    print("🎯 체제 기반 할당 결과")
    print("=" * 80)
    
    # 모드 분포
    total_periods = sum(mode_counts.values())
    print("[RESULT] 모드 분포:")
    for mode, count in mode_counts.items():
        if count > 0:
            percentage = count / total_periods * 100
            print(f"  {mode}: {percentage:.1f}% ({count:,} 구간)")
    
    print()
    print(f"[RESULT] 총 리밸런싱 횟수: {total_rebalances}")
    print(f"[RESULT] 평균 노출도: {avg_exposure * 100:.1f}%")
    print()
    
    # 성과 지표
    print("=" * 60)
    print(f"[RESULT] 최종 자산: ₩{final_equity:,.0f}")
    print(f"[RESULT] 총 수익률: {total_return:+.2f}%")
    print(f"[RESULT] 최대 드로우다운: {max_drawdown * 100:.2f}%")
    print("=" * 60)
    
    # 성과 비교
    print("📈 성과 비교:")
    print(f"   🎯 체제 전략: {total_return:+.2f}%")
    print(f"   📈 Buy & Hold: {buy_hold_return:+.2f}%")
    
    outperformance = total_return - buy_hold_return
    if outperformance > 0:
        print(f"   🎉 초과 성과: +{outperformance:.2f}%p")
    else:
        print(f"   🔴 저조한 성과: {outperformance:.2f}%p")
    
    print()
    print("💡 전략 요약:")
    print("   ✅ 단순한 체제 기반 할당")
    print("   ✅ RSI나 복잡한 필터 없음")
    print("   ✅ 체제 변화에 직접 노출")
    print("=" * 80)

if __name__ == "__main__":
    try:
        print("🚀 체제 기반 할당 전략 선택")
        print("=" * 50)
        print("선택하세요:")
        print("1. 기존 복잡한 RSI 전략")
        print("2. 새로운 단순 체제 기반 전략")
        
        choice = input("선택 (1 또는 2, 기본값 2): ").strip()
        
        if choice == "1":
            print("\n🔧 기존 복잡한 RSI 전략 실행...")
            backtest_with_fees()
        else:
            print("\n🎯 새로운 단순 체제 기반 전략 실행...")
            run_regime_based_backtest()
            
    except Exception as e:
        logger.error(f"백테스트 실행 실패: {e}")
        raise