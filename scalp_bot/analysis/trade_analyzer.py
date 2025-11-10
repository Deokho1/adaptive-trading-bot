"""
Trade Quality Analysis Module

# ANALYSIS: 트레이드 품질 분석 및 리포트 생성
Provides comprehensive trade analysis and reporting capabilities.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics


class TradeAnalyzer:
    """
    # ANALYSIS: 트레이드 품질 분석기
    Analyzes trade quality and generates comprehensive reports
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize trade analyzer
        
        Args:
            output_dir: Directory containing trade outputs
        """
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir  # Same as output_dir now
        
    def analyze_trades(self, trades_file: str = None) -> Dict[str, Any]:
        """
        # ANALYSIS: 트레이드 전체 분석
        Analyze all trades and generate comprehensive report
        
        Args:
            trades_file: Path to trades CSV file (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        if trades_file is None:
            trades_file = self.output_dir / "trades.csv"
        
        # Load trades data
        try:
            trades_df = pd.read_csv(trades_file)
        except FileNotFoundError:
            print(f"❌ Trades file not found: {trades_file}")
            return {}
        except Exception as e:
            print(f"❌ Error loading trades: {e}")
            return {}
        
        if trades_df.empty:
            print("⚠️ No trades found in CSV file")
            return {}
        
        print(f"📊 Analyzing {len(trades_df)} trades...")
        
        # Convert timestamps
        trades_df['timestamp_entry'] = pd.to_datetime(trades_df['timestamp_entry'])
        trades_df['timestamp_exit'] = pd.to_datetime(trades_df['timestamp_exit'])
        
        # Calculate extended metrics
        trades_df = self._calculate_extended_metrics(trades_df)
        
        # Perform analysis
        analysis = {
            'overall_stats': self._analyze_overall_performance(trades_df),
            'setup_analysis': self._analyze_by_setup_type(trades_df),
            'session_analysis': self._analyze_by_session(trades_df),
            'time_analysis': self._analyze_time_patterns(trades_df),
            'risk_analysis': self._analyze_risk_metrics(trades_df),
            'consecutive_analysis': self._analyze_consecutive_trades(trades_df)
        }
        
        return analysis
    
    def _calculate_extended_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        # ANALYSIS: 확장 지표 계산
        Calculate additional trade metrics
        """
        # Calculate holding time in minutes
        df['holding_minutes'] = (df['timestamp_exit'] - df['timestamp_entry']).dt.total_seconds() / 60
        
        # Calculate gross PnL (before fees/slippage)
        df['gross_pnl'] = df['pnl_abs']  # Assuming this is already gross
        
        # Calculate net PnL (after fees/slippage) - estimate 0.07% total cost
        trade_cost_pct = 0.0007  # 0.07% total cost (fees + slippage)
        df['net_pnl'] = df['gross_pnl'] * (1 - trade_cost_pct)
        
        # Calculate return percentage based on position size
        df['return_pct'] = df['pnl_pct']  # This should already be calculated
        
        # Ensure side column exists
        if 'side' not in df.columns:
            df['side'] = 'LONG'  # Default to LONG for scalping strategy
        
        # Add setup type based on entry reason
        df['setup_type'] = df['reason_entry'].apply(self._classify_setup_type)
        
        # Add session label based on entry time
        df['session_label'] = df['timestamp_entry'].apply(self._classify_session)
        
        return df
    
    def _classify_setup_type(self, reason: str) -> str:
        """
        # ANALYSIS: 진입 패턴 분류
        Classify setup type based on entry reason
        """
        reason_str = str(reason).lower()
        
        if 'dip' in reason_str or 'rebound' in reason_str:
            return 'dip_buy'
        elif 'spike' in reason_str or 'volume' in reason_str:
            return 'vol_spike'
        elif 'breakout' in reason_str:
            return 'breakout'
        elif 'reversal' in reason_str:
            return 'reversal'
        else:
            return 'other'
    
    def _classify_session(self, timestamp: pd.Timestamp) -> str:
        """
        # ANALYSIS: 거래 세션 분류
        Classify trading session based on UTC time
        """
        hour = timestamp.hour
        
        # UTC시간 기준 세션 분류
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'eu'
        elif 16 <= hour < 24:
            return 'us'
        else:
            return 'other'
    
    def _analyze_overall_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        # ANALYSIS: 전체 성과 분석
        Analyze overall trading performance
        """
        total_trades = len(df)
        winning_trades = len(df[df['net_pnl'] > 0])
        losing_trades = len(df[df['net_pnl'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Separate winners and losers
        winners = df[df['net_pnl'] > 0]
        losers = df[df['net_pnl'] <= 0]
        
        avg_return_all = df['return_pct'].mean() if not df.empty else 0
        avg_return_winners = winners['return_pct'].mean() if not winners.empty else 0
        avg_return_losers = losers['return_pct'].mean() if not losers.empty else 0
        
        avg_holding_minutes = df['holding_minutes'].mean() if not df.empty else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': round(win_rate, 1),
            'avg_return_pct_all': round(avg_return_all, 2),
            'avg_return_pct_winners': round(avg_return_winners, 2),
            'avg_return_pct_losers': round(avg_return_losers, 2),
            'avg_holding_minutes': round(avg_holding_minutes, 1),
            'total_gross_pnl': round(df['gross_pnl'].sum(), 2),
            'total_net_pnl': round(df['net_pnl'].sum(), 2),
            'profit_factor': round(winners['net_pnl'].sum() / abs(losers['net_pnl'].sum()), 2) if not losers.empty and losers['net_pnl'].sum() != 0 else float('inf'),
            'largest_win': round(df['net_pnl'].max(), 2) if not df.empty else 0,
            'largest_loss': round(df['net_pnl'].min(), 2) if not df.empty else 0
        }
    
    def _analyze_by_setup_type(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        # ANALYSIS: 진입 패턴별 분석
        Analyze performance by setup type
        """
        setup_analysis = {}
        
        for setup_type in df['setup_type'].unique():
            setup_trades = df[df['setup_type'] == setup_type]
            
            if setup_trades.empty:
                continue
                
            winning_trades = len(setup_trades[setup_trades['net_pnl'] > 0])
            total_trades = len(setup_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            setup_analysis[setup_type] = {
                'trade_count': total_trades,
                'win_rate_pct': round(win_rate, 1),
                'avg_net_pnl': round(setup_trades['net_pnl'].mean(), 2),
                'total_net_pnl': round(setup_trades['net_pnl'].sum(), 2),
                'avg_return_pct': round(setup_trades['return_pct'].mean(), 2),
                'avg_holding_minutes': round(setup_trades['holding_minutes'].mean(), 1)
            }
        
        return setup_analysis
    
    def _analyze_by_session(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        # ANALYSIS: 거래 세션별 분석
        Analyze performance by trading session
        """
        session_analysis = {}
        
        for session in df['session_label'].unique():
            session_trades = df[df['session_label'] == session]
            
            if session_trades.empty:
                continue
                
            winning_trades = len(session_trades[session_trades['net_pnl'] > 0])
            total_trades = len(session_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            session_analysis[session] = {
                'trade_count': total_trades,
                'win_rate_pct': round(win_rate, 1),
                'avg_net_pnl': round(session_trades['net_pnl'].mean(), 2),
                'total_net_pnl': round(session_trades['net_pnl'].sum(), 2),
                'avg_return_pct': round(session_trades['return_pct'].mean(), 2),
                'avg_holding_minutes': round(session_trades['holding_minutes'].mean(), 1)
            }
        
        return session_analysis
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        # ANALYSIS: 시간대별 패턴 분석
        Analyze trading patterns by time
        """
        # Hour of day analysis
        df['entry_hour'] = df['timestamp_entry'].dt.hour
        hourly_stats = df.groupby('entry_hour').agg({
            'net_pnl': ['count', 'mean', 'sum'],
            'return_pct': 'mean'
        }).round(2)
        
        # Day of week analysis
        df['entry_weekday'] = df['timestamp_entry'].dt.day_name()
        daily_stats = df.groupby('entry_weekday').agg({
            'net_pnl': ['count', 'mean', 'sum'],
            'return_pct': 'mean'
        }).round(2)
        
        return {
            'best_hour': int(df.groupby('entry_hour')['net_pnl'].sum().idxmax()) if not df.empty else 0,
            'worst_hour': int(df.groupby('entry_hour')['net_pnl'].sum().idxmin()) if not df.empty else 0,
            'best_day': df.groupby('entry_weekday')['net_pnl'].sum().idxmax() if not df.empty else 'Unknown',
            'hourly_distribution': df['entry_hour'].value_counts().to_dict(),
            'daily_distribution': df['entry_weekday'].value_counts().to_dict()
        }
    
    def _analyze_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        # ANALYSIS: 리스크 지표 분석
        Analyze risk-related metrics
        """
        if df.empty:
            return {}
            
        # Maximum Adverse Excursion analysis
        avg_mae = df['max_adverse_excursion_pct'].mean()
        max_mae = df['max_adverse_excursion_pct'].min()  # Most negative
        
        # Maximum Favorable Excursion analysis  
        avg_mfe = df['max_favorable_excursion_pct'].mean()
        max_mfe = df['max_favorable_excursion_pct'].max()
        
        # Risk-Return metrics
        returns = df['return_pct'].tolist()
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        return {
            'avg_max_adverse_excursion_pct': round(avg_mae, 2),
            'worst_adverse_excursion_pct': round(max_mae, 2),
            'avg_max_favorable_excursion_pct': round(avg_mfe, 2),
            'best_favorable_excursion_pct': round(max_mfe, 2),
            'return_volatility_pct': round(volatility, 2),
            'risk_adjusted_return': round(df['return_pct'].mean() / volatility, 2) if volatility > 0 else 0
        }
    
    def _analyze_consecutive_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        # ANALYSIS: 연속 거래 분석
        Analyze consecutive winning/losing streaks
        """
        if df.empty:
            return {'max_consecutive_losses': 0, 'max_consecutive_wins': 0, 'current_streak': 0}
        
        # Sort by entry time
        df_sorted = df.sort_values('timestamp_entry')
        
        # Determine win/loss for each trade
        trade_results = (df_sorted['net_pnl'] > 0).astype(int)  # 1 for win, 0 for loss
        
        # Calculate consecutive streaks
        max_losses = 0
        max_wins = 0
        current_loss_streak = 0
        current_win_streak = 0
        
        for is_winner in trade_results:
            if is_winner:
                current_win_streak += 1
                current_loss_streak = 0
                max_wins = max(max_wins, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_losses = max(max_losses, current_loss_streak)
        
        # Current streak (last trades)
        current_streak = current_win_streak if trade_results.iloc[-1] else -current_loss_streak
        
        return {
            'max_consecutive_losses': max_losses,
            'max_consecutive_wins': max_wins,
            'current_streak': current_streak
        }
    
    def generate_console_report(self, analysis: Dict[str, Any]) -> None:
        """
        # ANALYSIS: 콘솔 리포트 출력
        Generate human-readable console report
        """
        print("\n" + "="*80)
        print("🎯 TRADE QUALITY ANALYSIS REPORT")
        print("="*80)
        
        # Overall Performance
        overall = analysis.get('overall_stats', {})
        print(f"\n📊 OVERALL PERFORMANCE")
        print("-" * 40)
        print(f"Total Trades:           {overall.get('total_trades', 0):>8,}")
        print(f"Winning Trades:         {overall.get('winning_trades', 0):>8,}")
        print(f"Losing Trades:          {overall.get('losing_trades', 0):>8,}")
        print(f"Win Rate:               {overall.get('win_rate_pct', 0):>8.1f}%")
        print(f"Average Return:         {overall.get('avg_return_pct_all', 0):>8.2f}%")
        print(f"Average Winners:        {overall.get('avg_return_pct_winners', 0):>8.2f}%")
        print(f"Average Losers:         {overall.get('avg_return_pct_losers', 0):>8.2f}%")
        print(f"Average Holding:        {overall.get('avg_holding_minutes', 0):>8.1f} min")
        print(f"Profit Factor:          {overall.get('profit_factor', 0):>8.2f}")
        print(f"Total Net P&L:          {overall.get('total_net_pnl', 0):>8.2f}")
        
        # Setup Type Analysis
        setup_analysis = analysis.get('setup_analysis', {})
        if setup_analysis:
            print(f"\n🎲 SETUP TYPE ANALYSIS")
            print("-" * 40)
            for setup, stats in setup_analysis.items():
                print(f"{setup.upper():>12}: {stats['trade_count']:>3} trades, {stats['win_rate_pct']:>5.1f}% win, {stats['avg_net_pnl']:>+7.2f} avg")
        
        # Session Analysis
        session_analysis = analysis.get('session_analysis', {})
        if session_analysis:
            print(f"\n🌍 SESSION ANALYSIS")
            print("-" * 40)
            for session, stats in session_analysis.items():
                print(f"{session.upper():>8}: {stats['trade_count']:>3} trades, {stats['win_rate_pct']:>5.1f}% win, {stats['total_net_pnl']:>+7.2f} total")
        
        # Risk Analysis
        risk_analysis = analysis.get('risk_analysis', {})
        if risk_analysis:
            print(f"\n⚠️ RISK METRICS")
            print("-" * 40)
            print(f"Avg Max Adverse:        {risk_analysis.get('avg_max_adverse_excursion_pct', 0):>8.2f}%")
            print(f"Worst Adverse:          {risk_analysis.get('worst_adverse_excursion_pct', 0):>8.2f}%")
            print(f"Avg Max Favorable:      {risk_analysis.get('avg_max_favorable_excursion_pct', 0):>8.2f}%")
            print(f"Best Favorable:         {risk_analysis.get('best_favorable_excursion_pct', 0):>8.2f}%")
            print(f"Return Volatility:      {risk_analysis.get('return_volatility_pct', 0):>8.2f}%")
        
        # Consecutive Analysis
        consecutive = analysis.get('consecutive_analysis', {})
        if consecutive:
            print(f"\n🔄 CONSECUTIVE TRADES")
            print("-" * 40)
            print(f"Max Consecutive Losses: {consecutive.get('max_consecutive_losses', 0):>8}")
            print(f"Max Consecutive Wins:   {consecutive.get('max_consecutive_wins', 0):>8}")
            print(f"Current Streak:         {consecutive.get('current_streak', 0):>8}")
        
        # Time Analysis
        time_analysis = analysis.get('time_analysis', {})
        if time_analysis:
            print(f"\n⏰ TIME PATTERNS")
            print("-" * 40)
            print(f"Best Hour (UTC):        {time_analysis.get('best_hour', 0):>8}:00")
            print(f"Worst Hour (UTC):       {time_analysis.get('worst_hour', 0):>8}:00")
            print(f"Best Day:               {time_analysis.get('best_day', 'Unknown'):>12}")
        
        print("="*80)
    
    def save_json_report(self, analysis: Dict[str, Any], filename: str = "trade_report.json") -> None:
        """
        # ANALYSIS: JSON 리포트 저장
        Save analysis results to JSON file
        """
        output_file = self.results_dir / filename
        
        # Add metadata
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_version': '1.0',
            'trade_analysis': analysis
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📄 JSON report saved: {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving JSON report: {e}")
    
    def run_full_analysis(self, trades_file: str = None) -> Dict[str, Any]:
        """
        # ANALYSIS: 전체 분석 실행
        Run complete analysis and generate both console and JSON reports
        
        Args:
            trades_file: Path to trades CSV file (optional)
            
        Returns:
            Analysis results dictionary
        """
        print("🔍 Starting comprehensive trade analysis...")
        
        # Perform analysis
        analysis = self.analyze_trades(trades_file)
        
        if not analysis:
            print("⚠️ No analysis results - check if trades data exists")
            return {}
        
        # Generate reports
        self.generate_console_report(analysis)
        self.save_json_report(analysis)
        
        print("✅ Trade analysis completed!")
        
        return analysis


def analyze_backtest_results(output_dir: str = "results") -> Dict[str, Any]:
    """
    # ANALYSIS: 백테스트 결과 분석 진입점
    Convenience function to analyze backtest results
    
    Args:
        output_dir: Directory containing backtest outputs
        
    Returns:
        Analysis results dictionary
    """
    analyzer = TradeAnalyzer(output_dir)
    return analyzer.run_full_analysis()