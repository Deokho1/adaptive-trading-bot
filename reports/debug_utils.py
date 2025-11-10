"""
Debug utilities and logging functions.

백테스트 디버깅, 신호 추적, 성능 모니터링을 위한 유틸리티들입니다.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
import json
from pathlib import Path


class BacktestLogger:
    """백테스트 디버그 로거"""
    
    def __init__(self, output_dir: str = "debug_logs"):
        """
        디버그 로거 초기화
        
        Args:
            output_dir: 로그 출력 디렉터리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.events: List[Dict] = []
        self.performance_log: List[Dict] = []
        
    def log_event(self, 
                  timestamp: datetime,
                  event_type: str,
                  message: str,
                  data: Optional[Dict] = None) -> None:
        """
        이벤트 로깅
        
        Args:
            timestamp: 이벤트 시간
            event_type: 이벤트 타입 ("SIGNAL", "ORDER", "ERROR" 등)
            message: 메시지
            data: 추가 데이터
        """
        event = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'message': message,
            'data': data or {}
        }
        
        self.events.append(event)
        
    def log_performance_snapshot(self,
                               timestamp: datetime,
                               equity: float,
                               position_count: int,
                               indicators: Optional[Dict] = None) -> None:
        """
        성능 스냅샷 로깅
        
        Args:
            timestamp: 시간
            equity: 현재 자산
            position_count: 포지션 수
            indicators: 기술적 지표들
        """
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'position_count': position_count,
            'indicators': indicators or {}
        }
        
        self.performance_log.append(snapshot)
        
    def save_debug_logs(self) -> None:
        """디버그 로그를 파일로 저장"""
        
        # 이벤트 로그 CSV
        if self.events:
            events_file = self.output_dir / "debug_events.csv"
            with open(events_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'event_type', 'message', 'data'])
                writer.writeheader()
                
                for event in self.events:
                    # data 필드를 JSON 문자열로 변환
                    event_copy = event.copy()
                    event_copy['data'] = json.dumps(event_copy['data'], ensure_ascii=False)
                    writer.writerow(event_copy)
        
        # 성능 로그 CSV
        if self.performance_log:
            performance_file = self.output_dir / "performance_snapshots.csv"
            with open(performance_file, 'w', newline='', encoding='utf-8') as f:
                if self.performance_log:
                    fieldnames = ['timestamp', 'equity', 'position_count']
                    # 지표 컬럼들 추가
                    indicator_keys = set()
                    for log in self.performance_log:
                        if 'indicators' in log and log['indicators']:
                            indicator_keys.update(log['indicators'].keys())
                    fieldnames.extend(sorted(indicator_keys))
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for log in self.performance_log:
                        row = {
                            'timestamp': log['timestamp'],
                            'equity': log['equity'],
                            'position_count': log['position_count']
                        }
                        # 지표 데이터 추가
                        indicators = log.get('indicators', {})
                        for key in indicator_keys:
                            row[key] = indicators.get(key, '')
                        
                        writer.writerow(row)
        
        print(f"📁 디버그 로그 저장완료: {self.output_dir}")


def create_debug_signal_csv(signals: List[Dict], 
                           output_file: str = "debug_signals.csv") -> None:
    """
    신호 디버그 CSV 생성
    
    Args:
        signals: 신호 리스트
        output_file: 출력 파일명
    """
    if not signals:
        return
    
    # 필드명 정의
    fieldnames = [
        'timestamp', 'symbol', 'action', 'signal_type', 
        'strength', 'price', 'reason', 'indicators'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for signal in signals:
            row = {
                'timestamp': signal.get('timestamp', ''),
                'symbol': signal.get('symbol', ''),
                'action': signal.get('action', ''),
                'signal_type': signal.get('signal_type', ''),
                'strength': signal.get('strength', ''),
                'price': signal.get('price', ''),
                'reason': signal.get('reason', ''),
                'indicators': json.dumps(signal.get('indicators', {}), ensure_ascii=False)
            }
            writer.writerow(row)


def analyze_signal_distribution(signals: List[Dict]) -> Dict:
    """
    신호 분포 분석
    
    Args:
        signals: 신호 리스트
        
    Returns:
        Dict: 분석 결과
    """
    if not signals:
        return {}
    
    analysis = {
        'total_signals': len(signals),
        'action_distribution': {},
        'signal_type_distribution': {},
        'hourly_distribution': {},
        'strength_stats': {}
    }
    
    # 액션별 분포
    actions = [s.get('action', '') for s in signals]
    for action in set(actions):
        analysis['action_distribution'][action] = actions.count(action)
    
    # 신호 타입별 분포
    signal_types = [s.get('signal_type', '') for s in signals]
    for signal_type in set(signal_types):
        analysis['signal_type_distribution'][signal_type] = signal_types.count(signal_type)
    
    # 강도 통계
    strengths = [s.get('strength', 0) for s in signals if isinstance(s.get('strength'), (int, float))]
    if strengths:
        analysis['strength_stats'] = {
            'min': min(strengths),
            'max': max(strengths),
            'avg': sum(strengths) / len(strengths),
            'count': len(strengths)
        }
    
    return analysis


def generate_debug_summary(events: List[Dict],
                          signals: List[Dict],
                          trades: List[Dict]) -> str:
    """
    디버그 요약 리포트 생성
    
    Args:
        events: 이벤트 로그
        signals: 신호 로그  
        trades: 거래 로그
        
    Returns:
        str: 요약 리포트
    """
    signal_analysis = analyze_signal_distribution(signals)
    
    # 에러 이벤트 카운트
    error_count = len([e for e in events if e.get('event_type') == 'ERROR'])
    
    report = f"""
=== 백테스트 디버그 요약 ===

📊 데이터 처리 통계:
- 총 이벤트: {len(events)}개
- 에러 발생: {error_count}개
- 생성된 신호: {len(signals)}개
- 실행된 거래: {len(trades)}개

🎯 신호 분석:
- 총 신호: {signal_analysis.get('total_signals', 0)}개
- BUY 신호: {signal_analysis.get('action_distribution', {}).get('BUY', 0)}개
- SELL 신호: {signal_analysis.get('action_distribution', {}).get('SELL', 0)}개
- HOLD 신호: {signal_analysis.get('action_distribution', {}).get('HOLD', 0)}개

⚠️ 문제 감지:
- 에러율: {(error_count/len(events)*100):.1f}% if events else 0
- 신호 실행율: {(len(trades)/len(signals)*100):.1f}% if signals else 0

"""
    
    # 신호 타입별 분포
    if 'signal_type_distribution' in signal_analysis:
        report += "📈 신호 타입별 분포:\n"
        for signal_type, count in signal_analysis['signal_type_distribution'].items():
            if signal_type:  # 빈 문자열 제외
                report += f"- {signal_type}: {count}개\n"
    
    return report