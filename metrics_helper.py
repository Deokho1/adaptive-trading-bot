"""
Performance Metrics Helper Module
Provides clean, side-effect-free functions for computing trading strategy metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, List


def compute_total_return(equity_series: Union[np.ndarray, List, pd.Series]) -> float:
    """
    Compute total return from an equity series.
    
    Args:
        equity_series: Series of portfolio equity values over time
        
    Returns:
        Total return as a percentage (e.g., 50.0 for 50% return)
    """
    if len(equity_series) == 0:
        return 0.0
    
    equity_array = np.array(equity_series)
    initial_value = equity_array[0]
    final_value = equity_array[-1]
    
    if initial_value <= 0:
        return 0.0
    
    total_return = (final_value / initial_value - 1.0) * 100.0
    return total_return


def compute_max_drawdown(equity_series: Union[np.ndarray, List, pd.Series]) -> float:
    """
    Compute maximum drawdown from an equity series.
    
    Args:
        equity_series: Series of portfolio equity values over time
        
    Returns:
        Maximum drawdown as a positive percentage (e.g., 15.5 for -15.5% max DD)
    """
    if len(equity_series) <= 1:
        return 0.0
    
    equity_array = np.array(equity_series)
    
    # Calculate running maximum (peak values)
    rolling_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown at each point
    drawdown = equity_array / rolling_max - 1.0
    
    # Maximum drawdown is the most negative value
    max_dd = abs(np.min(drawdown)) * 100.0
    
    return max_dd


def compute_sharpe_ratio(equity_series: Union[np.ndarray, List, pd.Series], 
                        periods_per_year: int = 2190, 
                        risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio from an equity series.
    
    Args:
        equity_series: Series of portfolio equity values over time
        periods_per_year: Number of periods in a year (2190 for 4H data: 365 * 6)
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(equity_series) <= 1:
        return 0.0
    
    equity_array = np.array(equity_series)
    
    # Calculate returns
    returns = np.diff(equity_array) / equity_array[:-1]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate mean and std of returns
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Use sample standard deviation
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    # Calculate excess returns
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    
    # Annualized Sharpe ratio
    sharpe = (excess_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
    
    return sharpe


def validate_metrics_computation(equity_series: Union[np.ndarray, List, pd.Series],
                               expected_total_return: float = None,
                               expected_max_dd: float = None,
                               expected_sharpe: float = None,
                               tolerance: float = 0.01) -> dict:
    """
    Validate metrics computation against expected values.
    
    Args:
        equity_series: Series of portfolio equity values over time
        expected_total_return: Expected total return percentage
        expected_max_dd: Expected maximum drawdown percentage
        expected_sharpe: Expected Sharpe ratio
        tolerance: Tolerance for comparison (as fraction)
        
    Returns:
        Dictionary with computed metrics and validation results
    """
    computed_return = compute_total_return(equity_series)
    computed_dd = compute_max_drawdown(equity_series)
    computed_sharpe = compute_sharpe_ratio(equity_series)
    
    result = {
        'computed_total_return': computed_return,
        'computed_max_dd': computed_dd,
        'computed_sharpe': computed_sharpe,
        'validation': {}
    }
    
    if expected_total_return is not None:
        diff = abs(computed_return - expected_total_return) / max(abs(expected_total_return), 1.0)
        result['validation']['total_return_ok'] = diff <= tolerance
        result['validation']['total_return_diff'] = diff
    
    if expected_max_dd is not None:
        diff = abs(computed_dd - expected_max_dd) / max(abs(expected_max_dd), 1.0)
        result['validation']['max_dd_ok'] = diff <= tolerance
        result['validation']['max_dd_diff'] = diff
    
    if expected_sharpe is not None:
        diff = abs(computed_sharpe - expected_sharpe) / max(abs(expected_sharpe), 1.0)
        result['validation']['sharpe_ok'] = diff <= tolerance
        result['validation']['sharpe_diff'] = diff
    
    return result


def debug_equity_series(equity_series: Union[np.ndarray, List, pd.Series], 
                       name: str = "Series") -> dict:
    """
    Debug an equity series to understand its characteristics.
    
    Args:
        equity_series: Series of portfolio equity values over time
        name: Name for the series (for logging)
        
    Returns:
        Dictionary with debug information
    """
    if len(equity_series) == 0:
        return {'error': 'Empty series'}
    
    equity_array = np.array(equity_series)
    
    debug_info = {
        'name': name,
        'length': len(equity_array),
        'initial_value': equity_array[0],
        'final_value': equity_array[-1],
        'min_value': np.min(equity_array),
        'max_value': np.max(equity_array),
        'mean_value': np.mean(equity_array),
        'std_value': np.std(equity_array),
        'negative_values': np.sum(equity_array <= 0),
        'zero_values': np.sum(equity_array == 0),
        'inf_values': np.sum(np.isinf(equity_array)),
        'nan_values': np.sum(np.isnan(equity_array))
    }
    
    # Calculate basic metrics
    try:
        debug_info['total_return'] = compute_total_return(equity_array)
        debug_info['max_drawdown'] = compute_max_drawdown(equity_array)
        debug_info['sharpe_ratio'] = compute_sharpe_ratio(equity_array)
    except Exception as e:
        debug_info['metrics_error'] = str(e)
    
    return debug_info