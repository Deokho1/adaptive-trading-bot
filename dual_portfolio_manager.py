"""
Dual Portfolio Manager
Manages BTC and SOL as independent engines with adaptive weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

@dataclass
class AssetPerformance:
    """Track individual asset performance metrics"""
    returns: List[float]
    exposures: List[float]
    trades: int
    current_dd: float
    max_dd: float
    sharpe_ratio: float
    total_return: float

class DualPortfolioManager:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Portfolio allocation - Updated to BTC 80%, SOL 20%
        self.base_btc_weight = self.config.get('btc_weight', 0.8)
        self.base_sol_weight = self.config.get('sol_weight', 0.2)
        
        # Rebalancing settings
        self.rebalance_hours = self.config.get('rebalance_hours', 4)
        self.adaptive_weighting = self.config.get('adaptive_weighting', True)
        self.max_weight_deviation = self.config.get('max_weight_deviation', 0.15)  # ±15%
        
        # Performance tracking - Updated for SOL
        self.btc_performance = AssetPerformance([], [], 0, 0.0, 0.0, 0.0, 0.0)
        self.sol_performance = AssetPerformance([], [], 0, 0.0, 0.0, 0.0, 0.0)
        
        # Sharpe-based weighting parameters
        self.sharpe_lookback = self.config.get('sharpe_lookback', 42)  # 7 days in 4-hour periods
        self.sharpe_adjustment_factor = self.config.get('sharpe_adjustment_factor', 0.5)
        
        self.logger = logging.getLogger(f'{__class__.__name__}')
        self.logger.info(f"Dual Portfolio Manager: {self.base_btc_weight:.0%} BTC / {self.base_sol_weight:.0%} SOL")
        
    def calculate_adaptive_weights(self, current_hour: int) -> Tuple[float, float]:
        """Calculate adaptive weights based on Sharpe ratios"""
        if not self.adaptive_weighting or current_hour < self.sharpe_lookback:
            return self.base_btc_weight, self.base_sol_weight
        
        # Calculate recent Sharpe ratios
        btc_sharpe = self._calculate_recent_sharpe(self.btc_performance.returns[-self.sharpe_lookback:])
        sol_sharpe = self._calculate_recent_sharpe(self.sol_performance.returns[-self.sharpe_lookback:])
        
        if btc_sharpe == 0 and sol_sharpe == 0:
            return self.base_btc_weight, self.base_sol_weight
        
        # Calculate relative performance
        total_sharpe = abs(btc_sharpe) + abs(sol_sharpe)
        if total_sharpe == 0:
            return self.base_btc_weight, self.base_sol_weight
            
        # Adjust weights based on relative Sharpe performance
        if btc_sharpe > sol_sharpe:
            # BTC performing better
            sharpe_diff = (btc_sharpe - sol_sharpe) / total_sharpe
            weight_adjustment = sharpe_diff * self.sharpe_adjustment_factor
            
            new_btc_weight = self.base_btc_weight + weight_adjustment
            new_sol_weight = 1.0 - new_btc_weight
        else:
            # SOL performing better
            sharpe_diff = (sol_sharpe - btc_sharpe) / total_sharpe
            weight_adjustment = sharpe_diff * self.sharpe_adjustment_factor
            
            new_sol_weight = self.base_sol_weight + weight_adjustment
            new_btc_weight = 1.0 - new_sol_weight
        
        # Apply maximum deviation limits
        new_btc_weight = max(
            self.base_btc_weight - self.max_weight_deviation,
            min(self.base_btc_weight + self.max_weight_deviation, new_btc_weight)
        )
        new_sol_weight = 1.0 - new_btc_weight
        
        self.logger.debug(f"Adaptive weights: BTC {new_btc_weight:.1%} (Sharpe: {btc_sharpe:.2f}) / SOL {new_sol_weight:.1%} (Sharpe: {sol_sharpe:.2f})")
        
        return new_btc_weight, new_sol_weight
    
    def _calculate_recent_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for recent returns"""
        if len(returns) < 6:  # Need at least 6 periods (24 hours) of data
            return 0.0
        
        returns_array = np.array(returns)
        if np.std(returns_array) == 0:
            return 0.0
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Annualized Sharpe (assuming 4-hour returns)
        sharpe = (mean_return * 2190) / (std_return * np.sqrt(2190))  # 2190 = 4-hour periods per year
        return sharpe
    
    def should_rebalance(self, current_hour: int) -> bool:
        """Check if portfolio should be rebalanced"""
        return current_hour % self.rebalance_hours == 0
    
    def calculate_portfolio_exposure(self, btc_signal: float, sol_signal: float, current_hour: int) -> Tuple[float, float, Dict]:
        """Calculate final portfolio exposures for both assets"""
        
        # Get adaptive weights
        btc_weight, sol_weight = self.calculate_adaptive_weights(current_hour)
        
        # Apply weights to individual signals
        btc_portfolio_exposure = btc_signal * (btc_weight / self.base_btc_weight)
        sol_portfolio_exposure = sol_signal * (sol_weight / self.base_sol_weight)
        
        # Cap individual asset exposures at their engine limits
        btc_portfolio_exposure = min(btc_portfolio_exposure, 100.0)  # BTC max exposure
        sol_portfolio_exposure = min(sol_portfolio_exposure, 45.0)   # SOL max exposure
        
        # Portfolio-level risk management
        total_gross_exposure = btc_portfolio_exposure + sol_portfolio_exposure
        
        # Apply portfolio-level exposure limits
        max_total_exposure = self.config.get('max_total_exposure', 120.0)  # 120% max combined
        if total_gross_exposure > max_total_exposure:
            scale_factor = max_total_exposure / total_gross_exposure
            btc_portfolio_exposure *= scale_factor
            sol_portfolio_exposure *= scale_factor
        
        portfolio_info = {
            "btc_weight": btc_weight,
            "sol_weight": sol_weight,
            "total_exposure": btc_portfolio_exposure + sol_portfolio_exposure,
            "rebalanced": self.should_rebalance(current_hour)
        }
        
        return btc_portfolio_exposure, sol_portfolio_exposure, portfolio_info
    
    def update_performance(self, btc_return: float, sol_return: float, btc_exposure: float, sol_exposure: float):
        """Update performance tracking for both assets"""
        
        # Update BTC performance
        self.btc_performance.returns.append(btc_return)
        self.btc_performance.exposures.append(btc_exposure)
        
        # Update SOL performance  
        self.sol_performance.returns.append(sol_return)
        self.sol_performance.exposures.append(sol_exposure)
        
        # Calculate running statistics
        self._update_asset_stats(self.btc_performance, 'BTC')
        self._update_asset_stats(self.sol_performance, 'SOL')
    
    def _update_asset_stats(self, performance: AssetPerformance, asset_name: str):
        """Update statistics for an individual asset"""
        if len(performance.returns) < 2:
            return
        
        # Calculate cumulative return
        cumulative_returns = np.cumprod(1 + np.array(performance.returns))
        performance.total_return = cumulative_returns[-1] - 1
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        performance.current_dd = abs(drawdowns[-1]) * 100
        performance.max_dd = abs(np.min(drawdowns)) * 100
        
        # Calculate Sharpe ratio
        if len(performance.returns) >= 6:
            performance.sharpe_ratio = self._calculate_recent_sharpe(performance.returns[-42:])  # 7 days
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio performance summary"""
        
        # Calculate portfolio-level metrics
        btc_returns = np.array(self.btc_performance.returns)
        sol_returns = np.array(self.sol_performance.returns)
        btc_weights = np.array([self.base_btc_weight] * len(btc_returns))
        sol_weights = np.array([self.base_sol_weight] * len(sol_returns))
        
        if len(btc_returns) == 0 or len(sol_returns) == 0:
            return {"error": "Insufficient data for portfolio summary"}
        
        # Portfolio returns (weighted)
        portfolio_returns = btc_returns * btc_weights + sol_returns * sol_weights
        
        # Portfolio cumulative return
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_total_return = portfolio_cumulative[-1] - 1 if len(portfolio_cumulative) > 0 else 0
        
        # Portfolio drawdown
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdowns = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
        portfolio_current_dd = abs(portfolio_drawdowns[-1]) * 100 if len(portfolio_drawdowns) > 0 else 0
        portfolio_max_dd = abs(np.min(portfolio_drawdowns)) * 100 if len(portfolio_drawdowns) > 0 else 0
        
        # Portfolio Sharpe
        portfolio_sharpe = self._calculate_recent_sharpe(portfolio_returns.tolist()) if len(portfolio_returns) > 6 else 0
        
        # Correlation
        correlation = np.corrcoef(btc_returns, sol_returns)[0, 1] if len(btc_returns) > 10 else 0
        
        return {
            "portfolio": {
                "total_return": portfolio_total_return * 100,
                "current_dd": portfolio_current_dd,
                "max_dd": portfolio_max_dd,
                "sharpe_ratio": portfolio_sharpe,
                "correlation": correlation
            },
            "btc": {
                "total_return": self.btc_performance.total_return * 100,
                "current_dd": self.btc_performance.current_dd,
                "max_dd": self.btc_performance.max_dd,
                "sharpe_ratio": self.btc_performance.sharpe_ratio,
                "trades": self.btc_performance.trades,
                "avg_exposure": np.mean(self.btc_performance.exposures) if self.btc_performance.exposures else 0
            },
            "sol": {
                "total_return": self.sol_performance.total_return * 100,
                "current_dd": self.sol_performance.current_dd,
                "max_dd": self.sol_performance.max_dd,
                "sharpe_ratio": self.sol_performance.sharpe_ratio,
                "trades": self.sol_performance.trades,
                "avg_exposure": np.mean(self.sol_performance.exposures) if self.sol_performance.exposures else 0
            },
            "weights": {
                "btc_base": self.base_btc_weight,
                "sol_base": self.base_sol_weight,
                "adaptive_enabled": self.adaptive_weighting
            }
        }
    
    def get_current_drawdowns(self) -> Tuple[float, float]:
        """Get current drawdowns for both assets"""
        return self.btc_performance.current_dd, self.sol_performance.current_dd