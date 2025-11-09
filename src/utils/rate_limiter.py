"""
RateLimiter Module

Implements rate limiting for API calls to comply with Upbit's rate limits
"""

import time
from collections import deque
from typing import Callable
import logging
import functools


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, requests_per_second: int = 8, requests_per_minute: int = 200):
        """
        Initialize RateLimiter
        
        Args:
            requests_per_second: Maximum requests allowed per second
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        
        # Track request timestamps
        self.second_window = deque(maxlen=requests_per_second)
        self.minute_window = deque(maxlen=requests_per_minute)
        
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self):
        """Wait if rate limit is about to be exceeded"""
        current_time = time.time()
        
        # Check per-second limit
        if len(self.second_window) >= self.requests_per_second:
            oldest_request = self.second_window[0]
            time_diff = current_time - oldest_request
            if time_diff < 1.0:
                wait_time = 1.0 - time_diff
                self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s (per-second limit)")
                time.sleep(wait_time)
                current_time = time.time()
        
        # Check per-minute limit
        if len(self.minute_window) >= self.requests_per_minute:
            oldest_request = self.minute_window[0]
            time_diff = current_time - oldest_request
            if time_diff < 60.0:
                wait_time = 60.0 - time_diff
                self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s (per-minute limit)")
                time.sleep(wait_time)
                current_time = time.time()
        
        # Record this request
        self.second_window.append(current_time)
        self.minute_window.append(current_time)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply rate limiting to a function
        
        Args:
            func: Function to wrap with rate limiting
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        
        return wrapper
