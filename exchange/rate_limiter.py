"""
Rate limiter for API calls.

This module provides rate limiting functionality to ensure
we don't exceed exchange API rate limits.
"""

import time
from collections import deque
from typing import Deque


class RateLimiter:
    """
    Simple rate limiter using a sliding window approach.
    
    Maintains separate rate limits for public and private API calls.
    Uses a 1-second window to track and limit the number of calls.
    """
    
    def __init__(self, max_calls_per_sec_public: int, max_calls_per_sec_private: int) -> None:
        """
        Initialize the rate limiter.
        
        Args:
            max_calls_per_sec_public: Maximum public API calls per second
            max_calls_per_sec_private: Maximum private API calls per second
        """
        self.max_calls_per_sec_public = max_calls_per_sec_public
        self.max_calls_per_sec_private = max_calls_per_sec_private
        
        # Track timestamps of recent calls
        self._public_calls: Deque[float] = deque()
        self._private_calls: Deque[float] = deque()
    
    def wait_public(self) -> None:
        """
        Wait if necessary before making a public API call.
        
        This method should be called before every public API request.
        It will sleep if the rate limit would be exceeded.
        """
        self._wait_for_rate_limit(self._public_calls, self.max_calls_per_sec_public)
    
    def wait_private(self) -> None:
        """
        Wait if necessary before making a private API call.
        
        This method should be called before every private API request.
        It will sleep if the rate limit would be exceeded.
        """
        self._wait_for_rate_limit(self._private_calls, self.max_calls_per_sec_private)
    
    def _wait_for_rate_limit(self, calls_deque: Deque[float], max_calls: int) -> None:
        """
        Internal method to handle rate limiting logic.
        
        Args:
            calls_deque: Deque tracking recent call timestamps
            max_calls: Maximum calls allowed per second
        """
        current_time = time.time()
        
        # Remove calls older than 1 second
        while calls_deque and calls_deque[0] <= current_time - 1.0:
            calls_deque.popleft()
        
        # If we're at the limit, wait until we can make another call
        if len(calls_deque) >= max_calls:
            # Calculate how long to wait
            oldest_call = calls_deque[0]
            wait_time = oldest_call + 1.0 - current_time
            
            if wait_time > 0:
                time.sleep(wait_time)
                # Remove expired calls again after sleeping
                current_time = time.time()
                while calls_deque and calls_deque[0] <= current_time - 1.0:
                    calls_deque.popleft()
        
        # Record this call
        calls_deque.append(current_time)