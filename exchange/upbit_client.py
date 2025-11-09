"""
Upbit exchange client for public API access.

This module provides a client for interacting with Upbit's public API,
including fetching candle data and ticker prices.
"""

import requests
from datetime import datetime, timezone
from typing import List, Optional

from .models import Candle
from .rate_limiter import RateLimiter


class UpbitClient:
    """
    Client for Upbit public API.
    
    Provides methods to fetch market data from Upbit exchange
    with proper rate limiting.
    """
    
    def __init__(self, base_url: str, rate_limiter: RateLimiter) -> None:
        """
        Initialize the Upbit client.
        
        Args:
            base_url: Base URL for Upbit API (e.g., "https://api.upbit.com")
            rate_limiter: Rate limiter instance to control API call frequency
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'adaptive-trading-bot/1.0',
            'Accept': 'application/json',
        })
    
    def get_candles_4h(self, symbol: str, count: int = 200) -> List[Candle]:
        """
        Fetch 4-hour candles for the given symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "KRW-BTC")
            count: Number of candles to fetch (max 200, default 200)
        
        Returns:
            List of Candle objects, ordered from oldest to newest
            (index -1 is the most recent candle)
        
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        return self.get_candles_4h_page(symbol, count, to=None)
    
    def get_candles_4h_page(
        self,
        symbol: str,
        count: int = 200,
        to: Optional[datetime] = None,
    ) -> List[Candle]:
        """
        Fetch a single page of 4h candles for the given symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "KRW-BTC")
            count: Number of candles to fetch (max 200, default 200)
            to: End time for candles (UTC). If None, fetch most recent candles.
        
        Returns:
            List of Candle objects, ordered from oldest to newest
        
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        # Apply rate limiting
        self.rate_limiter.wait_public()
        
        # Prepare request
        url = f"{self.base_url}/v1/candles/minutes/240"
        params = {
            'market': symbol,
            'count': min(count, 200)  # Upbit max is 200
        }
        
        # Add 'to' parameter if specified
        if to is not None:
            # Convert to UTC and format as ISO8601
            to_utc = to.astimezone(timezone.utc)
            params['to'] = to_utc.strftime("%Y-%m-%dT%H:%M:%S%z")
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                raise ValueError(f"Expected list response, got {type(data)}")
            
            # Convert to Candle objects
            candles = []
            for item in data:
                try:
                    # Parse Upbit timestamp format (ISO 8601)
                    timestamp_str = item['candle_date_time_utc']
                    # Upbit returns UTC timestamps like "2023-01-01T00:00:00"
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    candle = Candle(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(item['opening_price']),
                        high=float(item['high_price']),
                        low=float(item['low_price']),
                        close=float(item['trade_price']),
                        volume=float(item['candle_acc_trade_volume'])
                    )
                    candles.append(candle)
                    
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(f"Invalid candle data format: {e}")
            
            # Upbit returns newest first, so reverse to get oldest first
            candles.reverse()
            
            return candles
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch candles for {symbol}: {e}")
    
    def get_ticker_price(self, symbol: str) -> float:
        """
        Fetch the latest trade price for the given symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "KRW-BTC")
        
        Returns:
            Latest trade price as float
        
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        # Apply rate limiting
        self.rate_limiter.wait_public()
        
        # Prepare request
        url = f"{self.base_url}/v1/ticker"
        params = {
            'markets': symbol
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError(f"Expected non-empty list response, got {type(data)}")
            
            ticker_data = data[0]  # Upbit returns list with one item for single market
            
            try:
                trade_price = float(ticker_data['trade_price'])
                return trade_price
                
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"Invalid ticker data format: {e}")
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch ticker for {symbol}: {e}")
    
    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()