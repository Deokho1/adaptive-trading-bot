import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_loader import load_config
from core.logger import setup_logger
from exchange.rate_limiter import RateLimiter
from exchange.upbit_client import UpbitClient

def main() -> None:
    config = load_config()
    logger = setup_logger(
        config["persistence"]["logs_dir"],
        config["persistence"]["log_level"],
    )

    rl = RateLimiter(
        max_calls_per_sec_public=config["exchange"]["public_rate_limit"]["max_calls_per_sec"],
        max_calls_per_sec_private=config["exchange"]["private_rate_limit"]["max_calls_per_sec"],
    )

    client = UpbitClient(
        base_url=config["exchange"]["base_url"],
        rate_limiter=rl,
    )

    candles = client.get_candles_4h("KRW-BTC", count=10)
    logger.info(f"Fetched {len(candles)} candles for KRW-BTC, last close={candles[0].close if candles else 'N/A'}")

    price = client.get_ticker_price("KRW-BTC")
    logger.info(f"Current KRW-BTC price={price}")

if __name__ == "__main__":
    main()