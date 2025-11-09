# Development Guide

## Project Structure

```
adaptive-trading-bot/
├── src/                    # Source code
│   ├── core/              # Core trading modules
│   │   ├── market_analyzer.py     # Technical analysis
│   │   ├── strategy_manager.py    # Trading strategies
│   │   ├── risk_manager.py        # Risk management
│   │   └── execution_engine.py    # Order execution
│   ├── utils/             # Utility modules
│   │   ├── rate_limiter.py        # API rate limiting
│   │   ├── position_tracker.py    # Position tracking
│   │   └── logger.py              # Logging setup
│   └── bot.py             # Main bot orchestrator
├── tests/                 # Test files
├── config.yaml            # Configuration
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## Running Tests

```bash
# Run all component tests
python tests/test_components.py

# Quick bot startup test
python tests/test_bot.py
```

## Code Architecture

### Core Modules

**MarketAnalyzer**
- Calculates technical indicators (RSI, ADX, ATR, Bollinger Bands)
- Determines market conditions (trending vs ranging)
- Pure calculation logic, no state

**StrategyManager**
- Implements trading strategies
- Generates buy/sell/hold signals
- Switches between strategies based on market conditions
- Maintains strategy state

**RiskManager**
- Position sizing calculations
- Stop-loss and take-profit checks
- Order validation
- Risk metrics calculation

**ExecutionEngine**
- Interfaces with Upbit API
- Executes buy/sell orders
- Handles both live and dry-run modes
- Manages simulated balances

### Utility Modules

**RateLimiter**
- Enforces API rate limits
- Prevents exceeding Upbit's limits (8/sec, 200/min)
- Can be used as decorator or called directly

**PositionTracker**
- Tracks current position
- Maintains trade history
- Persists state to JSON file
- Calculates performance metrics

**Logger**
- Configures logging for the bot
- Supports both console and file logging
- Configurable log levels

## Adding a New Strategy

1. Edit `src/core/strategy_manager.py`
2. Add a new method for your strategy:
   ```python
   def my_new_strategy_signal(self, df, current_price, has_position):
       # Your strategy logic here
       return signal, details
   ```
3. Update `get_trading_signal()` to include your strategy
4. Add tests in `tests/test_components.py`

## Adding a New Indicator

1. Edit `src/core/market_analyzer.py`
2. Add a new calculation method:
   ```python
   def calculate_my_indicator(self, df):
       # Your indicator calculation
       return indicator_series
   ```
3. Use in strategy or market analysis as needed

## Modifying Risk Rules

Edit `src/core/risk_manager.py`:
- `calculate_position_size()` - Change position sizing logic
- `check_stop_loss()` - Modify stop-loss conditions
- `check_take_profit()` - Modify take-profit conditions
- Add new risk checks as needed

## Configuration

All settings are in `config.yaml`:
- Trading parameters (ticker, intervals, investment ratio)
- Strategy parameters (k_value, RSI thresholds, ADX thresholds)
- Risk management (stop-loss, take-profit, position limits)
- Rate limiting (API call limits)
- Logging (level, file location)

## Debugging

### Enable Debug Logging

Edit `config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

Or set in `.env`:
```
LOG_LEVEL=DEBUG
```

### Common Issues

**Bot not trading:**
- Check logs for decision reasoning
- Verify market data is being fetched
- Check if signals are being generated
- Verify balance is sufficient

**API errors:**
- Check rate limiting
- Verify API credentials
- Check Upbit service status

**Import errors:**
- Ensure all dependencies are installed
- Check Python version (3.8+)
- Verify directory structure

## Testing Strategies

### Backtesting (Manual)

1. Get historical data
2. Create a test script that feeds data to strategies
3. Track hypothetical trades
4. Calculate performance metrics

### Forward Testing

1. Run in dry-run mode
2. Monitor for extended period (1+ weeks)
3. Analyze trade decisions
4. Compare with live market

### Paper Trading

1. Run bot in dry-run mode
2. Manually verify each decision is sound
3. Track performance vs real market
4. Adjust parameters as needed

## Contributing

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all functions/classes
- Keep functions focused and small

### Adding Features

1. Create a new branch
2. Implement feature with tests
3. Update documentation
4. Test thoroughly in dry-run mode
5. Submit pull request

### Reporting Issues

Include:
- Bot version
- Configuration (sanitized)
- Log excerpt showing issue
- Steps to reproduce
- Expected vs actual behavior

## Performance Optimization

### Database Integration

Consider adding a database for:
- Trade history
- Performance analytics
- Backtesting results

### Caching

Add caching for:
- Market data (avoid repeated API calls)
- Indicator calculations (reuse when possible)

### Parallel Processing

For multiple trading pairs:
- Run separate bot instances
- Use message queue for coordination
- Share rate limiter across instances

## Security Best Practices

1. **Never commit API keys**
   - Use `.env` for secrets
   - Keep `.env` in `.gitignore`

2. **Validate all inputs**
   - Check order amounts
   - Verify prices are reasonable
   - Validate configuration values

3. **Use dry-run first**
   - Test all changes in simulation
   - Verify behavior before going live

4. **Monitor regularly**
   - Check logs frequently
   - Set up alerts for errors
   - Review trades regularly

5. **Limit exposure**
   - Don't invest more than you can lose
   - Use stop-loss protection
   - Diversify across strategies/pairs

## Future Enhancements

Possible improvements:
- [ ] Web dashboard for monitoring
- [ ] Telegram notifications
- [ ] Backtesting framework
- [ ] Multiple trading pair support
- [ ] Machine learning for parameter optimization
- [ ] Advanced order types (limit, stop-limit)
- [ ] Portfolio rebalancing
- [ ] Tax reporting
- [ ] Performance analytics dashboard
