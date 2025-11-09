# Quick Start Guide

## Installation

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/Deokho1/adaptive-trading-bot.git
   cd adaptive-trading-bot
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Upbit API keys
   ```

## Running the Bot

### Dry-run Mode (Safe Testing)

```bash
# Make sure dry_run is set to true in config.yaml or .env
python main.py
```

The bot will simulate trading with virtual money (starts with 1,000,000 KRW).

### Live Trading Mode

⚠️ **Use with caution! Test thoroughly in dry-run mode first.**

1. Edit `.env`:
   ```
   DRY_RUN=False
   ```

2. Or edit `config.yaml`:
   ```yaml
   trading:
     dry_run: false
   ```

3. Run:
   ```bash
   python main.py
   ```

## Understanding the Output

```
2025-11-09 12:00:10 - Market Analysis - ADX: 32.45, BB Width: 0.0423, ATR: 1250000.50
2025-11-09 12:00:10 - Market Condition: TREND (High ADX and ATR)
2025-11-09 12:00:10 - Trend Strategy: BUY signal at 65000000.0
```

- **ADX > 25**: Trending market → Uses Volatility Breakout strategy
- **ADX < 25**: Ranging market → Uses RSI Mean Reversion strategy
- **BB Width**: Bollinger Band width indicates volatility
- **ATR**: Average True Range shows price movement

## Strategy Behavior

### Trend Mode (Volatility Breakout)
- Activated when: ADX > 25 and volatility is high
- Buy when: Price > yesterday_close + k × (yesterday_high - yesterday_low)
- Default k value: 0.5

### Range Mode (RSI Mean Reversion)
- Activated when: ADX < 25 or volatility is low
- Buy when: RSI < 30 (oversold)
- Sell when: RSI > 70 (overbought)

## Customization

### Change Trading Pair

Edit `config.yaml`:
```yaml
trading:
  ticker: "KRW-ETH"  # Or KRW-XRP, KRW-DOGE, etc.
```

### Adjust Risk Settings

Edit `config.yaml`:
```yaml
risk:
  stop_loss_pct: 0.03      # 3% stop loss
  take_profit_pct: 0.15    # 15% take profit
```

### Modify Strategy Parameters

For Trend Strategy:
```yaml
strategy:
  trend:
    k_value: 0.3  # More conservative (0.3-0.5 recommended)
```

For Range Strategy:
```yaml
strategy:
  range:
    rsi_oversold: 25   # More aggressive entry
    rsi_overbought: 75  # More aggressive exit
```

## Monitoring

### View Live Logs
```bash
tail -f logs/trading_bot.log
```

### Check Position Status
The bot creates `positions.json` which tracks:
- Current position details
- Entry price and time
- Trade history
- Performance metrics

### Stop the Bot
Press `Ctrl+C` to gracefully stop the bot. It will:
- Display performance summary
- Save current position state
- Exit cleanly

## Performance Metrics

When you stop the bot:
```
Performance Summary:
  Total Trades: 15
  Winning Trades: 10
  Losing Trades: 5
  Win Rate: 66.67%
  Total Profit: 125,000.00 KRW
  Final Balance: 1,125,000.00 KRW
```

## Tips for Success

1. **Start with Dry-run**: Test for at least a week before going live
2. **Monitor ADX**: High ADX means trending, low means ranging
3. **Adjust Parameters**: Backtest different k_value and RSI thresholds
4. **Set Reasonable Limits**: Don't make stop_loss too tight or take_profit too wide
5. **Check Logs**: Review why trades were made to understand bot behavior
6. **Paper Trading**: Run dry-run parallel to live trading to compare

## Troubleshooting

### "Insufficient market data"
- Normal on first run, bot needs historical data
- Wait 1-2 minutes for data to load

### "Upbit API error"
- Check your API keys in `.env`
- Ensure API keys have trading permissions
- Verify rate limits aren't exceeded

### Bot not making trades
- Check market conditions (may not meet strategy criteria)
- Review log file for detailed decision reasoning
- Verify balance is sufficient (minimum 5000 KRW)

## Advanced Usage

### Run as a Background Service

Using `screen`:
```bash
screen -S trading-bot
python main.py
# Press Ctrl+A then D to detach
# screen -r trading-bot to reattach
```

Using `nohup`:
```bash
nohup python main.py > bot.log 2>&1 &
```

### Multiple Trading Pairs

Run separate instances with different config files:
```bash
python main.py --config config_btc.yaml
python main.py --config config_eth.yaml
```

(Note: You'll need to modify main.py to accept config argument)
