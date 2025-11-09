# Adaptive Dual-Mode Crypto Trading Bot

**시장 상황에 맞게 전략을 자동으로 바꿔가며 코인을 자동으로 사고파는 봇**

An intelligent cryptocurrency trading bot that automatically switches between two trading strategies based on real-time market analysis using Upbit Open API.

## 🚀 Features

### Dual-Mode Trading Strategies
- **Trend Mode (Volatility Breakout)**: Active when market shows strong directional movement
  - Uses Larry Williams' Volatility Breakout strategy
  - Buys when price breaks above: `yesterday_close + k * (yesterday_high - yesterday_low)`
  - Optimal for trending markets with high ADX

- **Range Mode (RSI Mean Reversion)**: Active during sideways market conditions
  - Uses RSI (Relative Strength Index) for entry/exit signals
  - Buys when RSI < 30 (oversold)
  - Sells when RSI > 70 (overbought)
  - Optimal for ranging markets with low volatility

### Automatic Strategy Switching
The bot analyzes market conditions in real-time using:
- **ADX (Average Directional Index)**: Measures trend strength
- **ATR (Average True Range)**: Measures volatility
- **Bollinger Bands**: Identifies price range and volatility

When ADX > 25 and volatility is high → **Trend Strategy**  
When ADX < 25 or volatility is low → **Range Strategy**

### Modular Architecture
- **MarketAnalyzer**: Calculates technical indicators (ADX, ATR, Bollinger Bands, RSI)
- **StrategyManager**: Implements and switches between trading strategies
- **RiskManager**: Manages position sizing, stop-loss, and take-profit
- **ExecutionEngine**: Handles order execution via Upbit API
- **RateLimiter**: Ensures compliance with API rate limits
- **PositionTracker**: Tracks positions and maintains trade history

### Risk Management
- Position sizing based on available capital
- Stop-loss protection (default: 5%)
- Take-profit targets (default: 10%)
- Maximum position size limits
- Order validation before execution

### Additional Features
- **Dry-run simulation mode**: Test strategies without risking real money
- **Comprehensive logging**: Track all decisions and trades
- **Position persistence**: Maintains state across restarts
- **Rate limiting**: Respects Upbit API limits (8 req/sec, 200 req/min)
- **Performance tracking**: Win rate, total profit, trade history

## 📋 Requirements

- Python 3.8+
- Upbit account (for live trading)
- API keys from Upbit (for live trading)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Deokho1/adaptive-trading-bot.git
   cd adaptive-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Upbit API credentials:
   ```
   UPBIT_ACCESS_KEY=your_access_key_here
   UPBIT_SECRET_KEY=your_secret_key_here
   DRY_RUN=True
   LOG_LEVEL=INFO
   ```

4. **Configure the bot** (optional)
   
   Edit `config.yaml` to customize:
   - Trading pair (default: KRW-BTC)
   - Strategy parameters
   - Risk management settings
   - Check interval

## 🎯 Usage

### Dry-run Mode (Recommended for Testing)

Start the bot in simulation mode:
```bash
python main.py
```

The bot will:
- Simulate trades without using real money
- Start with 1,000,000 KRW virtual balance
- Log all decisions and trades
- Track performance metrics

### Live Trading Mode

⚠️ **WARNING**: Only use live mode after thoroughly testing in dry-run mode!

1. Set `DRY_RUN=False` in `.env` or set `dry_run: false` in `config.yaml`
2. Ensure your Upbit API keys are configured
3. Start the bot:
   ```bash
   python main.py
   ```

### Monitoring

The bot logs to:
- Console output (real-time)
- `logs/trading_bot.log` file

Monitor the bot's decisions:
```bash
tail -f logs/trading_bot.log
```

## 📊 Configuration

### Trading Settings (`config.yaml`)

```yaml
trading:
  ticker: "KRW-BTC"        # Trading pair
  investment_ratio: 0.95   # Use 95% of available KRW
  dry_run: true            # Simulation mode
  check_interval: 10       # Check market every 10 seconds
```

### Strategy Parameters

```yaml
strategy:
  trend:
    k_value: 0.5           # Volatility breakout coefficient
  
  range:
    rsi_period: 14         # RSI calculation period
    rsi_oversold: 30       # Buy when RSI < 30
    rsi_overbought: 70     # Sell when RSI > 70
  
  market_analysis:
    adx_period: 14         # ADX calculation period
    adx_threshold: 25      # Trend if ADX > 25
    atr_period: 14         # ATR calculation period
    bb_period: 20          # Bollinger Bands period
    bb_std: 2              # Bollinger Bands standard deviation
```

### Risk Management

```yaml
risk:
  max_position_size: 0.95  # Max 95% of portfolio
  stop_loss_pct: 0.05      # 5% stop loss
  take_profit_pct: 0.10    # 10% take profit
```

## 🏗️ Project Structure

```
adaptive-trading-bot/
├── main.py                 # Entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── bot.py             # Main bot orchestrator
│   ├── core/              # Core trading modules
│   │   ├── __init__.py
│   │   ├── market_analyzer.py      # Technical analysis
│   │   ├── strategy_manager.py     # Trading strategies
│   │   ├── risk_manager.py         # Risk management
│   │   └── execution_engine.py     # Order execution
│   └── utils/             # Utility modules
│       ├── __init__.py
│       ├── rate_limiter.py         # API rate limiting
│       ├── position_tracker.py     # Position tracking
│       └── logger.py               # Logging setup
└── logs/                  # Log files (created automatically)
```

## 🔍 How It Works

1. **Market Analysis**: Every cycle, the bot fetches OHLCV data and calculates technical indicators
2. **Strategy Selection**: Based on ADX and ATR, determines if market is trending or ranging
3. **Signal Generation**: Active strategy generates buy/sell/hold signals
4. **Risk Check**: Validates signals against risk management rules
5. **Order Execution**: Places orders via Upbit API (or simulates in dry-run mode)
6. **Position Monitoring**: Tracks positions and checks stop-loss/take-profit conditions
7. **Performance Tracking**: Records all trades and calculates performance metrics

## 📈 Example Output

```
============================================================
Adaptive Dual-Mode Crypto Trading Bot Starting
============================================================
2025-11-09 12:00:00 - Bot initialized - Ticker: KRW-BTC, Dry Run: True
2025-11-09 12:00:10 - Market Analysis - ADX: 32.45, BB Width: 0.0423, ATR: 1250000.50
2025-11-09 12:00:10 - Market Condition: TREND (High ADX and ATR)
2025-11-09 12:00:10 - Trend Strategy: BUY signal at 65000000.0 (target: 64500000.0)
2025-11-09 12:00:10 - ✓ BUY ORDER EXECUTED: 0.01461538 at 65000000.00 KRW
2025-11-09 12:10:10 - Current Price: 65500000.00 KRW | Strategy: trend | Signal: hold
2025-11-09 12:10:10 - Position P/L: 0.77% | Stop Loss: 61750000.00 | Take Profit: 71500000.00
```

## ⚠️ Disclaimer

This trading bot is for educational purposes. Cryptocurrency trading carries significant risk:
- Past performance does not guarantee future results
- You may lose all invested capital
- Only invest what you can afford to lose
- Test thoroughly in dry-run mode before live trading
- The authors are not responsible for any financial losses

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Trading! 📊💰**
