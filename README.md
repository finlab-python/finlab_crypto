<a href="https://github.com/finlab-python/finlab_crypto"><img src="https://i.ibb.co/qWCSkr9/Picture1.png" align="center" width="200px"></a>

[![tests](https://github.com/finlab-python/finlab_crypto/actions/workflows/tests.yml/badge.svg)](https://github.com/finlab-python/finlab_crypto/actions/workflows/tests.yml) [![PyPI version](https://badge.fury.io/py/finlab-crypto.svg)](https://badge.fury.io/py/finlab-crypto)

Develop, verify and deploy crypto trading strategies at a glance.

## Key Features
* One-line download of Binance history, cached locally, that also works on Google Colab and other US-hosted machines
* Pandas-vectorised backtests (uses [vectorbt](https://github.com/polakowo/vectorbt/) as backend)
* TA-Lib wrappers to compose strategies and filters in one line
* Interactive backtest charts and parameter-optimisation dashboards
* Probability of backtest overfitting ([combinatorially symmetric cross-validation](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf))
* Free on-chain metrics (hash rate, exchange flows, SOPR) without API keys
* Live trading on Binance spot, deployable to Google Cloud Functions
* Works on Colab, Jupyter and VS Code, Python 3.10 – 3.13
* Companion of the [Hahow online course](https://hahow.in/cr/crypto-python); its notebooks live in [hahow-crypto-course](https://github.com/finlab-python/hahow-crypto-course)

## Installation
```bash
pip install finlab_crypto              # research and backtesting
pip install "finlab_crypto[trade]"     # + live trading on Binance
```
TA-Lib ships as a binary wheel, no C library to install.

## Colab Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finlab-python/hahow-crypto-course/blob/main/4-1_backtest_platform.ipynb) basic example of backtesting and optimisation

## Usage
### Get Historical Price
``` python
import finlab_crypto

ohlcv = finlab_crypto.crawler.get_all_binance('BTCUSDT', '4h')
ohlcv.head()
```
Klines come from Binance's public market-data endpoint (`data-api.binance.vision`), which needs no API key and is not geo-restricted. They are cached in `./history/`, so later calls only download new candles.

### Trading Strategy
``` python
@finlab_crypto.Strategy(n1=20, n2=60)
def sma_strategy(ohlcv):
    sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
    sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
    return sma1 > sma2, sma1 < sma2
```
### Backtest
``` python
# default fee and slippage are 0.1% and 0.1%
portfolio = sma_strategy.backtest(ohlcv, {'n1': 20, 'n2': 60}, freq='4h', plot=True)
```
![image](https://media.giphy.com/media/tv4xpwJ3T1zJGV6Smj/giphy.gif)

Stop-loss (`sl_stop`), take-profit (`tp_stop`) and trailing-stop (`ts_stop`) exits are strategy variables too:
``` python
portfolio = sma_strategy.backtest(ohlcv, {'n1': 20, 'n2': 60, 'sl_stop': 0.05}, freq='4h')
```

### Optimization
``` python
import numpy as np

variables = {'n1': np.arange(10, 100, 5), 'n2': np.arange(10, 100, 5)}
portfolio = sma_strategy.backtest(ohlcv, variables, freq='4h', plot=True)
```
![cumulative returns](https://i.ibb.co/vxMV4yG/Screen-Shot-2020-11-23-at-9-49-06-AM.png)
![parameter performance](https://i.ibb.co/McrKYDc/Screen-Shot-2020-11-23-at-9-49-15-AM.png)
![parameter range view](https://i.ibb.co/q9d1YHG/Screen-Shot-2020-11-23-at-9-49-28-AM.png)

### TA-Lib strategies and filters
``` python
from finlab_crypto.talib_strategy import TalibStrategy
from finlab_crypto.talib_filter import TalibFilter

macd_strategy = TalibStrategy('MACD',
                              entries=lambda ohlcv, macd: macd.macdhist > 0,
                              exits=lambda ohlcv, macd: macd.macdhist < 0)
angle_filter = TalibFilter('LINEARREG_ANGLE', lambda ohlcv, angle: angle > 0)

macd_strategy.backtest(ohlcv, {'fastperiod': 12}, filters={'angle': angle_filter.create({'timeperiod': 14})},
                       freq='4h', plot=True)
```

### On-chain metrics
``` python
from finlab_crypto import onchain

hash_rate = onchain.coinmetrics(['HashRate']).HashRate   # Coin Metrics Community API
sopr = onchain.sopr()                                    # BGeometrics, latest ~4 years
hash_rate_4h = onchain.align(hash_rate, ohlcv.index)     # no look-ahead: known after the day closes
```

### Live Trading

To perform live trading of a strategy, the following 3 sections should be executed when any candle is complete.

Placing orders uses `api.binance.com`, which answers HTTP 451 to US IP addresses (including Google Colab). Run live trading from your own computer or a server outside the US (e.g. a Cloud Function in `asia-east1`). Without API keys, `TradingPortfolio` runs as a dry run: signals, positions and orders are computed but nothing is sent.

#### 1. Create TradingMethods
First, we need to encapsulate a strategy into `TradingMethod`
```py
from finlab_crypto.online import TradingMethod, TradingPortfolio, render_html

# create TradingMethod for live trading
tm_sma = TradingMethod(
    name='live-strategy-sma',
    symbols=['ADAUSDT', 'DOTBTC', 'ETHBTC'], freq='4h', lookback=1200,
    strategy=sma_strategy,
    variables=dict(n1=35, n2=105),
    weight=5000,
    weight_unit='USDT',
    execution_price='close',  # trade at close or open price
)
```

#### 2. register TradingMethods to TradingPortfolio
A `TradingPortfolio` can sync the virtual portfolio to your Binance trading account. A `TradingPortfolio` contains many `TradingMethod`s, which should be executed whenever any new candle is (going to) closed. You can decide when to rebalance the portfolio by giving `execute_before_candle_complete` when creating the `TradingPortfolio`:
* `execute_before_candle_complete=True`: rebalance right *before* a candle is closed (i.e. setting xx:59 for 1h frequency strategy), so you can execute orders faster than others. However, signal hazards may occur due to incomplete candles.
* `execute_before_candle_complete=False` (default): rebalance right *after* a candle is closed (i.e. setting xx:00 for 1h frequency strategy)

The above information is crucial to help `TradingPortfolio` decide whether to remove incomplete candles when generating trading signals or not. However, `TradingPortfolio` will *not* execute periodically for you. So, you should set up a crontab or cloud function to execute it.
We recommend you run the code by yourself before setting the crontab or cloud function.

```py
import os

tp = TradingPortfolio(os.environ.get('BINANCE_KEY'), os.environ.get('BINANCE_SECRET'),
                      execute_before_candle_complete=False)
tp.register(tm_sma)
# additional trading methods can be registered
# tp.register(tm1)
```

#### 3. view and execute orders
Finally, we could call `tp.get_ohlcvs()` to get history data of all trading assets and call `tp.get_latest_signals` to calculate the trading signals. The aggregate information is created using `tp.calculate_position_size`. All the information can be viewed by `render_html`.
```py
ohlcvs = tp.get_ohlcvs()
signals = tp.get_latest_signals(ohlcvs)
position, position_btc, new_orders = tp.calculate_position_size(signals)
```

If the result makes sense, use `tp.execute_orders` to sync the position of your real account. Please make an issue if there is any bug:
```py
# (order) mode can be either 'TEST', 'MARKET', 'LIMIT'
# TEST mode validates orders with Binance without executing them (or only prints them in a dry run).
order_results = tp.execute_orders(new_orders, mode='TEST')
render_html(signals, position, position_btc, new_orders, order_results)
```

## Testing

```bash
git clone https://github.com/finlab-python/finlab_crypto.git
cd finlab_crypto
pip install -e ".[test]"
pytest                  # all tests
pytest -m "not network" # offline tests only
```
No API key is needed: market data comes from public endpoints and trading tests run as a dry run.

## Release

```bash
python -m build
twine upload dist/*
```

See [CHANGELOG.md](CHANGELOG.md) for the version history.
