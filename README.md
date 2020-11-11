# FinLab Crypto
[![Build Status](https://travis-ci.com/finlab-python/finlab_crypto.svg?branch=master)](https://travis-ci.com/finlab-python/finlab_crypto) [![PyPI version](https://badge.fury.io/py/finlab-crypto.svg)](https://badge.fury.io/py/finlab-crypto) [![codecov](https://codecov.io/gh/finlab-python/finlab_crypto/branch/master/graph/badge.svg?token=POS648UJ10)](https://codecov.io/gh/finlab-python/finlab_crypto)
A backtest and verification framework for cryptocurrency.
## Key Features
* Pandas vectorize backtest
* Talib wrapper to composite strategies easily
* Backtest visualization and analysis (uses vectorbt and pyecharts as backend)
* Analaysis the probability of overfitting using CSCV (combinatorially symmetric cross validation )
* Easy to deploy strategies on google cloud function
* Colab and Jupyter compatable
## Installation
```
pip install finlab_crypto
```
## Usage
### Setup Research Environment (Recommend)
Create directory `./history/` for saving historical data. If Colab notebook is detected, it creates `GoogleDrive/crypto_workspace/history` and link the folder to `./history/`.
``` python
import finlab_crypto
finlab_crypto.setup()
```
### Get Historical Price
``` python
ohlcv = finlab_crypto.crawler.get_all_binance('BTCUSDT', '4h')
ohlcv.head()
```
### Trading Strategy
``` python
@finlab_crypto.Strategy(n1=20, n2=60)
def sma_strategy(ohlcv):
  n1 = sma_strategy.n1
  n2 = sma_strategy.n2
  
  sma1 = ohlcv.close.rolling(int(n1)).mean()
  sma2 = ohlcv.close.rolling(int(n2)).mean()
  return (sma1 > sma2), (sma1 < sma2)
```
### Backtest
``` python
# default fee and slipagge are 0.1% and 0.1%

vars =  {'n1': 20, 'n2': 60}
portfolio = sma_strategy.backtest(ohlcv, vars, freq='4h', plot=True)
```

### Optimization
``` python
import numpy as np
vars = {
  'n1': np.arange(10, 100, 5), 
  'n2': np.arange(10, 100, 5)
}
portfolio = sma_strategy.backtest(ohlcv, vars, freq='4h', plot=True)
```

## Todo
* comments in online.py
* add batch backtesting
* support shorting asset
* more tests

## Updates
Version 0.1.18
* fix(crawler): get_n_bars
* fix(TradingPortfolio): get_ohlcv
* fix(TradingPortfolio): portfolio_backtest

Version 0.1.17
* fix error for latest_signal asset_btc_value
* add unittest for latest_signal

Version 0.1.16
* fix web page error
* fix error for zero orders

Version 0.1.15
* fix web page error

Version 0.1.14
* refine render_html function

Version 0.1.13
* refine display html for TradingPortfolio

Version 0.1.12
* add delay when portfolio backtesting
* fix colab compatability
* improve interface of TradingPortfolio

Version 0.1.11
* fix portfolio backtest error
* add last date equity for backtest

Version 0.1.10
* add portfolio backtest
* rename online.py functions
* refactor error tolerance of different position in online.py functions
* set usdt to excluded asset when calculate position size

Version 0.1.9
* set 'filters' as an optional argument on TradingMethod
* set plot range dynamically
* portfolio backtest

Version 0.1.8
* fix talib parameter type incompatable issue

Version 0.1.7
* fix talib parameter type incompatable issue

Version 0.1.6
* fix talib-binary compatable issue using talib_strategy or talib_filter

Version 0.1.5
* add filters to online.py
* add lambda argument options to talib_filter
* move talib_filter to finlab_crypto package

Version 0.1.4
* fix talib filter and strategy pandas import error
* fix talib import error in indicators, talib_strategy, and talib_filter

Version 0.1.3
* remove progress bar when only single strategy is backtested
* adjust online portfolio to support leaverge
* new theme for overfitting plots
* fix online order with zero order amount
* fix SD2 for overfitting plots

Version 0.1.2
* fix strategy variables

Version 0.1.1
* fix talib error
* add filters folder
* add excluded assets when sync portfolio
* add filter folder to setup
* fix variable eval failure

Version 0.1.0
* add filter interface
* add talib strategy wrapper
* add talib filter wrapper

Version 0.0.9.dev1
* vectorbt heatmap redesign
* improve optimization plots
* redesign strategy interface
* add new function setup, to replace setup_colab

Version 0.0.8.dev1
* fix transaction duplicate bug

Version 0.0.7.dev1
* fix bugs of zero transaction

Version 0.0.6.dev1
* fix latest signal
* rename strategy.recent_signal
* restructure rebalance function in online.py

Version 0.0.5.dev1
* add init module
* add colab setup function
* set vectorbt default
* fix crawler duplicated index

Version 0.0.4.dev1
* add seaborn to dependencies
* remove talib-binary from dependencies
* fix padding style

Version 0.0.3.dev1
* remove logs when calculating portfolio
* add render html to show final portfolio changes
* add button in html to place real trade with google cloud function

Version 0.0.2.dev1
* skip heatmap if it is broken
* add portfolio strategies
* add talib dependency
