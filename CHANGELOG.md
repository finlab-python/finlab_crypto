# Changelog

## 0.3.0

Works again on Google Colab and current Python stacks (Python 3.10 – 3.13, numpy 2, pandas 2).

### Changed
* Klines, exchange info and ticker prices come from Binance's public market-data endpoint
  `data-api.binance.vision`: no API key, and no HTTP 451 on US machines such as Colab.
  `get_all_binance(symbol, freq)` keeps its signature and return format; downloads are
  paginated with retries and cached in `./history/`.
* `import finlab_crypto` no longer needs `python-binance` (it used to contact
  `api.binance.com` at import time). Trading needs `pip install "finlab_crypto[trade]"`.
* `TradingPortfolio()` without API keys runs as a dry run: orders are computed and
  printed, never sent. A clear error explains HTTP 451 when trading from a US IP.
* `setup()` only creates `./history/`; it no longer mounts Google Drive.
* TA-Lib is a regular dependency (`ta-lib>=0.6` binary wheels); `talib-binary` is no longer needed.
* Interactive charts render in an iframe, so they work in Colab, JupyterLab and VS Code,
  and `html=` saves a standalone page.
* Packaging moved to `pyproject.toml`; tests moved to pytest and GitHub Actions.

### Added
* `finlab_crypto.onchain`: Coin Metrics Community metrics (hash rate, issuance, exchange
  flows...), SOPR from BGeometrics, and `align()` to put daily metrics on a price index
  without look-ahead. Replaces `GlassnodeClient`, whose free API is gone.
* `crawler.get_exchange_info()` and `crawler.get_symbol_tickers()`.

### Fixed
* numpy 2: optimisation columns were labelled through `eval(str(...))`, which breaks on
  `np.int64(20)`; numpy scalars are also converted before reaching TA-Lib.
* pandas 2: `DataFrame.pivot` positional arguments, `groupby(axis=1)`, positional
  `Series[int]` access and silent downcasting.
* Python 3.11+: `inspect.getargspec` in `TalibFilter`; scipy 1.13+: `signal.gaussian` in `alma`.
* `ts_stop` now means a trailing stop (vectorbt `sl_stop` + `sl_trail=True`); stop columns
  are labelled `sl_stop`, `tp_stop`, `ts_stop`.
* `execution_price='open'` trades at the next bar's open instead of the signal bar's open.
* `CSCV(n_bins=...)` used 10 bins regardless of `n_bins`.
* `calculate_position_size` no longer grows a shared default `excluded_assets` list.
* `TradingPortfolio.status` plots the equity curves again.

### Removed
* `crawler.get_all_bitmex`, `crawler.minutes_of_new_data`, `crawler.GlassnodeClient`.
* The `client` argument of `get_all_binance` / `get_nbars_binance` (public data needs no client).
* `quote_assets` argument of `portfolio_backtest` (quote assets come from exchange info).

## 0.2.27
* support new version of pandas 3.8
* support 3.8 syntax

## 0.2.26
* fix dependency of python 3.10

## 0.2.25

## 0.2.24
* fix dependency in colab

## 0.2.23
* fix binance min_notional not found

## 0.2.22
* fix compatability of seaborn in cscv algorithm plotting

## 0.2.21
* fix pyecharts compatibility
* set the default argument `client` of get_nbars_binance

## 0.2.20
* fix get_all_binance last candle not updated

## 0.2.19
* fix bar color

## 0.2.18
* fix stop loss and take profit and add them into tests

## 0.2.17
* update vectorbt version

## 0.2.16
* update pandas version

## 0.2.15
* fix tp.portfolio_backtest

## 0.2.14
* add `execute_before_candle_complete`
* add `weight` and `weight_unit` for `TradingMethod`

## 0.2.12
* fix numba version

## 0.2.11

## 0.2.10
* fix numpy version

## 0.2.8
* merge transactions to reduce fees

## 0.2.7
* fix test error (request binance api too fast)
* add USDC as base stable coin (tp.set_default_stable_coin('USDC'))

## 0.2.6
* fix version of pandas==1.1.5, since pandas==1.2.0 is not compatable with vectorbt
* fix show_parameters function in Strategy and Filter

## 0.2.5
* fix weight_btc error
* fix strategy mutable input

## 0.2.4
* fix entry price online.py

## 0.2.3
* fix execution price issue

## stable
* improve syntax
* add execution price for the strategy

## 0.2.1
* fix vectorbt version

## 0.2.0
* update vectorbt to 0.14.4

## 0.1.19
* refactor(strategy.py): refactor strategy
* refactor(cscv.py): refactor cscv
* add cscv_nbins and cscv_objective to strategy.backtest
* add bitmex support

## 0.1.18
* fix(crawler): get_n_bars
* fix(TradingPortfolio): get_ohlcv
* fix(TradingPortfolio): portfolio_backtest

## 0.1.17
* fix error for latest_signal asset_btc_value
* add unittest for latest_signal

## 0.1.16
* fix web page error
* fix error for zero orders

## 0.1.15
* fix web page error

## 0.1.14
* refine render_html function

## 0.1.13
* refine display html for TradingPortfolio

## 0.1.12
* add delay when portfolio backtesting
* fix colab compatability
* improve interface of TradingPortfolio

## 0.1.11
* fix portfolio backtest error
* add last date equity for backtest

## 0.1.10
* add portfolio backtest
* rename online.py functions
* refactor error tolerance of different position in online.py functions
* set usdt to excluded asset when calculate position size

## 0.1.9
* set 'filters' as an optional argument on TradingMethod
* set plot range dynamically
* portfolio backtest

## 0.1.8
* fix talib parameter type incompatable issue

## 0.1.7
* fix talib parameter type incompatable issue

## 0.1.6
* fix talib-binary compatable issue using talib_strategy or talib_filter

## 0.1.5
* add filters to online.py
* add lambda argument options to talib_filter
* move talib_filter to finlab_crypto package

## 0.1.4
* fix talib filter and strategy pandas import error
* fix talib import error in indicators, talib_strategy, and talib_filter

## 0.1.3
* remove progress bar when only single strategy is backtested
* adjust online portfolio to support leaverge
* new theme for overfitting plots
* fix online order with zero order amount
* fix SD2 for overfitting plots

## 0.1.2
* fix strategy variables

## 0.1.1
* fix talib error
* add filters folder
* add excluded assets when sync portfolio
* add filter folder to setup
* fix variable eval failure

## 0.1.0
* add filter interface
* add talib strategy wrapper
* add talib filter wrapper

## 0.0.9.dev1
* vectorbt heatmap redesign
* improve optimization plots
* redesign strategy interface
* add new function setup, to replace setup_colab

## 0.0.8.dev1
* fix transaction duplicate bug

## 0.0.7.dev1
* fix bugs of zero transaction

## 0.0.6.dev1
* fix latest signal
* rename strategy.recent_signal
* restructure rebalance function in online.py

## 0.0.5.dev1
* add init module
* add colab setup function
* set vectorbt default
* fix crawler duplicated index

## 0.0.4.dev1
* add seaborn to dependencies
* remove talib-binary from dependencies
* fix padding style

## 0.0.3.dev1
* remove logs when calculating portfolio
* add render html to show final portfolio changes
* add button in html to place real trade with google cloud function

## 0.0.2.dev1
* skip heatmap if it is broken
* add portfolio strategies
* add talib dependency
