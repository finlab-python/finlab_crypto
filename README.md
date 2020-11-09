# A backtesting framework for crytpo currency


## Todo
* comments in online.py
* add batch backtesting
* support shorting asset
* more tests

## Updates
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
