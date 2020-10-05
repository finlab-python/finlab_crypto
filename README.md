# A backtesting framework for crytpo currency

## Todo
* add portfolio backtest

## Updates
Version 0.1.1
* fix talib error
* add filters folder
* add excluded assets when sync portfolio
* add filter folder to setup

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
