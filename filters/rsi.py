from finlab_crypto.talib_filter import TalibFilter
rsi_filter = TalibFilter("RSI", lambda ohlcv, rsi: rsi > 50)