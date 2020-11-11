from finlab_crypto.talib_filter import TalibFilter
rsi_filter = TalibFilter("RSI", 
                         lambda ohlcv, rsi, params: rsi > params['threshold'],
                         threshold=50)