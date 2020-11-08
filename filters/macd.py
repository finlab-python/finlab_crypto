from finlab_crypto.talib_filter import TalibFilter
macd_filter = TalibFilter("MACD", condition=lambda ohlcv, macd, params: macd.macdhist > 0, )
