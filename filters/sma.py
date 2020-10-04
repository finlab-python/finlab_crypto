from finlab_crypto.talib_filter import TalibFilter
sma_filter = TalibFilter("SMA", condition=lambda ohlcv, sma: ohlcv.close > sma)
