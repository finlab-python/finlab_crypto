from finlab_crypto.talib_strategy import TalibStrategy

macd_strategy = TalibStrategy('MACD', 
                             entries=lambda ohlcv, macd: macd.macdhist > 0, 
                             exits=lambda ohlcv, macd: macd.macdhist < 0)