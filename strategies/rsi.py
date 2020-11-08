import talib
import numpy as np
import pandas as pd
from finlab_crypto.strategy import Strategy

@Strategy(timeperiod=14, buy_threshold=52, sell_threshold=50)
def rsi_strategy(ohlcv):
  
  rsi = talib.RSI(ohlcv.close, timeperiod=rsi_strategy.timeperiod)
  
  entries = (rsi > rsi_strategy.buy_threshold) & (rsi.shift() < rsi_strategy.buy_threshold)
  exits = (rsi < rsi_strategy.sell_threshold) & (rsi.shift() > rsi_strategy.sell_threshold)

  figure = {
    'figures': {
        str(rsi_strategy.timeperiod) + '_rsi': rsi
    }
  }

  return entries, exits, figure
