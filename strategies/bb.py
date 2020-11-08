import talib
import numpy as np
import pandas as pd
from finlab_crypto.strategy import Strategy

@Strategy(window=14, nstd=2)
def bb_strategy(ohlcv):
  
  window = bb_strategy.window
  nstd = bb_strategy.nstd
  
  mean = ohlcv.close.rolling(window).mean()
  std = ohlcv.close.rolling(window).std()
  
  up = mean + nstd * std
  dn = mean - nstd * std

  entries = ohlcv.close > up
  exits = ohlcv.close < dn

  figure = {
    'overlaps': {
        'up': mean + nstd * std,
        'dn': mean - nstd * std,
    }
  }
  return entries, exits, figure