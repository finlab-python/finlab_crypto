import talib
import numpy as np
import pandas as pd
from finlab_crypto.strategy import Strategy

@Strategy(bb_sma_length=20, std_dev_mult=2)
def bb_strategy(ohlcv):
  
  bb_sma_length = bb_strategy.bb_sma_length
  std_dev_mult = bb_strategy.std_dev_mult
  
  bb_sma_values = ohlcv.close.rolling(bb_sma_length).mean()
  std_dev_value = ohlcv.close.rolling(bb_sma_length).std()
  
  upperBB_values = bb_sma_values + std_dev_mult * std_dev_value
  lowerBB_values = bb_sma_values - std_dev_mult * std_dev_value

  entries = ohlcv.close >= upperBB_values
  exits = ohlcv.close <= lowerBB_values
 
  #make chart
  figure = {
      'overlaps': {
           'BB SMA': bb_sma_values,
           'Upper BB': upperBB_values,
           'Lower BB': lowerBB_values,
      }
  }

  return entries, exits, figure