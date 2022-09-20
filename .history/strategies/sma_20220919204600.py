from finlab_crypto.strategy import Strategy

@Strategy(sma1_length=20, sma2_length=50)
def sma_strategy(ohlcv):
    
    sma1_length = sma_strategy.sma1_length
    sma2_length = sma_strategy.sma2_length
    
    sma1_values = ohlcv.open.rolling(sma1_length).mean()
    sma2_values = ohlcv.open.rolling(sma2_length).mean()
    
    # Compare current SMA values to the ones on the last candle to see if they've cross
    entries = (sma1_values > sma2_values) & (sma1_values.shift(1) <= sma2_values.shift(1))
    exits = (sma1_values < sma2_values) & (sma1_values.shift(1) >= sma2_values.shift(1))
  
    figure = {
        'overlaps': {
             str(sma1_length) + 'SMA': sma1_values,
             str(sma2_length) + 'SMA': sma2_values,
        }
    }
  
    return entries, exits, figure