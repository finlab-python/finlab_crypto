from finlab_crypto.strategy import Strategy

@Strategy(up_window=21, dn_window=144)
def breakout_strategy(ohlcv):
    
    v1 = breakout_strategy.up_window
    v2 = breakout_strategy.dn_window
    
    up = ohlcv.open.rolling(v1).max()
    dn = ohlcv.open.rolling(v2).min()
  
    entries = (ohlcv.close >= up) & (ohlcv.close.shift() < up.shift())
    exits = (ohlcv.close <= dn) & (ohlcv.close.shift() > dn.shift())
  
    figure = {
        'overlaps': {
             str(v1) + 'max': up,
             str(v2) + 'min': dn,
        }
    }

    return entries, exits, figure