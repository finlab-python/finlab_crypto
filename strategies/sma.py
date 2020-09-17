from finlab_crypto.strategy import Strategy

@Strategy(sma1=21, sma2=144)
def sma_strategy(ohlcv):
    
    v1 = sma_strategy.sma1
    v2 = sma_strategy.sma2
    
    sma1 = ohlcv.open.rolling(v1).mean()
    sma2 = ohlcv.open.rolling(v2).mean()
  
    entries = (sma1 > sma2) & (sma1.shift() < sma2.shift())
    exits = (sma1 < sma2) & (sma1.shift() > sma2.shift())
  
    figure = {
        'overlaps': {
             str(v1) + 'ma': sma1,
             str(v2) + 'ma': sma2,
        }
    }
  
    return entries, exits, figure