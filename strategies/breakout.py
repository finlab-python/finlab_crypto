from finlab_crypto.strategy import Strategy


@Strategy(long_window=30, short_window=30)
def breakout_strategy(ohlcv):
    lw = breakout_strategy.long_window
    sw = breakout_strategy.short_window
    
    ub = ohlcv.close.rolling(lw).max()
    lb = ohlcv.close.rolling(sw).min()
    
    entries = ohlcv.close == ub
    exits = ohlcv.close == lb
    
    figures = {
        'overlaps': {
            'ub': ub,
            'lb': lb
        }
    }
    
    return entries, exits, figures