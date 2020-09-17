from finlab_crypto.strategy import Strategy

@Strategy(sma1=50, sma2=1000, series=None)
def diff_strategy(ohlcv):
    
    series = diff_strategy.series
    sma1 = series.rolling(diff_strategy.sma1).mean()
    sma2 = series.rolling(diff_strategy.sma2).mean()
    
    entries = (sma1 < sma2) & (sma1.shift() > sma2.shift())
    exits = (sma1 > sma2) & (sma1.shift() < sma2.shift())
    
    figures = {
        'figures':{
            'indicator': {
                'value': series,
                'sma1': sma1,
                'sma2': sma2
            }
        }
    }

    return entries, exits, figures