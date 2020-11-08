from finlab_crypto.strategy import Filter
import talib

@Filter(side='long', fast=5, slow=3, matype=0)
def stoch_filter(ohlcv):
    
    side = stoch_filter.side
    fast = stoch_filter.fast
    slow = stoch_filter.slow
    matype = stoch_filter.matype
    
    k, d = talib.STOCH(ohlcv.high, ohlcv.low, ohlcv.close, 
                fastk_period=fast, slowk_period=slow, slowk_matype=matype, 
                slowd_period=slow, slowd_matype=matype)
    
    signals = k > d if side == 'long' else k < d
    
    fig = {
        'figures': {
            'kd': {'k': k, 'd': d}
        }
    }
    
    return signals, fig