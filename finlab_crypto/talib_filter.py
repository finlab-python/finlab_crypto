from talib import abstract
from finlab_crypto.strategy import Strategy, Filter

def TalibFilter(talib_function_name, condition=None):
    f = getattr(abstract, talib_function_name)

    @Filter(condition=condition, **f.parameters)
    def ret(ohlcv):
        parameters = {pn: int(getattr(ret, pn)) for pn in f.parameters.keys()}
        o = f(ohlcv, **parameters)
        
        signals = ret.condition(ohlcv, o)
        
        figures = {}
        group = 'overlaps' if f.info['group'] == 'Overlap Studies' else 'figures'
        figures[group] = {f.info['name']: o}
        
        return signals, figures
    return ret