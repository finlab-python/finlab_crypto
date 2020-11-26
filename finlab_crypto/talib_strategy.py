from finlab_crypto.strategy import Strategy, Filter
import pandas as pd
import numpy as np

def TalibStrategy(talib_function_name, entries, exits):
    """A strategy factory that makes strategies using talib indicator.

    Args:
      talib_function_name:
        A str of technical indicator function name in talib mudule.
      entries:
        A function that transfer indicator series to boolean signals (ex: lambda ohlcv, ma: ohlcv.close > ma)
      exits:
        A function that transfer indicator series to boolean signals (ex: lambda ohlcv, ma: ohlcv.close < ma)

    Returns:
      entries:
        A dataframe of entries point time series after add talib strategy function.
      exits:
        A dataframe of exits point time series after add talib strategy function.
      figures:
        A dict of tuple with filter signal dataframe and figures data.

    """
    from talib import abstract
    import talib
    f = getattr(abstract, talib_function_name)
    ff = getattr(talib, talib_function_name)

    @Strategy(entries=entries, exits=exits, **f.parameters)
    def ret(ohlcv):
        parameters = {pn: getattr(ret, pn) for pn, val in f.parameters.items()}
        try:
            o = f(ohlcv, **parameters)
        except:
            o = ff(ohlcv.close, **parameters)
            if isinstance(o, list) or isinstance(o, tuple):
                o = pd.DataFrame(np.array(o).T, index=ohlcv.index, columns=f.output_names)

        if isinstance(o, np.ndarray):
            o = pd.Series(o, index=ohlcv.index)

        entries = ret.entries(ohlcv, o)
        exits = ret.exits(ohlcv, o)

        figures = {}
        group = 'overlaps' if f.info['group'] == 'Overlap Studies' else 'figures'
        if group == 'overlaps' and isinstance(o, pd.DataFrame):
            figures['overlaps'] = {}
            for sname, s in o.items():
                figures['overlaps'][sname] = s

        else:
            figures[group] = {f.info['name']: o}

        return entries, exits, figures
    return ret
