from finlab_crypto.strategy import Strategy, Filter
import inspect
import pandas as pd
import numpy as np

def TalibFilter(talib_function_name, condition=None, **additional_parameters):
    """A filter factory that makes filter using talib indicator.

    Args:
      talib_function_name:
        A str of technical indicator function name in talib mudule.
      condition:
        A function that transfer indicators to bool signals (ex: lambda ohlcv, ma: ohlcv.close > ma)
      **additional_parameters:
        other parameters for parameter optimization.

    Returns:
      signals:
        A dataframe of filter signals.
      figures:
        A dict of required data for figure display.
    """
    from talib import abstract
    import talib
    f = getattr(abstract, talib_function_name)
    ff = getattr(talib, talib_function_name)

    @Filter(condition=condition, **f.parameters, additional_parameters=additional_parameters)
    def ret(ohlcv):
        parameters = {pn: (getattr(ret, pn)) for pn, val in f.parameters.items()}
        try:
            o = f(ohlcv, **parameters)
        except:
            o = ff(ohlcv.close, **parameters)
            if isinstance(o, list) or isinstance(o, tuple):
                o = pd.DataFrame(np.array(o).T, index=ohlcv.index, columns=f.output_names)

        if isinstance(o, np.ndarray):
            o = pd.Series(o, index=ohlcv.index)

        if len(inspect.getargspec(ret.condition)[0]) == 2:
            signals = ret.condition(ohlcv, o)
        else:
          try:
            parameters = ({pn: (getattr(ret, pn)) for pn, val in additional_parameters.items()})            
          except:
            parameters = additional_parameters
                       
          signals = ret.condition(ohlcv, o, parameters)

        figures = {}
        group = 'overlaps' if f.info['group'] == 'Overlap Studies' else 'figures'
        figures[group] = {f.info['name']: o}

        return signals, figures
    return ret
