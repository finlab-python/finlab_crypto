from talib import abstract
from finlab_crypto.strategy import Strategy, Filter

def TalibFilter(talib_function_name, condition=None):
    f = getattr(abstract, talib_function_name)

    @Filter(condition=condition, **f.parameters)
    def ret(ohlcv):
        parameters = {pn: int(getattr(ret, pn)) for pn in f.parameters.keys()}
        try:
            o = f(ohlcv, **parameters)
        except:
            o = f(ohlcv.close, **parameters)
            if isinstance(o, list):
                o = pd.DataFrame(np.array(o).T, index=ohlcv.index, columns=f.output_names)


        signals = ret.condition(ohlcv, o)

        figures = {}
        group = 'overlaps' if f.info['group'] == 'Overlap Studies' else 'figures'
        figures[group] = {f.info['name']: o}

        return signals, figures
    return ret
