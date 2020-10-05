from finlab_crypto.strategy import Strategy, Filter
from talib import abstract

def TalibStrategy(talib_function_name, entries, exits):
    f = getattr(abstract, talib_function_name)

    @Strategy(entries=entries, exits=exits, **f.parameters)
    def ret(ohlcv):
        parameters = {pn: (getattr(ret, pn)) for pn in f.parameters.keys()}
        try:
            o = f(ohlcv, **parameters)
        except:
            o = f(ohlcv.close, **parameters)
            if isinstance(o, list):
                o = pd.DataFrame(np.array(o).T, index=ohlcv.index, columns=f.output_names)

        entries = ret.entries(ohlcv, o)
        exits = ret.exits(ohlcv, o)

        figures = {}
        group = 'overlaps' if f.info['group'] == 'Overlap Studies' else 'figures'
        figures[group] = {f.info['name']: o}

        return entries, exits, figures
    return ret
