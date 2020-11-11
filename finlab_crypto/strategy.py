from finlab_crypto.utility import (enumerate_variables, enumerate_signal,
                                   stop_early, plot_combination, plot_strategy,
                                   variable_visualization, remove_pd_object
                                  )
import copy
import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from vectorbt.indicators.factory import compare

def Filter(**default_parameters):

    class Filter:
        def __init__(self, func,):
            self.func = func
            self._variables = None
            self.filters = {}
            self.set_parameters(default_parameters)

        def set_parameters(self, variables):
            if variables:
                for key, val in variables.items():
                    setattr(self, key, val)
            self._variables = variables

        def show_parameters(self):
            print(self._variables)

        def create(self, variables=None):

            def ret_f(ohlcv):

                variable_enumerate = enumerate_variables(variables)
                if len(variable_enumerate) == 0:
                    variable_enumerate.append(default_parameters)

                signals = {}
                fig_data = {}
                for v in variable_enumerate:

                    self.set_parameters(v)
                    results = self.func(ohlcv)

                    v = remove_pd_object(v)

                    if isinstance(results, Iterable):
                        signals[str(v)], fig_data = results
                    else:
                        signals[str(v)] = results

                signals = pd.DataFrame(signals)
                signals.columns.name = 'filter'

                param_names = list(eval(signals.columns[0]).keys())
                arrays = ([signals.columns.map(lambda s: eval(s)[p]) for p in param_names])
                tuples = list(zip(*arrays))
                columns = pd.MultiIndex.from_tuples(tuples, names=param_names)
                signals.columns = columns

                return signals, fig_data

            return ret_f

    def deco(func):
        return Filter(func)

    return deco

def Strategy(**default_parameters):

    class Strategy:
        def __init__(self, func):
            self.func = func
            self._variables = None
            self.filters = {}
            self.set_parameters(default_parameters)

        def set_parameters(self, variables):
            if variables:
                for key, val in variables.items():
                    setattr(self, key, val)
            self._variables = variables

        def show_parameters(self):
            print(self._variables)

        def backtest(self, ohlcv, variables=None, filters=None, plot=False, lookback=None, signals=False, side='long', **args):

            if variables is None:
                variables = {}

            variables_without_stop = copy.copy(variables)

            if 'stoploss' in variables_without_stop:
                variables_without_stop.pop('stoploss')

            if 'profit_targets' in variables_without_stop:
                variables_without_stop.pop('profit_targets')

            ohlcv_lookback = ohlcv.iloc[-lookback:] if lookback is not None else ohlcv

            if side == 'short':
                inv_close = ohlcv_lookback.close / ohlcv_lookback.ohlcv.close.shift(1)

            variable_enumerate = enumerate_variables(variables_without_stop)
            if len(variable_enumerate) == 0:
                variable_enumerate.append(default_parameters)
            entries, exits, fig_data = enumerate_signal(ohlcv_lookback, self, variable_enumerate)

            if filters is not None:
                for fname, f in filters.items():

                    # get filter signals and figures
                    filter_df, filter_figures = f(ohlcv)

                    # merge filter_df
                    filter_df.columns = filter_df.columns.set_names([fname + '_' + n for n in filter_df.columns.names])
                    entries = filter_df.vbt.tile(entries.shape[1]).vbt & entries.vbt.repeat(filter_df.shape[1]).vbt
                    exits = exits.vbt.repeat(filter_df.shape[1])
                    exits.columns = entries.columns

                    # merge figures
                    if filter_figures is not None:
                        if 'figures' in filter_figures:
                            if 'figures' not in fig_data:
                                fig_data['figures'] = {}
                            for name, fig in filter_figures['figures'].items():
                                fig_data['figures'][fname+'_'+name] = fig
                        if 'overlaps' in filter_figures:
                            if 'overlaps' not in fig_data:
                                fig_data['overlaps'] = {}
                            for name, fig in filter_figures['overlaps'].items():
                                fig_data['overlaps'][fname+'_'+name] = fig

            if 'stoploss' in variables:
                entries, exits = stop_early(ohlcv_lookback,
                                            'stoploss',
                                            variables['stoploss'], entries, exits)

            if 'profit_targets' in variables:
                entries, exits = stop_early(ohlcv_lookback,
                                            'profit_targets',
                                            variables['profit_targets'], entries, exits)

            entries = entries.squeeze()
            exits = exits.squeeze()

            if signals:
                return entries, exits, fig_data

            if side == 'long':
                portfolio = vbt.Portfolio.from_signals(
                    ohlcv_lookback.close, entries.fillna(False), exits.fillna(False), **args)
            elif side == 'short':
                portfolio = vbt.Portfolio.from_signals(
                    inv_close, exits.fillna(False), entries.fillna(False), **args)
            else:
                raise Exception("side should be 'long' or 'short'")


            if plot and isinstance(entries, pd.Series):
                plot_strategy(ohlcv_lookback, entries, exits, portfolio ,fig_data)
            elif plot:
                plot_combination(portfolio)
                plt.show()
                variable_visualization(portfolio)

            return portfolio


    def deco(func):
        return Strategy(func)

    return deco
