from finlab_crypto.utility import (enumerate_variables, enumerate_signal,
                                   stop_early, plot_combination, plot_strategy,
                                   variable_visualization, remove_pd_object
                                  )
from finlab_crypto.overfitting import CSCV
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

        @staticmethod
        def _enumerate_filters(ohlcv, filters):
            ret = {}
            for fname, f in filters.items():

                # get filter signals and figures
                filter_df, filter_figures = f(ohlcv)
                ret[fname] = (filter_df, filter_figures)
            return ret

        @staticmethod
        def _add_filters(entries, exits, fig_data, filters):

            for fname, (filter_df, filter_figures) in filters.items():
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

            return entries, exits, fig_data

        @staticmethod
        def _add_stops(ohlcv, entries, exits, variables):

            if 'stoploss' in variables:
                trailing = False
                if 'trailing' in variables:
                    trailing = variables['trailing']

                entries, exits = stop_early(ohlcv,
                                            'stoploss',
                                            variables['stoploss'], entries, exits, trailing=trailing)

            if 'profit_targets' in variables:
                entries, exits = stop_early(ohlcv,
                                            'profit_targets',
                                            variables['profit_targets'], entries, exits)
            entries = entries.squeeze()
            exits = exits.squeeze()
            return entries, exits

        def backtest(self, ohlcv, variables=dict(),
                filters=dict(), plot=False, lookback=None,
                signals=False, side='long', cscv_nbins=10, cscv_objective=lambda r:r.mean(), **args):

            variables_without_stop = copy.copy(variables)

            exit_vars = ['stoploss', 'profit_targets', 'trailing']
            for e in exit_vars:
                if e in variables_without_stop:
                    variables_without_stop.pop(e)

            ohlcv_lookback = ohlcv.iloc[-lookback:] if lookback else ohlcv

            variable_enumerate = enumerate_variables(variables_without_stop)

            if not variable_enumerate:
                variable_enumerate = [default_parameters]

            entries, exits, fig_data = enumerate_signal(ohlcv_lookback, self, variable_enumerate)

            if filters:
                filter_signals = self._enumerate_filters(ohlcv_lookback, filters)
                entries, exits, fig_data = self._add_filters(entries, exits, fig_data, filter_signals)

            entries, exits = self._add_stops(ohlcv_lookback, entries, exits, variables)

            if signals:
                return entries, exits, fig_data

            if side == 'long':
                portfolio = vbt.Portfolio.from_signals(
                    ohlcv_lookback.close, entries.fillna(False), exits.fillna(False), **args)

            elif side == 'short':
                raise Exception('Shorting is not support yet')

            else:
                raise Exception("side should be 'long' or 'short'")

            if plot and isinstance(entries, pd.Series):
                plot_strategy(ohlcv_lookback, entries, exits, portfolio ,fig_data)

            elif plot:

                # perform CSCV algorithm
                cscv = CSCV(n_bins=cscv_nbins, objective=cscv_objective)
                cscv.add_daily_returns(portfolio.daily_returns)
                cscv_result = cscv.estimate_overfitting(plot=False)

                # plot results
                plot_combination(portfolio, cscv_result)
                plt.show()
                variable_visualization(portfolio)

            return portfolio


    def deco(func):
        return Strategy(func)

    return deco
