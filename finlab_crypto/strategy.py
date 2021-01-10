"""Strategy function plug-in.

You can use Filter and Strategy function as decorator
that make strategies easy to construct filters layers
and common strategy detection methods, such as back-testing,
parameter tuning and analysis charts.

  Typical usage example:
```
  @Filter(timeperiod=20)
  def your_filter(ohlcv):
      your filter logic...
      return filter > filter_value, figures
  f60 = your_filter.create({'timeperiod': 60})
```
  -------------------------------
```
  @Strategy()
  def your_strategy(ohlcv):
     your strategy logic...
      return entries, exits, figures
  portfolio = your_strategy.backtest(ohlcv, freq='4h', plot=True)
```
"""

from finlab_crypto.utility import (enumerate_variables, enumerate_signal,
                                   stop_early, plot_combination, plot_strategy,
                                   variable_visualization, remove_pd_object
                                   )
from finlab_crypto.overfitting import CSCV
import copy
import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt
from collections import Iterable


class Filter(object):
    """Filter package features plug-in.

    Offer easy way to create filter to use in class Strategy.

    Attributes:
        default_parameters: customized filter attributes.
    """

    def __init__(self, **default_parameters):
        """inits filter."""
        self.func = None
        self.filters = {}
        self._default_parameters = default_parameters
        self.set_parameters(default_parameters)

    def __call__(self, func):
        """decorator function

        Args
          func: A function of the customized filter.
        """
        self.func = func
        return self

    def set_parameters(self, variables):
        """set your customized filter parameters.

        let filter class use variables dict to set method

        Args:
          variables: a dict of your customized filter attributes.

        """
        if self._default_parameters:
            for key, val in self._default_parameters.items():
                setattr(self, key, val)

        if variables:
            for key, val in variables.items():
                setattr(self, key, val)


    def show_parameters(self):
        parameters = {}
        for key, val in self._default_parameters.items():
            parameters[key] = getattr(self, key)
        print(parameters)

    def create(self, variables=None):
        """generate filter signals, fig_data.

        offer easy way to create filter signals, fig_data

        Args:
          variables: a dict of your customized filter attributes.
        Returns:
          signals: a dataframe of filter signals.
          fig_data: a dict of required data for figure display.
        """

        def ret_f(ohlcv):

            variable_enumerate = enumerate_variables(variables)
            if len(variable_enumerate) == 0:
                variable_enumerate.append(self._default_parameters)

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

class Strategy(object):
    """strategy features plug-in.

    offer common strategy detection methods, such as back-testing,
    parameter tuning and analysis charts.

    Attributes:
        default_parameters: customized strategy attributes.

    """

    def __init__(self, **default_parameters):
        """inits strategy."""
        self.filters = {}
        self._default_parameters = default_parameters
        self.set_parameters(default_parameters)

    def __call__(self, func):
        """decorator function

        Args
          func: A function of customized strategy.
        """
        self.func = func
        return self

    def set_parameters(self, variables):
        """set your customized strategy parameters.

        let strategy class use variables dict to set method.

        Args:
          variables: a dict of your customized strategy attributes.

        """

        # remove stop vars
        stop_vars = ['sl_stop', 'tp_stop', 'ts_stop']
        for svar in stop_vars:
            if hasattr(self, svar):
                delattr(self, svar)

        # set defualt variables
        if self._default_parameters:
            for key, val in self._default_parameters.items():
                setattr(self, key, val)

        # set custom variables
        if variables:
            for key, val in variables.items():
                setattr(self, key, val)

    def show_parameters(self):
        parameters = {}
        for key, val in self._default_parameters.items():
            parameters[key] = getattr(self, key)
        print(parameters)

    @staticmethod
    def _enumerate_filters(ohlcv, filters):
        """enumerate filters data.

        process filter dictionary data to prepare for adding filter signals.

        Args:
          ohlcv: a dataframe of your trading target.
          filters: a dict of your customized filter attributes.

        Returns:
          a dict that generate tuple with filter signal dataframe and figures data.
          for example:

        {'mmi': (timeperiod                    20
          timestamp
          2020-11-25 02:00:00+00:00   true
          2020-11-25 03:00:00+00:00   true
          2020-11-25 04:00:00+00:00   true

          [3 rows x 1 columns], {'figures': {'mmi_index': timestamp
            2020-11-25 02:00:00+00:00    0.7
            2020-11-25 03:00:00+00:00    0.7
            2020-11-25 04:00:00+00:00    0.7
            name: close, length: 28597, dtype: float64}})}

        """
        ret = {}
        for fname, f in filters.items():
            # get filter signals and figures
            filter_df, filter_figures = f(ohlcv)
            ret[fname] = (filter_df, filter_figures)
        return ret

    @staticmethod
    def _add_filters(entries, exits, fig_data, filters):
        """add filters in strategy.

        generate entries, exits, fig_data after add filters.

        Args:
          entries: A dataframe of entries point time series.
          exits: A dataframe of exits point time series.
          fig_data: A dict of your customized figure Attributes.
          filters: A dict of _enumerate_filters function return.

        Returns:
          entries: A dataframe of entries point time series after add filter function.
          exits: A dataframe of exits point time series after add filter function.
          fig_data: A dict of tuple with filter signal dataframe and figures data.

        """
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
                        fig_data['figures'][fname + '_' + name] = fig
                if 'overlaps' in filter_figures:
                    if 'overlaps' not in fig_data:
                        fig_data['overlaps'] = {}
                    for name, fig in filter_figures['overlaps'].items():
                        fig_data['overlaps'][fname + '_' + name] = fig

        return entries, exits, fig_data

    @staticmethod
    def _add_stops(ohlcv, entries, exits, variables):
        """Add early trading stop condition in strategy.

        Args:
          ohlcv: A dataframe of your trading target.
          entries: A dataframe of entry point time series.
          exits: A dataframe of exits point time series.
          variables: A dict of your customized strategy Attributes.

        Returns:
          entries: A dataframe of entries point time series after add stop_early function.
          exits: A dataframe of exits point time series after add stop_early function.

        """
        entries, exits = stop_early(ohlcv, entries, exits, variables)
        entries = entries.squeeze()
        exits = exits.squeeze()
        return entries, exits

    def backtest(self, ohlcv, variables=None,
                 filters=None, lookback=None, plot=False,
                 signals=False, side='long', cscv_nbins=10,
                 cscv_objective=lambda r: r.mean(), html=None, compounded=True, execution_price='close', **args):

        """Backtest analysis tool set.
        Use vectorbt as base module to create numerical operations features.
        Use seaborn and pyechart as base modules to create analysis charts platform.

        Args:
          ohlcv: A dataframe of your trading target.
          variables: A dict of your customized strategy Attributes.
            Default is None.
          filters: A dict of your customized filter Attributes.
            Default is None.
          lookback: A int of slice that you want to get recent ohlcv.
            Default is None.
          plot: A bool of control plot display.
            Default is False.
          signals: A bool of controlentries, exits, fig_data return.
            Default is False.
          side: A str of transaction direction,short or long.
            Default is long.
          cscv_nbins: A int of CSCV algorithm bin size to control overfitting calculation.
            Default is 10.
          cscv_objective: A function of in sample(is) and out of sample(oos) return benchmark algorithm.
            Default is lambda r:r.mean().
          html: A str of your customized html format file to show plot.
            Default is None.
          compounded: use compounded return as result of backtesting
            Default is True
          execution_price: price for trading operation ('open' or 'close').
            Default is 'open'
          **args:
            Other parameters.

        Returns:
            A dataframe of vectorbt.Portfolio.from_signals results
            Plot results display.

        Raises:
            'Shorting is not support yet':if side is 'short'.
            "side should be 'long' or 'short'":if side is not 'short' or 'long'.

        """
        variables = variables or dict()
        filters = filters or dict()

        variables_without_stop = copy.copy(variables)
        exit_vars = ['sl_stop', 'ts_stop', 'tp_stop']
        stop_vars = {}
        for e in exit_vars:
            if e in variables_without_stop:
                stop_vars[e] = variables[e]
                variables_without_stop.pop(e)

        ohlcv_lookback = ohlcv.iloc[-lookback:] if lookback else ohlcv

        variable_enumerate = enumerate_variables(variables_without_stop)

        if not variable_enumerate:
            variable_enumerate = [self._default_parameters]

        entries, exits, fig_data = enumerate_signal(ohlcv_lookback, self, variable_enumerate)

        if filters:
            filter_signals = self._enumerate_filters(ohlcv_lookback, filters)
            entries, exits, fig_data = self._add_filters(entries, exits, fig_data, filter_signals)

        entries, exits = self._add_stops(ohlcv_lookback, entries, exits, stop_vars)

        if signals:
            return entries, exits, fig_data

        if side == 'long':

            if not compounded:
                args['size'] = vbt.defaults.portfolio['init_cash'] /  ohlcv_lookback.close[0]

            assert execution_price == 'close' or execution_price == 'open'
            price = ohlcv_lookback[execution_price] if execution_price == 'close' else ohlcv_lookback[execution_price].shift(-1).bfill()

            portfolio = vbt.Portfolio.from_signals(
                ohlcv_lookback[execution_price], entries.fillna(False), exits.fillna(False), **args)

        elif side == 'short':
            raise Exception('Shorting is not support yet')

        else:
            raise Exception("side should be 'long' or 'short'")

        if (plot or html is not None) and isinstance(entries, pd.Series):
            plot_strategy(ohlcv_lookback, entries, exits, portfolio, fig_data, html=html)

        elif plot and isinstance(entries, pd.DataFrame):

            # perform CSCV algorithm
            cscv = CSCV(n_bins=cscv_nbins, objective=cscv_objective)
            cscv.add_daily_returns(portfolio.daily_returns())
            cscv_result = cscv.estimate_overfitting(plot=False)

            # plot results
            plot_combination(portfolio, cscv_result)
            plt.show()
            variable_visualization(portfolio)

        return portfolio
