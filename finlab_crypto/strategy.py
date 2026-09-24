"""``@Strategy`` and ``@Filter`` decorators.

A strategy returns entry and exit signals (plus optional figures); a filter
returns a boolean condition that entries must satisfy::

    @Strategy(n1=20, n2=60)
    def sma_strategy(ohlcv):
        sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
        sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
        return sma1 > sma2, sma1 < sma2

    @Filter(timeperiod=20)
    def mmi_filter(ohlcv):
        ...
        return mmi > 0.5, figures

    portfolio = sma_strategy.backtest(ohlcv, {'n1': 20, 'n2': 60},
                                      filters={'mmi': mmi_filter.create()},
                                      freq='4h', plot=True)
"""
import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt

from finlab_crypto.overfitting import CSCV
from finlab_crypto.utility import (STOP_VARIABLES, enumerate_signal, enumerate_variables, label_parameters,
                                   plot_combination, plot_strategy, signals_frame, stop_early,
                                   variable_visualization)

EXECUTION_PRICES = ('close', 'open')


def _as_bool(signals):
    return signals.where(signals.notna(), False).astype(bool)


class _Parametrized:
    """Decorated function whose parameters are read as attributes."""

    def __init__(self, **default_parameters):
        self.func = None
        self._default_parameters = default_parameters
        self.set_parameters(default_parameters)

    def __call__(self, func):
        self.func = func
        return self

    def set_parameters(self, variables):
        """Reset to the defaults, then apply ``variables`` as attributes."""
        for key, value in {**self._default_parameters, **(variables or {})}.items():
            setattr(self, key, value)

    def show_parameters(self):
        print({key: getattr(self, key) for key in self._default_parameters})


class Filter(_Parametrized):
    """Decorator for a filter ``func(ohlcv) -> condition`` or ``(condition, figures)``."""

    def create(self, variables=None):
        """Bind parameters; values given as lists or arrays are optimised over.

        Returns:
          A function ``ohlcv -> (signals, figures)`` accepted by
          ``Strategy.backtest(filters=...)``.
        """
        def run(ohlcv):
            parameter_sets = enumerate_variables(variables) or [self._default_parameters]
            signals, labels = [], []
            figures = {}
            for parameters in parameter_sets:
                self.set_parameters(parameters)
                result = self.func(ohlcv)
                signal, figures = result if isinstance(result, tuple) else (result, figures)
                signals.append(signal)
                labels.append(label_parameters(parameters))
            return signals_frame(signals, labels), figures

        return run


class Strategy(_Parametrized):
    """Decorator for a strategy ``func(ohlcv) -> (entries, exits[, figures])``."""

    @staticmethod
    def _add_filters(entries, exits, fig_data, filters):
        """AND each filter into the entries, testing every filter parameter set."""
        for name, (filter_df, filter_figures) in filters.items():
            filter_df.columns = filter_df.columns.set_names([f'{name}_{n}' for n in filter_df.columns.names])
            entries = filter_df.vbt.tile(entries.shape[1]).vbt & entries.vbt.repeat(filter_df.shape[1]).vbt
            exits = exits.vbt.repeat(filter_df.shape[1])
            exits.columns = entries.columns

            for group in ('figures', 'overlaps'):
                for figure_name, figure in (filter_figures or {}).get(group, {}).items():
                    fig_data.setdefault(group, {})[f'{name}_{figure_name}'] = figure
        return entries, exits, fig_data

    def backtest(self, ohlcv, variables=None, filters=None, lookback=None, plot=False, signals=False,
                 side='long', cscv_nbins=10, cscv_objective=lambda r: r.mean(), html=None, compounded=True,
                 execution_price='close', k_colors='world', **args):
        """Backtest the strategy with vectorbt.

        Args:
          ohlcv: OHLCV DataFrame.
          variables: Strategy parameters. Lists or arrays are optimised over;
            ``sl_stop``, ``tp_stop``, ``ts_stop`` add stop-loss, take-profit
            and trailing-stop exits.
          filters: {name: Filter.create(...)} conditions for entries.
          lookback: Only use the latest ``lookback`` bars.
          plot: Show the interactive chart (one parameter set) or the
            optimisation dashboard with overfitting analysis (many sets).
          signals: Return ``(entries, exits, figures)`` instead of backtesting.
          side: Only 'long' is supported.
          cscv_nbins: Number of blocks for the overfitting analysis.
          cscv_objective: Performance measure for the overfitting analysis.
          html: Also save the interactive chart to this file.
          compounded: Reinvest profits; otherwise trade a fixed size.
          execution_price: 'close' trades at the signal bar's close, 'open'
            at the next bar's open.
          k_colors: Candle colors, 'world', 'taiwan' or
            {'increasing_line': color, 'decreasing_line': color}.
          **args: Passed to ``vbt.Portfolio.from_signals`` (ex: ``freq='4h'``, ``fees``).

        Returns:
          vbt.Portfolio
        """
        if side == 'short':
            raise NotImplementedError('Shorting is not supported yet')
        if side != 'long':
            raise ValueError("side should be 'long' or 'short'")
        if execution_price not in EXECUTION_PRICES:
            raise ValueError(f'execution_price should be one of {EXECUTION_PRICES}')

        variables = dict(variables or {})
        stop_vars = {name: variables.pop(name) for name in STOP_VARIABLES if name in variables}
        ohlcv = ohlcv.iloc[-lookback:] if lookback else ohlcv

        parameter_sets = enumerate_variables(variables) or [self._default_parameters]
        entries, exits, fig_data = enumerate_signal(ohlcv, self, parameter_sets)
        if filters:
            filter_results = {name: create(ohlcv) for name, create in filters.items()}
            entries, exits, fig_data = self._add_filters(entries, exits, fig_data, filter_results)
        entries, exits = stop_early(ohlcv, entries, exits, stop_vars)
        entries, exits = entries.squeeze(axis=1), exits.squeeze(axis=1)

        if signals:
            return entries, exits, fig_data

        if not compounded:
            args['size'] = vbt.settings.portfolio['init_cash'] / ohlcv.close.iloc[0]
        price = ohlcv.close if execution_price == 'close' else ohlcv.open.shift(-1)
        portfolio = vbt.Portfolio.from_signals(price, _as_bool(entries), _as_bool(exits), **args)

        if isinstance(entries, pd.Series):
            if plot or html is not None:
                plot_strategy(ohlcv, entries, exits, portfolio, fig_data, html=html, k_colors=k_colors)
        elif plot:
            cscv = CSCV(n_bins=cscv_nbins, objective=cscv_objective)
            cscv.add_daily_returns(portfolio.daily_returns())
            plot_combination(portfolio, cscv.estimate_overfitting())
            plt.show()
            variable_visualization(portfolio)

        return portfolio
