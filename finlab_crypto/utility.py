"""Parameter enumeration, stop exits and backtest visualisation."""
from collections.abc import Hashable, Iterable
from html import escape
from itertools import product

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm
import vectorbt as vbt
from IPython.display import HTML, display

from finlab_crypto import chart, overfitting

STOP_VARIABLES = ('sl_stop', 'ts_stop', 'tp_stop', 'sl_trail')
STOP_LEVEL_PREFIX = 'stop'
CHART_BARS = 1000
CHART_FRAME_PADDING_PX = 40
TOP_N_CURVES = 10
DRAWDOWN_TOP_NS = (5, 10, 20, 30)
PARAMETER_METRICS = ('final_value', 'sharpe_ratio', 'sortino_ratio')
DASHBOARD_METRICS = ('final_value', 'calmar_ratio', 'max_drawdown', 'sharpe_ratio',
                     'downside_risk', 'omega_ratio', 'cond_value_at_risk')


def _is_grid(value):
    return isinstance(value, Iterable) and not isinstance(value, (str, pd.Series, pd.DataFrame))


def _is_label(value):
    return isinstance(value, Hashable) and not callable(value) and not isinstance(value, (pd.Series, pd.DataFrame))


def _native(value):
    """numpy scalars (from np.arange) as plain Python numbers, which TA-Lib requires."""
    return value.item() if isinstance(value, np.generic) else value


def enumerate_variables(variables):
    """Expand ``{'n1': [10, 20], 'n2': 30}`` into one dict per combination."""
    if not variables:
        return []
    grid = {name: [_native(v) for v in value] for name, value in variables.items() if _is_grid(value)}
    constants = {name: _native(value) for name, value in variables.items() if not _is_grid(value)}
    return [{**dict(zip(grid, values)), **constants} for values in product(*grid.values())]


def label_parameters(variables):
    """Parameters that can name a result column (drops Series and callables)."""
    return {name: value for name, value in variables.items() if _is_label(value)}


def signals_frame(signals, labels):
    """Stack signal Series into a DataFrame whose columns are the parameters."""
    frame = pd.concat(signals, axis=1)
    names = list(labels[0])
    if names:
        frame.columns = pd.MultiIndex.from_tuples([tuple(label.values()) for label in labels], names=names)
    return frame


def enumerate_signal(ohlcv, strategy, variables):
    """Run ``strategy`` once per parameter set.

    Returns:
      (entries, exits, figures): entries and exits have one column per
      parameter set; figures come from the last run.
    """
    entries, exits, labels = [], [], []
    figures = {}
    iteration = tqdm.tqdm(variables) if len(variables) > 1 else variables
    for variable in iteration:
        strategy.set_parameters(variable)
        results = strategy.func(ohlcv)
        entries.append(results[0])
        exits.append(results[1])
        labels.append(label_parameters(variable))
        if len(results) > 2:
            figures = results[2]
    return signals_frame(entries, labels), signals_frame(exits, labels), figures


def _stop_arguments(stop_vars):
    """Translate course stop names into ``vbt.OHLCSTX`` arguments.

    ``ts_stop`` (trailing stop) is a stop loss that follows the highest price,
    which vectorbt expresses as ``sl_stop`` plus ``sl_trail=True``.

    Returns:
      (arguments, level_names): OHLCSTX keyword arguments, and the course name
      of each argument for labelling result columns.
    """
    unknown = set(stop_vars) - set(STOP_VARIABLES)
    if unknown:
        raise ValueError(f'{sorted(unknown)} are not stop variables: {STOP_VARIABLES}')

    arguments = dict(stop_vars)
    level_names = {name: name for name in arguments}
    if 'ts_stop' in arguments:
        if 'sl_stop' in arguments:
            raise ValueError('sl_stop and ts_stop cannot be combined; choose one of them')
        arguments['sl_stop'] = arguments.pop('ts_stop')
        arguments['sl_trail'] = True
        level_names = {**level_names, 'sl_stop': 'ts_stop'}
    return arguments, level_names


def stop_early(ohlcv, entries, exits, stop_vars):
    """Add stop-loss / take-profit / trailing-stop exits to the signals.

    Every combination of the stop values is tested, e.g.
    ``{'sl_stop': [0.05, 0.1], 'tp_stop': 0.2}`` yields two variants.
    """
    if not stop_vars:
        return entries, exits

    combos = enumerate_variables(stop_vars)
    arguments, level_names = _stop_arguments({name: [combo[name] for combo in combos] for name in combos[0]})
    stops = vbt.OHLCSTX.run(entries, ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'],
                            short_name=STOP_LEVEL_PREFIX, hide_params=['sl_trail'], **arguments)

    stop_exits = stops.exits
    stop_exits.columns = stop_exits.columns.set_names(
        [level_names.get(str(name).removeprefix(f'{STOP_LEVEL_PREFIX}_'), name) for name in stop_exits.columns.names])
    repeats = stop_exits.shape[1] // entries.shape[1]
    entries = entries.vbt.tile(repeats)
    exits = stop_exits.vbt | exits.vbt.tile(repeats).values
    entries.columns = exits.columns
    return entries, exits


def display_chart(grid_chart, height, html=None):
    """Show a pyecharts chart inline (Colab, Jupyter, VS Code) and optionally save it."""
    page = chart.to_html(grid_chart, height)
    if html is not None:
        with open(html, 'w', encoding='utf-8') as file:
            file.write(page)
    display(HTML(f'<div><iframe srcdoc="{escape(page)}" width="100%" height="{height + CHART_FRAME_PADDING_PX}" '
                 'style="border:none"></iframe></div>'))


def plot_strategy(ohlcv, entries, exits, portfolio, fig_data, html=None, k_colors='world'):
    """Interactive candlestick chart with trades, indicators and equity."""
    trades = portfolio.positions.records
    entry_times = ohlcv.index[trades.entry_idx.to_numpy()]
    exit_times = ohlcv.index[trades.exit_idx.to_numpy()]
    mark_lines = [
        (trade[['entry_price', 'exit_price', 'return']].to_string(), [entry_time, exit_time],
         [trade.entry_price, trade.exit_price])
        for (_, trade), entry_time, exit_time in zip(trades.iterrows(), entry_times, exit_times)
    ]

    figures = {
        **fig_data.get('figures', {}),
        'entries & exits': pd.DataFrame({'entries': entries, 'exits': exits}),
        'performance': portfolio.cumulative_returns(),
    }
    grid_chart, size = chart.chart(ohlcv, overlaps=fig_data.get('overlaps', {}), figures=figures,
                                   markerlines=mark_lines,
                                   start_date=ohlcv.index[-min(CHART_BARS, len(ohlcv))],
                                   end_date=ohlcv.index[-1], k_colors=k_colors)
    display_chart(grid_chart, size['height'], html)
    return grid_chart


def _parameter_sensitivity(portfolio, metric):
    values = getattr(portfolio, metric)().replace([np.inf, -np.inf], np.nan)
    names = values.index.names
    results = {}
    for name in names:
        mean = values.groupby(level=name).mean()
        if all(isinstance(value, str) for value in mean.index):
            mean = mean.reset_index(drop=True)
        results[name] = mean
    return pd.DataFrame(results)


def plot_combination(portfolio, cscv_result=None, metric='final_value'):
    """Summary charts of a parameter optimisation."""
    sns.set_theme(style='whitegrid')

    ranking = getattr(portfolio, metric)().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('Backtest Results')
    ax = (portfolio.cumulative_returns()[ranking.index[-TOP_N_CURVES:]] * 100).plot(ax=axes[0])
    ax.set(xlabel='time', ylabel='cumulative return (%)')

    axes[1].set_title('Drawdown (%)')
    for n in DRAWDOWN_TOP_NS:
        (portfolio.drawdown()[ranking.index[-n:]].min(axis=1) * 100).plot(linewidth=1, ax=axes[1], label=f'top {n}')
    axes[1].set(xlabel='time', ylabel='drawdown (%)')
    axes[1].legend()
    plt.show()

    fig, axes = plt.subplots(1, len(PARAMETER_METRICS), figsize=(15, 3))
    fig.subplots_adjust(top=0.75)
    fig.suptitle('Partial Differentiation')
    for ax, item in zip(axes, PARAMETER_METRICS):
        ax.set_title(item)
        _parameter_sensitivity(portfolio, item).plot(ax=ax)
    plt.show()

    if cscv_result is not None:
        overfitting.plot_cscv(cscv_result)
        plt.show()


def variable_visualization(portfolio):
    """Interactive heatmap of a metric over two parameters."""
    param_names = list(portfolio.cumulative_returns().columns.names)
    axis1 = widgets.Dropdown(options=param_names, value=param_names[0], description='axis 1:')
    axis2 = widgets.Dropdown(options=param_names, value=param_names[-1], description='axis 2:')
    metric = widgets.Dropdown(options=DASHBOARD_METRICS, value=DASHBOARD_METRICS[0], description='performance')
    out = widgets.Output()

    def update(_):
        values = getattr(portfolio, metric.value)().replace([np.inf, -np.inf], np.nan)
        with out:
            out.clear_output(wait=True)
            if axis1.value != axis2.value:
                sns.heatmap(values.groupby(level=[axis1.value, axis2.value]).mean().unstack())
            else:
                values.groupby(level=axis1.value).mean().plot()
            plt.show()

    for dropdown in (axis1, axis2, metric):
        dropdown.observe(update, 'value')
    display(widgets.VBox([metric, widgets.HBox([axis1, axis2])]), out)
    update(None)
