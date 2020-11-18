from IPython.display import display, HTML, IFrame, clear_output
from itertools import compress, product
from collections.abc import Iterable
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import ipywidgets as widgets
import vectorbt as vbt
import seaborn as sns
import pandas as pd
import numpy as np
import copy

from . import chart
from . import overfitting


def is_evalable(obj):
    try:
        eval(str(obj))
        return True
    except:
        return False

def remove_pd_object(d):
    ret = {}
    for n, v in d.items():
        if not isinstance(v, pd.Series) and not isinstance(v, pd.DataFrame) and not callable(v) and is_evalable(v):
            ret[n] = v
        elif isinstance(v, str):
            ret[n] = v
    return ret

def enumerate_variables(variables):

    if len(variables) == 0:
        return []

    enumeration_name = []
    enumeration_vars = []

    constant_name = []
    constant_vars = []

    for name, v in variables.items():
        if (isinstance(v, Iterable) and not isinstance(v, str)
            and not isinstance(v, pd.Series)
            and not isinstance(v, pd.DataFrame)):

            enumeration_name.append(name)
            enumeration_vars.append(v)
        else:
            constant_name.append(name)
            constant_vars.append(v)

    variable_enumerations = [dict(**dict(zip(enumeration_name, ps)),
                                  **dict(zip(constant_name, constant_vars)))
                             for ps in list(product(*enumeration_vars))]

    return variable_enumerations



def enumerate_signal(ohlcv, strategy, variables, ):
    entries = {}
    exits = {}

    fig = {}

    iteration = tqdm.tqdm(variables) if len(variables) > 1 else variables
    for v in iteration:
        strategy.set_parameters(v)
        results = strategy.func(ohlcv)

        v = remove_pd_object(v)

        entries[str(v)], exits[str(v)] = results[0], results[1]
        if len(results) >= 3:
            fig = results[2]

    entries = pd.DataFrame(entries)
    exits = pd.DataFrame(exits)

    # setup columns
    param_names = list(eval(entries.columns[0]).keys())
    arrays = ([entries.columns.map(lambda s: eval(s)[p]) for p in param_names])
    tuples = list(zip(*arrays))
    if len(tuples) != 0:
        columns = pd.MultiIndex.from_tuples(tuples, names=param_names)
        exits.columns = columns
        entries.columns = columns
    return entries, exits, fig

def stop_early(ohlcv, stop_type, stop_percentages, entries, exits, trailing=False):

    nrepeat = 1

    if stop_type == 'stoploss':
        stop_exits = entries.vbt.signals.generate_stop_loss_exits(
            ohlcv.close, stop_percentages, trailing=trailing)
    elif stop_type == 'profit_targets':
        stop_exits = entries.vbt.signals.generate_take_profit_exits(
            ohlcv.close, stop_percentages)
    else:
        raise "stop_type can only be either 'stoploss' or 'profit_target'"

    if isinstance(stop_percentages, Iterable):
        exits = exits.vbt.tile(len(stop_percentages))
        nrepeat = len(stop_percentages)

    stop_exits = stop_exits.vbt | exits.values
    entries = entries.vbt.tile(nrepeat)
    entries.columns = stop_exits.columns

    return entries, stop_exits

def plot_strategy(ohlcv, entries, exits, portfolio ,fig_data):

    # format trade data
    txn = portfolio.positions.records
    txn['enter_time'] = ohlcv.iloc[portfolio.trades.records.entry_idx].index.values
    txn['exit_time'] = ohlcv.iloc[portfolio.trades.records.exit_idx].index.values

    # plot trade data
    mark_lines = []
    for name, t in txn.iterrows():
        x = [str(t.enter_time), str(t.exit_time)]
        y = [t.entry_price, t.exit_price]
        name = t.loc[['entry_price', 'exit_price', 'return']].to_string()
        mark_lines.append((name, x, y))

    # calculate overlap figures
    overlaps = {}
    if 'overlaps' in fig_data:
        overlaps = fig_data['overlaps']

    # calculate sub-figures
    figures = {}
    if 'figures' in fig_data:
        figures = fig_data['figures']

    figures['entries & exits'] = pd.DataFrame(
        {'entries':entries.squeeze(), 'exits': exits.squeeze()})
    figures['performance'] = portfolio.equity

    c, info = chart.chart(ohlcv, overlaps=overlaps,
                          figures=figures, markerlines=mark_lines,
                          start_date=ohlcv.index[-min(1000, len(ohlcv))], end_date=ohlcv.index[-1])
    c.load_javascript()
    c.render()
    display(HTML(filename="render.html"))
    return HTML(filename="render.html")

def plot_combination(portfolio, cscv_result=None, metric='final_equity'):

    sns.set()
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=False, sharex=False)
    fig.suptitle('Backtest Results')

    def heat_map(item, name1, name2, ax):
        if name1 != name2:
            sns.heatmap(item.reset_index().pivot(name1, name2)[0], cmap='magma_r', ax=ax)
        else:
            getattr(portfolio, item_name).groupby(name1).mean().plot(ax=ax)

    def best_n(portfolio, n):
        return getattr(portfolio, metric).sort_values().tail(n).index

    best_10 = best_n(portfolio, 10)

    ax = (portfolio.cumulative_returns[best_10] * 100).plot(ax=axes[0])
    ax.set(xlabel='time', ylabel='cumulative return (%)')

    axes[1].title.set_text('Drawdown (%)')
    for n, c in zip([5, 10, 20, 30], sns.color_palette("GnBu_d")):
        bests = best_n(portfolio, n)
        drawdown = portfolio.drawdown[bests].min(axis=1)
        ax = drawdown.plot(linewidth=1, ax=axes[1])
        # ax.fill_between(drawdown.index, 0, drawdown * 100, alpha=0.2, color=c)
    ax.set(xlabel='time', ylabel='drawdown (%)')

    plt.show()


    items = ['final_equity', 'sharpe_ratio', 'sortino_ratio']
    fig, axes = plt.subplots(1, len(items), figsize=(15, 3),
                             sharey=False, sharex=False, constrained_layout=False)
    fig.subplots_adjust(top=0.75)
    fig.suptitle('Partial Differentiation')

    if isinstance(portfolio.final_equity.index, pd.MultiIndex):
        index_names = portfolio.final_equity.index.names
    else:
        index_names = [portfolio.final_equity.index.name]

    for i, item in enumerate(items):
        results = {}
        for name in index_names:
            s = getattr(portfolio, item)
            s = s.replace([np.inf, -np.inf], np.nan)
            results[name] = s.groupby(name).mean()
        results = pd.DataFrame(results)
        axes[i].title.set_text(item)
        results.plot(ax=axes[i])

    if cscv_result is None:
        return

    results = cscv_result

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             sharey=False, sharex=False, constrained_layout=False)
    fig.subplots_adjust(bottom=0.5)
    fig.suptitle('Combinatorially Symmetric Cross-validation')

    axes[0].title.set_text('Probability of overfitting ')
    axes[0].hist(x=[l for l in results['logits'] if l > -10000], bins='auto')
    axes[0].set_xlabel('Logits')
    axes[0].set_ylabel('Frequency')

    # performance degradation
    axes[1].title.set_text('Performance degradation')
    sns.regplot(results['R_n_star'], results['R_bar_n_star'], ax=axes[1])
    axes[1].set_xlim(min(results['R_n_star']) * 1.2,max(results['R_n_star']) * 1.2)
    axes[1].set_ylim(min(results['R_bar_n_star']) * 1.2,max(results['R_bar_n_star']) * 1.2)
    axes[1].set_xlabel('In-sample Performance')
    axes[1].set_ylabel('Out-of-sample Performance')

    # first and second Stochastic dominance
    axes[2].title.set_text('Stochastic dominance')
    results['dom_df'].plot(ax=axes[2], secondary_y=['SD2'])
    axes[2].set_xlabel('Performance optimized vs non-optimized')
    axes[2].set_ylabel('Frequency')


def variable_visualization(portfolio):

    param_names = portfolio.cumulative_returns.columns.names
    dropdown1 = widgets.Dropdown(
        options=param_names,
        value=param_names[0],
        description='axis 1:',
        disabled=False,
    )
    dropdown2 = widgets.Dropdown(
        options=param_names,
        value=param_names[0],
        description='axis 2:',
        disabled=False,
    )

    performance_metric = ['final_equity',
        'calmar_ratio', 'max_drawdown', 'sharpe_ratio',
        'downside_risk', 'omega_ratio', 'conditional_value_at_risk']

    performance_dropdwon = widgets.Dropdown(
        options=performance_metric,
        value=performance_metric[0],
        description='performance',
        disabled=False,
    )

    out = widgets.Output()

    import matplotlib.pyplot as plt
    def update(v):
        name1 = dropdown1.value
        name2 = dropdown2.value
        performance = performance_dropdwon.value

        with out:
            out.clear_output()
            if name1 != name2:
                df = (getattr(portfolio, performance)
                      .reset_index().groupby([name1, name2]).mean()[0]
                      .reset_index().pivot(name1, name2)[0])

                df = df.replace([np.inf, -np.inf], np.nan)
                sns.heatmap(df)
            else:
                getattr(portfolio, performance).groupby(name1).mean().plot()
            plt.show()


    dropdown1.observe(update, 'value')
    dropdown2.observe(update, 'value')
    performance_dropdwon.observe(update, 'value')
    drawdowns = widgets.VBox([performance_dropdwon,
                 widgets.HBox([dropdown1, dropdown2])])
    display(drawdowns)
    display(out)
    update(0)
