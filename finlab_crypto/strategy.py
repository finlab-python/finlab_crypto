import numpy as np
import pandas as pd
import seaborn as sns
import vectorbt as vbt
import matplotlib.pyplot as plt
from collections.abc import Iterable
from IPython.display import display, HTML, IFrame, clear_output

from . import chart
from . import overfitting

def Strategy(**default_parameters):

  class Strategy:
    def __init__(self, func):
      self.stoploss = None
      self.profit_targets = None
      self.func = func
      self._variables = None
      self.set_parameters(default_parameters)

    def set_parameters(self, variables):
      self.stoploss = None
      self.profit_targets = None
      if variables:
        for key, val in variables.items():
          setattr(self, key, val)
      self._variables = variables

    def exits_sl_tp(self, entries, exits, price, variables):

      if 'stoploss' in variables:
        stoploss_exits = entries.vbt.signals.generate_stop_loss_exits(price, variables['stoploss'])
        exits = exits.vbt.signals.OR(stoploss_exits)

      if 'profit_targets' in variables:
        take_profit_exits = entries.vbt.signals.generate_take_profit_exits(price, variables['profit_targets'])
        exits = exits.vbt.signals.OR(take_profit_exits)

      return exits

    def backtest(self, ohlcv, variables=None, plot=True, price_decision='close', S=10, **args):

      if variables:
        self.set_parameters(variables)
      else:
        variables = self._variables
      # check if the backtest is an optimization of combinations of parameters
      is_combination = sum(isinstance(v, Iterable) and not isinstance(v, str) for name, v in variables.items()) != 0

      # generate entries and exits signals
      results = self.func(ohlcv, is_combination)

      if len(results) == 2:
        entries, exits = results
        fig_data = {}
      elif len(results) == 3:
        entries, exits, fig_data = results
      else:
        raise("strategy outputs multiple results!")

      # backtest price

      if 'trading' not in variables or ('trading' in variables and variables['trading'] == 'long'):
        price = ohlcv[price_decision]
      elif 'trading' in variables and variables['trading'] == 'short':
        price = (ohlcv[price_decision].shift()/ohlcv[price_decision]).cumprod()
      else:
        raise Exception('variable "trading" should be either "long" or "short". Get ' + variables['trading'] + ' instead.')

      exits = self.exits_sl_tp(entries, exits, price, variables)

      portfolio = vbt.Portfolio.from_signals(price, entries, exits, **args)

      if 'trading' in variables and variables['trading'] == 'short':
        portfolio.position_records['OpenPrice'] = (ohlcv[price_decision].iloc[portfolio.position_records.OpenAt] * (1 - vbt.defaults.portfolio['slippage'])).values
        portfolio.position_records['ClosePrice'] = (ohlcv[price_decision].iloc[portfolio.position_records.CloseAt] * (1 + vbt.defaults.portfolio['slippage'])).values

      if plot:
        if not is_combination:
          self.plot(ohlcv, entries, exits, portfolio , fig_data)
        else:
          self.plot_combination(portfolio)
          # Plot heatmap
          # try:
          #     i1, i2 = np.array([len(set(l)) for l in portfolio.total_return.index.levels]).argsort()[-2:][::-1]
          #     name1, name2 = portfolio.total_return.index.names[i1], portfolio.total_return.index.names[i2]
          #     portfolio.total_return.vbt.heatmap(
          #         x_level=name1, y_level=name2, symmetric=False,
          #         trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%'))
          #     ).show_png()
          # except:
          #     pass

          # plot best returns
          # best_n = 5
          # best_results = portfolio.total_return.sort_values().index[-best_n:][::-1]
          # portfolio.cumulative_returns[best_results].plot()

          # import matplotlib.pyplot as plt
          # plt.show()

          # # check the probability of backtest overfitting
          # from . import overfitting
          # results = overfitting.CSCV(portfolio.daily_returns, S=S)

      return portfolio

    def plot(self, ohlcv, entries, exits, portfolio ,fig_data):

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

      figures['entries & exits'] = pd.DataFrame({'entries':entries.squeeze(), 'exits': exits.squeeze()})
      figures['performance'] = portfolio.equity

      c, info = chart.chart(ohlcv, overlaps=overlaps, figures=figures, markerlines=mark_lines, start_date='2018', end_date='2020')
      c.load_javascript()
      c.render()
      display(HTML(filename="render.html"))
      return HTML(filename="render.html")

    def plot_combination(self, portfolio):

      if isinstance(portfolio.final_equity.index, pd.MultiIndex):
        n1 = portfolio.final_equity.index.names[0]
        n2 = portfolio.final_equity.index.names[1]
      else:
        n1 = portfolio.index.name
        n2 = n1


      fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False, sharex=False)
      fig.subplots_adjust(bottom=0.5)
      fig.suptitle('Backtest Results')

      def heat_map(item, name1, name2, ax):
        if name1 != name2:
          sns.heatmap(item.reset_index().pivot(name1, name2)[0], cmap='magma_r', ax=ax)
        else:
          getattr(portfolio, item_name).groupby(name1).mean().plot(ax=ax)

      def best_n(portfolio, n):
        return portfolio.final_equity.sort_values().tail(n).index

      best_10 = best_n(portfolio, 10)

      sns.set()
      with sns.axes_style("whitegrid"):
        sns.set(rc={'figure.figsize':(7.5,5)})
        axes[0][0].title.set_text('Cumulative Return')
        ax = (portfolio.cumulative_returns[best_10] * 100).plot(ax=axes[0][0])
        ax.set(xlabel='time', ylabel='cumulative return (%)')

        axes[0][1].title.set_text('Perfornace (return)')
        heat_map(portfolio.final_equity, n1, n2, axes[0][1])

        axes[1][1].title.set_text('Perfornace (sharpe ratio)')
        heat_map(portfolio.sharpe_ratio.clip(-3,3), n1, n2, axes[1][1])


      for n, c in zip([5, 10, 20, 30], sns.color_palette("GnBu_d")):
        bests = best_n(portfolio, n)
        axes[1][0].title.set_text('Drawdown (%)')
        drawdown = portfolio.drawdown[bests].min(axis=1)
        ax = drawdown.plot(linewidth=0, ax=axes[1][0])
        ax.fill_between(drawdown.index, 0, drawdown * 100, alpha=0.2, color=c)
        ax.set(xlabel='time', ylabel='drawdown (%)')

      plt.show(True)

      items = ['max_drawdown', 'final_equity', 'sharpe_ratio', 'sortino_ratio']
      fig, axes = plt.subplots(1, len(items), figsize=(15, 3), sharey=False, sharex=False, constrained_layout=False)
      fig.subplots_adjust(bottom=0.5)
      fig.suptitle('Partial Differentiation')

      if isinstance(portfolio.final_equity.index, pd.MultiIndex):
        index_names = portfolio.final_equity.index.names
      else:
        index_names = [portfolio.final_equity.index.name]

      for i, item in enumerate(items):
        results = {}
        for name in index_names:
          s = getattr(portfolio, item)
          if item == 'sharpe_ratio' or item == 'sortino_ratio':
              s = s.clip(-1, 2)
          results[name] = s.groupby(name).mean()
        results = pd.DataFrame(results)
        axes[i].title.set_text(item)
        results.plot(ax=axes[i])

      plt.show(True)

      def metric(x):
        return x.mean(axis=0)
      results = overfitting.CSCV(portfolio.daily_returns, S=10, metric_func=metric, plot=False)

      fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False, sharex=False, constrained_layout=False)
      fig.subplots_adjust(bottom=0.5)
      fig.suptitle('Combinatorially Symmetric Cross-validation')

      axes[0].title.set_text('Probability Distribution')
      axes[0].hist(x=[l for l in results['logits'] if l > -10000], bins='auto')
      plt.xlabel('Logits')
      plt.ylabel('Frequency')

      # performance degradation
      axes[1].title.set_text('Performance degradation')
      axes[1].scatter(results['R_n_star'], results['R_bar_n_star'])
      plt.xlabel('In-sample Performance')

      # first and second Stochastic dominance
      axes[2].title.set_text('Stochastic dominance')
      results['dom_df'].plot(ax=axes[2])
      plt.xlabel('Performance optimized vs non-optimized')
      plt.ylabel('Frequency')

    def test_function(self, ohlcv, variables=None):
      if variables:
        self.set_parameters(variables)
      else:
        variables = self._variables
      # check if the backtest is an optimization of combinations of parameters
      is_combination = sum(isinstance(v, list) for name, v in variables.items()) != 0

      # generate entries and exits signals
      return self.func(ohlcv, is_combination)

    def latest_signal(self, ohlcv, variables=None, lookback_period=0):
      portfolio = self.backtest(ohlcv.iloc[-abs(lookback_period):], variables=variables, plot=False)
      #return portfolio.shares.iloc[-1]
      return portfolio


  def deco(func):
    return Strategy(func)

  return deco
