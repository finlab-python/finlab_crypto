import vectorbt as vbt
from finlab_crypto import strategy
from finlab_crypto import overfitting
from finlab_crypto.strategy import Strategy

@Strategy(fast_window=21, slow_window=144)
def sma_strategy(ohlcv, combination=False):
  if combination:
    windows = sorted(list(set(list(sma_strategy.fast_window) + list(sma_strategy.slow_window))))
    fast_ma, slow_ma = vbt.MA.from_combs(ohlcv['open'], windows, 2)
  else:
    fast_ma = vbt.MA.from_params(ohlcv['open'], sma_strategy.fast_window, False)
    slow_ma = vbt.MA.from_params(ohlcv['open'], sma_strategy.slow_window, False)

  entries = fast_ma.ma_above(slow_ma, crossed=True)
  exits = fast_ma.ma_below(slow_ma, crossed=True)

  figure = {
    'overlaps': {
        str(sma_strategy.slow_window) + 'ma': fast_ma.ma,
        str(sma_strategy.fast_window) + 'ma': slow_ma.ma,
    }
  }

  return entries, exits, figure
