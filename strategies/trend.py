from finlab_crypto.strategy import Strategy
from finlab_crypto.indicators import trends

@Strategy(name='sma', n1=20, n2=40)
def trend_following_strategy(ohlcv):
  name = trend_following_strategy.name
  n1 = trend_following_strategy.n1
  n2 = trend_following_strategy.n2

  filtered1 = trends[name](ohlcv.close, n1)
  filtered2 = trends[name](ohlcv.close, n2)

  entries = (filtered1 > filtered2) & (filtered1.shift() < filtered2.shift())
  exit = (filtered1 < filtered2) & (filtered1.shift() > filtered2.shift())

  figures = {
      'overlaps': {
          'trend1': filtered1,
          'trend2': filtered2,
      }
  }
  return entries, exit, figures