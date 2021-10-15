import unittest
import warnings
import finlab_crypto
from finlab_crypto.strategy import Strategy
from finlab_crypto.indicators import trends
from finlab_crypto.online import TradingPortfolio, render_html
from finlab_crypto import online
import pandas as pd
import json
import os

warnings.filterwarnings(
    'ignore',
    category=ResourceWarning,
    message='ResourceWarning: unclosed'
)

class TestOnlineMethods(unittest.TestCase):

    def setUp(self):

        finlab_crypto.setup()

        @Strategy(name='sma', n1=20, n2=40)
        def trend_strategy(ohlcv):
          name = trend_strategy.name
          n1 = trend_strategy.n1
          n2 = trend_strategy.n2

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


        # altcoin strategy
        # --------------------
        # 'XRPBTC', 'ADABTC', 'LINKBTC', 'ETHBTC', 'VETBTC'
        # trend_strategy(ohlcv, variables={'name': 'sma', 'n1', 30, 'n2': 130}, freq='4h')

        from finlab_crypto.online import TradingMethod

        tm1 = TradingMethod(
            symbols=['ADABTC', 'VETBTC', 'ADAUSDT'],
            freq='4h',
            lookback=1000,
            strategy=trend_strategy,
            variables={'name': 'sma', 'n1': 30, 'n2': 130},
            weight_btc={'default': 0.01, 'ADABTC': 0.02},
            name='altcoin-trend-strategy-2020-10-31',
        )

        # btc strategy
        # --------------------
        # 'BTCUSDT'
        # trend_strategy(ohlcv, variables={'name': 'hullma', 'n1', 70, 'n2': 108}, freq='4h')

        tm2 = TradingMethod(
            symbols=['BTCUSDT'],
            freq='4h',
            lookback=1000,
            strategy=trend_strategy,
            variables={'name': 'hullma', 'n1': 70, 'n2': 108},
            weight=5000,
            weight_unit='USDT',
            name='btc-trend-strategy-2020-10-31',
        )

        key = os.environ.get('BINANCE_KEY')
        secret = os.environ.get('BINANCE_SECRET')



        tp = TradingPortfolio(key, secret)
        tp.register(tm1)
        tp.register(tm2)
        tp.register_margin('USDT', 1000)

        self.tp = tp
        self.ohlcvs = tp.get_full_ohlcvs()

    def test_adjust_quote_value(self):

        def trim_time(ohlcvs, time):
            ret = {}
            for key, df in self.ohlcvs.items():
                ret[key] = df.loc[:time]
            return ret

        dates = self.ohlcvs[('ADABTC', '4h')]['2019-01-01':'2019-01-5'].index

        for d1, d2 in zip(dates, dates[1:]):

            ohlcvs_temp = trim_time(self.ohlcvs, d1)
            signal1 = self.tp.get_latest_signals(ohlcvs_temp)

            ohlcvs_temp = trim_time(self.ohlcvs, d2)
            signal2 = self.tp.get_latest_signals(ohlcvs_temp)
            
            active_signals = (signal1.amount != 0) & (signal2.amount!=0)
            self.assertEqual((signal1.amount[active_signals] == signal2.amount[active_signals]).all(), True)

    def test_position_size(self):
        signals = pd.DataFrame(json.loads("""{"symbol":{"0":"XRPBTC","1":"ADABTC","2":"LINKBTC","3":"ETHBTC","4":"VETBTC","5":"ADAUSDT","6":"BTCUSDT"},"method name":{"0":"altcoin-trend-strategy-2020-10-31","1":"altcoin-trend-strategy-2020-10-31","2":"altcoin-trend-strategy-2020-10-31","3":"altcoin-trend-strategy-2020-10-31","4":"altcoin-trend-strategy-2020-10-31","5":"altcoin-trend-strategy-2020-10-31","6":"btc-trend-strategy-2020-10-31"},"latest_signal":{"0":false,"1":false,"2":false,"3":false,"4":true,"5":false,"6":true},"weight_btc":{"0":0.01,"1":0.01,"2":0.01,"3":0.01,"4":0.01,"5":0.01,"6":0.05},"freq":{"0":"4h","1":"4h","2":"4h","3":"4h","4":"4h","5":"4h","6":"4h"},"return":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0183486239,"5":0.0,"6":-0.0598689289},"value_in_btc":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0101834862,"5":0.0,"6":0.05},"latest_price":{"0":0.00009006,"1":0.00001112,"2":0.00011533,"3":0.030943,"4":0.00000111,"5":0.03818,"6":3434.28},"entry_price":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.00000109,"5":0.0,"6":3652.98},"entry_time":{"0":0,"1":0,"2":0,"3":0,"4":1547539200000,"5":0,"6":1545163200000}}"""))

        position_btc = self.tp.calculate_position_size(signals)[1]

        # test algo p
        self.assertEqual(position_btc.loc['BTC', 'algo_p'], 0.04)
        self.assertEqual(position_btc.loc['VET', 'algo_p'], 0.0101834862)

        print(position_btc)

        # test estimate p
        self.assertEqual((position_btc.algo_p + position_btc.margin_p - position_btc.estimate_p).sum() == 0, True)

        # test difference
        self.assertEqual((position_btc.estimate_p.clip(0, None) - position_btc.present_p - position_btc.difference).sum() == 0,True)

        # test excluded
        all_symbols = list(
                set(signals.symbol.map(self.tp.ticker_info.get_base_asset).values.tolist() +
                    signals.symbol.map(self.tp.ticker_info.get_quote_asset).values.tolist()))

        self.assertEqual(
                ((~position_btc.index.isin(all_symbols)) | (position_btc.index == 'USDT')
                 == position_btc.excluded).all(), True)

    def test_get_ohlcvs(self):
        ohlcvs = self.tp.get_ohlcvs()
        self.tp.status(ohlcvs)
        self.assertEqual(len(ohlcvs) != 0, True)

    def test_portfolio_backtest(self):
        result = self.tp.portfolio_backtest(self.ohlcvs, '4h')
        self.assertEqual(len(result) != 0, True)


