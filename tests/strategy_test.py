from finlab_crypto import Strategy, Filter
from datetime import timezone
import finlab_crypto
import pandas as pd
import numpy as np
import unittest
import warnings
import datetime
import time
import json
import os

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings(
    'ignore',
    category=ResourceWarning,
    message='ResourceWarning: unclosed'
)

class TestStrategyMethods(unittest.TestCase):

    def setUp(self):
        finlab_crypto.setup()
        np.random.seed(0)
        self.ohlcv = pd.DataFrame({
            'open': np.random.normal(1, 0.02, size=100).clip(0, None),
            'close': np.random.normal(1, 0.02, size=100).clip(0, None),
            'high': np.random.normal(1, 0.02, size=100).clip(0, None),
            'low': np.random.normal(1, 0.02, size=100).clip(0, None),
            'volume': np.random.normal(1, 0.02, size=100).clip(0, None)
            }, index=pd.date_range('2000-01-01', periods=100))

    def test_strategy_without_params(self):
        @Strategy()
        def sma_strategy(ohlcv):
            sma1 = ohlcv.close.rolling(10).mean()
            sma2 = ohlcv.close.rolling(20).mean()
            return (sma1 > sma2), (sma2 > sma1)

        ohlcv = self.ohlcv
        sma1 = ohlcv.close.rolling(10).mean()
        sma2 = ohlcv.close.rolling(20).mean()

        entries_signal = (sma1 > sma2).fillna(False)
        exits_signal = (sma1 > sma2).fillna(False)

        entries, exits, fig = sma_strategy.backtest(ohlcv, signals=True, freq='4h')

        self.assertEqual((entries_signal != entries).any(), False)
        self.assertEqual((exits_signal != exits).all(), False)
        self.assertEqual(len(fig), 0)

    def test_strategy_witht_default_params(self):
        @Strategy(n1=10, n2=20)
        def sma_strategy(ohlcv):
            sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
            sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
            return (sma1 > sma2), (sma2 > sma1)

        ohlcv = self.ohlcv

        sma1 = ohlcv.close.rolling(10).mean()
        sma2 = ohlcv.close.rolling(20).mean()

        entries_signal = (sma1 > sma2).fillna(False)
        exits_signal = (sma1 > sma2).fillna(False)

        entries, exits, fig = sma_strategy.backtest(ohlcv, signals=True, freq='4h')

        self.assertEqual((entries_signal != entries).any(), False)
        self.assertEqual((exits_signal != exits).all(), False)
        self.assertEqual(len(fig), 0)


    def test_strategy_with_setting_params(self):
        @Strategy(n1=20, n2=30)
        def sma_strategy(ohlcv):
            sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
            sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
            return (sma1 > sma2), (sma2 > sma1)

        ohlcv = self.ohlcv

        sma1 = ohlcv.close.rolling(10).mean()
        sma2 = ohlcv.close.rolling(20).mean()

        entries_signal = (sma1 > sma2).fillna(False)
        exits_signal = (sma1 > sma2).fillna(False)

        svars = {'n1': 10, 'n2': 20}

        entries, exits, fig = sma_strategy.backtest(ohlcv, variables=svars, signals=True, freq='4h')

        self.assertEqual((entries_signal != entries).any(), False)
        self.assertEqual((exits_signal != exits).all(), False)
        self.assertEqual(len(fig), 0)

    def test_strategy_with_signle_params(self):
        @Strategy(n1=20, n2=30)
        def sma_strategy(ohlcv):
            sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
            sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
            return (sma1 > sma2), (sma2 > sma1)

        ohlcv = self.ohlcv

        sma1 = ohlcv.close.rolling(10).mean()
        sma2 = ohlcv.close.rolling(20).mean()

        entries_signal = (sma1 > sma2).fillna(False)
        exits_signal = (sma1 > sma2).fillna(False)

        svars = {'n1': 10, 'n2': 20}

        import vectorbt as vbt
        portfolio = sma_strategy.backtest(ohlcv, variables=svars, freq='4h', plot=True)
        self.assertEqual(isinstance(portfolio, vbt.Portfolio), True)

    def test_strategy_with_multi_params(self):
        @Strategy(n1=20, n2=30)
        def sma_strategy(ohlcv):
            sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
            sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
            return (sma1 > sma2), (sma2 > sma1)

        ohlcv = self.ohlcv

        sma1 = ohlcv.close.rolling(10).mean()
        sma2 = ohlcv.close.rolling(20).mean()

        svars = {'n1': [10, 11, 12], 'n2': 20}

        import vectorbt as vbt
        portfolio = sma_strategy.backtest(ohlcv, variables=svars, freq='4h', plot=True)
        self.assertEqual(isinstance(portfolio, vbt.Portfolio), True)


class TestFilterMethods(unittest.TestCase):

    def setUp(self):
        finlab_crypto.setup()
        np.random.seed(0)
        self.ohlcv = pd.DataFrame({
            'open': np.random.normal(1, 0.02, size=100).clip(0, None),
            'close': np.random.normal(1, 0.02, size=100).clip(0, None),
            'high': np.random.normal(1, 0.02, size=100).clip(0, None),
            'low': np.random.normal(1, 0.02, size=100).clip(0, None),
            'volume': np.random.normal(1, 0.02, size=100).clip(0, None)
        }, index=pd.date_range('2000-01-01', periods=100))

    def test_filter_without_params(self):
        @Filter()
        def mmi_filter(ohlcv):
            median = ohlcv.close.rolling(20).median()
            p1 = ohlcv.close > median
            p2 = ohlcv.close.shift() > median
            mmi = (p1 & p2).astype(int).rolling(20).mean()

            # if you don't want to show any figures, use figures={}
            figures = {}
            return mmi > 0.5, figures

        ohlcv = self.ohlcv
        num = 20
        median = ohlcv.close.rolling(num).median()
        p1 = ohlcv.close > median
        p2 = ohlcv.close.shift() > median
        mmi = (p1 & p2).rolling(num).sum() / num
        result = mmi > 0.5

        f20 = mmi_filter.create({'timeperiod': 20})
        test_result = f20(ohlcv)

        self.assertEqual((test_result[0].iloc[:, 0] != result).any(), False)
        self.assertEqual(len(test_result[1]), 0)

    def test_filter_with_default_params(self):
        @Filter(timeperiod=20)
        def mmi_filter(ohlcv):
            median = ohlcv.close.rolling(mmi_filter.timeperiod).median()
            p1 = ohlcv.close > median
            p2 = ohlcv.close.shift() > median
            mmi = (p1 & p2).astype(int).rolling(mmi_filter.timeperiod).mean()

            # if you don't want to show any figures, use figures={}
            figures = {}
            return mmi > 0.5, figures

        ohlcv = self.ohlcv
        num = 20
        median = ohlcv.close.rolling(num).median()
        p1 = ohlcv.close > median
        p2 = ohlcv.close.shift() > median
        mmi = (p1 & p2).rolling(num).sum() / num
        result = mmi > 0.5

        f20 = mmi_filter.create({})
        test_result = f20(ohlcv)

        self.assertEqual((test_result[0].iloc[:, 0] != result).any(), False)
        self.assertEqual(len(test_result[1]), 0)

    def test_filter_with_setting_params(self):
        @Filter(timeperiod=20)
        def mmi_filter(ohlcv):
            median = ohlcv.close.rolling(mmi_filter.timeperiod).median()
            p1 = ohlcv.close > median
            p2 = ohlcv.close.shift() > median
            mmi = (p1 & p2).astype(int).rolling(mmi_filter.timeperiod).mean()

            # if you don't want to show any figures, use figures={}
            figures = {}
            return mmi > 0.5, figures

        ohlcv = self.ohlcv
        num = 60
        median = ohlcv.close.rolling(num).median()
        p1 = ohlcv.close > median
        p2 = ohlcv.close.shift() > median
        mmi = (p1 & p2).rolling(num).sum() / num
        result = mmi > 0.5

        f60 = mmi_filter.create({'timeperiod': 60})
        test_result = f60(ohlcv)

        self.assertEqual((test_result[0].iloc[:, 0] != result).any(), False)
        self.assertEqual(len(test_result[1]), 0)

