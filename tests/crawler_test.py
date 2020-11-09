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

class TestCrawlerMethods(unittest.TestCase):

    def setUp(self):
        finlab_crypto.setup()

    def test_crawler_binance(self):
        ohlcv = finlab_crypto.crawler.get_all_binance('ADAUSDT', '4h')
        self.assertEqual('close' in ohlcv.columns, True)
        self.assertEqual(len(ohlcv) > 0, True)
        self.assertEqual(os.path.exists('history/ADAUSDT-4h-data.csv'), True)

    # todo test glassnode
    # todo test bitmex
