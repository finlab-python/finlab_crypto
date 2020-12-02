from finlab_crypto.crawler import GlassnodeClient
from binance.client import Client
from datetime import timezone
import finlab_crypto
import pandas as pd
import unittest
import warnings
import datetime
import time
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

    def test_crawler_n_bars(self):
        key = os.environ.get('BINANCE_KEY')
        secret = os.environ.get('BINANCE_SECRET')
        client = Client(key, secret)
        for i in [10, 100, 300, 1000]:
            ohlcv = finlab_crypto.crawler.get_nbars_binance('ADAUSDT', '4h', i, client=client)
            self.assertEqual(len(ohlcv) >= i, True)
            time.sleep(1)

    def test_glassnode_api(self):

        def get_glassnode(url, api_key):

            gn = GlassnodeClient()
            gn.api_key = api_key
            ret = gn.get(url)
            return ret.astype(float)

        sopr = get_glassnode('https://api.glassnode.com/v1/metrics/indicators/sopr',
                             'c5846d6e-b7ed-4d84-9339-03e88e6db3af')

        self.assertEqual(len(sopr) >= 10, True)

    # todo test glassnode
    # todo test bitmex
