import pandas as pd
import math
import os.path
import time
from bitmex import bitmex
from binance.client import Client
from datetime import timedelta, datetime, timezone
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)

import json
import requests
import pandas as pd

### CONSTANTS
binsizes = {"1m": 1, "5m": 5,'15m': 15, '30m': 30, "1h": 60, '2h': 120, "4h": 240, "1d": 1440}
batch_size = 750

### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source, client):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
    if source == "binance": new = pd.to_datetime(client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
    return old, new

def get_all_binance(symbol, kline_size, save=True, client=Client()):
    filename = 'history/%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance", client=client)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df = data_df[~data_df.index.duplicated(keep='last')]
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    data_df.index = pd.to_datetime(data_df.index, utc=True)
    data_df = data_df[~data_df.index.duplicated(keep='last')]
    return data_df.astype(float)


def get_nbars_binance(symbol, interval, nbars, client):
  interval_to_seconds = lambda interval: int(interval[0]) * {'m': 60, 'h': 60*60, 'd': 60*60*24}[interval[1]]

  # calculate crawl time interval
  now = datetime.now(tz=timezone.utc)
  crawl_period = timedelta(seconds=interval_to_seconds(interval) * nbars) + timedelta(days=1)
  test_time = (now - crawl_period).strftime('%d %b %Y')

  # download results
  klines = client.get_historical_klines(symbol, interval, start_str=test_time)
  data = pd.DataFrame(klines,
    columns = ['timestamp', 'open', 'high', 'low', 'close',
               'volume', 'close_time', 'quote_av', 'trades',
               'tb_base_av', 'tb_quote_av', 'ignore' ], dtype=float)
  data.index = pd.to_datetime(data['timestamp'], unit='ms')
  return data


def get_all_bitmex(symbol, kline_size, save = True):
    bitmex_client = bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
    filename = 'history/%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "bitmex")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    rounds = math.ceil(available_data / batch_size)
    if rounds > 0:
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (delta_min, symbol, available_data, kline_size, rounds))
        for round_num in tqdm_notebook(range(rounds)):
            time.sleep(1)
            new_time = (oldest_point + timedelta(minutes = round_num * batch_size * binsizes[kline_size]))
            data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=batch_size, startTime = new_time).result()[0]
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)

    data_df.set_index('timestamp', inplace=True)
    data_df = data_df[~data_df.index.duplicated(keep='last')]

    if save and rounds > 0: data_df.to_csv(filename)
    print('All caught up..!')
    data_df.index = pd.to_datetime(data_df.index, utc=True)
    return data_df.astype(float)


class GlassnodeClient:

  def __init__(self):
    self._api_key = ''

  @property
  def api_key(self):
    return self._api_key

  def set_api_key(self, value):
    self._api_key = value

  def get(self, url, a='BTC', i='24h', c='native', s=None, u=None):
    p = dict()
    p['a'] = a
    p['i'] = i
    p['c'] = c

    p['api_key'] = self.api_key

    r = requests.get(url, params=p)

    try:
       r.raise_for_status()
    except Exception as e:
        print(e)
        print(r.text)

    try:
        df = pd.DataFrame(json.loads(r.text))
        df = df.set_index('t')
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.sort_index()
        s = df.v
        s.name = '_'.join(url.split('/')[-2:])
        return s
    except Exception as e:
        print(e)
