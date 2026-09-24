"""Binance market data from the public, key-free market-data endpoint.

``data-api.binance.vision`` serves the same klines, exchange info and ticker
prices as ``api.binance.com`` but is not geo-restricted, so downloads work on
Google Colab and other US-hosted machines where ``api.binance.com`` answers
HTTP 451.
"""
from pathlib import Path

import pandas as pd

from finlab_crypto._http import get_json

BINANCE_MARKET_DATA_URL = 'https://data-api.binance.vision/api/v3'
KLINE_LIMIT = 1000
KLINE_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
HISTORY_DIR = Path('history')


def _klines(symbol, interval, limit=KLINE_LIMIT, **params):
    return get_json(f'{BINANCE_MARKET_DATA_URL}/klines',
                    {'symbol': symbol, 'interval': interval, 'limit': limit, **params})


def _to_frame(rows):
    data = pd.DataFrame(rows, columns=KLINE_COLUMNS, dtype=float)
    data.index = pd.to_datetime(data.pop('timestamp'), unit='ms', utc=True)
    return data


def _read_cache(path):
    data = pd.read_csv(path, index_col='timestamp', float_precision='round_trip')
    data.index = pd.to_datetime(data.index, utc=True)
    return data.astype(float)


def _download_since(symbol, interval, start_ms):
    rows = []
    while True:
        page = _klines(symbol, interval, startTime=start_ms)
        rows += page
        if len(page) < KLINE_LIMIT:
            return _to_frame(rows)
        start_ms = page[-1][0] + 1


def get_all_binance(symbol, kline_size, save=True):
    """Download the full kline history of a Binance spot symbol.

    Rows already cached in ``history/{symbol}-{kline_size}-data.csv`` are
    reused, so later calls only fetch the newest candles.

    Args:
      symbol (str): Trading pair (ex: 'BTCUSDT').
      kline_size (str): Candle interval (ex: '1m', '15m', '1h', '4h', '1d').
      save (bool): Keep the result in ./history/ as a cache.

    Returns:
      pd.DataFrame: OHLCV and trade statistics indexed by UTC open time.
      The last row is the still-forming candle.
    """
    path = HISTORY_DIR / f'{symbol}-{kline_size}-data.csv'
    cached = _read_cache(path) if path.is_file() else None
    start_ms = int(cached.index[-1].timestamp() * 1000) if cached is not None else 0

    fresh = _download_since(symbol, kline_size, start_ms)
    data = fresh if cached is None else pd.concat([cached, fresh])
    data = data[~data.index.duplicated(keep='last')].sort_index()

    if save:
        HISTORY_DIR.mkdir(exist_ok=True)
        data.to_csv(path)

    print(f'{symbol} {kline_size}: {len(data)} candles '
          f'({data.index[0]:%Y-%m-%d} ~ {data.index[-1]:%Y-%m-%d %H:%M} UTC)')
    return data


def get_nbars_binance(symbol, interval, nbars):
    """Download the latest ``nbars`` candles of a Binance spot symbol.

    Args:
      symbol (str): Trading pair (ex: 'BTCUSDT').
      interval (str): Candle interval (ex: '1h', '4h', '1d').
      nbars (int): Number of candles, including the still-forming one.

    Returns:
      pd.DataFrame: Same format as ``get_all_binance``.
    """
    pages = []
    end_ms = None
    remaining = nbars
    while remaining > 0:
        page = _klines(symbol, interval, limit=min(remaining, KLINE_LIMIT), endTime=end_ms)
        if not page:
            break
        pages.insert(0, page)
        remaining -= len(page)
        end_ms = page[0][0] - 1
    data = _to_frame([row for page in pages for row in page])
    return data


def get_exchange_info():
    """Symbols, base/quote assets and trading filters of Binance spot."""
    return get_json(f'{BINANCE_MARKET_DATA_URL}/exchangeInfo')


def get_symbol_tickers():
    """Latest prices of every Binance spot symbol as ``[{'symbol', 'price'}]``."""
    return get_json(f'{BINANCE_MARKET_DATA_URL}/ticker/price')
