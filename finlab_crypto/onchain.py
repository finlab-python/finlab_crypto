"""Free on-chain metrics that need no API key.

* Coin Metrics Community API (https://coinmetrics.io/community-network-data/):
  daily hash rate, issuance, exchange flows, price... since 2009.
  Licensed CC BY-NC 4.0, attribute "Coin Metrics" when you publish results.
* BGeometrics (https://bitcoin-data.com): SOPR. The free tier covers roughly
  the latest four years, lags seven days and allows about ten requests per
  hour per IP, so results are cached in ``history/``.
"""
import time
import warnings

import pandas as pd
import requests

from finlab_crypto._http import get_json
from finlab_crypto.crawler import HISTORY_DIR

COINMETRICS_URL = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
COINMETRICS_PAGE_SIZE = 10000
BGEOMETRICS_URL = 'https://bitcoin-data.com/v1'
SOPR_CACHE_MAX_AGE_SECONDS = 12 * 60 * 60
PUBLICATION_DELAY = pd.Timedelta(days=1)


def coinmetrics(metrics, asset='btc'):
    """Daily metrics from the Coin Metrics Community API.

    Args:
      metrics (list[str]): Metric ids, ex: ['HashRate', 'FlowInExNtv'].
        Free metrics: https://docs.coinmetrics.io/api/v4 (community tier).
      asset (str): Asset id, ex: 'btc', 'eth'.

    Returns:
      pd.DataFrame: One float column per metric, indexed by UTC date.
    """
    rows = []
    url = COINMETRICS_URL
    params = {'assets': asset, 'metrics': ','.join(metrics), 'frequency': '1d',
              'page_size': COINMETRICS_PAGE_SIZE}
    while url:
        page = get_json(url, params)
        rows += page['data']
        url, params = page.get('next_page_url'), None

    data = pd.DataFrame(rows)
    data.index = pd.to_datetime(data['time'], utc=True).rename('timestamp')
    return data[metrics].astype(float)


def sopr():
    """Bitcoin Spent Output Profit Ratio (SOPR) from BGeometrics.

    Returns:
      pd.Series: Daily SOPR indexed by UTC date.
    """
    path = HISTORY_DIR / 'sopr.csv'
    is_fresh = path.is_file() and time.time() - path.stat().st_mtime < SOPR_CACHE_MAX_AGE_SECONDS
    if not is_fresh:
        try:
            rows = get_json(f'{BGEOMETRICS_URL}/sopr')
            HISTORY_DIR.mkdir(exist_ok=True)
            pd.DataFrame(rows)[['d', 'sopr']].to_csv(path, index=False)
        except requests.RequestException as error:
            if not path.is_file():
                raise
            warnings.warn(f'SOPR download failed ({error}); using cached {path}', stacklevel=2)

    data = pd.read_csv(path)
    return pd.Series(data['sopr'].to_numpy(), name='sopr',
                     index=pd.to_datetime(data['d'], utc=True).rename('timestamp'))


def align(series, index):
    """Put a daily metric on a price index without look-ahead.

    A daily value is only known after its day closes, so it is shifted by one
    day before being forward-filled onto ``index``.
    """
    shifted = series.copy()
    shifted.index = shifted.index + PUBLICATION_DELAY
    return shifted.reindex(shifted.index.union(index)).ffill().reindex(index)
