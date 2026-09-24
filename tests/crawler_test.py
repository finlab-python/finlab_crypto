import pandas as pd
import pytest

from finlab_crypto import crawler

pytestmark = pytest.mark.network


def test_get_all_binance_downloads_and_caches():
    ohlcv = crawler.get_all_binance('BTCUSDT', '1d')

    assert {'open', 'high', 'low', 'close', 'volume'} <= set(ohlcv.columns)
    assert ohlcv.index.tz is not None and ohlcv.index.is_monotonic_increasing
    assert not ohlcv.index.duplicated().any()
    assert ohlcv.index[0] == pd.Timestamp('2017-08-17', tz='UTC')
    assert (crawler.HISTORY_DIR / 'BTCUSDT-1d-data.csv').is_file()


def test_get_all_binance_extends_the_cache():
    full = crawler.get_all_binance('ETHBTC', '1d')
    path = crawler.HISTORY_DIR / 'ETHBTC-1d-data.csv'
    full.iloc[:-30].to_csv(path)

    resumed = crawler.get_all_binance('ETHBTC', '1d')

    pd.testing.assert_frame_equal(resumed.iloc[:-1], full.iloc[:-1])


@pytest.mark.parametrize('nbars', [10, 1000, 2500])
def test_get_nbars_binance(nbars):
    ohlcv = crawler.get_nbars_binance('BTCUSDT', '4h', nbars)

    assert len(ohlcv) == nbars
    assert ohlcv.index.is_monotonic_increasing and not ohlcv.index.duplicated().any()
    assert (ohlcv.index.to_series().diff().dropna() == pd.Timedelta('4h')).all()


def test_exchange_info_and_tickers():
    symbols = {s['symbol']: s for s in crawler.get_exchange_info()['symbols']}
    prices = {t['symbol']: float(t['price']) for t in crawler.get_symbol_tickers()}

    assert symbols['ETHBTC']['baseAsset'] == 'ETH'
    assert prices['BTCUSDT'] > 0
