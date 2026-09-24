import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

SYNTHETIC_BARS = 500
SYNTHETIC_SEED = 0


@pytest.fixture(scope='session', autouse=True)
def workspace(tmp_path_factory):
    """Run in a scratch directory so ./history/ caches never touch the repo."""
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('workspace'))
    yield
    os.chdir(previous)


@pytest.fixture
def ohlcv():
    """Random-walk OHLCV with a 4h UTC index."""
    rng = np.random.default_rng(SYNTHETIC_SEED)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, SYNTHETIC_BARS))),
                      index=pd.date_range('2020-01-01', periods=SYNTHETIC_BARS, freq='4h', tz='UTC'))
    spread = close * rng.uniform(0, 0.01, SYNTHETIC_BARS)
    return pd.DataFrame({
        'open': close.shift(fill_value=close.iloc[0]),
        'high': close + spread,
        'low': close - spread,
        'close': close,
        'volume': rng.uniform(1, 10, SYNTHETIC_BARS),
    })
