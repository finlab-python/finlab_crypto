import math

import numpy as np
import pytest

from finlab_crypto import indicators

PERIOD = 30


def highpass_loop(price, n):
    a = 0.707 * 2 * math.pi / n
    alpha = (math.cos(a) + math.sin(a) - 1) / math.cos(a)
    b, c = 1 - alpha / 2, 1 - alpha
    x, out = price.to_numpy(), np.zeros(len(price))
    for i in range(2, len(x)):
        out[i] = b * b * (x[i] - 2 * x[i - 1] + x[i - 2]) + 2 * c * out[i - 1] - c * c * out[i - 2]
    return out


def lowpass_loop(price, n):
    a = 2 / (1 + n)
    x = price.to_numpy()
    out = x.copy()
    for i in range(2, len(x)):
        out[i] = ((a - a * a / 4) * x[i] + a * a / 2 * x[i - 1] - (a - 3 * a * a / 4) * x[i - 2]
                  + 2 * (1 - a) * out[i - 1] - (1 - a) ** 2 * out[i - 2])
    return out


def test_digital_filters_match_their_recurrence(ohlcv):
    np.testing.assert_allclose(indicators.highpass(ohlcv.close, PERIOD), highpass_loop(ohlcv.close, PERIOD))
    np.testing.assert_allclose(indicators.lowpass(ohlcv.close, PERIOD), lowpass_loop(ohlcv.close, PERIOD))


@pytest.mark.parametrize('name', list(indicators.trends))
def test_trends_keep_the_index(ohlcv, name):
    trend = indicators.trends[name](ohlcv.close, PERIOD)

    assert trend.index.equals(ohlcv.index)
    assert trend.iloc[-1] == pytest.approx(ohlcv.close.iloc[-PERIOD:].mean(), rel=0.5)
