"""Trend filters (moving averages and digital filters) used in the course.

Every function takes a price ``pd.Series`` and a period ``n`` and returns a
smoothed ``pd.Series`` on the same index. ``trends`` maps names to functions.
"""
import math

import numpy as np
import pandas as pd
import scipy.signal as sg
import talib

HIGHPASS_POLE = 0.707
ALMA_SIGMA_RATIO = 0.2


def sma(price, n):
    """Simple moving average."""
    return price.rolling(n).mean()


def wma(price, n):
    """Exponentially weighted moving average with centre of mass ``n``."""
    return price.ewm(com=n).mean()


def highpass(price, n=48):
    """John F. Ehlers' 2-pole high-pass filter (removes the trend, keeps the noise)."""
    a = HIGHPASS_POLE * 2 * math.pi / n
    alpha = (math.cos(a) + math.sin(a) - 1) / math.cos(a)
    b = 1 - alpha / 2
    c = 1 - alpha

    x = price.to_numpy(dtype=float)
    drive = np.zeros_like(x)
    drive[2:] = b * b * (x[2:] - 2 * x[1:-1] + x[:-2])
    return pd.Series(sg.lfilter([1], [1, -2 * c, c * c], drive), index=price.index)


def lowpass(price, n):
    """John F. Ehlers' 2-pole low-pass filter."""
    a = 2 / (1 + n)
    x = price.to_numpy(dtype=float)
    feedforward = [a - a * a / 4, a * a / 2, -(a - 3 * a * a / 4)]
    feedback = [1, -2 * (1 - a), (1 - a) ** 2]

    filtered = x.copy()
    drive = sg.lfilter(feedforward, [1], x)[2:]
    initial = sg.lfiltic([1], feedback, y=[x[1], x[0]])
    filtered[2:] = sg.lfilter([1], feedback, drive, zi=initial)[0]
    return pd.Series(filtered, index=price.index)


def hullma(price, n):
    """Hull moving average."""
    return wma(wma(price, n // 2) * 2 - wma(price, n), int(math.sqrt(n)))


def zlma(price, n):
    """John Ehlers' zero-lag exponential moving average."""
    lag = (n - 1) // 2
    return wma(2 * price - price.shift(lag), n)


def alma(price, n):
    """Arnaud Legoux moving average (Gaussian-weighted, causal)."""
    window = sg.windows.gaussian(n, n * ALMA_SIGMA_RATIO)
    filtered = sg.convolve(price.to_numpy(dtype=float), window / window.sum())
    return pd.Series(filtered[:len(price)], index=price.index)


def detrend(price, n):
    """Price minus its high-pass component."""
    return price - highpass(price, n)


def linear_reg(price, n):
    """End point of an ``n``-bar linear regression."""
    return talib.LINEARREG(price, timeperiod=n)


trends = {
    'sma': sma,
    'wma': wma,
    'lowpass': lowpass,
    'hullma': hullma,
    'zlma': zlma,
    'alma': alma,
    'detrend': detrend,
    'linear_reg': linear_reg,
}
