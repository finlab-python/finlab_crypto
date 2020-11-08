import sys
import math
import scipy
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt

def sma(price, n):
  return price.rolling(n).mean()
def wma(price, n):
  return price.ewm(com=n).mean()

# Highpass filter by John F. Ehlers, converted by DdlV
def highpass(Data, n=48):
  a	= (0.707*2*math.pi) / n

  alpha1 = (math.cos(a)+math.sin(a)-1)/math.cos(a);
  b	= 1-alpha1/2
  c	= 1-alpha1

  ret = [0] * len(Data)
  for i in range(2, len(Data)):
    ret[i] = b*b*(Data.iloc[i]-2*Data[i-1]+Data.iloc[i-2])+2*c*ret[i-1]-c*c*ret[i-2]

  return pd.Series(ret, index=Data.index)

# lowpass filter
def lowpass(Data,n):
  a = 2.0/(1+n)

  lp = [Data[0], Data[1]] + [0] * (len(Data) - 2)
  for i in range(2, len(Data)):
    lp[i] = (a-0.25*a*a)*Data[i]+ 0.5*a*a*Data[i-1]\
      - (a-0.75*a*a)*Data[i-2]\
      + 2*(1.-a)*lp[i-1]\
      - (1.-a)*(1.-a)*lp[i-2]

  return pd.Series(lp, index=Data.index)

def hullma(price, n):
  wma1 = wma(price, n//2)
  wma2 = wma(price, n)
  return wma(wma1 * 2 - wma2, int(math.sqrt(n)))

def zlma(price, n):
  """
  John Ehlers' Zero lag (exponential) moving average
  https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
  """
  lag = (n - 1) // 2
  series = 2 * price - price.shift(lag)
  return wma(series, n)

def alma(price, n):
  # triangular window with 60 samples.
  h = sg.gaussian(n, n*0.2)

  # We convolve the signal with this window.
  fil = sg.convolve(price, h / h.sum())
  filtered = pd.Series(fil[:len(price)], index=price.index)
  return filtered

def detrend(price, n):
  return price - highpass(price, n)

def linear_reg(price, n):
  import talib
  return talib.LINEARREG(price, timeperiod=n)

trends = {
    'sma': sma,
    'wma': wma,
    'lowpass': lowpass,
    'hullma': hullma,
    'zlma': zlma,
    'alma': alma,
    'detrend': detrend,
    'linear_reg': linear_reg
}
