import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from finlab_crypto import Filter, Strategy
from finlab_crypto.talib_filter import TalibFilter
from finlab_crypto.talib_strategy import TalibStrategy


@Strategy(n1=10, n2=20)
def sma_strategy(ohlcv):
    sma1 = ohlcv.close.rolling(sma_strategy.n1).mean()
    sma2 = ohlcv.close.rolling(sma_strategy.n2).mean()
    return sma1 > sma2, sma1 < sma2, {'overlaps': {'sma1': sma1, 'sma2': sma2}}


@Filter(window=20)
def momentum_filter(ohlcv):
    momentum = ohlcv.close.pct_change(momentum_filter.window)
    return momentum > 0, {'figures': {'momentum': momentum}}


def expected_signals(ohlcv, n1, n2):
    sma1 = ohlcv.close.rolling(n1).mean()
    sma2 = ohlcv.close.rolling(n2).mean()
    return sma1 > sma2, sma1 < sma2


def test_default_parameters(ohlcv):
    entries, exits, figures = sma_strategy.backtest(ohlcv, signals=True)

    expected_entries, expected_exits = expected_signals(ohlcv, 10, 20)
    assert (entries.to_numpy() == expected_entries.to_numpy()).all()
    assert (exits.to_numpy() == expected_exits.to_numpy()).all()
    assert set(figures['overlaps']) == {'sma1', 'sma2'}


def test_variables_override_defaults(ohlcv):
    entries, _, _ = sma_strategy.backtest(ohlcv, {'n1': 5, 'n2': 30}, signals=True)

    assert (entries.to_numpy() == expected_signals(ohlcv, 5, 30)[0].to_numpy()).all()
    assert sma_strategy.n1 == 5


def test_strategy_without_parameters(ohlcv):
    @Strategy()
    def always_in(ohlcv):
        return pd.Series(True, ohlcv.index), pd.Series(False, ohlcv.index)

    portfolio = always_in.backtest(ohlcv, freq='4h')

    assert isinstance(portfolio, vbt.Portfolio)
    assert portfolio.total_return() == pytest.approx(
        portfolio.close.iloc[-1] / portfolio.close.iloc[0] - 1, rel=0.01)


def test_optimisation_labels_columns_with_parameters(ohlcv):
    portfolio = sma_strategy.backtest(ohlcv, {'n1': np.arange(5, 15, 5), 'n2': [20, 30]}, freq='4h', plot=True)

    assert portfolio.wrapper.columns.names == ['n1', 'n2']
    assert list(portfolio.wrapper.columns) == [(5, 20), (5, 30), (10, 20), (10, 30)]


def test_single_backtest_saves_interactive_chart(ohlcv, tmp_path):
    html = tmp_path / 'chart.html'

    sma_strategy.backtest(ohlcv, freq='4h', plot=True, html=str(html))

    page = html.read_text()
    assert 'echarts.init' in page and 'sma1' in page


def test_stop_variables_add_stop_columns(ohlcv):
    portfolio = sma_strategy.backtest(ohlcv, {'sl_stop': [0.01, 0.05], 'tp_stop': 0.1}, freq='4h')
    assert portfolio.wrapper.columns.names[:2] == ['sl_stop', 'tp_stop']

    trailing = sma_strategy.backtest(ohlcv, {'ts_stop': [0.01, 0.05]}, freq='4h')
    assert trailing.wrapper.columns.names[0] == 'ts_stop'


def test_stops_only_shorten_positions(ohlcv):
    plain = sma_strategy.backtest(ohlcv, freq='4h')
    stopped = sma_strategy.backtest(ohlcv, {'sl_stop': 0.001}, freq='4h')

    assert stopped.positions.records.shape[0] >= plain.positions.records.shape[0]


def test_invalid_stop_combination(ohlcv):
    with pytest.raises(ValueError):
        sma_strategy.backtest(ohlcv, {'sl_stop': 0.1, 'ts_stop': 0.1}, freq='4h')


def test_filters_are_applied_to_entries(ohlcv):
    entries, _, figures = sma_strategy.backtest(
        ohlcv, filters={'mom': momentum_filter.create({'window': [10, 30]})}, signals=True)

    assert entries.columns.names == ['mom_window', 'n1', 'n2']
    unfiltered = expected_signals(ohlcv, 10, 20)[0]
    assert (entries.to_numpy() <= unfiltered.to_numpy()[:, None]).all()
    assert 'mom_momentum' in figures['figures']


def test_execution_at_next_open(ohlcv):
    portfolio = sma_strategy.backtest(ohlcv, freq='4h', execution_price='open')
    first = portfolio.orders.records.iloc[0]

    assert first.price == pytest.approx(ohlcv.open.iloc[int(first.idx) + 1] * (1 + vbt.settings.portfolio['slippage']))


def test_short_side_is_not_supported(ohlcv):
    with pytest.raises(NotImplementedError):
        sma_strategy.backtest(ohlcv, side='short')


def test_talib_strategy_and_filter(ohlcv):
    macd = TalibStrategy('MACD', lambda ohlcv, macd: macd.macdhist > 0, lambda ohlcv, macd: macd.macdhist < 0)
    bbands = TalibStrategy('BBANDS', lambda ohlcv, bb: ohlcv.close > bb.upperband,
                           lambda ohlcv, bb: ohlcv.close < bb.lowerband)
    angle = TalibFilter('LINEARREG_ANGLE', lambda ohlcv, angle: angle > 0)
    rsi = TalibFilter('RSI', lambda ohlcv, rsi, params: rsi > params['threshold'], threshold=50)

    assert isinstance(bbands.backtest(ohlcv, {'timeperiod': 20}, freq='4h', plot=True), vbt.Portfolio)
    portfolio = macd.backtest(ohlcv, {'fastperiod': [6, 12]},
                              filters={'angle': angle.create({'timeperiod': 14}),
                                       'rsi': rsi.create({'threshold': [40, 60]})},
                              freq='4h')
    assert portfolio.wrapper.columns.names[:2] == ['rsi_threshold', 'angle_timeperiod']
    assert portfolio.wrapper.shape[1] == 4
