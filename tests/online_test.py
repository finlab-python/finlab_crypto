import pandas as pd
import pytest

from finlab_crypto.indicators import trends
from finlab_crypto.online import TradingMethod, TradingPortfolio, render_html
from finlab_crypto.strategy import Strategy

pytestmark = pytest.mark.network

LOOKBACK = 500


@Strategy(name='sma', n1=20, n2=40)
def trend_strategy(ohlcv):
    fast = trends[trend_strategy.name](ohlcv.close, trend_strategy.n1)
    slow = trends[trend_strategy.name](ohlcv.close, trend_strategy.n2)
    return (fast > slow) & (fast.shift() < slow.shift()), (fast < slow) & (fast.shift() > slow.shift())


@pytest.fixture(scope='module')
def portfolio():
    tp = TradingPortfolio()
    tp.register(TradingMethod(symbols=['ADABTC', 'ETHBTC', 'ADAUSDT'], freq='4h', lookback=LOOKBACK,
                              strategy=trend_strategy, variables={'name': 'sma', 'n1': 30, 'n2': 130},
                              weight_btc={'default': 0.01, 'ADABTC': 0.02}, name='altcoin'))
    tp.register(TradingMethod(symbols=['BTCUSDT'], freq='4h', lookback=LOOKBACK, strategy=trend_strategy,
                              variables={'name': 'hullma', 'n1': 70, 'n2': 108}, weight=5000,
                              weight_unit='USDT', name='btc'))
    tp.register_margin('USDT', 1000)
    return tp


@pytest.fixture(scope='module')
def ohlcvs(portfolio):
    return portfolio.get_ohlcvs()


def test_dry_run_without_key(portfolio):
    assert portfolio.is_dry_run
    assert portfolio.ticker_info.balances().empty


def test_lookback_covers_conversion_symbols(portfolio, ohlcvs):
    assert {('ADABTC', '4h'), ('ADAUSDT', '4h'), ('BTCUSDT', '4h')} <= set(ohlcvs)
    assert all(len(ohlcv) == LOOKBACK for ohlcv in ohlcvs.values())


def test_signals_are_stable_while_in_a_position(portfolio, ohlcvs):
    times = ohlcvs[('ADABTC', '4h')].index[-6:]
    signals = [portfolio.get_latest_signals({k: v.loc[:t] for k, v in ohlcvs.items()}) for t in times]

    for before, after in zip(signals, signals[1:]):
        held = (before.amount != 0) & (after.amount != 0)
        assert (before.amount[held] == after.amount[held]).all()


def test_position_size_and_orders(portfolio):
    info = portfolio.ticker_info
    signals = pd.DataFrame({
        'symbol': ['ADABTC', 'ETHBTC', 'ADAUSDT', 'BTCUSDT'],
        'latest_signal': [True, False, False, True],
        'weight_btc': [0.02, 0.01, 0.01, 0.05],
        'value_in_btc': [0.021, 0.0, 0.0, 0.05],
    })

    position, position_btc, orders = portfolio.calculate_position_size(signals)

    assert position_btc.loc['BTC', 'algo_p'] == pytest.approx(0.05 - 0.02)
    assert position_btc.loc['ADA', 'algo_p'] == pytest.approx(0.021)
    assert (position_btc.algo_p + position_btc.margin_p - position_btc.estimate_p).abs().sum() == pytest.approx(0)
    difference = position_btc.estimate_p.clip(0, None) - position_btc.present_p - position_btc.difference
    assert difference.abs().sum() == pytest.approx(0)
    assert position_btc.loc['USDT', 'excluded']
    assert position.loc['ADA', 'algo_p'] == pytest.approx(0.021 / info.get_asset_price_in_btc('ADA'))
    assert set(orders.index) <= {'ADABTC', 'ADAUSDT', 'BTCUSDT'}

    results = portfolio.execute_orders(orders, mode='TEST')
    assert (results.result == 'dry run').all()
    assert '<h1>Crypto Portfolio</h1>' in render_html(signals, position, position_btc, orders, results)


def test_live_orders_need_a_key(portfolio):
    empty = pd.DataFrame({'final_value': pd.Series(dtype=float)})
    with pytest.raises(RuntimeError):
        portfolio.execute_orders(empty, mode='MARKET')


def test_status_and_portfolio_backtest(portfolio, ohlcvs):
    assert portfolio.status(ohlcvs) is not None
    results = portfolio.portfolio_backtest(ohlcvs, '4h')
    assert len(results) == 4
