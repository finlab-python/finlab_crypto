import pandas as pd
import pytest
import requests

from finlab_crypto import onchain

SOPR_ROWS = [{'d': '2024-01-01', 'unixTs': 1704067200, 'sopr': 1.01},
             {'d': '2024-01-02', 'unixTs': 1704153600, 'sopr': 0.99}]


@pytest.mark.network
def test_coinmetrics_daily_metrics():
    metrics = onchain.coinmetrics(['HashRate', 'FlowInExNtv'])

    assert list(metrics.columns) == ['HashRate', 'FlowInExNtv']
    assert metrics.index.tz is not None
    assert metrics.loc['2020-01-01':, 'HashRate'].notna().all()


def test_sopr_is_cached_and_survives_outages(monkeypatch):
    calls = []

    def fake_get_json(url, params=None):
        calls.append(url)
        return SOPR_ROWS

    monkeypatch.setattr(onchain, 'get_json', fake_get_json)
    first = onchain.sopr()
    assert onchain.sopr().equals(first)
    assert len(calls) == 1

    def failing_get_json(url, params=None):
        raise requests.HTTPError('429 Too Many Requests')

    monkeypatch.setattr(onchain, 'get_json', failing_get_json)
    monkeypatch.setattr(onchain, 'SOPR_CACHE_MAX_AGE_SECONDS', 0)
    with pytest.warns(UserWarning, match='using cached'):
        assert onchain.sopr().equals(first)
    assert first.to_list() == [1.01, 0.99]


def test_align_waits_for_the_day_to_close():
    daily = pd.Series([1.0, 2.0], index=pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True))
    index = pd.date_range('2024-01-01', periods=12, freq='8h', tz='UTC')

    aligned = onchain.align(daily, index)

    assert aligned.iloc[:3].isna().all()
    assert (aligned.iloc[3:6] == 1.0).all()
    assert (aligned.iloc[6:] == 2.0).all()
