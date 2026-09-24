import numpy as np
import pandas as pd

from finlab_crypto import overfitting

STRATEGIES = 10
DAYS = 4001


def random_returns(seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f's{i}': rng.normal(0, 0.02, DAYS) for i in range(STRATEGIES)})


def test_a_dominant_strategy_is_not_overfitted():
    returns = random_returns(0)
    returns['s1'] += 0.02
    cscv = overfitting.CSCV()
    cscv.add_daily_returns(returns)

    result = cscv.estimate_overfitting(plot=True)

    assert result['pbo_test'] == 0
    assert len(result['logits']) == len(cscv.bins_enumeration)


def test_pure_noise_is_overfitted():
    cscv = overfitting.CSCV()
    cscv.add_daily_returns(random_returns(1))

    assert cscv.estimate_overfitting()['pbo_test'] > 0


def test_n_bins_sets_the_number_of_splits():
    cscv = overfitting.CSCV(n_bins=6)
    cscv.add_daily_returns(random_returns(2))

    assert len(cscv.estimate_overfitting()['R_n_star']) == 20
