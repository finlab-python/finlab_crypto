from finlab_crypto import overfitting
import finlab_crypto
import pandas as pd
import numpy as np
import unittest
import warnings
import datetime
import time
import json
import os

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings(
    'ignore',
    category=ResourceWarning,
    message='ResourceWarning: unclosed'
)

class TestOverfittingMethods(unittest.TestCase):

    def setUp(self):
        finlab_crypto.setup()

    def test_overfitting(self):

        nstrategy = 10
        nreturns = 4001
        returns = pd.DataFrame({'s' + str(i): np.random.normal(0, 0.02, size=nreturns) for i in range(nstrategy)})
        returns['s1'] += 0.02

        results = overfitting.CSCV(returns, S=10, plot=False)

        self.assertEqual(results['pbo_test'], 0)


        returns = pd.DataFrame({'s' + str(i): np.random.normal(0, 0.02, size=nreturns) for i in range(nstrategy)})

        results = overfitting.CSCV(returns, S=10, plot=True)

        self.assertEqual(results['pbo_test'] > 0, True)
