from statsmodels.distributions.empirical_distribution import ECDF
import itertools as itr
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import math

sharpe_ratio = lambda r: r.mean() / (r.std()+0.0000001) * (365 ** 0.5)

class CSCV(object):
    """Combinatorially symmetric cross-validation algorithm.

    Calculate backtesting about overfitting probability distribution and performance degradation.

    Attributes:
        n_bins:A int of CSCV algorithm bin size to control overfitting calculation.Default is 10.
        objective:A function of in sample(is) and out of sample(oos) return benchmark algorithm.Default is lambda r:r.mean().

    """
    def __init__(self, n_bins=10, objective=sharpe_ratio):
        self.n_bins = n_bins
        self.objective = objective
        self.bins_enumeration = [set(x) for x in itr.combinations(np.arange(10), 10 // 2)]

        self.Rs = [pd.Series(dtype=float) for i in range(len(self.bins_enumeration))]
        self.R_bars = [pd.Series(dtype=float) for i in range(len(self.bins_enumeration))]

    def add_daily_returns(self, daily_returns):
        """Add daily_returns in algorithm.

        Args:
          daily_returns: A dataframe of trading daily_returns.

        """
        bin_size = daily_returns.shape[0] // self.n_bins
        bins = [daily_returns.iloc[i*bin_size: (i+1) * bin_size] for i in range(self.n_bins)]

        for set_id, is_set in enumerate(self.bins_enumeration):
            oos_set = set(range(10)) - is_set
            is_returns = pd.concat([bins[i] for i in is_set])
            oos_returns = pd.concat([bins[i] for i in oos_set])
            R = self.objective(is_returns)
            R_bar = self.objective(oos_returns)
            self.Rs[set_id] = self.Rs[set_id].append(R)
            self.R_bars[set_id] = self.R_bars[set_id].append(R_bar)

    def estimate_overfitting(self, plot=False):
        """Estimate overfitting probability.

        Generate the result on Combinatorially symmetric cross-validation algorithm.
        Display related analysis charts.

        Args:
          plot: A bool of control plot display. Default is False.

        Returns:
          A dict of result include:
          pbo_test: A float of overfitting probability.
          logits: A float of estimated logits of OOS rankings.
          R_n_star: A list of IS performance of th trategies that has the best ranking in IS.
          R_bar_n_star: A list of find the OOS performance of the strategies that has the best ranking in IS.
          dom_df: A dataframe of optimized_IS, non_optimized_OOS data.

        """
        # calculate strategy performance in IS(R_df) and OOS(R_bar_df)
        R_df = pd.DataFrame(self.Rs)
        R_bar_df = pd.DataFrame(self.R_bars)

        # calculate ranking of the strategies
        R_rank_df = R_df.rank(axis=1, ascending=False, method='first')
        R_bar_rank_df = R_bar_df.rank(axis=1, ascending=False, method='first')

        # find the IS performance of th trategies that has the best ranking in IS
        r_star_series = (R_df * (R_rank_df == 1)).unstack().dropna()
        r_star_series = r_star_series[r_star_series != 0].sort_index(level=-1)

        # find the OOS performance of the strategies that has the best ranking in IS
        r_bar_star_series = (R_bar_df * (R_rank_df == 1)).unstack().dropna()
        r_bar_star_series = r_bar_star_series[r_bar_star_series != 0].sort_index(level=-1)

        # find the ranking of strategies which has the best ranking in IS
        r_bar_rank_series = (R_bar_rank_df * (R_rank_df == 1)).unstack().dropna()
        r_bar_rank_series = r_bar_rank_series[r_bar_rank_series != 0].sort_index(level=-1)

        # probability of overfitting

        # estimate logits of OOS rankings
        logits = (1-((r_bar_rank_series)/(len(R_df.columns)+1))).map(lambda p: math.log(p/(1-p)))
        prob = (logits < 0).sum() / len(logits)

        # stochastic dominance

        # caluclate
        if len(r_bar_star_series) != 0:
            y = np.linspace(
                min(r_bar_star_series), max(r_bar_star_series), endpoint=True, num=1000
            )

            # build CDF performance of best candidate in IS
            R_bar_n_star_cdf = ECDF(r_bar_star_series.values)
            optimized = R_bar_n_star_cdf(y)

            # build CDF performance of average candidate in IS
            R_bar_mean_cdf = ECDF(R_bar_df.median(axis=1).values)
            non_optimized = R_bar_mean_cdf(y)

            #
            dom_df = pd.DataFrame(
                dict(optimized_IS=optimized, non_optimized_OOS=non_optimized)
            , index=y)
            dom_df["SD2"] = -(dom_df.non_optimized_OOS - dom_df.optimized_IS).cumsum()
        else:
            dom_df = pd.DataFrame(columns=['optimized_IS', 'non_optimized_OOS', 'SD2'])

        ret = {
            'pbo_test': (logits < 0).sum() / len(logits),
            'logits': logits.to_list(),
            'R_n_star': r_star_series.to_list(),
            'R_bar_n_star': r_bar_star_series.to_list(),
            'dom_df': dom_df,
        }

        if plot:
            # probability distribution
            plt.title('Probability Distribution')
            plt.hist(x=[l for l in ret['logits'] if l > -10000], bins='auto')
            plt.xlabel('Logits')
            plt.ylabel('Frequency')
            plt.show()

            # performance degradation
            plt.title('Performance degradation')
            plt.scatter(ret['R_n_star'], ret['R_bar_n_star'])
            plt.xlabel('In-sample Performance')
            plt.ylabel('Out-of-sample Performance')

            # first and second Stochastic dominance
            plt.title('Stochastic dominance')
            ret['dom_df'].plot(secondary_y=['SD2'])
            plt.xlabel('Performance optimized vs non-optimized')
            plt.ylabel('Frequency')
            plt.show()

        return ret
