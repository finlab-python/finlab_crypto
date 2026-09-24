"""Probability of backtest overfitting (Bailey et al., 2016).

Combinatorially symmetric cross-validation (CSCV): split the returns into
``n_bins`` blocks, use every half of the blocks as in-sample (IS) and the
rest as out-of-sample (OOS), and check how the IS winner ranks OOS.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DAYS_PER_YEAR = 365
EPSILON = 1e-7
DOMINANCE_GRID_POINTS = 1000
LOGIT_DISPLAY_FLOOR = -10000


def sharpe_ratio(returns):
    """Annualised Sharpe ratio of daily returns."""
    return returns.mean() / (returns.std() + EPSILON) * DAYS_PER_YEAR ** 0.5


def _ecdf(samples, points):
    return np.searchsorted(np.sort(samples), points, side='right') / len(samples)


class CSCV:
    """Combinatorially symmetric cross-validation.

    Attributes:
      n_bins: Number of blocks the return history is split into.
      objective: Performance measure of a block of returns, ex: ``lambda r: r.mean()``.
    """

    def __init__(self, n_bins=10, objective=sharpe_ratio):
        self.n_bins = n_bins
        self.objective = objective
        self.bins_enumeration = [set(c) for c in combinations(range(n_bins), n_bins // 2)]
        self._is_performance = [[] for _ in self.bins_enumeration]
        self._oos_performance = [[] for _ in self.bins_enumeration]

    def add_daily_returns(self, daily_returns):
        """Add strategies, one column of daily returns each."""
        bin_size = len(daily_returns) // self.n_bins
        bins = [daily_returns.iloc[i * bin_size:(i + 1) * bin_size] for i in range(self.n_bins)]

        for split, is_set in enumerate(self.bins_enumeration):
            is_returns = pd.concat([bins[i] for i in sorted(is_set)])
            oos_returns = pd.concat([bins[i] for i in range(self.n_bins) if i not in is_set])
            self._is_performance[split].append(self.objective(is_returns))
            self._oos_performance[split].append(self.objective(oos_returns))

    def estimate_overfitting(self, plot=False):
        """Estimate the probability of backtest overfitting.

        Returns:
          dict with
            pbo_test: probability that the IS-best strategy ranks below the OOS median.
            logits: logit of the OOS rank of the IS-best strategy, per split.
            R_n_star: IS performance of the IS-best strategy, per split.
            R_bar_n_star: OOS performance of the IS-best strategy, per split.
            dom_df: CDFs of OOS performance for optimised vs. non-optimised
              selection, and their second-order stochastic dominance ``SD2``.
        """
        is_perf = pd.DataFrame([pd.concat(split) for split in self._is_performance])
        oos_perf = pd.DataFrame([pd.concat(split) for split in self._oos_performance])
        oos_rank = oos_perf.rank(axis=1, ascending=False, method='first')

        best = is_perf.dropna(how='all').idxmax(axis=1)
        splits = best.index
        position = oos_perf.columns.get_indexer(best)
        r_star = pd.Series(is_perf.to_numpy()[splits, position])
        r_bar_star = pd.Series(oos_perf.to_numpy()[splits, position])
        best_oos_rank = pd.Series(oos_rank.to_numpy()[splits, position]).dropna()

        relative_rank = 1 - best_oos_rank / (is_perf.shape[1] + 1)
        logits = np.log(relative_rank / (1 - relative_rank))

        observed = r_bar_star.dropna().to_numpy()
        if len(observed):
            grid = np.linspace(observed.min(), observed.max(), DOMINANCE_GRID_POINTS)
            dom_df = pd.DataFrame({
                'optimized_IS': _ecdf(observed, grid),
                'non_optimized_OOS': _ecdf(oos_perf.median(axis=1).to_numpy(), grid),
            }, index=grid)
            dom_df['SD2'] = -(dom_df.non_optimized_OOS - dom_df.optimized_IS).cumsum()
        else:
            dom_df = pd.DataFrame(columns=['optimized_IS', 'non_optimized_OOS', 'SD2'])

        result = {
            'pbo_test': (logits < 0).mean(),
            'logits': logits.to_list(),
            'R_n_star': r_star.to_list(),
            'R_bar_n_star': r_bar_star.to_list(),
            'dom_df': dom_df,
        }
        if plot:
            plot_cscv(result)
            plt.show()
        return result


def plot_cscv(result):
    """Plot the logit histogram, performance degradation and stochastic dominance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.5)
    fig.suptitle('Combinatorially Symmetric Cross-validation')

    axes[0].set_title(f'Probability of overfitting: {result["pbo_test"] * 100:.2f} %')
    axes[0].hist(x=[logit for logit in result['logits'] if logit > LOGIT_DISPLAY_FLOOR], bins='auto')
    axes[0].set(xlabel='Logits', ylabel='Frequency')

    axes[1].set_title('Performance degradation')
    degradation = pd.DataFrame({'IS': result['R_n_star'], 'OOS': result['R_bar_n_star']}).dropna()
    sns.regplot(x=degradation.IS.to_numpy(), y=degradation.OOS.to_numpy(), ax=axes[1])
    axes[1].set(xlabel='In-sample Performance', ylabel='Out-of-sample Performance')

    axes[2].set_title('Stochastic dominance')
    if len(result['dom_df']):
        result['dom_df'].plot(ax=axes[2], secondary_y=['SD2'])
    axes[2].set(xlabel='Performance optimized vs non-optimized', ylabel='Frequency')
    return fig
