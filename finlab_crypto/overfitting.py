from statsmodels.distributions.empirical_distribution import ECDF
import tqdm.notebook as tqdm
import itertools as itr
import scipy.stats as ss
import scipy.special as spec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sharp_ratio(x):
  mean = x.mean(axis=0)
  std = x.std(axis=0)
  return (mean / std)[std != 0]


def CSCV(
    M,
    S,
    threshold=1,
    metric_func=sharp_ratio,
    plot=True,
):
    """
    Based on http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
    Features:
    * training and test sets are of equal size, providing comparable accuracy
    to both IS and OOS Sharpe ratios.
    * CSCV is symmetric, decline in performance can only result from
    overfitting, not arbitrary discrepancies between the training and test
    sets.
    * CSCV respects the time-dependence and other season-dependent features
    present in the data.
    * Results are deterministic, can be replicated.
    * Dispersion in the distribution of logits conveys relevant info regarding
    the robustness of the strategy selection process.
    * Model-free, non-parametric. Logits distribution resembles the cumulative
    Normal distribution if w_bar are close to uniform distribution (i.e. the
    backtest appears to be information-less). Therefore, for good backtesting,
    the distribution of logits will be centered in a significantly positive
    value, and its tail will marginally cover the region of negative logit
    values.
    Limitations:
    * CSCV is symmetric, for some strategies, K-fold CV might be better.
    * Not suitable for time series with strong auto-correlation, especially
    when S is large.
    * Assumes all the sample statistics carry the same weight.
    * Entirely possible that all the N strategy configs have high but similar
    Sharpe ratios. Therefore, PBO may appear high, however, 'overfitting' here
    is among many 'skilful' strategies.
    Parameters:
    M:
        returns data, numpy or dataframe format.
    S:
        chuncks to devided M into, must be even number. Paper suggests setting
        S = 16. See paper for details of choice of S.
    metric_func:
        evaluation function for returns data
    threshold:
        used as prob. of OOS Loss calculation cutoff. For Sharpe ratio,
        this should be 0 to indicate probabilty of loss.
    n_jobs:
        if greater than 1 then enable parallel mode
    hist:
        Default False, whether to plot histogram for rank of logits.
        Some problems exist when S >= 10. Need to look at why numpy /
        matplotlib does it.
    Returns:
    PBO result in namedtuple, instance of PBO.
    """
    if S % 2 == 1:
        raise ValueError(
            "S must be an even integer, {:.1f} was given".format(S)
        )
    n_jobs = 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        n_jobs = max(1, ps.cpu_count(logical=False))

    if isinstance(M, pd.DataFrame):
        # conver to numpy values
        print("Convert from DataFrame to numpy array.")
        M = M.values

    # Paper suggests T should be 2x the no. of observations used by investor
    # to choose a model config, due to the fact that CSCV compares combinations
    # of T/2 observations with their complements.
    T, N = M.shape
    residual = T % S
    if residual != 0:
        M = M[residual:]
        T, N = M.shape

    sub_T = T // S

    print("Total sample size: {:,d}, chunck size: {:,d}".format(T, sub_T))

    # generate subsets, each of length sub_T
    Ms = []
    #Ms_values = []
    for i in range(S):
        start, end = i * sub_T, (i + 1) * sub_T
        #Ms.append((i, M[start:end, :]))
        Ms.append((i, start, end))
        #Ms_values.append(M[start:end, :])
    #Ms_values = np.array(Ms_values)

    #if verbose:
    print("No. of Chuncks: {:,d}".format(len(Ms)))

    # generate combinations
    Cs = [x for x in itr.combinations(Ms, S // 2)]

    #if verbose:
    print("No. of combinations = {:,d}".format(len(Cs)))

    # Ms_index used to find J_bar (complementary OOS part)
    Ms_index = set([x for x in range(len(Ms))])

    get_sub_m = lambda i: M[Ms[i][1]: Ms[i][2], :]

    # create J and J_bar
    if n_jobs < 2:
        #J = []
        #J_bar = []
        #R = []
        #R_bar = []
        rn = []
        rn_bar = []
        R_n_star = []
        R_bar_mean = []
        R_bar_n_star = []

        for i in tqdm.tqdm(range(len(Cs))):
            # make sure chucks are concatenated in their original order
            order = [x for x, _, _ in Cs[i]]
            sort_ind = sorted(order)

            Cs_values = np.array([get_sub_m(j) for j in sort_ind])
            # if verbose:
            #     print('Cs index = {}, '.format(order), end='')
            joined = np.concatenate(Cs_values)
            #J.append(joined)
            R = metric_func(joined)
            R_rank = ss.rankdata(R)

            # find Cs_bar
            Cs_bar_index = list(sorted(Ms_index - set(order)))
            Cs_values2 = np.array([get_sub_m(i) for i in Cs_bar_index])
            # if verbose:
            # print('Cs_bar_index = {}'.format(Cs_bar_index))
            #J_bar.append(np.concatenate(Ms_values[Cs_bar_index, :]))
            R_bar = metric_func(np.concatenate(Cs_values2))
            R_bar_rank = ss.rankdata(R_bar)

            best_rank_id = np.argmax(R_rank)
            best_rank_oos = R_bar_rank[best_rank_id]

            rn.append(best_rank_id)
            rn_bar.append(best_rank_oos)
            R_n_star.append(R[best_rank_id])
            R_bar_n_star.append(R_bar[best_rank_id])
            R_bar_mean.append(np.mean(R_bar))


        # compute matrices for J and J_bar, e.g. Sharpe ratio
        #R = [metric_func(j) for j in J]
        #R_bar = [metric_func(j) for j in J_bar]

        # compute ranks of metrics
        #R_rank = [ss.rankdata(x) for x in R]
        #R_bar_rank = [ss.rankdata(x) for x in R_bar]

        # find highest metric, rn contains the index position of max value
        # in each set of R (IS)
        #rn = [np.argmax(r) for r in R_rank]
        # use above index to find R_bar (OOS) in same index position
        # i.e. the same config / setting
        #rn_bar = [R_bar_rank[i][rn[i]] for i in range(len(R_bar_rank))]

        # formula in paper used N+1 as the denominator for w_bar. For good reason
        # to avoid 1.0 in w_bar which leads to inf in logits. Intuitively, just
        # because all of the samples have outperformed one cannot be 100% sure.
        w_bar = [float(r) / (N+1) for r in rn_bar]
        # logit(.5) gives 0 so if w_bar value is equal to median logits is 0
        logits = [spec.logit(w) for w in w_bar]
    else:
        pass

    # prob of overfitting
    phi = np.array([1.0 if lam <= 0 else 0.0 for lam in logits]) / len(Cs)
    pbo_test = np.sum(phi)

    print('probability of backtest overfitting', pbo_test)
    
    # performance degradation
    #R_n_star = np.array([R[i][rn[i]] for i in range(len(R))])
    #R_bar_n_star = np.array([R_bar[i][rn[i]] for i in range(len(R_bar))])
    lm = ss.linregress(x=R_n_star, y=R_bar_n_star)

    prob_oos_loss = np.sum(
        [1.0 if r < threshold else 0.0 for r in R_bar_n_star]
    ) / len(R_bar_n_star)

    # Stochastic dominance
    y = np.linspace(
        min(R_bar_n_star), max(R_bar_n_star), endpoint=True, num=1000
    )
    R_bar_n_star_cdf = ECDF(R_bar_n_star)
    optimized = R_bar_n_star_cdf(y)

    R_bar_cdf = ECDF(R_bar_mean)
    non_optimized = R_bar_cdf(y)

    dom_df = pd.DataFrame(
        dict(optimized_IS=optimized, non_optimized_OOS=non_optimized)
    )
    dom_df.index = y
    # visually, non_optimized curve above optimized curve indicates good
    # backtest with low overfitting.
    dom_df["SD2"] = dom_df.non_optimized_OOS - dom_df.optimized_IS
    

    pbox = {
        'pbo_test': pbo_test,
        'prob_oos_loss': prob_oos_loss,
        'lm': lm,
        'dom_df':dom_df,
        'rn':rn,
        'rn_bar': rn_bar,
        'w_bar': w_bar,
        'logits': logits,
        'R_n_star':R_n_star,
        'R_bar_n_star':R_bar_n_star,
        'Ms': Ms
    }

    if plot:

      # probability distribution
      plt.title('Probability Distribution')
      plt.hist(x=[l for l in pbox['logits'] if l > -10000], bins='auto')
      plt.xlabel('Logits')
      plt.ylabel('Frequency')
      plt.show()

      # performance degradation
      plt.title('Performance degradation')
      plt.scatter(pbox['R_n_star'], pbox['R_bar_n_star'])
      plt.xlabel('In-sample Performance')
      plt.ylabel('Out-of-sample Performance')

      # first and second Stochastic dominance
      plt.title('Stochastic dominance')
      pbox['dom_df'].plot()
      plt.xlabel('Performance optimized vs non-optimized')
      plt.ylabel('Frequency')
      plt.show()

    return pbox