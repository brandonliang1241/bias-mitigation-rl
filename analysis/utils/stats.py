import numpy as np
from scipy.stats import ttest_ind

def bootstrap_mean_diff(a, b, n_bootstrap=10000):
    diffs = []
    n = len(a)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        diffs.append(np.mean(np.array(a)[idx]) - np.mean(np.array(b)[idx]))
    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def p_value(a, b):
    _, p = ttest_ind(a, b, equal_var=False)
    return p
