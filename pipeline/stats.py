"""Statistical analysis: paired t-tests, permutation tests, cluster-based
permutation tests over the ERP time course, and effect sizes — used to
determine which EEG features best distinguish READING vs NON-READING.
"""

import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_test
from scipy import stats as sp_stats

from pipeline import config
from pipeline.features_erp import _roi_waveform


def cohens_d(X, y):
    """Cohen's d (reading - non_reading) for every column of X. Returns a Series."""
    x1, x0 = X[y == 1], X[y == 0]
    n1, n0 = len(x1), len(x0)
    pooled_std = np.sqrt(((n1 - 1) * x1.var(axis=0, ddof=1) + (n0 - 1) * x0.var(axis=0, ddof=1)) / (n1 + n0 - 2))
    pooled_std_safe = pooled_std.replace(0, np.nan) if isinstance(pooled_std, pd.Series) else np.where(pooled_std == 0, np.nan, pooled_std)
    d = (x1.mean(axis=0) - x0.mean(axis=0)) / pooled_std_safe
    return d


def paired_ttest_by_subject(X, y, subjects):
    """Per-feature paired t-test across subjects: for each subject, the mean
    reading-feature minus mean non-reading-feature; then a paired t-test of
    those per-subject differences against zero. Returns a DataFrame indexed
    by feature with columns t_stat, p_value, mean_diff.
    """
    subj_diffs = []
    for subj in sorted(set(subjects)):
        mask = subjects == subj
        x1 = X[mask & (y == 1)]
        x0 = X[mask & (y == 0)]
        if len(x1) == 0 or len(x0) == 0:
            continue
        subj_diffs.append(x1.mean(axis=0) - x0.mean(axis=0))

    diffs = pd.DataFrame(subj_diffs)  # (n_subjects, n_features)
    t_stat, p_value = sp_stats.ttest_1samp(diffs, popmean=0, axis=0, nan_policy="omit")
    return pd.DataFrame({
        "t_stat": t_stat, "p_value": p_value, "mean_diff": diffs.mean(axis=0),
    }, index=X.columns)


def permutation_test_all_features(X, y, n_perm=1000, seed=config.SEED):
    """Vectorized label-shuffling permutation test for every feature at once.
    Returns a DataFrame indexed by feature with columns observed_diff, p_value.
    """
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy()
    y = np.asarray(y)
    observed = Xv[y == 1].mean(axis=0) - Xv[y == 0].mean(axis=0)

    n = len(y)
    ge_count = np.zeros(Xv.shape[1])
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        diff = Xv[perm_y == 1].mean(axis=0) - Xv[perm_y == 0].mean(axis=0)
        ge_count += (np.abs(diff) >= np.abs(observed))

    p_values = (ge_count + 1) / (n_perm + 1)
    return pd.DataFrame({"observed_diff": observed, "p_value": p_values}, index=X.columns)


def feature_significance_summary(X, y, subjects, n_perm=1000, seed=config.SEED):
    """Combines Cohen's d, paired t-test, and permutation test into one
    ranked table (most significant / largest effect first).
    """
    d = cohens_d(X, y)
    ttest = paired_ttest_by_subject(X, y, subjects)
    perm = permutation_test_all_features(X, y, n_perm=n_perm, seed=seed)
    out = pd.DataFrame({
        "cohens_d": d,
        "paired_t": ttest["t_stat"],
        "paired_p": ttest["p_value"],
        "perm_observed_diff": perm["observed_diff"],
        "perm_p": perm["p_value"],
    })
    out["abs_cohens_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_cohens_d", ascending=False)


def cluster_permutation_erp(epochs, roi_channels_dict, roi, n_permutations=1000, seed=config.SEED):
    """Cluster-based permutation test (independent samples) over the ERP time
    course for one ROI: READING vs NON-READING. Returns (T_obs, clusters,
    cluster_p_values, H0, times_ms).
    """
    data = epochs.get_data(copy=False) * 1e6
    labels = epochs.metadata["label"].to_numpy()
    wave = _roi_waveform(data, epochs.ch_names, roi_channels_dict[roi])

    reading = wave[labels == "reading"]
    non_reading = wave[labels == "non_reading"]

    rng = np.random.default_rng(seed)
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [reading, non_reading], n_permutations=n_permutations, tail=0,
        seed=int(rng.integers(0, 2**31 - 1)), out_type="mask", verbose=False,
    )
    return T_obs, clusters, cluster_p_values, H0, epochs.times * 1000.0
