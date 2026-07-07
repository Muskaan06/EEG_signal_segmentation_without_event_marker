"""Assemble the full feature matrix (ERP + spectral + connectivity +
temporal-statistics features) for a set of epochs, plus normalization,
feature selection, PCA, and importance-ranking utilities.

Fold-aware scaling: `standardize_fit`/`standardize_apply` are split so
cross-validation code (pipeline/models_classical.py) can fit scaling on the
training fold only and apply it to the held-out fold, avoiding leakage.
`select_top_features`/`run_pca`/`rank_importance` here operate on a full
matrix and are meant for exploratory reporting (pipeline/report.py), not as
part of the CV-safe model-training path.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from pipeline import config, features_connectivity, features_erp, features_spectral
from pipeline.features_erp import _roi_waveform


def compute_temporal_stats(epochs, roi_channels_dict):
    """Basic post-stimulus temporal statistics per ROI: mean, variance,
    skewness, kurtosis (broadband, on the ROI-averaged waveform).
    """
    data = epochs.get_data(copy=False) * 1e6
    times = epochs.times
    post_mask = times >= 0
    ch_names = epochs.ch_names

    out = {}
    for roi, chans in roi_channels_dict.items():
        wave = _roi_waveform(data, ch_names, chans)
        if wave is None:
            continue
        seg = wave[:, post_mask]
        out[f"{roi}_temporal_mean"] = seg.mean(axis=1)
        out[f"{roi}_temporal_var"] = seg.var(axis=1)
        out[f"{roi}_temporal_skew"] = sp_stats.skew(seg, axis=1)
        out[f"{roi}_temporal_kurtosis"] = sp_stats.kurtosis(seg, axis=1)

    return pd.DataFrame(out, index=epochs.metadata.index if epochs.metadata is not None else None)


def build_feature_matrix(epochs, roi_channels_dict):
    """Returns (X, y, meta): X = full feature DataFrame (one row per epoch),
    y = binary label array (1=reading, 0=non_reading), meta = the epochs'
    metadata DataFrame (subject/sentence_id/word_id/etc.), all row-aligned.
    """
    erp = features_erp.compute_erp_features(epochs, roi_channels_dict)
    spectral = features_spectral.compute_spectral_features(epochs, roi_channels_dict)
    conn = features_connectivity.compute_connectivity_features(epochs, roi_channels_dict)
    temporal = compute_temporal_stats(epochs, roi_channels_dict)

    X = pd.concat([erp, spectral, conn, temporal], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (epochs.metadata["label"].to_numpy() == "reading").astype(int)
    meta = epochs.metadata.reset_index(drop=True).copy()
    X = X.reset_index(drop=True)
    return X, y, meta


def standardize_fit(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def standardize_apply(scaler, X):
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def select_top_features(X, y, k=30):
    k = min(k, X.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    return scores.head(k)


def run_pca(X, n_components=0.95, seed=config.SEED):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(Xz)
    return pca, scores


def rank_importance(X, y, seed=config.SEED):
    """Quick RandomForest impurity importance, for exploratory reporting
    (the real model comparison/importance lives in models_classical.py).
    """
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    return pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
