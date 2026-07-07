#!/usr/bin/env python3
"""Resume run_all.py from the already-saved epochs/features (skips the
expensive raw-loading/preprocessing stage) — used when only the deep
learning / stats / report stages need to be re-run.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mne
import pandas as pd

from pipeline import config, rois, models_classical, models_deep, stats as pstats, report, io_raw

mne.set_log_level("WARNING")


def main(task="NR", n_permutations=1000, deep_epochs=25):
    combined = mne.read_epochs(
        os.path.join(config.DIRS["epochs"], f"{task}_validation-epo.fif"), preload=True, verbose=False)
    X = pd.read_parquet(os.path.join(config.DIRS["features"], f"{task}_X.parquet"))
    meta = pd.read_parquet(os.path.join(config.DIRS["features"], f"{task}_meta.parquet"))
    y = (meta["label"].to_numpy() == "reading").astype(int)
    subjects_arr = meta["subject"].to_numpy()
    roi_channels = rois.compute_roi_channels(ch_names=combined.ch_names, save=False)
    results = pd.read_csv(os.path.join(config.DIRS["models"], f"{task}_classical_results.csv"))

    print("=== Deep learning (subject-independent) ===")
    si_splits = list(models_classical.subject_independent_splits(y, subjects_arr))
    raw_data = combined.get_data(copy=False) * 1e6
    deep_rows = []
    for name in models_deep.MODEL_CLASSES:
        for tr, te_idx, fold in si_splits:
            m, _pred, _proba, _model = models_deep.train_and_eval(
                name, raw_data[tr], y[tr], raw_data[te_idx], y[te_idx], n_epochs=deep_epochs)
            deep_rows.append({"model": name, "fold": fold, **m})
            print(f"  {name} / {fold}: {m}")
    deep_results = pd.DataFrame(deep_rows)
    deep_results.to_csv(os.path.join(config.DIRS["models"], f"{task}_deep_results.csv"), index=False)

    print("=== Statistics ===")
    sig_summary = pstats.feature_significance_summary(X, y, subjects_arr, n_perm=n_permutations)
    sig_summary.to_csv(os.path.join(config.DIRS["stats"], f"{task}_feature_significance.csv"))

    cluster_rows = []
    for roi in roi_channels:
        try:
            _T, clusters, cluster_p, _H0, _times = pstats.cluster_permutation_erp(
                combined, roi_channels, roi, n_permutations=n_permutations)
            n_sig = int(sum(p < 0.05 for p in cluster_p))
            cluster_rows.append({"roi": roi, "n_clusters": len(clusters),
                                  "n_significant": n_sig,
                                  "min_p": float(min(cluster_p)) if len(cluster_p) else None})
        except Exception as exc:
            print(f"cluster permutation failed for {roi}: {exc}")
    pd.DataFrame(cluster_rows).to_csv(
        os.path.join(config.DIRS["stats"], f"{task}_cluster_permutation_erp.csv"), index=False)
    print(pd.DataFrame(cluster_rows))

    ds_stats = io_raw.dataset_stats(task, subjects=sorted(set(subjects_arr)))
    preprocessing_summary = {}  # not persisted from stage 1; report notes QA plots are on disk separately
    report.build_report(ds_stats, preprocessing_summary, sig_summary, results, deep_results, roi_channels)
    print("DONE")


if __name__ == "__main__":
    main()
