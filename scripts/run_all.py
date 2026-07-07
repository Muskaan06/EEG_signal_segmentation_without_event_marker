#!/usr/bin/env python3
"""End-to-end pipeline runner: raw EEG -> preprocessing -> epochs -> features
-> classical + deep models -> stats -> report.

Usage:
    python scripts/run_all.py --subjects ZAB ZDN ZJM --task NR
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mne
import pandas as pd

from pipeline import (config, io_raw, alignment, preprocessing, epochs as epochs_mod,
                       rois, features_matrix, features_erp, features_spectral,
                       features_connectivity, models_classical, models_deep,
                       stats as pstats, report)

mne.set_log_level("WARNING")


def process_subject(subj, task, preprocessing_summary):
    we, te = alignment.align_subject_task(subj, task)
    subj_reading, subj_non_reading = [], []

    for block_path in io_raw.raw_eeg_paths(subj, task):
        block_num = io_raw._block_number(block_path)
        tag = f"{subj}_{task}{block_num}"
        raw, _ = io_raw.load_raw_eeg_block(block_path)
        qa_dir = os.path.join(config.DIRS["figures"], "qa", tag)
        cleaned, info = preprocessing.preprocess_raw(raw, qa_dir=qa_dir, tag=tag)
        preprocessing_summary[tag] = info

        we_b = we[we["block_path"] == block_path]
        te_b = te[te["block_path"] == block_path]
        r_ep, nr_ep = epochs_mod.build_block_epochs(cleaned, we_b, te_b, subj, task, block_num)
        if r_ep is not None:
            subj_reading.append(r_ep)
        if nr_ep is not None:
            subj_non_reading.append(nr_ep)
        del raw, cleaned

    return subj_reading, subj_non_reading


def main(subjects, task, n_permutations, deep_epochs):
    config.ensure_dirs()
    ds_stats = io_raw.dataset_stats(task, subjects=subjects)

    preprocessing_summary = {}
    all_reading, all_non_reading = [], []
    for subj in subjects:
        print(f"\n=== Processing subject {subj} ({task}) ===")
        subj_reading, subj_non_reading = process_subject(subj, task, preprocessing_summary)
        if not subj_reading or not subj_non_reading:
            print(f"[run_all] WARNING: subject {subj} produced no usable epochs of one class, skipping")
            continue
        all_reading.append(mne.concatenate_epochs(subj_reading, verbose=False))
        all_non_reading.append(mne.concatenate_epochs(subj_non_reading, verbose=False))

    combined = epochs_mod.balance_and_concatenate(all_reading, all_non_reading)
    epochs_path = os.path.join(config.DIRS["epochs"], f"{task}_validation-epo.fif")
    combined.save(epochs_path, overwrite=True, verbose=False)
    print(f"[run_all] saved epochs -> {epochs_path}")

    roi_channels = rois.compute_roi_channels(ch_names=combined.ch_names)

    print("\n=== Feature extraction ===")
    X, y, meta = features_matrix.build_feature_matrix(combined, roi_channels)
    X.to_parquet(os.path.join(config.DIRS["features"], f"{task}_X.parquet"))
    meta.to_parquet(os.path.join(config.DIRS["features"], f"{task}_meta.parquet"))
    print(f"[run_all] feature matrix: {X.shape}")

    fig_dir = config.DIRS["figures"]
    features_erp.plot_grand_average_erp(combined, roi_channels, os.path.join(fig_dir, "erp", "grand_average.png"))
    features_spectral.plot_band_topomaps(combined, os.path.join(fig_dir, "spectral", "topomaps.png"))
    features_connectivity.plot_connectivity_matrices_and_graphs(
        combined, roi_channels, os.path.join(fig_dir, "connectivity", "matrices_alpha.png"), band=(8.0, 12.0))

    print("\n=== Classical ML ===")
    subjects_arr = meta["subject"].to_numpy()
    sd_splits = list(models_classical.subject_dependent_splits(y, subjects_arr))
    si_splits = list(models_classical.subject_independent_splits(y, subjects_arr))

    res_sd, preds_sd = models_classical.run_cv(X, y, subjects_arr, sd_splits, "subject_dependent")
    res_si, preds_si = models_classical.run_cv(X, y, subjects_arr, si_splits, "subject_independent")
    results = pd.concat([res_sd, res_si], ignore_index=True)
    results.to_csv(os.path.join(config.DIRS["models"], f"{task}_classical_results.csv"), index=False)

    models_classical.plot_model_comparison(results, os.path.join(fig_dir, "models", "comparison.png"))
    models_classical.plot_confusion_matrices(preds_si, "subject_independent",
                                              os.path.join(fig_dir, "models", "confusion_matrices_si.png"))
    models_classical.plot_roc_curves(preds_si, "subject_independent",
                                      os.path.join(fig_dir, "models", "roc_curves_si.png"))

    print("\n=== Deep learning (subject-independent) ===")
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

    print("\n=== Statistics ===")
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
            print(f"[run_all] cluster permutation failed for {roi}: {exc}")
    pd.DataFrame(cluster_rows).to_csv(
        os.path.join(config.DIRS["stats"], f"{task}_cluster_permutation_erp.csv"), index=False)

    print("\n=== Report ===")
    report.build_report(ds_stats, preprocessing_summary, sig_summary, results,
                         deep_results, roi_channels)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=config.VALIDATION_SUBJECTS)
    parser.add_argument("--task", default="NR", choices=["NR", "TSR"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--deep-epochs", type=int, default=25)
    args = parser.parse_args()
    main(args.subjects, args.task, args.n_permutations, args.deep_epochs)
