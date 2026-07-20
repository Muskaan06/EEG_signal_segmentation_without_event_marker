#!/usr/bin/env python3
"""Real block-level temporal-overlap accuracy for the combined VWFA +
Wernicke-ERP detector's output (word_events.csv, with a `source` column of
"vwfa" or "wernicke_erp"), vs. exact eye-tracking word onset/offset samples
recovered via pipeline/alignment.py's trigger-based method -- same ground
truth and greedy-overlap-matching method as
vwfa_word_detection/word_detection_output/eval/real_temporal_accuracy.py,
just also splitting out how many matches came specifically from the
Wernicke-ERP addition (source == "wernicke_erp"), so the union's actual
contribution is visible, not just the combined total.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = "/home/muskaan06/Desktop/Research/Mindscribe-IITIndore"
DATASET_ROOT = os.path.join(ROOT, "dataset/ZuCo/zuco1")
VWFA_DIR = os.path.join(ROOT, "EEG_research/vwfa_word_detection")
COMBO_DIR = os.path.join(ROOT, "EEG_research/vwfa_wernicke_word_detection")
PIPELINE_DIR = os.path.join(ROOT, "EEG_research/pipeline")
WORD_EVENTS_CSV = os.path.join(COMBO_DIR, "word_detection_output/word_events.csv")

_pkg = types.ModuleType("pipeline")
_pkg.__path__ = [VWFA_DIR]
sys.modules["pipeline"] = _pkg
from pipeline import config, io_raw  # noqa: E402

config.DATASET_ROOT = DATASET_ROOT
config.TASK_DIRS = {
    "NR": {
        "raw_data": os.path.join(DATASET_ROOT, "task2-NR", "Raw data"),
        "matlab_files": os.path.join(DATASET_ROOT, "task2-NR", "Matlab_files"),
        "results_suffix": "_NR.mat", "wordbounds_prefix": "wordbounds_NR_", "n_blocks": 6,
    },
    "TSR": {
        "raw_data": os.path.join(DATASET_ROOT, "task3-TSR", "Raw data"),
        "matlab_files": os.path.join(DATASET_ROOT, "task3-TSR", "Matlab_files"),
        "results_suffix": "_TSR.mat", "wordbounds_prefix": "wordbounds_TSR_", "n_blocks": 9,
    },
}

_CANDIDATE_MATLAB_DIRS = [
    os.path.join(DATASET_ROOT, "task2-NR", "Matlab_files"),
    os.path.join(DATASET_ROOT, "task3-TSR", "Matlab_files"),
]


def _robust_load_results_mat(subject, task):
    suffix = config.TASK_DIRS[task]["results_suffix"]
    fname = f"results{subject}{suffix}"
    for d in _CANDIDATE_MATLAB_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            mat = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
            return np.atleast_1d(mat["sentenceData"])
    raise FileNotFoundError(f"No results.mat for {subject}/{task}")


io_raw.load_results_mat = _robust_load_results_mat

import importlib.util
_spec = importlib.util.spec_from_file_location("pipeline.alignment", os.path.join(PIPELINE_DIR, "alignment.py"))
alignment = importlib.util.module_from_spec(_spec)
sys.modules["pipeline.alignment"] = alignment
_spec.loader.exec_module(alignment)

_orig_fixation_segments = alignment._fixation_segments


def _patched_fixation_segments(raw_eeg_obj):
    segs = _orig_fixation_segments(raw_eeg_obj)
    fixed = []
    for seg in segs:
        seg = np.asarray(seg)
        if seg.ndim == 1:
            seg = seg.reshape(-1, 1)
        if seg.ndim == 2 and seg.shape[1] > 0:
            fixed.append(seg)
    return fixed


alignment._fixation_segments = _patched_fixation_segments
_orig_find_offset = alignment._find_offset


def _safe_find_offset(haystack, needle, search_from=0):
    needle = np.asarray(needle)
    if needle.ndim != 2 or needle.shape[1] == 0:
        return None
    try:
        return _orig_find_offset(haystack, needle, search_from)
    except (IndexError, ValueError):
        return None


alignment._find_offset = _safe_find_offset

ALL_SUBJECTS = config.ALL_SUBJECTS


def greedy_match_with_source(true_windows, pred_windows, pred_sources):
    """Same greedy 1:1 overlap matching as real_temporal_accuracy.py, but
    also returns how many matches came from each source label."""
    pairs = []
    for ti, (ta, tb) in enumerate(true_windows):
        for pi, (pa, pb) in enumerate(pred_windows):
            ov = min(tb, pb) - max(ta, pa)
            if ov > 0:
                pairs.append((ov, ti, pi))
    pairs.sort(reverse=True)
    used_t, used_p = set(), set()
    matched_by_source = {}
    for ov, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        src = pred_sources[pi]
        matched_by_source[src] = matched_by_source.get(src, 0) + 1
    return matched_by_source


def main():
    pred_df = pd.read_csv(WORD_EVENTS_CSV)

    rows = []
    for subj in ALL_SUBJECTS:
        subj_true = subj_pred = subj_matched = subj_matched_wernicke_erp = 0

        for task in ["NR", "TSR"]:
            word_events, _ = alignment.align_subject_task(subj, task, verbose=False)
            if word_events.empty:
                print(f"[eval] {subj} {task}: no aligned word events, skipping")
                continue

            word_events = word_events.copy()
            word_events["onset_end"] = word_events["onset_sample"] + word_events["duration_samples"]
            task_pred = pred_df[(pred_df["subject"] == subj) & (pred_df["task"] == task)]

            n_true = n_pred = n_matched = n_matched_erp = 0
            for block_num, block_true in word_events.groupby("block"):
                true_windows = list(zip(block_true["onset_sample"], block_true["onset_end"]))
                block_pred = task_pred[task_pred["block"] == block_num]
                pred_windows = list(zip(block_pred["start_sample"], block_pred["end_sample"]))
                pred_sources = list(block_pred["source"])

                n_true += len(true_windows)
                n_pred += len(pred_windows)
                matched_by_source = greedy_match_with_source(true_windows, pred_windows, pred_sources)
                n_matched += sum(matched_by_source.values())
                n_matched_erp += matched_by_source.get("wernicke_erp", 0)

            aligned_blocks = set(word_events["block"].unique())
            leftover_pred = task_pred[~task_pred["block"].isin(aligned_blocks)]
            n_pred += len(leftover_pred)

            subj_true += n_true
            subj_pred += n_pred
            subj_matched += n_matched
            subj_matched_wernicke_erp += n_matched_erp
            print(f"[eval] {subj} {task}: true={n_true} pred={n_pred} matched={n_matched} "
                  f"(of which wernicke_erp-sourced: {n_matched_erp})")

        if subj_true == 0 and subj_pred == 0:
            continue
        precision = subj_matched / subj_pred if subj_pred else 0.0
        recall = subj_matched / subj_true if subj_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        denom = subj_true + subj_pred - subj_matched
        jaccard = subj_matched / denom if denom else 0.0

        rows.append({
            "subject": subj,
            "true_words": subj_true,
            "predicted_words": subj_pred,
            "matched": subj_matched,
            "matched_from_wernicke_erp": subj_matched_wernicke_erp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "accuracy_jaccard": round(jaccard, 3),
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(COMBO_DIR, "word_detection_output/eval/real_temporal_accuracy_combined.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))
    print(f"\nwrote {out_path}")

    tt = out["true_words"].sum()
    pp = out["predicted_words"].sum()
    mm = out["matched"].sum()
    mm_erp = out["matched_from_wernicke_erp"].sum()
    prec, rec = mm / pp if pp else 0, mm / tt if tt else 0
    f1o = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"\nOverall: true={tt} predicted={pp} matched={mm} "
          f"(of which {mm_erp} came only from the Wernicke-ERP addition) "
          f"precision={prec:.3f} recall={rec:.3f} f1={f1o:.3f}")


if __name__ == "__main__":
    main()
