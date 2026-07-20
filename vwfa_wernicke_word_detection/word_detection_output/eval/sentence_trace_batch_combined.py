#!/usr/bin/env python3
"""Same "5 random NR sentences per subject" diagnostic as
vwfa_word_detection/word_detection_output/eval/sentence_trace_batch.py, but
for the combined VWFA + Wernicke-ERP (union) detector -- same random seed
and sentence-selection filter, so the *same* 60 sentences are shown, letting
the two artifacts be compared side by side.

Detection runs at the real block level for both detectors (matching
run_combined_zuco1.py exactly), then each chosen sentence's window is
sliced out for inspection. Per ground-truth word, classifies:
  - matched_vwfa        : a VWFA marker overlaps it
  - matched_wernicke_erp: no VWFA marker overlaps, but a Wernicke-ERP marker
                           does (the word the union specifically added)
  - missed               : neither overlaps it

VWFA code is imported from vwfa_word_detection/, untouched by this script.
"""
import json
import os
import random
import sys
import types

import numpy as np
import pandas as pd

ROOT = "/home/muskaan06/Desktop/Research/Mindscribe-IITIndore"
DATASET_ROOT = os.path.join(ROOT, "dataset/ZuCo/zuco1")
VWFA_DIR = os.path.join(ROOT, "EEG_research/vwfa_word_detection")
COMBO_DIR = os.path.join(ROOT, "EEG_research/vwfa_wernicke_word_detection")
PIPELINE_DIR = os.path.join(ROOT, "EEG_research/pipeline")
OUT_PATH = os.path.join(COMBO_DIR, "word_detection_output/eval/sentence_traces_combined.json")

_pkg = types.ModuleType("pipeline")
_pkg.__path__ = [VWFA_DIR]
sys.modules["pipeline"] = _pkg
from pipeline import config, io_raw, rois  # noqa: E402
from pipeline import word_detection as vwfa_word_detection  # noqa: E402

sys.path.insert(0, COMBO_DIR)
import wernicke_erp_detection  # noqa: E402
from combine import combine_detections  # noqa: E402

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
    import scipy.io as sio
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

SRATE = config.SRATE
TASK = "NR"
N_SENTENCES_PER_SUBJECT = 5
MIN_WORDS_PER_SENTENCE = 6
PAD_MS = 400
RNG_SEED = config.SEED  # same seed as sentence_trace_batch.py -> same sentences chosen


def roi_signal(raw, channels):
    picks = [ch for ch in channels if ch in raw.ch_names]
    data_uv = raw.get_data(picks=picks) * 1e6
    return data_uv.mean(axis=0)


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def greedy_match_sources(true_windows, pred_windows, pred_sources):
    """Exclusive 1:1 greedy overlap matching (same rule as
    real_temporal_accuracy_combined.py's greedy_match_with_source) -- each
    predicted marker can match at most one true word and vice versa. Using
    non-exclusive "does any marker overlap" per word (what an earlier
    version of this script did) drastically overstates recall: VWFA's
    dense, wide (+/-200ms) marker windows can each spuriously overlap
    several adjacent short words at once. Returns {true_word_index: source}
    for matched words only.
    """
    pairs = []
    for ti, (ta, tb) in enumerate(true_windows):
        for pi, (pa, pb) in enumerate(pred_windows):
            ov = min(tb, pb) - max(ta, pa)
            if ov > 0:
                pairs.append((ov, ti, pi))
    pairs.sort(reverse=True)
    used_t, used_p = set(), set()
    result = {}
    for ov, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        result[ti] = pred_sources[pi]
    return result


def main():
    rng = random.Random(RNG_SEED)
    subjects_out = []

    for subj in config.ALL_SUBJECTS:
        print(f"[batch] {subj}: aligning ground truth...")
        word_events, _ = alignment.align_subject_task(subj, TASK, verbose=False)
        if word_events.empty:
            print(f"[batch] {subj}: no aligned word events, skipping")
            continue

        word_events = word_events.copy()
        word_events["onset_end"] = word_events["onset_sample"] + word_events["duration_samples"]

        groups = list(word_events.groupby(["block", "sentence_id"]))
        candidates = [(b, sid, g) for (b, sid), g in groups if len(g) >= MIN_WORDS_PER_SENTENCE]
        if not candidates:
            print(f"[batch] {subj}: no sentence with >= {MIN_WORDS_PER_SENTENCE} words, skipping")
            continue

        rng.shuffle(candidates)
        chosen = candidates[:N_SENTENCES_PER_SUBJECT]
        chosen.sort(key=lambda c: (c[0], c[1]))

        needed_blocks = sorted(set(b for b, sid, g in chosen))
        block_data = {}
        for block_num in needed_blocks:
            paths = [p for p in io_raw.raw_eeg_paths(subj, TASK) if io_raw._block_number(p) == block_num]
            if not paths:
                continue
            raw, _ = io_raw.load_raw_eeg_block(paths[0])
            roi_channels = rois.compute_roi_channels(ch_names=raw.ch_names, save=False)
            sig_vwfa = roi_signal(raw, roi_channels["vwfa"])
            sig_wernicke = roi_signal(raw, roi_channels["wernicke"])

            vwfa_thr = vwfa_word_detection.calibrate_signal_thresholds(sig_vwfa, SRATE)
            _, vwfa_markers = vwfa_word_detection.detect_word_events(sig_vwfa, SRATE, vwfa_thr)
            _, erp_markers = wernicke_erp_detection.detect_word_events_wernicke_erp(sig_wernicke, SRATE)

            envelope = vwfa_word_detection.classify_signal_levels(sig_vwfa, SRATE, vwfa_thr)[1]

            vwfa_intervals = list(zip(vwfa_markers["marker_start_sample"], vwfa_markers["marker_end_sample"])) if not vwfa_markers.empty else []
            erp_intervals_raw = list(zip(erp_markers["marker_start_sample"], erp_markers["marker_end_sample"])) if not erp_markers.empty else []
            # same exclusion rule as combine.combine_detections: an erp
            # interval only survives into the combined pool if it doesn't
            # overlap ANY vwfa interval. Applying this *before* matching
            # (not pooling raw erp+vwfa into one competitive greedy match)
            # is essential -- otherwise an erp marker can spuriously "win" a
            # word in a region vwfa already covers, which does not reflect
            # what combine.py + run_combined_zuco1.py actually produced.
            erp_intervals = [(a, b) for a, b in erp_intervals_raw
                              if not any(overlaps(a, b, va, vb) for va, vb in vwfa_intervals)]
            pred_windows = vwfa_intervals + erp_intervals
            pred_sources = ["vwfa"] * len(vwfa_intervals) + ["wernicke_erp"] * len(erp_intervals)

            # exclusive greedy matching against EVERY true word in this block
            # (not just the chosen sentences) -- same rule and same full
            # candidate pool real_temporal_accuracy_combined.py uses, so a
            # marker matched to a word outside a chosen sentence can't also
            # be double-credited to a word inside one.
            block_true = word_events[word_events["block"] == block_num].sort_values("onset_sample").reset_index(drop=True)
            true_windows = list(zip(block_true["onset_sample"], block_true["onset_end"]))
            match_result = greedy_match_sources(true_windows, pred_windows, pred_sources)
            block_true["stage"] = ["missed"] * len(block_true)
            for ti, src in match_result.items():
                block_true.loc[ti, "stage"] = "matched_vwfa" if src == "vwfa" else "matched_wernicke_erp"

            block_data[block_num] = {
                "n_samples": len(sig_vwfa),
                "envelope": envelope,
                "threshold": {"low_high": vwfa_thr.low_high, "high_very_high": vwfa_thr.high_very_high},
                "vwfa_intervals": vwfa_intervals,
                "erp_intervals": erp_intervals,
                "true_words_with_stage": block_true,
            }
            n_block_matched = sum(1 for s in block_true["stage"] if s != "missed")
            print(f"[batch] {subj} block {block_num}: traced ({len(sig_vwfa)} samples, "
                  f"{len(vwfa_intervals)} vwfa + {len(erp_intervals)} wernicke_erp markers, "
                  f"block recall {n_block_matched}/{len(block_true)})")

        sentence_entries = []
        for block_num, sentence_id, g in chosen:
            bd = block_data.get(block_num)
            if bd is None:
                continue
            block_true = bd["true_words_with_stage"]
            words = block_true[block_true["sentence_id"] == sentence_id].sort_values("onset_sample")
            w_min = int(words["onset_sample"].min()) - int(round(PAD_MS / 1000.0 * SRATE))
            w_max = int(words["onset_end"].max()) + int(round(PAD_MS / 1000.0 * SRATE))
            w_min, w_max = max(0, w_min), min(bd["n_samples"], w_max)

            true_words = []
            stage_counts = {}
            for _, row in words.iterrows():
                ws, we = int(row["onset_sample"]), int(row["onset_end"])
                stage = row["stage"]
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                true_words.append({"word_text": row["word_text"], "onset_sample": ws,
                                    "duration_samples": we - ws, "stage": stage})

            step = 2
            env_slice = bd["envelope"][w_min:w_max:step]
            sentence_entries.append({
                "block": int(block_num), "sentence_id": int(sentence_id),
                "content": " ".join(w["word_text"] for w in true_words),
                "window_start_sample": w_min, "window_end_sample": w_max,
                "envelope_step": step,
                "envelope": [None if np.isnan(v) else round(float(v), 3) for v in env_slice],
                "threshold": bd["threshold"],
                "true_words": true_words,
                "vwfa_markers": [{"start": a, "end": b} for a, b in bd["vwfa_intervals"] if overlaps(a, b, w_min, w_max)],
                "wernicke_erp_markers": [{"start": a, "end": b} for a, b in bd["erp_intervals"] if overlaps(a, b, w_min, w_max)],
                "stage_counts": stage_counts,
            })

        subjects_out.append({"subject": subj, "srate": SRATE, "sentences": sentence_entries})
        print(f"[batch] {subj}: {len(sentence_entries)} sentences traced\n")

    with open(OUT_PATH, "w") as f:
        json.dump(subjects_out, f)
    print(f"[batch] wrote {OUT_PATH}")
    total_sentences = sum(len(s["sentences"]) for s in subjects_out)
    print(f"[batch] {len(subjects_out)} subjects, {total_sentences} sentences total")


if __name__ == "__main__":
    main()
