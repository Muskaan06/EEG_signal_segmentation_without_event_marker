#!/usr/bin/env python3
"""Runs the combined VWFA + Wernicke-ERP detector over every ZuCo1 NR/TSR
raw block for every subject. Mirrors
vwfa_word_detection/word_detection_output/eval/run_word_detection_local.py's
block-loading loop exactly, but computes both ROI signals, runs both
detectors, and unions them (combine.combine_detections) before writing.

VWFA detection code is imported from vwfa_word_detection/, not copied --
that folder is untouched by this script.
"""
import os
import sys
import time
import traceback
import types

import numpy as np
import pandas as pd

ROOT = "/home/muskaan06/Desktop/Research/Mindscribe-IITIndore"
DATASET_ROOT = os.path.join(ROOT, "dataset/ZuCo/zuco1")
VWFA_DIR = os.path.join(ROOT, "EEG_research/vwfa_word_detection")
COMBO_DIR = os.path.join(ROOT, "EEG_research/vwfa_wernicke_word_detection")
OUT_DIR = os.path.join(COMBO_DIR, "word_detection_output")

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
TASKS = ["NR", "TSR"]


def roi_signal(raw, channels):
    picks = [ch for ch in channels if ch in raw.ch_names]
    data_uv = raw.get_data(picks=picks) * 1e6
    return data_uv.mean(axis=0)


def main():
    sample_path = io_raw.raw_eeg_paths(config.ALL_SUBJECTS[0], "NR")[0]
    sample_raw, _ = io_raw.load_raw_eeg_block(sample_path)
    roi_channels = rois.compute_roi_channels(ch_names=sample_raw.ch_names, save=False)
    vwfa_ch = roi_channels["vwfa"]
    wernicke_ch = roi_channels["wernicke"]
    print(f"[run] vwfa channels ({len(vwfa_ch)}): {vwfa_ch}")
    print(f"[run] wernicke channels ({len(wernicke_ch)}): {wernicke_ch}")

    block_rows = []
    event_rows = []

    for task in TASKS:
        for subj in config.ALL_SUBJECTS:
            paths = io_raw.raw_eeg_paths(subj, task)
            if not paths:
                print(f"[run] {subj} {task}: no raw blocks found, skipping")
                continue

            for path in paths:
                block_num = io_raw._block_number(path)
                tag = f"{subj}_{task}{block_num}"
                t0 = time.time()
                try:
                    raw, events_df = io_raw.load_raw_eeg_block(path)
                    srate = float(raw.info["sfreq"])
                    n_samples = len(raw.times)
                    duration_s = n_samples / srate

                    sig_vwfa = roi_signal(raw, vwfa_ch)
                    sig_wernicke = roi_signal(raw, wernicke_ch)

                    vwfa_thr = vwfa_word_detection.calibrate_signal_thresholds(sig_vwfa, srate)
                    _, vwfa_markers = vwfa_word_detection.detect_word_events(sig_vwfa, srate, vwfa_thr)

                    _, erp_markers = wernicke_erp_detection.detect_word_events_wernicke_erp(sig_wernicke, srate)

                    combined = combine_detections(vwfa_markers, erp_markers)

                    n_vwfa = len(vwfa_markers)
                    n_erp_added = int((combined["source"] == "wernicke_erp").sum()) if not combined.empty else 0
                    n_total = len(combined)

                    block_rows.append({
                        "zuco_version": "zuco_1", "subject": subj, "task": task, "block": block_num,
                        "source_file": path, "srate_hz": srate, "n_samples": n_samples,
                        "duration_s": round(duration_s, 3),
                        "n_vwfa_words": n_vwfa, "n_wernicke_erp_added": n_erp_added,
                        "n_words_detected": n_total, "status": "ok",
                        "elapsed_s": round(time.time() - t0, 2),
                    })

                    for i, row in combined.iterrows():
                        event_rows.append({
                            "zuco_version": "zuco_1", "subject": subj, "task": task, "block": block_num,
                            "word_index": int(i),
                            "start_time_s": round(row["marker_start_sample"] / srate, 4),
                            "end_time_s": round(row["marker_end_sample"] / srate, 4),
                            "start_sample": int(row["marker_start_sample"]),
                            "end_sample": int(row["marker_end_sample"]),
                            "source": row["source"],
                        })

                    print(f"[run] {tag}: {n_total} words ({n_vwfa} vwfa + {n_erp_added} wernicke_erp added) "
                          f"({duration_s:.1f}s, {time.time()-t0:.1f}s elapsed)")

                except Exception as exc:
                    print(f"[run] {tag}: FAILED - {exc}")
                    traceback.print_exc()
                    block_rows.append({
                        "zuco_version": "zuco_1", "subject": subj, "task": task, "block": block_num,
                        "source_file": path, "srate_hz": None, "n_samples": None, "duration_s": None,
                        "n_vwfa_words": None, "n_wernicke_erp_added": None, "n_words_detected": None,
                        "status": f"error: {exc}", "elapsed_s": round(time.time() - t0, 2),
                    })

    block_df = pd.DataFrame(block_rows)
    event_df = pd.DataFrame(event_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    block_path = os.path.join(OUT_DIR, "block_summary.csv")
    event_path = os.path.join(OUT_DIR, "word_events.csv")
    block_df.to_csv(block_path, index=False)
    event_df.to_csv(event_path, index=False)

    print(f"\n[run] wrote {block_path} ({len(block_df)} rows)")
    print(f"[run] wrote {event_path} ({len(event_df)} rows)")
    print(f"[run] total words detected: {block_df['n_words_detected'].sum(skipna=True)} "
          f"({block_df['n_vwfa_words'].sum(skipna=True)} vwfa + "
          f"{block_df['n_wernicke_erp_added'].sum(skipna=True)} wernicke_erp added)")


if __name__ == "__main__":
    main()
