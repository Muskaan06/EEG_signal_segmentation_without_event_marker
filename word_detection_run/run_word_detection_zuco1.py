#!/usr/bin/env python3
"""Ad-hoc runner: apply pipeline/word_detection.py to every continuous raw
EEG block (NR + TSR) for every ZuCo 1 subject, using only raw_data/ (no
preprocessing.py cleaning, no epoching) — VWFA-ROI and Wernicke-ROI signals
are the plain channel-average of the raw microvolt traces, matching
word_detection's own "source-agnostic 1-D signal + srate" contract.

Does not modify any pipeline/* file. Writes two CSVs:
  - block_summary.csv : one row per continuous block -> n_words_detected
  - word_events.csv   : one row per confirmed word event -> start/end timestamp
"""

import os
import sys
import time
import traceback

REPO_ROOT = "/media/kpdubey/8.0 TB Volume/Muskaan/EEG_research"
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import mne

from pipeline import config, io_raw, rois, word_detection

mne.set_log_level("WARNING")

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "word_detection")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["NR", "TSR"]


def roi_signal(raw, channels):
    picks = [ch for ch in channels if ch in raw.ch_names]
    data_uv = raw.get_data(picks=picks) * 1e6  # back to microvolts
    return data_uv.mean(axis=0)


def main():
    # Full 128-channel raw montage -> ROI channel lists (save=False: don't
    # touch the repo's existing outputs/rois/roi_channels.json).
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

                    vwfa_thr = word_detection.calibrate_signal_thresholds(sig_vwfa, srate)
                    wernicke_thr = word_detection.calibrate_signal_thresholds(sig_wernicke, srate)

                    all_candidates, markers = word_detection.detect_word_events(
                        sig_vwfa, sig_wernicke, srate, vwfa_thr, wernicke_thr,
                    )

                    n_candidates = len(all_candidates)
                    n_confirmed = len(markers)

                    block_rows.append({
                        "zuco_version": "zuco_1",
                        "subject": subj,
                        "task": task,
                        "block": block_num,
                        "source_file": path,
                        "srate_hz": srate,
                        "n_samples": n_samples,
                        "duration_s": round(duration_s, 3),
                        "n_vwfa_candidates": n_candidates,
                        "n_words_detected": n_confirmed,
                        "status": "ok",
                        "elapsed_s": round(time.time() - t0, 2),
                    })

                    for i, row in markers.iterrows():
                        start_s = row["marker_start_sample"] / srate
                        end_s = row["marker_end_sample"] / srate
                        vwfa_peak_s = row["sample"] / srate
                        event_rows.append({
                            "zuco_version": "zuco_1",
                            "subject": subj,
                            "task": task,
                            "block": block_num,
                            "word_index": int(i),
                            "vwfa_peak_time_s": round(vwfa_peak_s, 4),
                            "start_time_s": round(start_s, 4),
                            "end_time_s": round(end_s, 4),
                            "start_sample": int(row["marker_start_sample"]),
                            "end_sample": int(row["marker_end_sample"]),
                            "peak_amplitude": row["peak_amplitude"],
                            "wernicke_lag_ms": row["wernicke_lag_ms"],
                        })

                    print(f"[run] {tag}: {n_confirmed} words / {n_candidates} candidates "
                          f"({duration_s:.1f}s, {time.time()-t0:.1f}s elapsed)")

                except Exception as exc:
                    print(f"[run] {tag}: FAILED - {exc}")
                    traceback.print_exc()
                    block_rows.append({
                        "zuco_version": "zuco_1",
                        "subject": subj,
                        "task": task,
                        "block": block_num,
                        "source_file": path,
                        "srate_hz": None,
                        "n_samples": None,
                        "duration_s": None,
                        "n_vwfa_candidates": None,
                        "n_words_detected": None,
                        "status": f"error: {exc}",
                        "elapsed_s": round(time.time() - t0, 2),
                    })

    block_df = pd.DataFrame(block_rows)
    event_df = pd.DataFrame(event_rows)

    block_path = os.path.join(OUT_DIR, "block_summary.csv")
    event_path = os.path.join(OUT_DIR, "word_events.csv")
    block_df.to_csv(block_path, index=False)
    event_df.to_csv(event_path, index=False)

    print(f"\n[run] wrote {block_path} ({len(block_df)} rows)")
    print(f"[run] wrote {event_path} ({len(event_df)} rows)")
    print(f"[run] total words detected across all blocks: {block_df['n_words_detected'].sum(skipna=True)}")


if __name__ == "__main__":
    main()
