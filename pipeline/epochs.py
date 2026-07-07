"""Build READING and NON-READING epochs from cleaned raw EEG + the
alignment tables (pipeline/alignment.py).

READING epochs are centered on each word's first-fixation onset sample.
NON-READING epochs are drawn from three genuinely-non-reading intervals
found in the same recordings (no text is being read in any of them):
  - "pre_block"   : before the first trial of a block (post block-start trigger)
  - "inter_trial" : between one trial's reading-offset and the next trial's
                    reading-onset (inter-stimulus / question-answering interval)
  - "inter_word"  : within a trial, gaps between consecutive word fixations
                    that are long enough to host a full control window
                    (rare, since ZuCo saccades are short, but included per spec)
"""

import numpy as np
import pandas as pd
import mne

from pipeline import config

mne.set_log_level("WARNING")

EPOCH_LEN_S = config.EPOCH_TMAX - config.EPOCH_TMIN
_PAD_S = 0.05


def _windows_from_gap(gap_start_sample, gap_end_sample, srate, pad_s=_PAD_S, step_s=EPOCH_LEN_S):
    """Candidate onset samples, spaced `step_s` apart, s.t. [onset+tmin,
    onset+tmax] fits inside [gap_start, gap_end] with a small padding margin.
    `step_s == EPOCH_LEN_S` gives non-overlapping windows; a smaller step
    allows (mildly correlated) overlapping windows to harvest more control
    epochs from long, but rare, non-reading intervals.
    """
    step = int(round(step_s * srate))
    pad = int(round(pad_s * srate))
    tmin_samples = int(round(config.EPOCH_TMIN * srate))  # negative

    usable_start = gap_start_sample + pad - tmin_samples
    usable_end = gap_end_sample - pad - int(round(config.EPOCH_TMAX * srate))

    onsets = []
    t = usable_start
    while t <= usable_end:
        onsets.append(int(t))
        t += step
    return onsets


def non_reading_windows_for_block(word_events_block, trial_events_block, srate, block_start_sample=0,
                                   overlap_step_s=EPOCH_LEN_S / 2):
    """Returns a list of dicts: sample, source ('pre_block'/'inter_trial'/'inter_word'), sentence_id.

    `pre_block`/`inter_trial` windows are allowed to overlap (step=overlap_step_s)
    since genuinely non-reading intervals are scarce relative to reading
    events in this experimental paradigm; `inter_word` gaps stay
    non-overlapping since they are short and adjacent to active reading.
    """
    windows = []
    trials = trial_events_block.sort_values("onset_sample").reset_index(drop=True)

    if len(trials):
        for s in _windows_from_gap(block_start_sample, int(trials.iloc[0]["onset_sample"]), srate,
                                    step_s=overlap_step_s):
            windows.append({"sample": s, "source": "pre_block", "sentence_id": -1})

    for i in range(len(trials) - 1):
        gap_start = int(trials.iloc[i]["offset_sample"])
        gap_end = int(trials.iloc[i + 1]["onset_sample"])
        for s in _windows_from_gap(gap_start, gap_end, srate, step_s=overlap_step_s):
            windows.append({"sample": s, "source": "inter_trial", "sentence_id": int(trials.iloc[i]["sentence_id"])})

    for sentence_id, group in word_events_block.groupby("sentence_id"):
        words = group.sort_values("onset_sample")
        onsets = words["onset_sample"].to_numpy()
        durations = words["duration_samples"].to_numpy()
        for i in range(len(words) - 1):
            gap_start = int(onsets[i] + durations[i])
            gap_end = int(onsets[i + 1])
            for s in _windows_from_gap(gap_start, gap_end, srate):
                windows.append({"sample": s, "source": "inter_word", "sentence_id": int(sentence_id)})

    return windows


def _make_epochs(raw, samples, event_label, extra_metadata):
    """Build an MNE Epochs object anchored at the given sample indices."""
    if len(samples) == 0:
        return None

    samples = np.asarray(samples)
    # De-duplicate identical onset samples (rare: a fixation shared by two
    # adjacent word entries at a word-boundary) and sort chronologically —
    # np.unique returns first_idx aligned with the sorted unique values, so
    # this also fixes event ordering for mne.Epochs in one step.
    samples, first_idx = np.unique(samples, return_index=True)
    extra_metadata = extra_metadata.iloc[first_idx].reset_index(drop=True)

    events = np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])
    epochs = mne.Epochs(
        raw, events, event_id={event_label: 1},
        tmin=config.EPOCH_TMIN, tmax=config.EPOCH_TMAX,
        baseline=config.BASELINE, preload=True, verbose=False,
        on_missing="warn",
    )
    # mne.Epochs silently drops out-of-bounds events; keep metadata in sync
    # (samples are now unique, so isin-based positional masking is safe)
    kept_mask = np.isin(samples, epochs.events[:, 0])
    meta = extra_metadata.iloc[kept_mask].reset_index(drop=True)
    epochs.metadata = meta
    return epochs


def build_block_epochs(raw, word_events_block, trial_events_block, subject, task, block):
    """Build (reading_epochs, non_reading_epochs) for one cleaned raw block."""
    srate = raw.info["sfreq"]

    reading_meta = word_events_block.assign(
        label="reading", subject=subject, task=task, block=block,
        condition_source="word_fixation",
    )[["label", "subject", "task", "block", "sentence_id", "word_id",
       "word_text", "n_fixations", "condition_source"]]
    reading_epochs = _make_epochs(
        raw, word_events_block["onset_sample"].to_numpy(), "reading", reading_meta,
    )

    nr_windows = non_reading_windows_for_block(word_events_block, trial_events_block, srate)
    nr_df = pd.DataFrame(nr_windows)
    if len(nr_df):
        nr_meta = pd.DataFrame({
            "label": "non_reading", "subject": subject, "task": task, "block": block,
            "sentence_id": nr_df["sentence_id"], "word_id": -1, "word_text": "",
            "n_fixations": 0, "condition_source": nr_df["source"],
        })
        non_reading_epochs = _make_epochs(raw, nr_df["sample"].to_numpy(), "non_reading", nr_meta)
    else:
        non_reading_epochs = None

    return reading_epochs, non_reading_epochs


def balance_and_concatenate(reading_list, non_reading_list, seed=config.SEED):
    """Concatenate all per-block epochs, then randomly subsample the majority
    class so classes are balanced. Returns one combined mne.Epochs.
    """
    reading_list = [e for e in reading_list if e is not None and len(e) > 0]
    non_reading_list = [e for e in non_reading_list if e is not None and len(e) > 0]
    if not reading_list or not non_reading_list:
        raise ValueError("Need at least one non-empty epoch set per class to balance.")

    reading = mne.concatenate_epochs(reading_list, verbose=False)
    non_reading = mne.concatenate_epochs(non_reading_list, verbose=False)

    rng = np.random.default_rng(seed)
    n = min(len(reading), len(non_reading))
    r_idx = rng.choice(len(reading), size=n, replace=False)
    nr_idx = rng.choice(len(non_reading), size=n, replace=False)

    reading = reading[np.sort(r_idx)]
    non_reading = non_reading[np.sort(nr_idx)]

    combined = mne.concatenate_epochs([reading, non_reading], verbose=False)
    print(f"[epochs] balanced: {len(reading)} reading + {len(non_reading)} non_reading "
          f"= {len(combined)} total")
    return combined
