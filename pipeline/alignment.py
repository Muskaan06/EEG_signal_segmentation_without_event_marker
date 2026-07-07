"""Recover exact word/fixation onset sample indices in the raw continuous
EEG recording, using only the raw file's own event triggers plus the
ZuCo-team results .mat — no eye-tracker/EEG clock synchronization needed.

Validated empirically on subject ZDN (byte-exact array matching against the
continuous ICA-cleaned `gip_*` file, which shares the raw file's exact
sample indexing) and cross-checked structurally on ZAB across all 6 blocks
(0 mismatches over 300/300 sentences):

  - Every sentence trial's EEG (`sentenceData[i].rawData`) spans exactly one
    (onset, offset) pair drawn from the raw file's event stream filtered to
    trigger types {10, 11, 12, 13} and sorted by sample — taken two at a
    time, in chronological order, mapping 1:1 onto `sentenceData` rows in
    order. Onsets are always type '10' or '12' (fixation-cross-terminated
    first sentence of a "paragraph group" uses 12/13; subsequent sentences
    in the same group use 10/11, with an intervening '15' cue that carries
    no timing information of its own).
  - `word.rawEEG` (per fixation) is an exact contiguous slice of that
    sentence's `rawData` — this offset is not marked by any trigger and is
    recovered by array search.
  - EEGLAB event latencies are 1-based MATLAB indices; `sample = latency - 1`
    gives the 0-based numpy/MNE sample index (io_raw.load_raw_eeg_block
    already performs this conversion).

So: trial (onset, offset) sample = the k-th (onset,offset) trigger pair in
a block, assigned in order to sentenceData rows; word/fixation onset sample
= trial onset sample + offset-of-word.rawEEG-within-rawData.
"""

import numpy as np
import pandas as pd

from pipeline import config, io_raw


def _fixation_segments(raw_eeg_obj):
    """Normalize a word's `rawEEG` field (scipy squeeze_me) into a list of
    (n_channels, n_samples) float64 arrays, one per fixation, in order.
    """
    if raw_eeg_obj is None:
        return []
    arr = np.asarray(raw_eeg_obj)
    if arr.size == 0:
        return []
    if arr.dtype == object:
        # multiple fixations: 1-D object array of 2-D arrays
        return [np.asarray(seg, dtype=np.float64) for seg in arr.ravel() if np.asarray(seg).size > 0]
    if arr.ndim == 2:
        # single fixation: already (n_channels, n_samples)
        return [arr.astype(np.float64)]
    return []


def _find_offset(haystack, needle, search_from=0):
    """Find the sample offset at which `needle` (n_ch, k) occurs as an exact
    contiguous slice of `haystack` (n_ch, T), searching from `search_from`.
    Returns None if not found.
    """
    k = needle.shape[1]
    t = haystack.shape[1]
    if k == 0 or k > t:
        return None
    ref = needle[:, 0]
    # Candidate positions: first channel's first sample matches (cheap filter)
    window = haystack[0, search_from:t - k + 1]
    candidates = np.where(np.isclose(window, ref[0], atol=1e-6))[0] + search_from
    for start in candidates:
        if np.allclose(haystack[:, start:start + k], needle, atol=1e-3):
            return int(start)
    return None


_TRIAL_ONSET_TYPES = {config.TRIGGER_MISC_A, config.TRIGGER_FIXCROSS_ON}      # '10' or '12'
_TRIAL_OFFSET_TYPES = {config.TRIGGER_MISC_B, config.TRIGGER_FIXCROSS_OFF}    # '11' or '13'
_TRIAL_PAIR_TYPES = _TRIAL_ONSET_TYPES | _TRIAL_OFFSET_TYPES                  # {'10','11','12','13'}


def _trial_pairs_for_block(events_df):
    """(onset_sample, offset_sample) for each sentence trial in a block, in
    chronological order — see module docstring for the validated pairing rule.
    """
    sub = events_df[events_df["type"].isin(_TRIAL_PAIR_TYPES)].sort_values("sample")
    types = sub["type"].to_numpy()
    samples = sub["sample"].to_numpy()

    n_pairs = len(types) // 2
    pairs = []
    for k in range(n_pairs):
        onset_t, onset_s = types[2 * k], samples[2 * k]
        offset_t, offset_s = types[2 * k + 1], samples[2 * k + 1]
        if onset_t not in _TRIAL_ONSET_TYPES or offset_t not in _TRIAL_OFFSET_TYPES:
            print(f"[alignment] WARNING: unexpected trigger pairing at pair {k} "
                  f"({onset_t}@{onset_s}, {offset_t}@{offset_s}) — stopping block early.")
            break
        pairs.append((int(onset_s), int(offset_s)))
    return pairs


def align_subject_task(subject, task, verbose=True):
    """Build (word_events, trial_events) DataFrames for one subject/task.

    word_events columns: subject, task, block, block_path, sentence_id,
        word_id, word_text, n_fixations, onset_sample, duration_samples
    trial_events columns: subject, task, block, block_path, sentence_id,
        onset_sample, offset_sample (the sentence's own reading-window
        onset/offset trigger samples, see module docstring)
    All `*_sample` values are 0-based sample indices into that block's raw
    file (as loaded by io_raw.load_raw_eeg_block).
    """
    sentence_data = io_raw.load_results_mat(subject, task)
    block_paths = io_raw.raw_eeg_paths(subject, task)

    word_rows = []
    trial_rows = []
    cursor = 0  # index into sentence_data, consumed sequentially across blocks

    for block_path in block_paths:
        block_num = io_raw._block_number(block_path)
        _, events_df = io_raw.load_raw_eeg_block(block_path)
        trial_pairs = _trial_pairs_for_block(events_df)

        n_trials = len(trial_pairs)
        block_sentences = sentence_data[cursor:cursor + n_trials]
        if len(block_sentences) < n_trials:
            print(f"[alignment] WARNING: {subject}/{task} block {block_num}: "
                  f"{n_trials} trigger pairs but only {len(block_sentences)} "
                  f"sentences remain in results.mat — truncating.")
            n_trials = len(block_sentences)

        for k in range(n_trials):
            sentence = block_sentences[k]
            trial_start, trial_offset_trigger = trial_pairs[k]
            sentence_id = cursor + k

            trial_rows.append({
                "subject": subject, "task": task, "block": block_num,
                "block_path": block_path, "sentence_id": sentence_id,
                "onset_sample": trial_start,
                "offset_sample": trial_offset_trigger,
            })

            raw_data = np.asarray(sentence.rawData, dtype=np.float64)
            if raw_data.ndim != 2 or raw_data.shape[1] == 0:
                continue  # sentence excluded by ZuCo's own QC (empty rawData)
            words = np.atleast_1d(sentence.word) if getattr(sentence, "word", None) is not None else []
            for word_id, w in enumerate(words):
                nfix = getattr(w, "nFixations", 0)
                if isinstance(nfix, np.ndarray) or not nfix or nfix <= 0:
                    continue
                segs = _fixation_segments(getattr(w, "rawEEG", None))
                if not segs:
                    continue
                first_seg = segs[0]
                offset = _find_offset(raw_data, first_seg)
                if offset is None:
                    continue
                word_rows.append({
                    "subject": subject, "task": task, "block": block_num,
                    "block_path": block_path, "sentence_id": sentence_id,
                    "word_id": word_id, "word_text": str(getattr(w, "content", "")),
                    "n_fixations": int(nfix),
                    "onset_sample": trial_start + offset,
                    "duration_samples": first_seg.shape[1],
                })

        cursor += n_trials

    if cursor != len(sentence_data) and verbose:
        print(f"[alignment] NOTE: {subject}/{task}: aligned {cursor} of "
              f"{len(sentence_data)} sentences to raw blocks "
              f"(remainder likely belongs to blocks not present on disk).")

    word_events = pd.DataFrame(word_rows)
    trial_events = pd.DataFrame(trial_rows)
    if verbose:
        print(f"[alignment] {subject}/{task}: {len(trial_events)} trials, "
              f"{len(word_events)} word/fixation onsets recovered.")
    return word_events, trial_events
