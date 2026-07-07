"""Raw ZuCo 1 data access: EEGLAB raw EEG, event triggers, and the ZuCo-team
Matlab results (sentence/word/fixation-level eye-tracking + EEG metadata).

Nothing here does any signal processing — it only parses files into
Python-native structures (MNE RawArray, pandas DataFrames, numpy arrays).
"""

import glob
import os
import re

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from pipeline import config


def _block_number(path):
    m = re.search(r"(?:NR|TSR)(\d+)_EEG\.mat$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def raw_eeg_paths(subject, task):
    """Sorted list of per-block raw EEG .mat paths for a subject/task."""
    task_dir = os.path.join(config.TASK_DIRS[task]["raw_data"], subject)
    paths = glob.glob(os.path.join(task_dir, f"{subject}_{task}*_EEG.mat"))
    return sorted(paths, key=_block_number)


def load_raw_eeg_block(path):
    """Load one raw EEGLAB .mat file -> (mne.io.RawArray, events_df).

    events_df columns: sample (0-based, python-indexed), type (str), latency_matlab (1-based, as stored).
    """
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    eeg = mat["EEG"]

    data_uv = np.asarray(eeg.data, dtype=np.float64)   # (n_channels, n_samples), microvolts
    srate = float(eeg.srate)
    ch_names = [str(c.labels) for c in eeg.chanlocs]

    info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)  # MNE expects volts

    montage = mne.channels.make_standard_montage(config.MONTAGE_NAME)
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    events = np.atleast_1d(eeg.event)
    rows = []
    for e in events:
        lat_matlab = int(e.latency)
        rows.append({
            "sample": lat_matlab - 1,     # MATLAB 1-based -> python 0-based sample index
            "type": str(e.type).strip(),
            "latency_matlab": lat_matlab,
        })
    events_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)

    return raw, events_df


def load_results_mat(subject, task):
    """ZuCo-team-computed sentenceData struct array: per-sentence, per-word,
    per-fixation eye-tracking + EEG metadata (the ground-truth timing source
    used by pipeline/alignment.py).
    """
    task_cfg = config.TASK_DIRS[task]
    path = os.path.join(task_cfg["matlab_files"], f"results{subject}{task_cfg['results_suffix']}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results .mat for subject={subject} task={task}: {path}")
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return np.atleast_1d(mat["sentenceData"])


def load_wordbounds(subject, task):
    task_cfg = config.TASK_DIRS[task]
    task_dir = os.path.join(task_cfg["raw_data"], subject)
    path = os.path.join(task_dir, f"{task_cfg['wordbounds_prefix']}{subject}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No wordbounds .mat for subject={subject} task={task}: {path}")
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return np.atleast_1d(mat["wordbounds"])


def available_subjects(task):
    """Subjects that have both raw EEG blocks and a results .mat for this task."""
    subs = []
    for subj in config.ALL_SUBJECTS:
        has_raw = len(raw_eeg_paths(subj, task)) > 0
        task_cfg = config.TASK_DIRS[task]
        results_path = os.path.join(task_cfg["matlab_files"], f"results{subj}{task_cfg['results_suffix']}")
        has_results = os.path.exists(results_path)
        if has_raw and has_results:
            subs.append(subj)
        elif has_raw or has_results:
            print(f"[io_raw] WARNING: subject {subj} task {task} incomplete "
                  f"(raw={has_raw}, results={has_results}) — skipping")
    return subs


def dataset_stats(task, subjects=None):
    """Compute and print dataset statistics for a task, purely from the data
    (no hardcoded counts). Returns a dict of the same numbers.
    """
    subjects = subjects or available_subjects(task)
    n_sentences = 0
    n_word_events = 0
    srate = None
    n_channels = None

    for subj in subjects:
        sd = load_results_mat(subj, task)
        n_sentences += len(sd)
        for sentence in sd:
            words = np.atleast_1d(sentence.word) if hasattr(sentence, "word") and sentence.word is not None else []
            for w in words:
                nfix = getattr(w, "nFixations", 0)
                if isinstance(nfix, np.ndarray):
                    continue  # empty numpy array means 0 fixations
                if nfix and nfix > 0:
                    n_word_events += 1

        paths = raw_eeg_paths(subj, task)
        if paths and srate is None:
            raw, _ = load_raw_eeg_block(paths[0])
            srate = raw.info["sfreq"]
            n_channels = len(raw.ch_names)

    stats = {
        "task": task,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_sentences": n_sentences,
        "n_word_events": n_word_events,
        "sampling_rate_hz": srate,
        "n_channels": n_channels,
    }

    print(f"=== ZuCo 1 {task} dataset statistics ===")
    print(f"  Subjects        : {stats['n_subjects']}  {subjects}")
    print(f"  Sentences       : {stats['n_sentences']}")
    print(f"  Word events     : {stats['n_word_events']} (words with >=1 fixation)")
    print(f"  Sampling rate   : {stats['sampling_rate_hz']} Hz")
    print(f"  Channels (raw)  : {stats['n_channels']}")
    return stats
