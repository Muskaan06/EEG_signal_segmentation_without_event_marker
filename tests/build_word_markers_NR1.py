"""Build word-reading markers over the continuous raw EEG for subject NR1.

Method:
  1. resultsYHS_NR.mat (sentenceData, HDF5/v7.3) gives, per sentence: the
     actual word text, per-word fixation count, and a per-word raw EEG
     segment (word.rawEEG) that is an exact contiguous slice of that
     sentence's own cut-out EEG (sentence.rawData).
  2. YHS_Raw/YHS_NR1_EEG.mat (continuous, 128-channel, EEGLAB struct) gives
     the raw continuous recording plus its own trigger events. Trigger
     types {10,11,12,13} sorted by sample and taken two-at-a-time give each
     trial's (onset, offset) sample pair, in the same order as the first 50
     sentences in resultsYHS_NR.mat (NR1's block).
  3. For each word with >=1 fixation, find the sample offset of its first
     fixation segment inside the sentence's rawData via exact array search,
     then word_onset_sample = trial_onset_sample + offset. This recovers,
     for every word actually read, the exact sample in the continuous raw
     EEG at which reading of that word began -- without needing any
     eye-tracker/EEG clock synchronization (validated approach, see
     pipeline/alignment.py).
  4. Write a marker channel (same length as the continuous EEG) that is 0
     everywhere except a +/-1 sample marker at each word onset, plus a
     parallel table of (word_text, sentence_id, word_id, onset_sample,
     duration_samples, n_fixations) -- both saved into one .mat file.
"""

import os

import h5py
import numpy as np
import scipy.io as sio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "data_YHS", "resultsYHS_NR.mat")
RAW_EEG_PATH = os.path.join(REPO_ROOT, "data_YHS", "YHS_Raw", "YHS_NR1_EEG.mat")
OUT_PATH = os.path.join(REPO_ROOT, "outputs", "word_markers_NR1.mat")

N_SENTENCES_NR1 = 50  # NR1 occupies sentenceData[0:50] of the shared results file
TRIAL_PAIR_TYPES = {"10", "11", "12", "13"}
ONSET_TYPES = {"10", "12"}
OFFSET_TYPES = {"11", "13"}


def h5_text(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(x)) for x in arr.flatten())


def h5_scalar(f, ref):
    arr = f[ref][()]
    return float(arr.flatten()[0]) if arr.size else 0.0


def find_offset(haystack, needle):
    """Sample offset (axis 0) at which `needle` occurs as an exact
    contiguous slice of `haystack`. Both are (n_samples, n_channels)."""
    k = needle.shape[0]
    t = haystack.shape[0]
    if k == 0 or k > t:
        return None
    ref_row = needle[0]
    candidates = np.where(np.all(np.isclose(haystack[:t - k + 1], ref_row, atol=1e-6), axis=1))[0]
    for start in candidates:
        if np.allclose(haystack[start:start + k], needle, atol=1e-3):
            return int(start)
    return None


def trial_pairs(events_df_rows):
    rows = sorted(events_df_rows, key=lambda r: r[0])
    sub = [r for r in rows if r[1] in TRIAL_PAIR_TYPES]
    pairs = []
    for k in range(len(sub) // 2):
        onset_s, onset_t = sub[2 * k]
        offset_s, offset_t = sub[2 * k + 1]
        if onset_t not in ONSET_TYPES or offset_t not in OFFSET_TYPES:
            print(f"WARNING: unexpected trigger pairing at pair {k}: "
                  f"{sub[2*k]} {sub[2*k+1]} -- stopping.")
            break
        pairs.append((onset_s, offset_s))
    return pairs


def main():
    # ---- continuous raw EEG (128 ch) + its own trigger events ----
    eeg_mat = sio.loadmat(RAW_EEG_PATH, struct_as_record=False, squeeze_me=True)
    eeg = eeg_mat["EEG"]
    raw_data = np.asarray(eeg.data, dtype=np.float32)     # (128, n_samples), microvolts
    srate = float(eeg.srate)
    ch_names = [str(c.labels) for c in eeg.chanlocs]
    n_samples = raw_data.shape[1]

    events = np.atleast_1d(eeg.event)
    event_rows = [(int(e.latency) - 1, str(e.type).strip()) for e in events]  # 1-based -> 0-based
    pairs = trial_pairs(event_rows)
    print(f"Continuous raw EEG: {raw_data.shape[0]} ch x {raw_data.shape[1]} samples "
          f"@ {srate} Hz, {len(pairs)} trial trigger pairs found.")
    assert len(pairs) == N_SENTENCES_NR1, (
        f"Expected {N_SENTENCES_NR1} trial pairs for NR1, got {len(pairs)}")

    # ---- word-level content + fixation-segmented EEG (v7.3 / HDF5) ----
    f = h5py.File(RESULTS_PATH, "r")
    sd = f["sentenceData"]

    marker_channel = np.zeros(n_samples, dtype=np.int32)  # 0 = no word onset here
    sentence_ids, word_ids, word_texts = [], [], []
    onset_samples, duration_samples, n_fixations_list, marker_codes = [], [], [], []

    marker_code = 0
    for sent_idx in range(N_SENTENCES_NR1):
        trial_onset, _trial_offset = pairs[sent_idx]

        sentence_rawdata = f[sd["rawData"][sent_idx, 0]][()]  # (n_samples_sentence, 105)
        word_group = f[sd["word"][sent_idx, 0]]
        content_field = word_group["content"]
        raweeg_field = word_group["rawEEG"]
        nfix_field = word_group["nFixations"]

        n_words = content_field.shape[0]
        for word_idx in range(n_words):
            nfix_arr = f[nfix_field[word_idx, 0]][()]
            nfix = int(nfix_arr.flatten()[0]) if nfix_arr.size == 1 else 0
            if nfix <= 0:
                continue

            fix_cell = f[raweeg_field[word_idx, 0]]
            if fix_cell.size == 0:
                continue
            first_fix = np.asarray(f[fix_cell[0, 0]][()], dtype=np.float64)  # (n_samp, 105)

            offset = find_offset(sentence_rawdata, first_fix)
            if offset is None:
                continue

            onset_sample = trial_onset + offset
            duration = first_fix.shape[0]
            word_text = h5_text(f, content_field[word_idx, 0])

            marker_code += 1
            end_sample = min(onset_sample + duration, n_samples)
            marker_channel[onset_sample:end_sample] = marker_code

            sentence_ids.append(sent_idx)
            word_ids.append(word_idx)
            word_texts.append(word_text)
            onset_samples.append(onset_sample)
            duration_samples.append(duration)
            n_fixations_list.append(nfix)
            marker_codes.append(marker_code)

    f.close()
    print(f"Recovered {len(word_texts)} word-reading onsets across {N_SENTENCES_NR1} sentences.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sio.savemat(OUT_PATH, {
        "eeg_data": raw_data,                                    # (128, n_samples) float32, uV
        "channel_names": np.array(ch_names, dtype=object),
        "srate": srate,
        "marker_channel": marker_channel.reshape(1, -1),         # (1, n_samples) int32
        "sentence_id": np.array(sentence_ids, dtype=np.int32),
        "word_id": np.array(word_ids, dtype=np.int32),
        "word_text": np.array(word_texts, dtype=object),
        "onset_sample": np.array(onset_samples, dtype=np.int32),
        "duration_samples": np.array(duration_samples, dtype=np.int32),
        "n_fixations": np.array(n_fixations_list, dtype=np.int32),
        "marker_code": np.array(marker_codes, dtype=np.int32),
    }, do_compression=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
