"""
Extract word-level EEG signals for the first 5 sentences (examples) of subject ZDN.

Uses:
  - wordbounds_NR_ZDN.mat  : word bounding boxes (pixel coords) per sentence
  - resultsZDN_NR.mat      : processed results with rawEEG per word/fixation

Output structure:
  word_eeg_ZDN/
    example_0/
      word_00_With/
        eeg.npy          (105 channels x total_fixation_samples)
        metadata.txt     (word text, num_fixations, shape)
      word_01_his/
        ...
    example_1/
      ...
"""

import os
import numpy as np
import scipy.io as sio

ZDN_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.dirname(ZDN_DIR)
NR_DIR = os.path.dirname(PREPROCESSED_DIR)
MATLAB_DIR = os.path.join(NR_DIR, "Matlab_files")

WORDBOUNDS_FILE = os.path.join(ZDN_DIR, "wordbounds_NR_ZDN.mat")
RESULTS_FILE = os.path.join(MATLAB_DIR, "resultsZDN_NR.mat")
OUTPUT_DIR = os.path.join(ZDN_DIR, "word_eeg_ZDN")

NUM_EXAMPLES = 5


def sanitize(text):
    """Make word text safe for use as a directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def main():
    print("Loading wordbounds file...")
    wb_data = sio.loadmat(WORDBOUNDS_FILE)
    wordbounds = wb_data["wordbounds"]   # shape (1, 300)

    print("Loading Matlab results file...")
    results = sio.loadmat(RESULTS_FILE)
    sentence_data = results["sentenceData"]   # shape (1, 300)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ex_idx in range(NUM_EXAMPLES):
        sentence = sentence_data[0, ex_idx]
        sentence_text = str(sentence["content"].flat[0])
        words_arr = sentence["word"][0]   # array of word structs
        word_bounds = wordbounds[0, ex_idx]   # shape (N_words, 4): [x1, y1, x2, y2]

        n_words = len(words_arr)
        assert word_bounds.shape[0] == n_words, (
            f"Sentence {ex_idx}: wordbounds has {word_bounds.shape[0]} entries "
            f"but word array has {n_words} words"
        )

        example_dir = os.path.join(OUTPUT_DIR, f"example_{ex_idx}")
        os.makedirs(example_dir, exist_ok=True)

        # Write sentence-level metadata
        with open(os.path.join(example_dir, "sentence.txt"), "w") as f:
            f.write(f"Sentence {ex_idx}: {sentence_text}\n")
            f.write(f"Number of words: {n_words}\n")

        print(f"\nExample {ex_idx}: \"{sentence_text}\"")
        print(f"  {n_words} words")

        for w_idx in range(n_words):
            word_struct = words_arr[w_idx]
            word_text = str(word_struct["content"].flat[0])
            bounds = word_bounds[w_idx]   # [x1, y1, x2, y2]

            raw_eeg = word_struct["rawEEG"]   # shape (1, num_fixations)
            num_fixations = raw_eeg.shape[1]

            word_dir_name = f"word_{w_idx:02d}_{sanitize(word_text)}"
            word_dir = os.path.join(example_dir, word_dir_name)
            os.makedirs(word_dir, exist_ok=True)

            if num_fixations == 0:
                # No fixations recorded for this word — save empty array
                eeg_combined = np.zeros((105, 0), dtype=np.float32)
                print(f"  word {w_idx:02d} '{word_text}': no fixations")
            else:
                fixation_segments = [raw_eeg[0, j] for j in range(num_fixations)]
                eeg_combined = np.concatenate(fixation_segments, axis=1).astype(np.float32)
                print(f"  word {w_idx:02d} '{word_text}': {num_fixations} fixation(s), EEG shape {eeg_combined.shape}")

            np.save(os.path.join(word_dir, "eeg.npy"), eeg_combined)

            # Save per-fixation segments individually as well
            for j in range(num_fixations):
                seg = raw_eeg[0, j].astype(np.float32)
                np.save(os.path.join(word_dir, f"fixation_{j}.npy"), seg)

            # Write word metadata
            with open(os.path.join(word_dir, "metadata.txt"), "w") as f:
                f.write(f"Word index: {w_idx}\n")
                f.write(f"Word text: {word_text}\n")
                f.write(f"Bounding box (x1,y1,x2,y2): {bounds.tolist()}\n")
                f.write(f"Num fixations: {num_fixations}\n")
                f.write(f"EEG shape (channels x samples): {eeg_combined.shape}\n")
                f.write(f"Channels: 105\n")
                f.write(f"EEG file: eeg.npy  (concatenated across all fixations)\n")
                for j in range(num_fixations):
                    seg = raw_eeg[0, j]
                    f.write(f"  fixation_{j}.npy: shape {seg.shape}\n")

    print(f"\nDone. Output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
