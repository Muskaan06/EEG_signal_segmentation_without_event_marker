"""Central configuration: paths, constants, ROI/band/epoch definitions, seed."""

import os

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── Dataset paths ────────────────────────────────────────────────────────────
DATASET_ROOT = "/media/kpdubey/8.0 TB Volume/Muskaan/dataset/ZuCo/zuco_1"

TASK_DIRS = {
    "NR": {
        "raw_data": os.path.join(DATASET_ROOT, "NR", "raw_data"),
        "preprocessed": os.path.join(DATASET_ROOT, "NR", "preprocessed"),
        "matlab_files": os.path.join(DATASET_ROOT, "NR", "Matlab_files"),
        "results_suffix": "_NR.mat",
        "wordbounds_prefix": "wordbounds_NR_",
        "n_blocks": 6,
    },
    "TSR": {
        "raw_data": os.path.join(DATASET_ROOT, "TSR", "raw_data"),
        "preprocessed": os.path.join(DATASET_ROOT, "TSR", "matlab_files"),
        "matlab_files": os.path.join(DATASET_ROOT, "TSR", "matlab_files", "Matlab_files"),
        "results_suffix": "_TSR.mat",
        "wordbounds_prefix": "wordbounds_TSR_",
        "n_blocks": 9,
    },
}

ALL_SUBJECTS = [
    "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN",
    "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH",
]

VALIDATION_SUBJECTS = ["ZAB", "ZDN", "ZJM"]

# ── Repo paths ───────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

DIRS = {
    "alignment": os.path.join(OUTPUTS_DIR, "alignment"),
    "cleaned_eeg": os.path.join(OUTPUTS_DIR, "cleaned_eeg"),
    "epochs": os.path.join(OUTPUTS_DIR, "epochs"),
    "features": os.path.join(OUTPUTS_DIR, "features"),
    "models": os.path.join(OUTPUTS_DIR, "models"),
    "figures": os.path.join(OUTPUTS_DIR, "figures"),
    "stats": os.path.join(OUTPUTS_DIR, "stats"),
    "report": os.path.join(OUTPUTS_DIR, "report"),
    "rois": os.path.join(OUTPUTS_DIR, "rois"),
}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


# ── Recording constants ──────────────────────────────────────────────────────
SRATE = 500.0
N_CHANNELS_RAW = 128
MONTAGE_NAME = "GSN-HydroCel-129"

# Event trigger codes observed in raw EEGLAB .event.type for ZuCo 1 NR/TSR
TRIGGER_BLOCK_START = "90"
TRIGGER_BLOCK_END = "18"
TRIGGER_FIXCROSS_ON = "12"
TRIGGER_FIXCROSS_OFF = "13"
TRIGGER_SENTENCE_ON = "15"
TRIGGER_MISC_A = "10"
TRIGGER_MISC_B = "11"

# ── Epoching ─────────────────────────────────────────────────────────────────
EPOCH_TMIN = -0.2   # s
EPOCH_TMAX = 0.8    # s
BASELINE = (-0.2, 0.0)

# Minimum gap (s) for a window to qualify as a "resting"/"inter-word" non-reading
# control interval (must be at least as long as the epoch window itself).
MIN_CONTROL_GAP_S = EPOCH_TMAX - EPOCH_TMIN

# ── Frequency bands (Hz) ─────────────────────────────────────────────────────
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ── ERP components (ms relative to word/fixation onset) ─────────────────────
ERP_COMPONENTS = {
    "P1":   (80, 120),
    "N170": (140, 220),
    "N250": (200, 300),
    "N400": (300, 500),
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
BANDPASS = (0.1, 40.0)
NOTCH_CANDIDATES = (50.0, 60.0)

# ── ROI pairs for connectivity (see pipeline/rois.py for exact channel lists) ─
CONNECTIVITY_PAIRS = [
    ("vwfa", "pstg"),
    ("pstg", "wernicke"),
    ("wernicke", "broca"),
    ("parietal", "pstg"),
    ("parietal", "wernicke"),
]
