# EEG_research
Trying new EEG stuff

## Reading vs. Non-Reading EEG classification (ZuCo 1)

A modular pipeline that determines, from raw EEG alone, whether a subject is
reading a word (READING) vs. not (NON-READING). Binary reading-state
classification, not word-identity decoding.

### Environment

Everything runs in the `eeg2` conda environment (mne, torch, xgboost, etc.):

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eeg2
cd "/media/kpdubey/8.0 TB Volume/Muskaan/EEG_research"
```

### Run the full pipeline

```bash
python scripts/run_all.py --subjects ZAB ZDN ZJM --task NR
```

This does everything in one go: loads raw EEGLAB `.mat` files for each
subject/block, recovers exact word-onset timing (`pipeline/alignment.py`),
runs full preprocessing (notch/bandpass/bad-channel/ICA/interpolation,
`pipeline/preprocessing.py`), builds balanced READING/NON-READING epochs
(`pipeline/epochs.py`), extracts ERP/spectral/connectivity features
(`pipeline/features_*.py`), trains classical models (LR/SVM/RF/XGBoost/MLP,
subject-dependent + Leave-One-Subject-Out), trains deep models
(EEGNet/SimpleCNN/CNN-LSTM), runs statistics (`pipeline/stats.py`), and
writes `outputs/report/report.md`.

Flags:
- `--subjects` — any subset of `ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH`
  (ZJN has no NR results file and is skipped automatically for NR).
- `--task` — `NR` (Normal Reading) or `TSR` (Task-Specific Reading).
- `--n-permutations` — permutation/cluster-test iterations (default 1000).
- `--deep-epochs` — training epochs for the deep models (default 25).

Preprocessing (especially ICA) is the slow part — budget ~15-20s/block. For
just 3 subjects x 6 NR blocks that's ~15-20 minutes total; running all 12
subjects (NR+TSR) is a multi-hour job best kicked off in the background.

### Resuming without re-preprocessing

If raw loading/preprocessing already completed and only deep learning/stats/
report need re-running (e.g. after tuning a model), use:

```bash
python scripts/resume_stage2.py
```

It reloads `outputs/epochs/NR_validation-epo.fif` and
`outputs/features/NR_{X,meta}.parquet` instead of redoing preprocessing.

### Outputs

```
outputs/
  cleaned_eeg/        # (not persisted per-block by default; add raw.save() in preprocessing if needed)
  epochs/              # balanced READING/NON-READING epochs (-epo.fif)
  features/            # feature matrix (X) + per-epoch metadata, parquet
  models/              # classical + deep model results (csv)
  figures/
    qa/<subj>_<task><n>/   # PSD before/after, ICA components, before/after overlay
    erp/                   # grand-average ERP by ROI
    spectral/              # band-power topomaps
    connectivity/          # ROI connectivity matrices + graphs
    models/                # model comparison, confusion matrices, ROC curves
  stats/               # feature significance (Cohen's d / t-test / permutation), cluster-permutation ERP
  rois/roi_channels.json   # exact channel list per ROI
  report/report.md     # final summary report
```

### Scaling up

Once validation-subset results look right, scale to the full dataset:

```bash
python scripts/run_all.py --subjects ZAB ZDM ZDN ZGW ZJM ZJS ZKB ZKH ZKW ZMG ZPH --task NR
```

(omit ZJN for NR; run again with `--task TSR` for Task-Specific Reading,
where all 12 subjects have results files). This is a long, unattended run —
launch it in the background and monitor `outputs/`.

### Modules (`pipeline/`)

| Module | Purpose |
|---|---|
| `config.py` | paths, constants, ROI/band/epoch-window definitions, seed |
| `io_raw.py` | raw EEGLAB `.mat` loader, ET/wordbounds/results readers, dataset stats |
| `alignment.py` | recovers exact word/fixation onset sample in the raw recording from event triggers + the ZuCo results `.mat` (no eye-tracker clock sync needed — see module docstring) |
| `preprocessing.py` | notch/bandpass/bad-channel-detection/average-reference/ICA/interpolation + QA plots |
| `epochs.py` | builds balanced READING (word-onset) and NON-READING (pre-block/inter-trial/inter-word) epochs |
| `rois.py` | ROI channel groups from montage 3D coordinates (scalp-location approximations) |
| `features_erp.py` | P1/N170/N250/N400 peak/mean/latency/AUC per ROI + grand-average plots |
| `features_spectral.py` | band power (absolute/relative), ERD/ERS, topomaps |
| `features_connectivity.py` | coherence/PLV/wPLI between ROIs, matrices + graphs |
| `features_matrix.py` | assembles/normalizes the full feature matrix, feature selection, PCA, importance |
| `models_classical.py` | LR/SVM/RF/XGBoost/MLP, subject-dependent + Leave-One-Subject-Out CV |
| `models_deep.py` | EEGNet/SimpleCNN/CNN-LSTM trained on raw epochs (torch, CPU by default) |
| `stats.py` | paired t-tests, permutation tests, cluster-based permutation, effect sizes |
| `report.py` | assembles `outputs/report/report.md` |
