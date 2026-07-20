# VWFA–Wernicke EEG Word-Event Segmentation

Unsupervised detection of word-reading events directly from continuous scalp EEG,
using two independent, signal-processing-based detectors targeting the Visual Word
Form Area (VWFA) and Wernicke's area scalp projections, combined by simple union.
Evaluated against exact eye-tracking-derived word onset/offset times on the ZuCo 1
dataset (12 subjects, Normal Reading + Task-Specific Reading).

**Status: private research repository, work in progress.** Every detection
threshold here is empirically calibrated from the data itself (never hardcoded),
and every claim below is explicitly labeled as either a measured result or an
untested hypothesis — see [Limitations](#limitations--honest-caveats).

---

## 1. Motivation

Datasets like ZuCo provide word-level EEG features, but only because they are
*aligned to eye-tracking fixations recorded by dedicated hardware* — the EEG signal
itself is never used to figure out *when* a word was being read. This project asks
a narrower question: **can word-reading onset be located from the EEG signal
alone**, with no eye-tracker, using only two scalp regions motivated by the reading
literature —

- **VWFA** (left occipito-temporal cortex) — implicated in visual/orthographic
  word-form recognition.
- **Wernicke's area** (left posterior superior temporal / temporo-parietal cortex)
  — implicated in lexical-semantic access, indexed here via a classic ERP
  component complex (P1/N170/N250/N400).

Two independent detectors, one per region, are run over every raw continuous EEG
block and combined by union (Wernicke-ERP catches words VWFA misses; it never
overrides or duplicates a VWFA detection).

### 1.1 How this differs from standard EEG segmentation

Most EEG segmentation falls into one of two families, and this project is
deliberately neither:

| | **Event-marker segmentation** | **Fixed-window segmentation** | **This project** |
|---|---|---|---|
| How a segment boundary is decided | An external trigger says so (stimulus code, button press, or — in ZuCo's case — an eye-tracker fixation timestamp) | A clock says so (chop into 1s / 250ms / N-sample epochs, evenly spaced) | The EEG signal itself says so (a detector decides a word-reading event just occurred, right now) |
| Needs a second, non-EEG data source for timing? | Yes, always — the whole point is to borrow ground truth from stimulus/behavioral logs | No | No — this is the constraint being tested |
| Segment spacing | Irregular, but *given*, not discovered | Regular / constant | Irregular *and discovered* (variable inter-word gaps, refractory-period spaced, exactly like real reading behavior) |
| What happens with no external ground truth at all | Not applicable — the method requires it | Still works, but boundaries are meaningless with respect to cognitive events | Still works — this is the intended use case |
| Failure mode | None, by construction (boundaries are exact) | Boundaries frequently split or straddle the actual event of interest | Boundaries can be wrong (early/late/missed/spurious) — quantified in [Results](#4-results) |

Concretely: ZuCo's own word-level EEG features (used throughout the rest of this
research program, e.g. the EEG-to-text decoding pipelines) are event-marker
segments — exact, because they're sliced using the eye-tracker's fixation clock,
not the EEG. A fixed-window baseline would instead chop each block into
equal-length epochs regardless of when words actually occurred, which is
adequate for tasks like sleep-stage or overall mental-state classification but
cannot answer "did a word-reading event happen at time *t*" — the very question
this project asks. What's evaluated here is *whether the EEG signal alone can
recover event timing close enough to the event-marker ground truth to be useful
without ever consulting it* — reported honestly in Section 4, including where it
falls short.

## 2. Method

### 2.1 ROI definition (`vwfa_word_detection/rois.py`)

Scalp ROIs are computed from the `GSN-HydroCel-129` montage's 3D channel
coordinates — nearest-channel-to-anatomical-target, **not** a source-localized
parcellation (the net has no direct correspondence to real cortical geometry). A
fixed 24-channel periocular/face ring is excluded first.

| ROI | Channels | Approx. site |
|---|---|---|
| VWFA | 4 (E50, E58, E64, E69) | left occipito-temporal (~PO7/P7) |
| Wernicke | 3 | left temporo-parietal (~CP5/TP7) |
| pSTG | 3 | left posterior superior temporal (~T5/TP7) |
| Occipital | 5 | visual cortex (Oz/O1/O2) |
| Broca | 3 | left inferior frontal |
| Parietal | 5 | left angular/supramarginal |

### 2.2 VWFA detector (`vwfa_word_detection/word_detection.py`)

Hypothesis under test: the VWFA-ROI-averaged signal's amplitude envelope has
three distinguishable levels over time — *no word* / *word read* / *word present
but not understood*. Pipeline:

1. Bandpass the VWFA-ROI-averaged signal (1–30 Hz), take the Hilbert envelope,
   boxcar-smooth (60 ms window).
2. Classify every sample into low / medium-high / very-high via two percentile
   cut-points, calibrated **per block** from that block's own envelope
   distribution (never a fixed threshold).
3. Take the peak-amplitude sample of every "medium-high" run lasting ≥ 40 ms as a
   candidate word event.
4. Non-max-suppress candidates within a 250 ms refractory window (strongest wins).
5. Place a ±200 ms marker window around each surviving candidate.

A fourth stage — requiring the Wernicke-ROI signal to also spike 100–150 ms later
as *confirmation* before accepting a VWFA candidate — was tried and **removed**:
traced against 625 ground-truth word onsets, it discarded 17.8% of true words
while only modestly improving precision (F1 0.612 with confirmation vs. **0.691
without**). The code (`confirm_with_wernicke`) is kept but unused for this reason.

### 2.3 Wernicke-ERP detector (`vwfa_wernicke_word_detection/wernicke_erp_detection.py`)

A deliberately *different* signal-processing paradigm from the VWFA detector (ERP
component analysis, not envelope thresholding), applied to the Wernicke-ROI signal:

1. Bandpass, **sign preserved** (no Hilbert envelope — ERP polarity carries the
   signal here).
2. Find prominent troughs (`scipy.find_peaks` on the inverted signal) as candidate
   N400-like deflections, spaced ≥ 200 ms apart.
3. Each trough is presumed to be the N400 peak of a word read ~400 ms earlier;
   build a short baseline-corrected epoch around that hypothesized onset and score
   the peak amplitude of every component window (P1 80–120 ms, N170 140–220 ms,
   N250 200–300 ms, N400 300–500 ms), summed into one candidate-strength score.
4. Percentile-calibrate an accept cutoff on the pooled score distribution (top
   decile survives).
5. Refractory non-max-suppression, then a ±200 ms marker window per survivor.

### 2.4 Union combination (`vwfa_wernicke_word_detection/combine.py`)

Additive, not confirmatory: VWFA's markers are the base set; a Wernicke-ERP marker
is added **only if it does not already overlap a VWFA marker**. Wernicke-ERP can
therefore only ever catch words VWFA missed — it can never duplicate or override a
VWFA detection.

## 3. Data

ZuCo 1, both reading tasks, all 12 subjects (`ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH`):

| | |
|---|---|
| Blocks processed | 179 (100% `status: ok`, zero failures) |
| Total recording | 14.27 hours of continuous 500 Hz EEG |
| Ground-truth word onsets | 92,320 (recovered via eye-tracking-aligned ZuCo metadata) |

## 4. Results

Real block-level **temporal overlap** accuracy: a detected marker window counts as
a match if it overlaps the ground-truth word's true onset/offset interval
(greedy overlap matching, same method for both detectors).

| Subject | True words | Predicted | Matched | via Wernicke-ERP | Precision | Recall | F1 | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZAB | 7,819 | 7,063 | 5,161 | 2 | 0.731 | 0.660 | 0.694 | 0.531 |
| ZDM | 7,124 | 7,151 | 4,951 | 4 | 0.692 | 0.695 | 0.694 | 0.531 |
| ZDN | 6,210 | 5,862 | 4,366 | 6 | 0.745 | 0.703 | 0.723 | 0.567 |
| ZGW | 6,855 | 9,981 | 4,962 | 3 | 0.497 | 0.724 | 0.589 | 0.418 |
| ZJM | 10,067 | 12,978 | 7,404 | 5 | 0.571 | 0.735 | 0.643 | 0.473 |
| ZJN | 10,615 | 15,006 | 8,833 | 10 | 0.589 | 0.832 | 0.690 | 0.526 |
| ZJS | 4,295 | 10,960 | 2,932 | 9 | 0.268 | 0.683 | 0.384 | 0.238 |
| ZKB | 6,650 | 8,860 | 4,565 | 3 | 0.515 | 0.686 | 0.589 | 0.417 |
| ZKH | 8,931 | 10,170 | 6,768 | 4 | 0.665 | 0.758 | 0.709 | 0.549 |
| ZKW | 10,428 | 14,188 | 8,260 | 7 | 0.582 | 0.792 | 0.671 | 0.505 |
| ZMG | 7,888 | 7,629 | 5,447 | 6 | 0.714 | 0.691 | 0.702 | 0.541 |
| ZPH | 5,438 | 6,984 | 3,662 | 6 | 0.524 | 0.673 | 0.590 | 0.418 |
| **Mean (per-subject)** | | | | | **0.591** | **0.719** | **0.640** | **0.476** |
| **Micro (pooled)** | 92,320 | 116,832 | 67,311 | 65 | **0.576** | **0.729** | — | — |

**Note on the Wernicke-ERP contribution:** of 116,832 total predicted events,
only 283 came from the Wernicke-ERP addition (0.24%), and only 65 of the 67,311
total matches (0.1%) were attributable to it. In this combined system, **VWFA
drives essentially all correct detections**; the Wernicke-ERP union adds marginal
recall at best. This is reported as-is, not smoothed over.

## 5. Limitations & honest caveats

- **Scalp ROI ≠ source localization.** "VWFA channels" are the nearest EEG
  electrodes to an approximate scalp target, not a verified cortical source.
  Volume conduction means this is a noisy proxy, not a clean isolation of VWFA
  activity.
- **Exploratory, not an established neuroscience result.** The "three
  distinguishable VWFA signal levels" hypothesis and the ERP-component-sum
  scoring are both empirical hypotheses being tested here, not textbook facts.
- **Subject-variable performance.** Per-subject F1 ranges from 0.38 (ZJS) to 0.72
  (ZDN) — no single threshold/parameter set works equally well for everyone.
- **Precision is the weaker side of the trade-off.** Recall (0.72–0.73) is
  consistently higher than precision (0.58–0.59); the detector over-predicts more
  than it misses.
- **Small-sample calibration.** Thresholds are calibrated from within-block
  percentiles (no true resting/no-word reference exists in this data), which is
  an explicitly acknowledged limitation of the calibration method, not a design
  goal.
- **No word identity.** Output is *when* a word-reading event likely occurred —
  never *which* word, and never aligned to a specific sentence position without
  additional (unimplemented) alignment logic.

## 6. Repository layout

```
vwfa_word_detection/                       # VWFA envelope-threshold detector (self-contained)
├── config.py                              # paths, ROI/band/epoch constants, calibration percentiles
├── io_raw.py                              # raw ZuCo EEGLAB .mat / trigger / results loader
├── rois.py                                 # montage-coordinate-based ROI channel assignment
├── word_detection.py                       # core detection algorithm (Section 2.2)
├── run_word_detection_zuco1.py             # batch runner over all ZuCo1 NR/TSR blocks
└── word_detection_output/
    ├── block_summary.csv                   # per-block stats
    └── word_events.csv                     # every detected marker window

vwfa_wernicke_word_detection/               # Combined VWFA + Wernicke-ERP detector
├── wernicke_erp_detection.py               # Wernicke-ERP detector (Section 2.3)
├── combine.py                              # union combination logic (Section 2.4)
├── run_combined_zuco1.py                   # batch runner (imports vwfa_word_detection as `pipeline`)
└── word_detection_output/
    ├── block_summary.csv
    ├── word_events.csv
    └── eval/
        ├── real_temporal_accuracy_combined.py    # ground-truth overlap scoring (Section 4)
        ├── real_temporal_accuracy_combined.csv   # results table above, raw
        ├── sentence_trace_batch_combined.py       # per-sentence diagnostic trace generator
        ├── sentence_traces_combined.json          # 60 sampled sentence traces (5/subject)
        ├── viz/                                    # interactive HTML trace viewers
        └── report/                                  # rendered report (HTML + PDF)
```

## 7. Reproducing

Both runners expect the ZuCo 1 raw dataset locally (`dataset/ZuCo/zuco1/{task2-NR,task3-TSR}/`)
and the `mne` environment used elsewhere in this research (`config.py` paths will
need adjusting to your local dataset location — they are currently hardcoded to
the original machine's paths).

```bash
# VWFA-only detector, all subjects, both tasks
python vwfa_word_detection/run_word_detection_zuco1.py

# Combined VWFA + Wernicke-ERP detector
python vwfa_wernicke_word_detection/run_combined_zuco1.py

# Score against ground truth
python vwfa_wernicke_word_detection/word_detection_output/eval/real_temporal_accuracy_combined.py
```
