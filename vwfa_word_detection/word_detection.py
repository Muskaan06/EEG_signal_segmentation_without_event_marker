"""VWFA-based word-reading-event detection.

EXPLORATORY / EMPIRICAL, not an established neuroscience result. The
operating hypothesis being tested here (not assumed true) is that the
VWFA-ROI signal shows three distinguishable levels over time:

  - low          : no word currently being visually processed
  - medium_high   : a word is being read and understood
  - very_high     : a word is present but not readable/understood

Detection works by: (1) computing a smoothed amplitude envelope of the
VWFA-ROI-averaged signal, (2) classifying it into the three levels via
thresholds calibrated from data (never hardcoded), (3) picking peak
"medium_high" events as candidates with a refractory period so one word
isn't detected twice, and (4) placing a fixed +/-200ms marker window
(config.WORD_MARKER_HALF_WINDOW_MS) around each candidate.

Previously a 4th stage required the Wernicke-ROI signal to also show
elevated activity ~100-150ms later (config.VWFA_WERNICKE_LAG_MS_RANGE) as
independent confirmation before a candidate became a marker. Traced against
625 ground-truth word onsets (subject YHS, NR block 1), that step discarded
17.8% of true words outright (accounting for more lost recall than the
envelope-thresholding and minimum-duration stages combined) while only
modestly improving precision: with confirmation, precision 0.736 / recall
0.523 / F1 0.612; without it, precision 0.683 / recall 0.699 / F1 0.691 --
a net F1 improvement from dropping it. `confirm_with_wernicke` is kept below
(unused by `detect_word_events`) in case Wernicke confirmation is wanted
again for a specific analysis.

Every function here is source-agnostic: it takes a plain 1-D signal array
(optionally containing NaN gaps) + a sample rate, never an mne object or a
specific dataset's file layout. This lets the same code run against the
local ZDN pseudo-continuous demo data (tests/word_eeg_ZDN/) today and
against real continuous raw EEG blocks (pipeline/io_raw.py) once the full
dataset is available.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

from pipeline import config


@dataclass
class SignalThresholds:
    """Two cut-points partitioning envelope amplitude into 3 ordered bins:
    [0, low_high) -> low, [low_high, high_very_high) -> medium_high,
    [high_very_high, inf) -> very_high.
    """
    low_high: float
    high_very_high: float


def _finite_runs(sig):
    """Yield (start, end) index pairs (end exclusive) of contiguous
    non-NaN runs in a 1-D array."""
    finite = np.isfinite(sig)
    if not finite.any():
        return
    edges = np.diff(finite.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if finite[0]:
        starts = [0] + starts
    if finite[-1]:
        ends = ends + [len(sig)]
    yield from zip(starts, ends)


def _envelope(sig, srate, band, window_ms):
    """Bandpass -> Hilbert envelope -> boxcar smooth, computed per contiguous
    finite run (never across a NaN gap), NaN elsewhere. Returns an array the
    same length as `sig`.
    """
    out = np.full(sig.shape, np.nan, dtype=np.float64)
    nyq = srate / 2.0
    lo, hi = band
    win_n = max(1, int(round(window_ms / 1000.0 * srate)))
    kernel = np.ones(win_n) / win_n

    for start, end in _finite_runs(sig):
        seg = sig[start:end].astype(np.float64)
        n = len(seg)
        if n < 9:
            # Too short to filter meaningfully; envelope undefined.
            continue
        # filtfilt needs len(seg) > 3 * max(len(a), len(b)); order-4 butter
        # has ~9-tap coefficients, so guard with a generous minimum length.
        pad = 3 * 9
        if n <= pad:
            continue
        b, a = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
        filtered = filtfilt(b, a, seg)
        env = np.abs(hilbert(filtered))
        if win_n > 1:
            env = np.convolve(env, kernel, mode="same")
        out[start:end] = env

    return out


def classify_signal_levels(sig, srate, thresholds,
                            band=config.VWFA_BAND,
                            window_ms=config.SIGNAL_LEVEL_WINDOW_MS):
    """Classify each sample of `sig` into a signal level.

    Returns (labels, envelope): labels is an int array with -1 = no data
    (NaN gap or too-short run to filter), 0 = low, 1 = medium_high,
    2 = very_high. envelope is the smoothed amplitude envelope (NaN where
    labels == -1), returned so callers can find peak samples without
    recomputing it.
    """
    sig = np.asarray(sig, dtype=np.float64)
    envelope = _envelope(sig, srate, band, window_ms)

    labels = np.full(sig.shape, -1, dtype=int)
    finite = np.isfinite(envelope)
    labels[finite & (envelope < thresholds.low_high)] = 0
    labels[finite & (envelope >= thresholds.low_high) &
           (envelope < thresholds.high_very_high)] = 1
    labels[finite & (envelope >= thresholds.high_very_high)] = 2

    return labels, envelope


def calibrate_signal_thresholds(sig, srate, mode="within_reading_percentile",
                                 band=config.VWFA_BAND,
                                 window_ms=config.SIGNAL_LEVEL_WINDOW_MS,
                                 **kwargs):
    """Derive SignalThresholds from data. Never hardcode cutoffs.

    mode="within_reading_percentile" (local ZDN demo, small-sample
        limitation: no true resting/no-word reference exists in the data,
        so this pools the envelope across all provided word segments and
        takes within-distribution percentiles config.LOCAL_CALIB_LOW_PCT /
        config.LOCAL_CALIB_HIGH_PCT as the two cut-points. "Low" here means
        "low relative to other words in this sample," not resting baseline.

    mode="baseline_vs_reading" (future, full continuous dataset): pass
        `non_reading_envelope` and `reading_n170_envelope` arrays (already
        extracted by the caller using pipeline/epochs.py's reading vs
        non_reading distinction, restricted to the N170 window for the
        reading side) as kwargs; thresholds are percentiles of those two
        pools (config.BASELINE_CALIB_LOW_PCT / BASELINE_CALIB_HIGH_PCT).
    """
    if mode == "within_reading_percentile":
        envelope = _envelope(np.asarray(sig, dtype=np.float64), srate, band, window_ms)
        pool = envelope[np.isfinite(envelope)]
        if pool.size == 0:
            raise ValueError("No finite envelope samples to calibrate thresholds from.")
        low_high = float(np.percentile(pool, config.LOCAL_CALIB_LOW_PCT))
        high_very_high = float(np.percentile(pool, config.LOCAL_CALIB_HIGH_PCT))
        return SignalThresholds(low_high=low_high, high_very_high=high_very_high)

    if mode == "baseline_vs_reading":
        non_reading_envelope = kwargs["non_reading_envelope"]
        reading_n170_envelope = kwargs["reading_n170_envelope"]
        low_high = float(np.percentile(non_reading_envelope, config.BASELINE_CALIB_LOW_PCT))
        high_very_high = float(np.percentile(reading_n170_envelope, config.BASELINE_CALIB_HIGH_PCT))
        return SignalThresholds(low_high=low_high, high_very_high=high_very_high)

    raise ValueError(f"Unknown calibration mode: {mode!r}")


def detect_vwfa_candidates(sig, srate, thresholds,
                            band=config.VWFA_BAND,
                            window_ms=config.SIGNAL_LEVEL_WINDOW_MS,
                            min_sustained_ms=config.MIN_SUSTAINED_MS,
                            refractory_ms=config.VWFA_REFRACTORY_MS):
    """Find "medium_high" runs lasting >= min_sustained_ms, take the peak
    envelope sample within each run as the candidate, then apply non-max
    suppression with refractory_ms spacing (strongest candidate in a
    cluster wins). Returns a DataFrame:
    {sample, level, peak_amplitude, run_start_sample, run_end_sample}.
    """
    labels, envelope = classify_signal_levels(sig, srate, thresholds, band, window_ms)

    min_len = max(1, int(round(min_sustained_ms / 1000.0 * srate)))
    refractory_n = int(round(refractory_ms / 1000.0 * srate))

    runs = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j < n and labels[j] == 1:
                j += 1
            if (j - i) >= min_len:
                seg = envelope[i:j]
                peak_offset = int(np.argmax(seg))
                runs.append({
                    "sample": i + peak_offset,
                    "level": 1,
                    "peak_amplitude": float(seg[peak_offset]),
                    "run_start_sample": i,
                    "run_end_sample": j,
                })
            i = j
        else:
            i += 1

    if not runs:
        return pd.DataFrame(columns=["sample", "level", "peak_amplitude",
                                      "run_start_sample", "run_end_sample"])

    runs.sort(key=lambda r: r["peak_amplitude"], reverse=True)
    accepted = []
    accepted_samples = []
    for r in runs:
        if all(abs(r["sample"] - s) >= refractory_n for s in accepted_samples):
            accepted.append(r)
            accepted_samples.append(r["sample"])

    df = pd.DataFrame(accepted).sort_values("sample").reset_index(drop=True)
    return df


def confirm_with_wernicke(vwfa_candidates, wernicke_sig, srate, wernicke_thresholds,
                           band=config.VWFA_BAND,
                           window_ms=config.SIGNAL_LEVEL_WINDOW_MS,
                           lag_range_ms=config.VWFA_WERNICKE_LAG_MS_RANGE):
    """For each VWFA candidate at t0, confirm if the Wernicke-ROI signal
    reaches medium_high or above anywhere in [t0+lag_min, t0+lag_max].
    Any-crossing rule (not sustained-run), since the Wernicke ROI is a
    small channel average and a stricter rule would likely reject almost
    everything. Unconfirmed rows are kept (confirmed=False), not dropped,
    so diagnostics can inspect why confirmation failed.
    """
    labels, envelope = classify_signal_levels(wernicke_sig, srate, wernicke_thresholds, band, window_ms)
    n = len(labels)
    lag_min_n = int(round(lag_range_ms[0] / 1000.0 * srate))
    lag_max_n = int(round(lag_range_ms[1] / 1000.0 * srate))

    out = vwfa_candidates.copy()
    confirmed = []
    confirm_sample = []
    lag_ms = []

    for _, row in vwfa_candidates.iterrows():
        t0 = int(row["sample"])
        w_start = max(0, t0 + lag_min_n)
        w_end = min(n, t0 + lag_max_n + 1)
        if w_start >= w_end:
            confirmed.append(False)
            confirm_sample.append(np.nan)
            lag_ms.append(np.nan)
            continue

        window_labels = labels[w_start:w_end]
        window_env = envelope[w_start:w_end]
        hits = np.where(window_labels >= 1)[0]
        if hits.size == 0:
            confirmed.append(False)
            confirm_sample.append(np.nan)
            lag_ms.append(np.nan)
            continue

        hit_env = window_env[hits]
        best_hit = hits[np.nanargmax(hit_env)] if np.isfinite(hit_env).any() else hits[0]
        w_sample = w_start + best_hit
        confirmed.append(True)
        confirm_sample.append(w_sample)
        lag_ms.append((w_sample - t0) / srate * 1000.0)

    out["confirmed"] = confirmed
    out["wernicke_confirm_sample"] = confirm_sample
    out["wernicke_lag_ms"] = lag_ms
    return out


def build_marker_windows(confirmed_candidates, srate,
                          half_window_ms=config.WORD_MARKER_HALF_WINDOW_MS,
                          n_samples=None):
    """Build +/-half_window_ms marker windows around each confirmed==True
    candidate's vwfa sample. Returns a DataFrame shaped like
    pipeline.alignment's word_events for consistency:
    {sample (=vwfa_sample), level, peak_amplitude, confirmed,
     wernicke_confirm_sample, wernicke_lag_ms, marker_start_sample,
     marker_end_sample, marker_duration_samples}.
    """
    confirmed = confirmed_candidates[confirmed_candidates["confirmed"]].copy()
    half_n = int(round(half_window_ms / 1000.0 * srate))

    starts = confirmed["sample"] - half_n
    ends = confirmed["sample"] + half_n
    if n_samples is not None:
        starts = starts.clip(lower=0)
        ends = ends.clip(upper=n_samples - 1)
    else:
        starts = starts.clip(lower=0)

    confirmed["marker_start_sample"] = starts.astype(int)
    confirmed["marker_end_sample"] = ends.astype(int)
    confirmed["marker_duration_samples"] = confirmed["marker_end_sample"] - confirmed["marker_start_sample"]
    return confirmed.reset_index(drop=True)


def detect_word_events(sig_vwfa, srate, vwfa_thresholds,
                        band=config.VWFA_BAND,
                        window_ms=config.SIGNAL_LEVEL_WINDOW_MS,
                        min_sustained_ms=config.MIN_SUSTAINED_MS,
                        refractory_ms=config.VWFA_REFRACTORY_MS,
                        half_window_ms=config.WORD_MARKER_HALF_WINDOW_MS):
    """Convenience wrapper chaining detect_vwfa_candidates ->
    build_marker_windows. Single entry point for both the local demo and
    the full-pipeline caller.

    No longer requires Wernicke confirmation (see module docstring for the
    empirical reason: it cost more recall than it gained in precision).
    Every VWFA candidate that survives refractory non-max-suppression is
    treated as confirmed.

    Returns (all_candidates, marker_windows): all_candidates has every VWFA
    candidate with its confirmed flag (always True) for diagnostics;
    marker_windows has each one's +/-half_window_ms marker samples.
    """
    candidates = detect_vwfa_candidates(
        sig_vwfa, srate, vwfa_thresholds, band, window_ms, min_sustained_ms, refractory_ms,
    )
    if candidates.empty:
        return candidates, candidates

    candidates = candidates.copy()
    candidates["confirmed"] = True
    markers = build_marker_windows(
        candidates, srate, half_window_ms, n_samples=len(sig_vwfa),
    )
    return candidates, markers
