"""Wernicke-area word-reading-event detection via ERP (event-related
potential) analysis -- a second, independent detector from the VWFA envelope
method in vwfa_word_detection/word_detection.py. Deliberately a different
signal-processing paradigm, not the same envelope-threshold technique
pointed at a different ROI (that "confirmation" approach was tried and
removed -- see EEG_research/README.md).

EXPLORATORY / EMPIRICAL, matching vwfa_word_detection/word_detection.py's
stance: this is a hypothesis being tested (that word reading produces a
locatable multi-component ERP complex -- P1/N170/N250/N400 -- in the
Wernicke ROI), not an established result.

Method:
  1. Band-pass the Wernicke-ROI-averaged signal, sign preserved (no Hilbert
     envelope -- ERP polarity is the whole signal here, unlike VWFA's
     amplitude-only approach). Find prominent local minima (troughs) --
     candidate N400-like deflections -- via scipy.signal.find_peaks on the
     inverted signal, spaced by a minimum distance so adjacent noise wiggles
     don't each become a separate candidate.
  2. Each trough at time t_peak is presumed to be the N400 component of a
     word read at word_onset = t_peak - N400_latency_midpoint (the midpoint
     of config.ERP_COMPONENTS["N400"], ~400ms) -- ERP components peak well
     after the triggering event, so detected events must be reported on
     that onset-time basis to be comparable to VWFA markers and to ground
     truth.
  3. Build a short epoch around the hypothesized onset (config.EPOCH_TMIN/
     EPOCH_TMAX, baseline-corrected against config.BASELINE by subtracting
     the pre-onset mean), then score every config.ERP_COMPONENTS window's
     peak amplitude + latency in that epoch (same per-window logic
     pipeline/features_erp.py uses for single-trial ERP features -- ported
     here directly since the `pipeline` package doesn't import locally, not
     imported). Sum the four components' absolute peak amplitudes into one
     candidate-strength score.
  4. Percentile-calibrate an accept cutoff on the pooled score distribution
     (config.LOCAL_CALIB_HIGH_PCT, same "never hardcode" philosophy as
     word_detection.calibrate_signal_thresholds) -- only the top decile of
     candidates by combined ERP strength survive.
  5. Refractory non-max-suppression among survivors, spaced by hypothesized
     onset time (config.VWFA_REFRACTORY_MS, reused for symmetry with VWFA).
  6. +/-config.WORD_MARKER_HALF_WINDOW_MS marker windows around each
     survivor's hypothesized onset -- reuses
     word_detection.build_marker_windows directly (already signal-agnostic).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

_VWFA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vwfa_word_detection")
if _VWFA_DIR not in sys.path:
    sys.path.insert(0, _VWFA_DIR)

import config  # noqa: E402  (vwfa_word_detection/config.py)
from word_detection import build_marker_windows  # noqa: E402  (reused unmodified)

N400_LATENCY_MID_MS = sum(config.ERP_COMPONENTS["N400"]) / 2.0  # 400ms
MIN_TROUGH_DISTANCE_MS = 200.0  # candidate troughs can't be closer than this
ERP_ACCEPT_PCT = config.LOCAL_CALIB_HIGH_PCT  # top decile of candidate strength, by default


def _bandpassed(sig, srate, band=config.BANDPASS):
    nyq = srate / 2.0
    lo, hi = band
    b, a = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
    return filtfilt(b, a, sig.astype(np.float64))


def _score_epoch(sig, srate, onset_sample):
    """Baseline-corrected epoch around onset_sample; per-component signed
    peak amplitude + latency (ms) for every config.ERP_COMPONENTS window.
    Returns None if the epoch would run off either end of the signal.
    """
    tmin_n = int(round(config.EPOCH_TMIN * srate))
    tmax_n = int(round(config.EPOCH_TMAX * srate))
    start, end = onset_sample + tmin_n, onset_sample + tmax_n
    if start < 0 or end > len(sig):
        return None

    epoch = sig[start:end].astype(np.float64)
    times_ms = (np.arange(len(epoch)) + tmin_n) / srate * 1000.0

    base_lo, base_hi = config.BASELINE
    base_mask = (times_ms >= base_lo * 1000.0) & (times_ms < base_hi * 1000.0)
    if base_mask.any():
        epoch = epoch - epoch[base_mask].mean()

    components = {}
    for name, (lo_ms, hi_ms) in config.ERP_COMPONENTS.items():
        mask = (times_ms >= lo_ms) & (times_ms < hi_ms)
        if not mask.any():
            components[name] = {"peak": 0.0, "latency_ms": np.nan}
            continue
        window = epoch[mask]
        window_times = times_ms[mask]
        peak_idx = int(np.argmax(np.abs(window)))
        components[name] = {"peak": float(window[peak_idx]), "latency_ms": float(window_times[peak_idx])}

    return components


def detect_erp_candidates(sig_wernicke, srate,
                           band=config.BANDPASS,
                           min_trough_distance_ms=MIN_TROUGH_DISTANCE_MS,
                           accept_pct=ERP_ACCEPT_PCT,
                           refractory_ms=config.VWFA_REFRACTORY_MS):
    """Find Wernicke-ERP word-reading candidates. Returns a DataFrame:
    {sample (=hypothesized onset), level, peak_amplitude (=combined ERP
    score), trough_sample, p1_peak, p1_latency_ms, n170_peak, ...} sorted by
    sample -- same column shape as word_detection.detect_vwfa_candidates's
    output (`sample`, `level`, `peak_amplitude`) so build_marker_windows and
    downstream matching code work unmodified, plus extra ERP diagnostic
    columns.
    """
    filtered = _bandpassed(sig_wernicke, srate, band)
    min_distance_n = max(1, int(round(min_trough_distance_ms / 1000.0 * srate)))

    trough_idx, _ = find_peaks(-filtered, distance=min_distance_n)
    if trough_idx.size == 0:
        return pd.DataFrame(columns=["sample", "level", "peak_amplitude", "trough_sample"])

    rows = []
    onset_offset_n = int(round(N400_LATENCY_MID_MS / 1000.0 * srate))
    for t_peak in trough_idx:
        onset_sample = int(t_peak) - onset_offset_n
        components = _score_epoch(filtered, srate, onset_sample)
        if components is None:
            continue
        combined_score = sum(abs(c["peak"]) for c in components.values())
        row = {"sample": onset_sample, "level": 1, "peak_amplitude": combined_score,
               "trough_sample": int(t_peak)}
        for name, c in components.items():
            row[f"{name.lower()}_peak"] = c["peak"]
            row[f"{name.lower()}_latency_ms"] = c["latency_ms"]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["sample", "level", "peak_amplitude", "trough_sample"])

    all_candidates = pd.DataFrame(rows)

    accept_cutoff = float(np.percentile(all_candidates["peak_amplitude"], accept_pct))
    accepted = all_candidates[all_candidates["peak_amplitude"] >= accept_cutoff].copy()
    if accepted.empty:
        return accepted

    accepted = accepted.sort_values("peak_amplitude", ascending=False)
    refractory_n = int(round(refractory_ms / 1000.0 * srate))
    kept, kept_samples = [], []
    for _, r in accepted.iterrows():
        if all(abs(r["sample"] - s) >= refractory_n for s in kept_samples):
            kept.append(r)
            kept_samples.append(r["sample"])

    out = pd.DataFrame(kept).sort_values("sample").reset_index(drop=True)
    return out


def detect_word_events_wernicke_erp(sig_wernicke, srate,
                                     band=config.BANDPASS,
                                     min_trough_distance_ms=MIN_TROUGH_DISTANCE_MS,
                                     accept_pct=ERP_ACCEPT_PCT,
                                     refractory_ms=config.VWFA_REFRACTORY_MS,
                                     half_window_ms=config.WORD_MARKER_HALF_WINDOW_MS):
    """Entry point mirroring word_detection.detect_word_events's shape:
    (all_candidates, marker_windows) = detect_word_events_wernicke_erp(sig, srate).
    """
    candidates = detect_erp_candidates(sig_wernicke, srate, band, min_trough_distance_ms,
                                        accept_pct, refractory_ms)
    if candidates.empty:
        return candidates, candidates

    candidates = candidates.copy()
    candidates["confirmed"] = True
    markers = build_marker_windows(candidates, srate, half_window_ms, n_samples=len(sig_wernicke))
    return candidates, markers
