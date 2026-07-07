"""Spectral power features (absolute/relative band power, ERD/ERS) per ROI,
and full-scalp topomap figures comparing READING vs NON-READING.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.signal as ssig

from pipeline import config
from pipeline.features_erp import _roi_waveform

mne.set_log_level("WARNING")


def _band_power_batch(wave, srate, band, nperseg=None):
    """wave: (n_epochs, n_times). Returns (n_epochs,) power in `band` via Welch,
    vectorized across epochs.
    """
    nperseg = nperseg or min(wave.shape[-1], 256)
    freqs, psd = ssig.welch(wave, fs=srate, nperseg=nperseg, axis=-1)
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(wave.shape[0])
    return np.trapezoid(psd[:, mask], freqs[mask], axis=-1)


def _total_power_batch(wave, srate, nperseg=None, fmax=45.0):
    nperseg = nperseg or min(wave.shape[-1], 256)
    freqs, psd = ssig.welch(wave, fs=srate, nperseg=nperseg, axis=-1)
    mask = freqs <= fmax
    return np.trapezoid(psd[:, mask], freqs[mask], axis=-1)


def compute_spectral_features(epochs, roi_channels_dict):
    """Returns a DataFrame (one row per epoch) with, per ROI and band:
    absolute power, relative power (post-stimulus window), and ERD/ERS
    percent change relative to the pre-stimulus baseline window.
    """
    data = epochs.get_data(copy=False)  # volts, (n_ep, n_ch, n_t)
    times = epochs.times
    srate = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    post_mask = times >= 0
    base_mask = times < 0

    out = {}
    for roi, chans in roi_channels_dict.items():
        wave = _roi_waveform(data, ch_names, chans)
        if wave is None:
            continue
        post = wave[:, post_mask]
        base = wave[:, base_mask]
        total_post = _total_power_batch(post, srate)
        total_post_safe = np.where(total_post == 0, 1e-20, total_post)

        for band_name, band in config.BANDS.items():
            abs_power = _band_power_batch(post, srate, band)
            rel_power = abs_power / total_post_safe
            base_power = _band_power_batch(base, srate, band)
            base_safe = np.where(base_power == 0, 1e-20, base_power)
            erd_ers = (abs_power - base_power) / base_safe * 100.0

            prefix = f"{roi}_{band_name}"
            out[f"{prefix}_abs_power"] = abs_power
            out[f"{prefix}_rel_power"] = rel_power
            out[f"{prefix}_erd_ers_pct"] = erd_ers

    return pd.DataFrame(out, index=epochs.metadata.index if epochs.metadata is not None else None)


def plot_band_topomaps(epochs, out_path):
    """Per-band scalp topomaps of mean absolute post-stimulus power:
    READING, NON-READING, and their difference.
    """
    info = epochs.info
    data = epochs.get_data(copy=False)
    times = epochs.times
    post_mask = times >= 0
    labels = epochs.metadata["label"].to_numpy()
    srate = info["sfreq"]

    bands = list(config.BANDS.items())
    fig, axes = plt.subplots(len(bands), 3, figsize=(10, 3.2 * len(bands)))

    for i, (band_name, band) in enumerate(bands):
        post = data[:, :, post_mask]  # (n_ep, n_ch, n_t)
        # vectorize band power per channel: reshape (n_ep*n_ch, n_t)
        n_ep, n_ch, n_t = post.shape
        flat = post.reshape(n_ep * n_ch, n_t)
        power_flat = _band_power_batch(flat, srate, band)
        power = power_flat.reshape(n_ep, n_ch)

        read_p = power[labels == "reading"].mean(axis=0)
        nread_p = power[labels == "non_reading"].mean(axis=0)
        diff_p = read_p - nread_p

        # Robust color scale: a single noisy/incompletely-cleaned channel can
        # have power orders of magnitude above the rest of the scalp and would
        # otherwise wash out every other channel's color.
        vmax = float(np.percentile(np.concatenate([read_p, nread_p]), 95))
        for j, (title, vals, cmap, vlim) in enumerate([
            ("READING", read_p, "Reds", (0, vmax)),
            ("NON-READING", nread_p, "Blues", (0, vmax)),
            ("Difference", diff_p, "RdBu_r", (-np.abs(diff_p).max(), np.abs(diff_p).max())),
        ]):
            ax = axes[i, j]
            mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap,
                                  vlim=vlim, contours=4, sphere="auto")
            if i == 0:
                ax.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(band_name, fontsize=9)

    fig.suptitle("Band power topomaps: READING vs NON-READING (post-stimulus)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[features_spectral] saved {out_path}")
