"""Functional connectivity (coherence, PLV, wPLI) between ROI-averaged
signals, per epoch (feature extraction) and grand-averaged (visualization).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.signal as ssig

from pipeline import config
from pipeline.features_erp import _roi_waveform

CONNECTIVITY_MEASURES = ("coherence", "plv", "wpli")


def _bandpass_batch(wave, srate, band, order=4):
    """wave: (n_epochs, n_times). Zero-phase bandpass via filtfilt, vectorized
    across epochs (axis=-1).
    """
    lo, hi = band
    nyq = srate / 2.0
    b, a = ssig.butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype="bandpass")
    return ssig.filtfilt(b, a, wave, axis=-1)


def _coherence_scalar_batch(x, y, srate, band, nperseg=None):
    nperseg = nperseg or min(x.shape[-1], 256)
    freqs, cxy = ssig.coherence(x, y, fs=srate, nperseg=nperseg, axis=-1)
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(x.shape[0])
    return cxy[:, mask].mean(axis=-1)


def _plv_scalar_batch(x_band, y_band):
    """x_band, y_band: (n_epochs, n_times), already band-limited. PLV over time per epoch."""
    phase_x = np.angle(ssig.hilbert(x_band, axis=-1))
    phase_y = np.angle(ssig.hilbert(y_band, axis=-1))
    return np.abs(np.mean(np.exp(1j * (phase_x - phase_y)), axis=-1))


def _wpli_scalar_batch(x_band, y_band):
    """x_band, y_band: (n_epochs, n_times), already band-limited."""
    ax = ssig.hilbert(x_band, axis=-1)
    ay = ssig.hilbert(y_band, axis=-1)
    cross = ax * np.conj(ay)
    imag = np.imag(cross)
    num = np.abs(np.mean(imag, axis=-1))
    den = np.mean(np.abs(imag), axis=-1)
    den_safe = np.where(den == 0, 1e-20, den)
    return num / den_safe


def compute_connectivity_features(epochs, roi_channels_dict, pairs=None):
    """Returns a DataFrame (one row per epoch): {roi1}_{roi2}_{band}_{measure}."""
    pairs = pairs or config.CONNECTIVITY_PAIRS
    data = epochs.get_data(copy=False)
    times = epochs.times
    post_mask = times >= 0
    srate = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    roi_waves = {}
    for roi, chans in roi_channels_dict.items():
        w = _roi_waveform(data, ch_names, chans)
        if w is not None:
            roi_waves[roi] = w[:, post_mask]

    out = {}
    for roi_a, roi_b in pairs:
        if roi_a not in roi_waves or roi_b not in roi_waves:
            continue
        xa, xb = roi_waves[roi_a], roi_waves[roi_b]
        pair_prefix = f"{roi_a}_{roi_b}"
        for band_name, band in config.BANDS.items():
            xa_b = _bandpass_batch(xa, srate, band)
            xb_b = _bandpass_batch(xb, srate, band)
            out[f"{pair_prefix}_{band_name}_coherence"] = _coherence_scalar_batch(xa, xb, srate, band)
            out[f"{pair_prefix}_{band_name}_plv"] = _plv_scalar_batch(xa_b, xb_b)
            out[f"{pair_prefix}_{band_name}_wpli"] = _wpli_scalar_batch(xa_b, xb_b)

    return pd.DataFrame(out, index=epochs.metadata.index if epochs.metadata is not None else None)


def _full_roi_matrix(epochs, roi_channels_dict, measure="plv", band=(8.0, 12.0), label=None):
    data = epochs.get_data(copy=False)
    times = epochs.times
    post_mask = times >= 0
    srate = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    if label is not None:
        sel = epochs.metadata["label"].to_numpy() == label
        data = data[sel]

    rois = list(roi_channels_dict.keys())
    waves = {}
    for roi in rois:
        w = _roi_waveform(data, ch_names, roi_channels_dict[roi])
        waves[roi] = w[:, post_mask] if w is not None else None

    n = len(rois)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            wa, wb = waves[rois[i]], waves[rois[j]]
            if wa is None or wb is None:
                continue
            if measure == "coherence":
                val = _coherence_scalar_batch(wa, wb, srate, band).mean()
            else:
                wa_b, wb_b = _bandpass_batch(wa, srate, band), _bandpass_batch(wb, srate, band)
                val = (_plv_scalar_batch(wa_b, wb_b) if measure == "plv"
                       else _wpli_scalar_batch(wa_b, wb_b)).mean()
            mat[i, j] = val
    return rois, mat


def plot_connectivity_matrices_and_graphs(epochs, roi_channels_dict, out_path,
                                           measure="plv", band=(8.0, 12.0)):
    """Grand-average ROI x ROI connectivity heatmaps + graphs for READING vs
    NON-READING (and their difference).
    """
    rois, mat_read = _full_roi_matrix(epochs, roi_channels_dict, measure, band, label="reading")
    _, mat_nread = _full_roi_matrix(epochs, roi_channels_dict, measure, band, label="non_reading")
    mat_diff = mat_read - mat_nread

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (title, mat, cmap, vlim) in enumerate([
        ("READING", mat_read, "viridis", (0, 1)),
        ("NON-READING", mat_nread, "viridis", (0, 1)),
        ("Difference", mat_diff, "RdBu_r", (-np.abs(mat_diff).max(), np.abs(mat_diff).max())),
    ]):
        ax = axes[0, col]
        im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(len(rois))); ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rois))); ax.set_yticklabels(rois, fontsize=7)
        ax.set_title(f"{title}\n{measure} ({band[0]}-{band[1]}Hz)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax_g = axes[1, col]
        G = nx.Graph()
        G.add_nodes_from(rois)
        thresh = np.percentile(np.abs(mat[np.triu_indices(len(rois), k=1)]), 60)
        for i in range(len(rois)):
            for j in range(i + 1, len(rois)):
                if abs(mat[i, j]) >= thresh:
                    G.add_edge(rois[i], rois[j], weight=mat[i, j])
        pos = nx.circular_layout(G)
        weights = [abs(G[u][v]["weight"]) * 4 for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax_g, node_color="#5dade2", node_size=600)
        nx.draw_networkx_labels(G, pos, ax=ax_g, font_size=7)
        nx.draw_networkx_edges(G, pos, ax=ax_g, width=weights, alpha=0.6, edge_color="#555")
        ax_g.set_title(f"{title} graph (top 40% edges)", fontsize=9)
        ax_g.axis("off")

    fig.suptitle("ROI connectivity: READING vs NON-READING", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[features_connectivity] saved {out_path}")
