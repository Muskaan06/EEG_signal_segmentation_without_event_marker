"""ERP feature extraction (P1/N170/N250/N400) per ROI, and grand-average
ERP plots comparing READING vs NON-READING.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import config


def _roi_waveform(epochs_data, ch_names, roi_channels):
    """epochs_data: (n_epochs, n_channels, n_times). Returns (n_epochs, n_times)
    averaged over the ROI's channels (channels missing from ch_names are skipped).
    """
    idx = [ch_names.index(c) for c in roi_channels if c in ch_names]
    if not idx:
        return None
    return epochs_data[:, idx, :].mean(axis=1)


def compute_erp_features(epochs, roi_channels_dict):
    """Returns a DataFrame (one row per epoch, aligned with epochs.metadata index)
    of ERP features: {roi}_{component}_{peak,latency,mean,auc} in microvolts / ms.
    """
    data = epochs.get_data(copy=False) * 1e6  # -> microvolts, (n_ep, n_ch, n_t)
    times_ms = epochs.times * 1000.0
    ch_names = epochs.ch_names

    out = {}
    for roi, chans in roi_channels_dict.items():
        wave = _roi_waveform(data, ch_names, chans)
        if wave is None:
            continue
        for comp, (lo, hi) in config.ERP_COMPONENTS.items():
            mask = (times_ms >= lo) & (times_ms <= hi)
            if not mask.any():
                continue
            seg = wave[:, mask]           # (n_ep, n_t_in_window)
            seg_t = times_ms[mask]

            mean_amp = seg.mean(axis=1)
            peak_idx = np.argmax(np.abs(seg), axis=1)
            peak_amp = seg[np.arange(seg.shape[0]), peak_idx]
            peak_lat = seg_t[peak_idx]
            auc = np.trapezoid(seg, seg_t, axis=1)

            prefix = f"{roi}_{comp}"
            out[f"{prefix}_mean"] = mean_amp
            out[f"{prefix}_peak"] = peak_amp
            out[f"{prefix}_latency"] = peak_lat
            out[f"{prefix}_auc"] = auc

    return pd.DataFrame(out, index=epochs.metadata.index if epochs.metadata is not None else None)


def plot_grand_average_erp(epochs, roi_channels_dict, out_path, rois_to_plot=None):
    """Grand-average ROI-averaged ERP waveforms, READING vs NON-READING overlay,
    with shaded ERP component windows.
    """
    rois_to_plot = rois_to_plot or list(roi_channels_dict.keys())
    data = epochs.get_data(copy=False) * 1e6
    times_ms = epochs.times * 1000.0
    ch_names = epochs.ch_names
    labels = epochs.metadata["label"].to_numpy()

    n = len(rois_to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    comp_colors = plt.cm.Pastel1.colors

    for i, roi in enumerate(rois_to_plot):
        ax = axes[i // ncols][i % ncols]
        wave = _roi_waveform(data, ch_names, roi_channels_dict[roi])
        if wave is None:
            ax.set_visible(False)
            continue
        for label, color in (("reading", "#c0392b"), ("non_reading", "#2980b9")):
            m = labels == label
            if m.sum() == 0:
                continue
            mean_wave = wave[m].mean(axis=0)
            sem = wave[m].std(axis=0) / np.sqrt(m.sum())
            ax.plot(times_ms, mean_wave, label=f"{label} (n={m.sum()})", color=color, lw=1.3)
            ax.fill_between(times_ms, mean_wave - sem, mean_wave + sem, color=color, alpha=0.2)

        for j, (comp, (lo, hi)) in enumerate(config.ERP_COMPONENTS.items()):
            ax.axvspan(lo, hi, color=comp_colors[j % len(comp_colors)], alpha=0.35, lw=0)
            ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, comp,
                    fontsize=6, ha="center", va="bottom")

        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(roi, fontsize=10)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Amplitude (uV)", fontsize=8)
        ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    handles, hlabels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, hlabels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Grand-average ERP by ROI: READING vs NON-READING", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[features_erp] saved {out_path}")
