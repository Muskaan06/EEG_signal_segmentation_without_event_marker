#!/usr/bin/env python3
"""
EEG Visualization Suite — ZuCo ZDN Subject
12 visualization types × 5 examples, one plot per example with per-word subplots.
Time axes are CONTINUOUS (words placed sequentially, never reset to zero).
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm
import scipy.signal as ssig
import h5py
import networkx as nx
from sklearn.manifold import TSNE
import pywt
import umap as umap_lib
import mne
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
SRATE       = 500.0
N_CH        = 105
GAP_SEC     = 0.15        # gap inserted for no-fixation words
NUM_EXAMPLES = 5
REF_CH      = 52          # roughly central channel for single-channel plots

ZDN_DIR   = os.path.dirname(os.path.abspath(__file__))
EEG_FILE  = os.path.join(ZDN_DIR, 'gip_ZDN_NR1_EEG.mat')
WORD_DIR  = os.path.join(ZDN_DIR, 'word_eeg_ZDN')
VIZ_DIR   = os.path.join(WORD_DIR, 'visualizations')

BANDS = {
    'Delta (0.5–4 Hz)' : (0.5,  4),
    'Theta (4–8 Hz)'   : (4,    8),
    'Alpha (8–13 Hz)'  : (8,   13),
    'Beta (13–30 Hz)'  : (13,  30),
    'Gamma (30–45 Hz)' : (30,  45),
}
BAND_COLS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
WORD_CMAP = plt.cm.tab20

# ── Build MNE Info (once) ──────────────────────────────────────────────────────
def build_mne_info():
    """Use the standard GSN-HydroCel-129 montage — channel names match."""
    with h5py.File(EEG_FILE, 'r') as f:
        cl = f['EEG']['chanlocs']
        labs = [''.join(chr(int(c)) for c in f[r][()].flatten())
                for r in cl['labels'][:, 0]]
    info = mne.create_info(ch_names=labs, sfreq=SRATE, ch_types='eeg')
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    info.set_montage(montage, on_missing='ignore', verbose=False)
    return info

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_example(ex_idx):
    """Load word EEG files → list of (word_text, eeg[N_CH × T]) in order."""
    ex_dir = os.path.join(WORD_DIR, f'example_{ex_idx}')
    word_dirs = sorted(d for d in os.listdir(ex_dir) if d.startswith('word_'))
    items = []
    for wd in word_dirs:
        wt = wd
        with open(os.path.join(ex_dir, wd, 'metadata.txt')) as mf:
            for line in mf:
                if line.startswith('Word text:'):
                    wt = line.split(':', 1)[1].strip()
                    break
        eeg = np.load(os.path.join(ex_dir, wd, 'eeg.npy'))
        items.append((wt, eeg))
    return items


def build_continuous(items):
    """
    Concatenate word EEGs into a single continuous signal.
    Words with no fixation get a GAP_SEC NaN-filled placeholder.

    Returns
    -------
    eeg_cont : (N_CH, total_samples)  – NaN where no fixation
    t_arr    : (total_samples,)       – seconds, monotonically increasing
    bounds   : list of (word_text, t_start, t_end, has_data)
    """
    segs, ts, bounds = [], [], []
    t = 0.0
    n_gap = int(GAP_SEC * SRATE)
    for wt, eeg in items:
        t0 = t
        if eeg.shape[1] > 0:
            n = eeg.shape[1]
            segs.append(eeg)
            ts.append(np.arange(n) / SRATE + t)
            t += n / SRATE
            hd = True
        else:
            segs.append(np.full((N_CH, n_gap), np.nan))
            ts.append(np.arange(n_gap) / SRATE + t)
            t += GAP_SEC
            hd = False
        bounds.append((wt, t0, t, hd))
    return np.concatenate(segs, axis=1), np.concatenate(ts), bounds


def get_sentence(ex_idx):
    path = os.path.join(WORD_DIR, f'example_{ex_idx}', 'sentence.txt')
    with open(path) as f:
        line = f.readline()
    return line.split(':', 1)[1].strip() if ':' in line else line.strip()


def grid_layout(n, max_cols=5):
    n_cols = min(n, max_cols)
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def blank_ax(ax, wt, reason='No fixation'):
    ax.set_facecolor('#f5f5f5')
    ax.text(0.5, 0.5, f'"{wt}"\n{reason}', ha='center', va='center',
            fontsize=8, style='italic', color='#888888', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linestyle('--')
        sp.set_color('#cccccc')


def annotate_words(ax, bounds, y_label, fontsize=6.5, rotate=50):
    """Overlay word boundary lines and labels on a time-series axis."""
    for i, (wt, t0, t1, hd) in enumerate(bounds):
        c = WORD_CMAP.colors[i % len(WORD_CMAP.colors)]
        ax.axvline(t0, color=c, lw=0.6, ls='--', alpha=0.65)
        mid = (t0 + t1) / 2
        ax.text(mid, y_label, wt, ha='center', va='bottom',
                fontsize=fontsize, color=c, rotation=rotate, clip_on=True)
        if not hd:
            ax.axvspan(t0, t1, alpha=0.12, color='gray')


def try_topomap(data, info, ax, title='', cmap='RdBu_r', contours=4):
    try:
        mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            cmap=cmap, contours=contours, sphere='auto',
            extrapolate='head', outlines='head',
        )
        ax.set_title(title, fontsize=7)
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:60], ha='center', va='center',
                fontsize=6, transform=ax.transAxes, wrap=True)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1 — Raw EEG Traces
# One row per selected channel; continuous time; word labels on top
# ══════════════════════════════════════════════════════════════════════════════
def viz_raw_eeg(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_show = 6
    ch_idx = np.linspace(0, N_CH - 1, n_show, dtype=int)
    ch_names = [info.ch_names[i] for i in ch_idx]

    fig, axes = plt.subplots(n_show, 1, figsize=(24, 3 * n_show), sharex=True)
    fig.suptitle(
        f'Raw EEG Traces — Example {ex_idx}\n"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold', y=1.01,
    )

    p_lo = np.nanpercentile(eeg_cont[ch_idx], 1)
    p_hi = np.nanpercentile(eeg_cont[ch_idx], 99)
    pad  = (p_hi - p_lo) * 0.18

    for row, (ci, cn) in enumerate(zip(ch_idx, ch_names)):
        ax = axes[row]
        ax.plot(t_arr, eeg_cont[ci], color='#2166ac', lw=0.5, alpha=0.85)
        ax.set_ylabel(cn, fontsize=8, labelpad=2)
        ax.set_ylim(p_lo - pad, p_hi + pad * 3.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(axis='x', lw=0.25, alpha=0.35)
        annotate_words(ax, bounds, p_hi + pad * 0.5, fontsize=6, rotate=55)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2 — Power Spectral Density (per word)
# Grid of PSD plots; x = freq 0–45 Hz; frequency-band shading
# ══════════════════════════════════════════════════════════════════════════════
def viz_psd(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_words = len(items)
    n_rows, n_cols = grid_layout(n_words, max_cols=5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f'Power Spectral Density — Example {ex_idx}\n"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )

    for i, (wt, eeg) in enumerate(items):
        ax = axes[i]
        t0, t1 = bounds[i][1], bounds[i][2]
        if eeg.shape[1] < 10:
            blank_ax(ax, wt)
            continue
        nperseg = min(256, eeg.shape[1])
        freqs, psd = ssig.welch(eeg, fs=SRATE, nperseg=nperseg, axis=1)
        psd_mean = psd.mean(axis=0)

        ax.semilogy(freqs, psd_mean, color='#222222', lw=1.0, zorder=4)
        for (bname, (blo, bhi)), bc in zip(BANDS.items(), BAND_COLS):
            mask = (freqs >= blo) & (freqs < bhi)
            ax.fill_between(freqs, psd_mean, where=mask,
                            alpha=0.55, color=bc, zorder=3)

        ax.set_xlim(0.5, 45)
        ax.set_xlabel('Freq (Hz)', fontsize=6.5)
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=6.5)
        ax.set_title(f'"{wt}"  [{t0:.2f}–{t1:.2f} s]', fontsize=7)
        ax.tick_params(labelsize=5.5)
        ax.grid(lw=0.25, alpha=0.35)

    for j in range(n_words, len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(color=c, alpha=0.7, label=b)
               for (b, _), c in zip(BANDS.items(), BAND_COLS)]
    fig.legend(handles=handles, loc='lower right', fontsize=7, ncol=5,
               bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3 — Spectrograms (continuous time)
# STFT on one reference channel; words marked with vertical lines
# ══════════════════════════════════════════════════════════════════════════════
def viz_spectrograms(ex_idx, items, eeg_cont, t_arr, bounds, info):
    sig = eeg_cont[REF_CH]
    sig_filled = np.where(np.isnan(sig), 0.0, sig)

    nperseg  = 128
    noverlap = 96
    f, t_spec, Sxx = ssig.spectrogram(
        sig_filled, fs=SRATE, nperseg=nperseg, noverlap=noverlap,
    )
    t_spec = t_spec + t_arr[0]   # shift to match continuous time axis
    f_mask = f <= 50

    fig, ax = plt.subplots(figsize=(24, 5.5))
    fig.suptitle(
        f'Spectrogram (ch {info.ch_names[REF_CH]}) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    im = ax.pcolormesh(t_spec, f[f_mask],
                       10 * np.log10(Sxx[f_mask] + 1e-15),
                       cmap='inferno', shading='auto', vmin=-10)
    annotate_words(ax, bounds, f[f_mask][-1] * 0.92, fontsize=7, rotate=0)

    for _, (_, bhi) in BANDS.items():
        if bhi <= 50:
            ax.axhline(bhi, color='white', lw=0.5, ls='--', alpha=0.55)

    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.85)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 4 — Wavelet Scalograms (continuous time)
# Complex Morlet CWT on reference channel; log-frequency y-axis
# ══════════════════════════════════════════════════════════════════════════════
def viz_wavelet(ex_idx, items, eeg_cont, t_arr, bounds, info):
    sig = eeg_cont[REF_CH]
    sig_filled = np.where(np.isnan(sig), 0.0, sig)

    freqs_target = np.logspace(np.log10(0.5), np.log10(50), 64)

    # Subsample for speed (max 2 000 points)
    step = max(1, len(sig_filled) // 2000)
    sig_sub = sig_filled[::step]
    t_sub   = t_arr[::step]
    samp_period = step / SRATE

    wavelet = 'cmor1.5-1.0'
    scales  = pywt.frequency2scale(wavelet, freqs_target * samp_period)
    coef, _ = pywt.cwt(sig_sub, scales, wavelet, sampling_period=samp_period)
    power   = np.abs(coef) ** 2

    vmin = np.percentile(power[power > 0], 2) if (power > 0).any() else 1e-6
    vmax = np.percentile(power, 99)

    fig, ax = plt.subplots(figsize=(24, 5.5))
    fig.suptitle(
        f'Wavelet Scalogram (ch {info.ch_names[REF_CH]}) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    im = ax.pcolormesh(t_sub, freqs_target, power,
                       cmap='plasma', shading='auto',
                       norm=LogNorm(vmin=max(vmin, 1e-9), vmax=max(vmax, 1e-8)))
    ax.set_yscale('log')
    ax.set_ylim(0.5, 50)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    annotate_words(ax, bounds, freqs_target[-1] * 0.85, fontsize=7, rotate=0)
    plt.colorbar(im, ax=ax, label='Power (µV²)', shrink=0.85)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 5 — Topographic Maps (per word)
# Mean absolute amplitude per channel; uses GSN-HydroCel-129 positions
# ══════════════════════════════════════════════════════════════════════════════
def viz_topomaps(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_words = len(items)
    n_rows, n_cols = grid_layout(n_words, max_cols=6)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f'Topographic Maps (Mean |Amplitude|) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    for i, (wt, eeg) in enumerate(items):
        ax = axes[i]
        t0, t1 = bounds[i][1], bounds[i][2]
        if eeg.shape[1] < 2:
            blank_ax(ax, wt)
            continue
        data = np.abs(eeg).mean(axis=1)
        try_topomap(data, info, ax,
                    title=f'"{wt}"  [{t0:.2f}–{t1:.2f} s]',
                    cmap='hot', contours=4)

    for j in range(n_words, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 6 — Connectivity Heatmaps (per word)
# 105×105 Pearson correlation matrix across channels
# ══════════════════════════════════════════════════════════════════════════════
def viz_connectivity(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_words = len(items)
    n_rows, n_cols = grid_layout(n_words, max_cols=5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f'Connectivity Heatmaps (Pearson r) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    for i, (wt, eeg) in enumerate(items):
        ax = axes[i]
        t0, t1 = bounds[i][1], bounds[i][2]
        if eeg.shape[1] < 5:
            blank_ax(ax, wt)
            continue
        corr = np.corrcoef(eeg)
        ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                  aspect='auto', interpolation='nearest')
        ax.set_title(f'"{wt}"  [{t0:.2f}–{t1:.2f} s]', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_words, len(axes)):
        axes[j].set_visible(False)

    # shared colorbar
    cbar_ax = fig.add_axes([0.925, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=Normalize(-1, 1))
    fig.colorbar(sm, cax=cbar_ax, label='r')
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 7 — Brain Network Graphs (per word)
# Nodes = channels (colored by amplitude); edges = correlation > 0.70
# Node positions use the GSN-129 2D projected x/y from MNE
# ══════════════════════════════════════════════════════════════════════════════
def viz_network(ex_idx, items, eeg_cont, t_arr, bounds, info):
    # Extract 2D electrode positions from MNE info (head-projected)
    pos_3d = np.array([ch['loc'][:3] for ch in info['chs']])
    # Project onto 2D plane (x→right, y→anterior)
    pos_2d = {i: (pos_3d[i, 1], pos_3d[i, 2]) for i in range(N_CH)}

    n_words = len(items)
    n_rows, n_cols = grid_layout(n_words, max_cols=5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f'Brain Network Graphs (r > 0.70) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    THRESH = 0.70

    for wi, (wt, eeg) in enumerate(items):
        ax = axes[wi]
        t0, t1 = bounds[wi][1], bounds[wi][2]
        if eeg.shape[1] < 5:
            blank_ax(ax, wt)
            continue

        corr = np.corrcoef(eeg)
        node_act = np.abs(eeg).mean(axis=1)
        act_norm = (node_act - node_act.min()) / (node_act.max() - node_act.min() + 1e-9)

        G = nx.Graph()
        G.add_nodes_from(range(N_CH))
        edges = [(a, b) for a in range(N_CH)
                 for b in range(a + 1, N_CH) if corr[a, b] > THRESH]
        G.add_edges_from(edges)

        node_colors = plt.cm.RdYlBu_r(act_norm)
        node_sizes  = 5 + 30 * act_norm

        nx.draw_networkx_nodes(G, pos_2d, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        if 0 < len(edges) < 600:
            nx.draw_networkx_edges(G, pos_2d, ax=ax, width=0.25,
                                   alpha=0.18, edge_color='#555555')

        ax.set_title(
            f'"{wt}"  [{t0:.2f}–{t1:.2f} s]\n{len(edges)} edges',
            fontsize=7,
        )
        ax.axis('off')

    for j in range(n_words, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 8 — ICA Component Maps
# Fit ICA on full example; top-row = component topomaps;
# heatmap = component activation per word
# ══════════════════════════════════════════════════════════════════════════════
def viz_ica(ex_idx, items, eeg_cont, t_arr, bounds, info):
    data_words = [(wt, eeg, bounds[i])
                  for i, (wt, eeg) in enumerate(items) if eeg.shape[1] > 0]

    if len(data_words) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Not enough fixation data for ICA',
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        return fig

    eeg_all = np.concatenate([e for _, e, _ in data_words], axis=1)
    N_COMP  = min(10, eeg_all.shape[1] // 20, N_CH)

    try:
        raw_full = mne.io.RawArray(eeg_all * 1e-6, info, verbose=False)
        ica = mne.preprocessing.ICA(n_components=N_COMP, method='fastica',
                                     random_state=42, max_iter=500)
        ica.fit(raw_full, verbose=False)
        mixing = ica.get_components()     # (n_ch, n_comp)

        n_data   = len(data_words)
        n_comp_s = min(N_COMP, 8)

        fig = plt.figure(figsize=(max(18, 2.6 * n_comp_s), 5 + 3))
        fig.suptitle(
            f'ICA Component Maps — Example {ex_idx}\n"{get_sentence(ex_idx)}"',
            fontsize=11, fontweight='bold',
        )
        gs = gridspec.GridSpec(2, n_comp_s, figure=fig,
                               height_ratios=[2, 1.5], hspace=0.55, wspace=0.35)

        # Row 0: component topomaps
        for ci in range(n_comp_s):
            ax = fig.add_subplot(gs[0, ci])
            try_topomap(mixing[:, ci], info, ax,
                        title=f'IC {ci}', cmap='RdBu_r', contours=3)

        # Row 1: per-word activation heatmap
        act_matrix = np.zeros((n_comp_s, n_data))
        for wi, (_, eeg, _) in enumerate(data_words):
            raw_w = mne.io.RawArray(eeg * 1e-6, info, verbose=False)
            src   = ica.get_sources(raw_w).get_data()   # (n_comp, T)
            act_matrix[:, wi] = np.abs(src[:n_comp_s]).mean(axis=1)

        ax_h = fig.add_subplot(gs[1, :])
        im   = ax_h.imshow(act_matrix, aspect='auto', cmap='hot',
                           interpolation='nearest')
        ax_h.set_xticks(range(n_data))
        ax_h.set_xticklabels(
            [f'{w}\n[{b[1]:.2f}–{b[2]:.2f}s]' for w, _, b in data_words],
            rotation=40, ha='right', fontsize=6.5,
        )
        ax_h.set_yticks(range(n_comp_s))
        ax_h.set_yticklabels([f'IC {k}' for k in range(n_comp_s)], fontsize=7)
        ax_h.set_xlabel('Word (fixated)', fontsize=9)
        ax_h.set_ylabel('ICA Component', fontsize=9)
        ax_h.set_title('Mean |source activation| per word', fontsize=9)
        plt.colorbar(im, ax=ax_h, label='|activation|', shrink=0.9, pad=0.01)

    except Exception as exc:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'ICA error:\n{exc}', ha='center', va='center',
                fontsize=9, transform=ax.transAxes)
        ax.axis('off')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 9 — Source Localization Maps
# Current Source Density (surface Laplacian) topomap per word
# ══════════════════════════════════════════════════════════════════════════════
def viz_source(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_words = len(items)
    n_rows, n_cols = grid_layout(n_words, max_cols=6)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f'Source Localization (Current Source Density) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    for i, (wt, eeg) in enumerate(items):
        ax = axes[i]
        t0, t1 = bounds[i][1], bounds[i][2]
        if eeg.shape[1] < 5:
            blank_ax(ax, wt)
            continue
        try:
            epoch = mne.EpochsArray(
                eeg[np.newaxis] * 1e-6, info,
                tmin=t0, verbose=False,
            )
            csd_ep  = mne.preprocessing.compute_current_source_density(
                epoch, verbose=False,
            )
            csd_val = csd_ep.get_data()[0].mean(axis=1)
            try_topomap(csd_val, info, ax,
                        title=f'"{wt}"  [{t0:.2f}–{t1:.2f} s]',
                        cmap='RdBu_r', contours=4)
        except Exception:
            data = np.abs(eeg).mean(axis=1)
            try_topomap(data, info, ax,
                        title=f'"{wt}"  [{t0:.2f}–{t1:.2f} s]',
                        cmap='hot', contours=4)

    for j in range(n_words, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 10 — UMAP / t-SNE Embeddings
# 100ms sliding windows → 2D scatter colored by word index
# ══════════════════════════════════════════════════════════════════════════════
def viz_umap_tsne(ex_idx, items, eeg_cont, t_arr, bounds, info):
    win = int(0.1 * SRATE)
    features, word_ids, word_labels = [], [], []

    for i, (wt, eeg) in enumerate(items):
        if eeg.shape[1] < win:
            continue
        n_wins = eeg.shape[1] // win
        for w in range(n_wins):
            seg = eeg[:, w * win : (w + 1) * win]
            features.append(
                np.concatenate([seg.mean(axis=1), seg.std(axis=1)])
            )
            word_ids.append(i)
            word_labels.append(wt)

    if len(features) < 10:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Not enough windows for embedding',
                ha='center', va='center')
        return fig

    X = np.array(features, dtype=np.float32)
    y = np.array(word_ids)
    n_words_total = len(items)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f'UMAP & t-SNE (100 ms windows, colored by word) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )

    # UMAP
    try:
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                n_neighbors=min(15, len(X) - 1))
        emb_u = reducer.fit_transform(X)
        sc = ax1.scatter(emb_u[:, 0], emb_u[:, 1],
                         c=y, cmap='tab20', s=20, alpha=0.75,
                         vmin=0, vmax=n_words_total - 1)
        ax1.set_title('UMAP', fontsize=11)
        ax1.set_xlabel('UMAP-1', fontsize=9)
        ax1.set_ylabel('UMAP-2', fontsize=9)
    except Exception as e:
        ax1.text(0.5, 0.5, f'UMAP failed:\n{e}', ha='center', va='center', fontsize=8)

    # t-SNE
    try:
        perp  = max(5, min(30, len(X) // 3))
        tsne  = TSNE(n_components=2, random_state=42, perplexity=perp)
        emb_t = tsne.fit_transform(X)
        ax2.scatter(emb_t[:, 0], emb_t[:, 1],
                    c=y, cmap='tab20', s=20, alpha=0.75,
                    vmin=0, vmax=n_words_total - 1)
        ax2.set_title('t-SNE', fontsize=11)
        ax2.set_xlabel('t-SNE-1', fontsize=9)
        ax2.set_ylabel('t-SNE-2', fontsize=9)
    except Exception as e:
        ax2.text(0.5, 0.5, f't-SNE failed:\n{e}', ha='center', va='center', fontsize=8)

    # Legend (unique word ids that have data)
    seen, handles = set(), []
    for wid, wt in zip(word_ids, word_labels):
        if wid not in seen:
            seen.add(wid)
            c = WORD_CMAP.colors[wid % len(WORD_CMAP.colors)]
            handles.append(mpatches.Patch(color=c, label=f'{wid}: {wt}'))
    fig.legend(handles=handles, loc='lower center', ncol=min(8, len(handles)),
               fontsize=6.5, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 11 — Hypnogram (Band Power over Time)
# One sub-panel per frequency band; sliding-window power; continuous time
# ══════════════════════════════════════════════════════════════════════════════
def viz_hypnogram(ex_idx, items, eeg_cont, t_arr, bounds, info):
    n_bands = len(BANDS)
    fig, axes = plt.subplots(n_bands, 1, figsize=(24, 3.2 * n_bands),
                             sharex=True)
    fig.suptitle(
        f'Band Power Over Time (Hypnogram-style) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold', y=1.01,
    )

    win  = int(0.10 * SRATE)  # 100 ms window
    step = int(0.05 * SRATE)  # 50 ms step
    sig  = np.nanmean(eeg_cont, axis=0)
    sig_f = np.where(np.isnan(sig), 0.0, sig)
    n_wins = max(1, (len(sig_f) - win) // step)
    t_centers = np.array([
        (k * step + win // 2) / SRATE + t_arr[0] for k in range(n_wins)
    ])

    for ax, (bname, (blo, bhi)), bc in zip(axes, BANDS.items(), BAND_COLS):
        powers = []
        for k in range(n_wins):
            seg = sig_f[k * step : k * step + win]
            fw, ps = ssig.welch(seg, fs=SRATE, nperseg=win)
            mask = (fw >= blo) & (fw < bhi)
            powers.append(float(ps[mask].mean()) if mask.any() else 0.0)
        powers = np.array(powers)

        ax.fill_between(t_centers, powers, alpha=0.65, color=bc)
        ax.plot(t_centers, powers, color=bc, lw=0.9)
        ax.set_ylabel(bname, fontsize=7.5, labelpad=2)
        ax.set_ylim(bottom=0)
        ymax = max(powers.max() * 1.15, 1e-9)
        annotate_words(ax, bounds, ymax * 0.92, fontsize=5.5, rotate=50)
        ax.grid(axis='x', lw=0.25, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 12 — Density Spectral Array (DSA)
# Sliding-window Welch PSD averaged across channels; freq × time heatmap
# ══════════════════════════════════════════════════════════════════════════════
def viz_dsa(ex_idx, items, eeg_cont, t_arr, bounds, info):
    sig   = np.nanmean(eeg_cont, axis=0)
    sig_f = np.where(np.isnan(sig), 0.0, sig)

    win  = int(0.5 * SRATE)   # 500 ms window
    step = int(0.1 * SRATE)   # 100 ms step
    n_wins = max(1, (len(sig_f) - win) // step)

    t_centers = np.array([
        (k * step + win // 2) / SRATE + t_arr[0] for k in range(n_wins)
    ])

    fw, _ = ssig.welch(sig_f[:win], fs=SRATE, nperseg=win)
    f_mask = fw <= 50
    fw_f   = fw[f_mask]

    dsa = np.zeros((f_mask.sum(), n_wins))
    for k in range(n_wins):
        seg = sig_f[k * step : k * step + win]
        _, ps = ssig.welch(seg, fs=SRATE, nperseg=win)
        dsa[:, k] = 10 * np.log10(ps[f_mask] + 1e-15)

    fig, ax = plt.subplots(figsize=(24, 6))
    fig.suptitle(
        f'Density Spectral Array (mean across channels) — Example {ex_idx}\n'
        f'"{get_sentence(ex_idx)}"',
        fontsize=11, fontweight='bold',
    )
    im = ax.pcolormesh(t_centers, fw_f, dsa,
                       cmap='jet', shading='auto')

    for bname, (blo, bhi) in BANDS.items():
        ax.axhline(blo, color='white', lw=0.5, ls='--', alpha=0.5)
        ax.text(t_centers[0] - 0.02, (blo + bhi) / 2,
                bname.split('(')[0].strip(),
                va='center', ha='right', fontsize=6, color='#222222')

    annotate_words(ax, bounds, fw_f[-1] * 0.92, fontsize=7, rotate=0)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.85)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
VIZ_TYPES = [
    ('01_raw_eeg',            'Raw EEG Traces',              viz_raw_eeg),
    ('02_psd',                'Power Spectral Density',       viz_psd),
    ('03_spectrograms',       'Spectrograms',                 viz_spectrograms),
    ('04_wavelet_scalograms', 'Wavelet Scalograms',           viz_wavelet),
    ('05_topomaps',           'Topographic Maps',             viz_topomaps),
    ('06_connectivity',       'Connectivity Heatmaps',        viz_connectivity),
    ('07_network',            'Brain Network Graphs',         viz_network),
    ('08_ica',                'ICA Component Maps',           viz_ica),
    ('09_source',             'Source Localization Maps',     viz_source),
    ('10_umap_tsne',          'UMAP/t-SNE Embeddings',        viz_umap_tsne),
    ('11_hypnogram',          'Hypnograms (Band Power)',       viz_hypnogram),
    ('12_dsa',                'Density Spectral Arrays',      viz_dsa),
]


def main():
    print('Building MNE info ...')
    info = build_mne_info()

    for folder, _, _ in VIZ_TYPES:
        os.makedirs(os.path.join(VIZ_DIR, folder), exist_ok=True)

    for ex_idx in range(NUM_EXAMPLES):
        print(f'\n{"="*60}\nExample {ex_idx}: {get_sentence(ex_idx)[:80]}\n{"="*60}')
        items          = load_example(ex_idx)
        eeg_cont, t_arr, bounds = build_continuous(items)

        for folder, name, fn in VIZ_TYPES:
            print(f'  [{name:35s}] ...', end=' ', flush=True)
            try:
                fig = fn(ex_idx, items, eeg_cont, t_arr, bounds, info)
                out = os.path.join(VIZ_DIR, folder, f'example_{ex_idx}.png')
                fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print('OK')
            except Exception as exc:
                print(f'FAILED — {exc}')
                plt.close('all')

    print(f'\n✓ All done. Plots saved to:\n  {VIZ_DIR}')


if __name__ == '__main__':
    main()
