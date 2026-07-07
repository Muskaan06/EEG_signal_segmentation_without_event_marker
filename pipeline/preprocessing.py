"""Full preprocessing chain applied to the genuinely raw EEG (independent of
any cleaning already done by the ZuCo team): notch, bandpass, bad-channel
detection, average reference, ICA (EOG/muscle removal), interpolation.

QA figures are written at every stage so preprocessing quality can be
visually audited before trusting downstream epochs/features.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from pipeline import config

mne.set_log_level("WARNING")


def detect_line_freq(raw, candidates=config.NOTCH_CANDIDATES):
    """Pick 50 vs 60 Hz by comparing PSD power at each candidate (+harmonic)."""
    psd = raw.compute_psd(fmin=45, fmax=65, n_fft=2048, verbose=False)
    freqs = psd.freqs
    power = psd.get_data().mean(axis=0)
    scores = {}
    for f in candidates:
        idx = np.argmin(np.abs(freqs - f))
        scores[f] = power[idx]
    return max(scores, key=scores.get)


def detect_bad_channels(raw, var_mad_thresh=5.0, min_corr=0.4, flat_std_v=1e-9):
    """Heuristic bad-channel detection (flat / extreme-variance / poorly
    correlated with the rest of the scalp), in the spirit of PREP's
    lite/robust-reference bad-channel checks.
    """
    data = raw.get_data()  # (n_ch, n_samples), volts
    ch_names = np.array(raw.ch_names)

    stds = data.std(axis=1)
    flat = ch_names[stds < flat_std_v].tolist()

    log_var = np.log(np.maximum(stds ** 2, 1e-20))
    med = np.median(log_var)
    mad = np.median(np.abs(log_var - med)) + 1e-12
    robust_z = np.abs(log_var - med) / (1.4826 * mad)
    extreme = ch_names[robust_z > var_mad_thresh].tolist()

    # Robust reference: the per-sample *median* across channels, not the mean —
    # a single very-high-amplitude bad channel (e.g. a broken electrode at
    # thousands of uV) can otherwise dominate the mean and make every good
    # channel look poorly correlated with it.
    reference_signal = np.median(data, axis=0)
    corrs = np.array([
        np.corrcoef(data[i], reference_signal)[0, 1] if stds[i] > 0 else 0.0
        for i in range(len(ch_names))
    ])
    low_corr = ch_names[corrs < min_corr].tolist()

    bads = sorted(set(flat) | set(extreme) | set(low_corr))
    if len(bads) > 0.4 * len(ch_names):
        # The correlation heuristic can itself misfire (e.g. it did before the
        # median-reference fix above); never let bad-channel detection eat
        # nearly the whole montage — fall back to the safer flat/extreme-only
        # set and log loudly so it can be inspected.
        print(f"[preprocessing] WARNING: {len(bads)}/{len(ch_names)} channels flagged bad "
              f"(likely a broken heuristic, not a broken cap) — falling back to "
              f"flat+extreme-variance channels only: {sorted(set(flat) | set(extreme))}")
        bads = sorted(set(flat) | set(extreme))
    return bads


def run_ica(raw_good, n_components=0.99, seed=config.SEED):
    """Fit ICA on the good (non-bad) channels and auto-flag EOG/muscle ICs.

    Returns (ica, excluded_dict) where excluded_dict maps reason -> component indices.
    """
    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica",
        random_state=seed, max_iter="auto",
    )
    ica.fit(raw_good, verbose=False)

    excluded = {"eog": [], "muscle": []}

    # Proxy EOG channels: ZuCo's frontal/periocular net channels (no dedicated EOG channel type)
    eog_proxies = [ch for ch in ("E1", "E8", "E25", "E32", "E126", "E127") if ch in raw_good.ch_names]
    if eog_proxies:
        try:
            eog_idx, _scores = ica.find_bads_eog(raw_good, ch_name=eog_proxies, verbose=False)
            excluded["eog"] = list(eog_idx)
        except Exception as exc:
            print(f"[preprocessing] EOG detection failed: {exc}")

    try:
        muscle_idx, _scores = ica.find_bads_muscle(raw_good, verbose=False)
        excluded["muscle"] = list(muscle_idx)
    except Exception as exc:
        print(f"[preprocessing] Muscle-artifact detection failed: {exc}")

    all_bad = sorted(set(excluded["eog"]) | set(excluded["muscle"]))
    ica.exclude = all_bad
    return ica, excluded


def preprocess_raw(raw, qa_dir=None, tag=""):
    """Full chain: notch -> bandpass -> bad channels -> average reference ->
    ICA -> interpolate. Returns the cleaned Raw (all original channels
    present, bads interpolated).
    """
    raw = raw.copy().load_data()
    if qa_dir:
        os.makedirs(qa_dir, exist_ok=True)
        fig = raw.compute_psd(fmax=80, verbose=False).plot(show=False)
        fig.suptitle(f"PSD before filtering — {tag}")
        fig.savefig(os.path.join(qa_dir, "01_psd_before.png"), dpi=110)
        plt.close(fig)

    line_freq = detect_line_freq(raw)
    raw.notch_filter(freqs=[line_freq, 2 * line_freq], verbose=False)
    raw.filter(l_freq=config.BANDPASS[0], h_freq=config.BANDPASS[1],
               fir_design="firwin", verbose=False)

    if qa_dir:
        fig = raw.compute_psd(fmax=80, verbose=False).plot(show=False)
        fig.suptitle(f"PSD after notch({line_freq}Hz)+bandpass — {tag}")
        fig.savefig(os.path.join(qa_dir, "02_psd_after_filter.png"), dpi=110)
        plt.close(fig)

    bads = detect_bad_channels(raw)
    raw.info["bads"] = bads
    print(f"[preprocessing] {tag}: line_freq={line_freq}Hz, bad channels={bads}")

    good_ch = [ch for ch in raw.ch_names if ch not in bads]
    raw.set_eeg_reference(ref_channels="average", projection=False,
                           verbose=False, ch_type="eeg")

    raw_good = raw.copy().pick(good_ch)
    ica, excluded = run_ica(raw_good)

    if qa_dir:
        try:
            fig = ica.plot_components(show=False)
            figs = fig if isinstance(fig, list) else [fig]
            for i, f in enumerate(figs):
                f.suptitle(f"ICA components — {tag} (excluded={ica.exclude})")
                f.savefig(os.path.join(qa_dir, f"03_ica_components_{i}.png"), dpi=110)
                plt.close(f)
        except Exception as exc:
            print(f"[preprocessing] ICA plot failed: {exc}")

    raw_before_ica = raw.copy()
    ica.apply(raw, verbose=False)

    if bads:
        raw.interpolate_bads(reset_bads=True, verbose=False)

    if qa_dir:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ch_show = raw.ch_names[len(raw.ch_names) // 2]
        t = raw.times[:2000]
        axes[0].plot(t, raw_before_ica.get_data(picks=[ch_show])[0, :2000] * 1e6)
        axes[0].set_title(f"Before ICA removal ({ch_show})")
        axes[1].plot(t, raw.get_data(picks=[ch_show])[0, :2000] * 1e6)
        axes[1].set_title(f"After ICA removal + interpolation ({ch_show})")
        for ax in axes:
            ax.set_ylabel("uV")
        axes[1].set_xlabel("Time (s)")
        fig.suptitle(f"Before/after overlay — {tag}")
        fig.tight_layout()
        fig.savefig(os.path.join(qa_dir, "04_before_after_overlay.png"), dpi=110)
        plt.close(fig)

    raw.info["description"] = (
        f"preprocessed: notch={line_freq}Hz, bandpass={config.BANDPASS}, "
        f"bads={bads}, ica_excluded={excluded}"
    )
    return raw, {"line_freq": line_freq, "bads": bads, "ica_excluded": excluded}
