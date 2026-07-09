#!/usr/bin/env python3
"""
VWFA -> Wernicke-confirmed word-detection demo on the local ZDN example data.

EXPLORATORY: the ZuCo continuous raw dataset is not present on this machine
(pipeline/config.py's DATASET_ROOT points to a remote path). The only real
EEG data available locally is the 5 example sentences under
tests/word_eeg_ZDN/, as pre-cut per-fixation .npy arrays. This script:

  1. reconstructs a pseudo-continuous signal per sentence (same trick as
     visualize_word_eeg.py's build_continuous: concatenate word segments,
     insert NaN-filled gaps for words with no fixation) -- this also gives
     us the ground-truth word-boundary timing straight from the eye-fixation
     data (each word's segment IS its fixation span);
  2. averages the VWFA-ROI and Wernicke-ROI channels (pipeline/rois.py /
     outputs/rois/roi_channels.json) from that signal;
  3. runs pipeline.word_detection.detect_word_events on the VWFA/Wernicke
     traces to get confirmed word-reading-event markers (+/-200ms windows);
  4. plots ground truth vs. detected markers on one continuous timeline per
     sentence, and writes a per-word offset summary CSV.

Channel-mapping caveat (loudly flagged, not silently assumed): the source
gip_ZDN_NR1_EEG.mat (which would give the true chanlocs) is not present
locally. eeg.npy's 105 channels are assumed to be the GSN-HydroCel-129
montage's E-channels in ascending numeric order, minus the 24-channel
PERIOCULAR_OUTER_RING, with Cz appended as the 105th channel (104 + 1 = 105,
matching the observed shape). This ordering is UNVERIFIED against ground
truth and should be re-checked once the real .mat file is available.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import config, rois, word_detection as wd

ZDN_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_DIR = ZDN_DIR
VIZ_DIR = os.path.join(WORD_DIR, "visualizations", "13_word_detection")
NUM_EXAMPLES = 5
GAP_SEC = 0.15
SRATE = 500.0


# ── Channel-mapping (see module docstring caveat) ────────────────────────────
def build_channel_order():
    """Assumed 105-channel order for eeg.npy: ascending E-number GSN-HydroCel-129
    channels minus PERIOCULAR_OUTER_RING, with Cz appended. UNVERIFIED -- flagged
    loudly since the source .mat with true chanlocs isn't available locally.
    """
    import mne
    montage = mne.channels.make_standard_montage(config.MONTAGE_NAME)
    pos = montage.get_positions()["ch_pos"]

    def e_num(name):
        m = re.match(r"E(\d+)$", name)
        return int(m.group(1)) if m else None

    e_channels = sorted(
        (n for n in pos if e_num(n) is not None and n not in rois.PERIOCULAR_OUTER_RING),
        key=e_num,
    )
    order = e_channels + ["Cz"]
    assert len(order) == 105, (
        f"Expected 105 channels (104 E-channels + Cz), got {len(order)}. "
        "The eeg.npy channel-order assumption may be wrong -- do not trust "
        "ROI averaging results until this is resolved."
    )
    print("[word_detection_demo] WARNING: channel order for eeg.npy is an "
          "UNVERIFIED assumption (ascending E-number GSN-HydroCel-129 minus "
          "periocular ring, + Cz appended) -- the source .mat with true "
          "chanlocs is not available on this machine.")
    return order


# ── Data loading (reimplemented minimally to avoid visualize_word_eeg.py's
#    heavy optional deps -- umap/networkx/sklearn/pywt/h5py -- none of which
#    are needed for this script) ─────────────────────────────────────────────
def load_example(ex_idx):
    ex_dir = os.path.join(WORD_DIR, f"example_{ex_idx}")
    word_dirs = sorted(d for d in os.listdir(ex_dir) if d.startswith("word_"))
    items = []
    for wd_name in word_dirs:
        word_text = wd_name
        with open(os.path.join(ex_dir, wd_name, "metadata.txt")) as mf:
            for line in mf:
                if line.startswith("Word text:"):
                    word_text = line.split(":", 1)[1].strip()
                    break
        eeg = np.load(os.path.join(ex_dir, wd_name, "eeg.npy"))
        items.append((word_text, eeg))
    return items


def get_sentence(ex_idx):
    path = os.path.join(WORD_DIR, f"example_{ex_idx}", "sentence.txt")
    with open(path) as f:
        line = f.readline()
    return line.split(":", 1)[1].strip() if ":" in line else line.strip()


def build_continuous(items, n_ch=105, srate=SRATE, gap_sec=GAP_SEC):
    """Ground-truth word timing comes straight from this: each word's
    (t_start, t_end) IS its real fixation span from the eye-tracking data.
    """
    segs, ts, bounds = [], [], []
    t = 0.0
    n_gap = int(gap_sec * srate)
    for word_text, eeg in items:
        t0 = t
        if eeg.shape[1] > 0:
            n = eeg.shape[1]
            segs.append(eeg)
            ts.append(np.arange(n) / srate + t)
            t += n / srate
            has_data = True
        else:
            segs.append(np.full((n_ch, n_gap), np.nan))
            ts.append(np.arange(n_gap) / srate + t)
            t += gap_sec
            has_data = False
        bounds.append((word_text, t0, t, has_data))
    return np.concatenate(segs, axis=1), np.concatenate(ts), bounds


def annotate_words(ax, bounds, y_label, fontsize=6.5, rotate=50):
    cmap = plt.cm.tab20
    for i, (word_text, t0, t1, has_data) in enumerate(bounds):
        c = cmap.colors[i % len(cmap.colors)]
        ax.axvline(t0, color=c, lw=0.6, ls="--", alpha=0.65)
        mid = (t0 + t1) / 2
        ax.text(mid, y_label, word_text, ha="center", va="bottom",
                fontsize=fontsize, color=c, rotation=rotate, clip_on=True)
        if not has_data:
            ax.axvspan(t0, t1, alpha=0.12, color="gray")


# ── ROI averaging ─────────────────────────────────────────────────────────
def roi_average(eeg_cont, channel_order, roi_channels):
    idx = [channel_order.index(c) for c in roi_channels if c in channel_order]
    if not idx:
        raise ValueError(f"None of {roi_channels} found in channel_order.")
    with np.errstate(invalid="ignore"):
        # All-NaN columns are the intentional no-fixation gap placeholders
        # from build_continuous(); nanmean over them is NaN by design.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            return np.nanmean(eeg_cont[idx], axis=0)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VIZ_DIR, exist_ok=True)
    channel_order = build_channel_order()

    with open(os.path.join(config.DIRS["rois"], "roi_channels.json")) as f:
        roi_channels = json.load(f)
    vwfa_channels = roi_channels["vwfa"]
    wernicke_channels = roi_channels["wernicke"]
    print(f"[word_detection_demo] vwfa channels: {vwfa_channels}")
    print(f"[word_detection_demo] wernicke channels: {wernicke_channels}")

    # Load all 5 examples up front so thresholds can be calibrated pooled
    # across all of them (more samples -> more stable percentiles than
    # calibrating per-sentence on ~25-35 words each).
    examples = {}
    all_vwfa, all_wernicke = [], []
    for ex_idx in range(NUM_EXAMPLES):
        items = load_example(ex_idx)
        eeg_cont, t_arr, bounds = build_continuous(items)
        sig_vwfa = roi_average(eeg_cont, channel_order, vwfa_channels)
        sig_wernicke = roi_average(eeg_cont, channel_order, wernicke_channels)
        examples[ex_idx] = dict(items=items, eeg_cont=eeg_cont, t_arr=t_arr,
                                 bounds=bounds, sig_vwfa=sig_vwfa, sig_wernicke=sig_wernicke)
        all_vwfa.append(sig_vwfa)
        all_wernicke.append(sig_wernicke)

    pooled_vwfa = np.concatenate(all_vwfa)
    pooled_wernicke = np.concatenate(all_wernicke)

    print("[word_detection_demo] CAVEAT: calibrating thresholds via "
          "mode='within_reading_percentile' -- there is no true resting/"
          "no-word reference in this local data, so 'low' means 'low "
          "relative to other words in these 5 sentences,' not a validated "
          "absolute baseline. Full validation requires the real continuous "
          "dataset + pipeline/epochs.py's non_reading windows "
          "(mode='baseline_vs_reading').")
    vwfa_thresholds = wd.calibrate_signal_thresholds(pooled_vwfa, SRATE)
    wernicke_thresholds = wd.calibrate_signal_thresholds(pooled_wernicke, SRATE)
    print(f"[word_detection_demo] vwfa thresholds: {vwfa_thresholds}")
    print(f"[word_detection_demo] wernicke thresholds: {wernicke_thresholds}")

    summary_rows = []
    total_words_with_data = 0
    total_matched = 0
    offsets_ms = []

    for ex_idx in range(NUM_EXAMPLES):
        ex = examples[ex_idx]
        bounds, t_arr = ex["bounds"], ex["t_arr"]
        sig_vwfa, sig_wernicke = ex["sig_vwfa"], ex["sig_wernicke"]

        candidates, markers = wd.detect_word_events(
            sig_vwfa, sig_wernicke, SRATE, vwfa_thresholds, wernicke_thresholds,
        )
        n_confirmed = int(markers.shape[0]) if not markers.empty else 0
        n_candidates = int(candidates.shape[0]) if not candidates.empty else 0
        print(f"[word_detection_demo] example {ex_idx}: {n_candidates} VWFA "
              f"candidates, {n_confirmed} Wernicke-confirmed -> markers")

        marker_times = (markers["sample"].to_numpy() / SRATE) if n_confirmed else np.array([])
        marker_confirmed_flags = np.ones_like(marker_times, dtype=bool)

        # Nearest-marker matching per ground-truth word (has_data words only)
        for word_text, t0, t1, has_data in bounds:
            if not has_data:
                continue
            total_words_with_data += 1
            word_center = (t0 + t1) / 2
            if marker_times.size:
                nearest_idx = int(np.argmin(np.abs(marker_times - word_center)))
                offset_ms = (marker_times[nearest_idx] - word_center) * 1000.0
                within_500 = abs(offset_ms) <= 500
            else:
                offset_ms = np.nan
                within_500 = False
            if within_500:
                total_matched += 1
                offsets_ms.append(offset_ms)
            summary_rows.append({
                "example": ex_idx, "word_text": word_text,
                "gt_t_start_s": t0, "gt_t_end_s": t1,
                "nearest_marker_offset_ms": offset_ms if within_500 else np.nan,
                "matched_within_500ms": within_500,
            })

        # ── Plot ──
        fig, (ax_vwfa, ax_wernicke) = plt.subplots(
            2, 1, figsize=(24, 8), sharex=True,
        )
        fig.suptitle(
            f"VWFA/Wernicke Word Detection — Example {ex_idx}\n"
            f'"{get_sentence(ex_idx)}"\n'
            f"(channel-order assumption UNVERIFIED — see script docstring)",
            fontsize=10, fontweight="bold",
        )

        for ax, sig, label, color in (
            (ax_vwfa, sig_vwfa, "VWFA ROI", "#2166ac"),
            (ax_wernicke, sig_wernicke, "Wernicke ROI", "#b2182b"),
        ):
            ax.plot(t_arr, sig, color=color, lw=0.6, alpha=0.85)
            ax.set_ylabel(f"{label}\n(uV)", fontsize=8)
            finite = sig[np.isfinite(sig)]
            if finite.size:
                lo, hi = np.percentile(finite, [1, 99])
                pad = (hi - lo) * 0.2 + 1e-9
                ax.set_ylim(lo - pad, hi + pad * 3)
            annotate_words(ax, bounds, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1, fontsize=6, rotate=55)

            for _, row in markers.iterrows():
                t0_marker = row["marker_start_sample"] / SRATE
                t1_marker = row["marker_end_sample"] / SRATE
                t_center = row["sample"] / SRATE
                ax.axvspan(t0_marker, t1_marker, color="#2ca02c", alpha=0.18, lw=0)
                ax.axvline(t_center, color="#2ca02c", lw=1.2, ls="-", alpha=0.9)

            if not candidates.empty:
                unconfirmed = candidates[~candidates["confirmed"]]
                for _, row in unconfirmed.iterrows():
                    ax.axvline(row["sample"] / SRATE, color="#888888", lw=0.8, ls=":", alpha=0.6)

            ax.grid(axis="x", lw=0.25, alpha=0.3)

        handles = [
            plt.Line2D([0], [0], color="#2ca02c", lw=1.5, label="Confirmed VWFA→Wernicke marker (center)"),
            plt.matplotlib.patches.Patch(color="#2ca02c", alpha=0.18, label="400ms marker window"),
            plt.Line2D([0], [0], color="#888888", lw=0.8, ls=":", label="Unconfirmed VWFA candidate"),
            plt.matplotlib.patches.Patch(color="gray", alpha=0.12, label="No-fixation gap (ground truth)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        ax_wernicke.set_xlabel("Time (s)", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])

        out_path = os.path.join(VIZ_DIR, f"example_{ex_idx}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[word_detection_demo] saved {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(WORD_DIR, "detected_vs_ground_truth.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[word_detection_demo] wrote {csv_path}")

    pct_matched = 100.0 * total_matched / total_words_with_data if total_words_with_data else 0.0
    print(f"\n[word_detection_demo] SUMMARY (5 sentences, subject ZDN, exploratory):")
    print(f"  Ground-truth words with fixation data : {total_words_with_data}")
    print(f"  Matched to a confirmed marker (<=500ms): {total_matched} ({pct_matched:.1f}%)")
    if offsets_ms:
        offsets_arr = np.array(offsets_ms)
        print(f"  Median offset                          : {np.median(offsets_arr):.1f} ms")
        print(f"  IQR offset                              : "
              f"[{np.percentile(offsets_arr,25):.1f}, {np.percentile(offsets_arr,75):.1f}] ms")
    else:
        print("  No matches within 500ms -- see caveats in script docstring "
              "(small-sample threshold calibration, unverified channel order).")


if __name__ == "__main__":
    main()
