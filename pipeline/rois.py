"""Define scalp ROIs approximating the reading network, from the montage's
3D channel coordinates rather than hand-picked channel numbers.

The GSN-HydroCel net has no direct correspondence to 10-20 sites or to
underlying cortex, so these are explicitly *scalp-location approximations*
(nearest-channel-to-anatomical-target), not a source-localized ROI parcellation.
Each ROI's exact channel list is computed once and saved to
outputs/rois/roi_channels.json for inspection/reporting.

MNE head-coordinate convention: x = right(+)/left(-), y = anterior(+)/
posterior(-), z = superior(+)/inferior(-), in meters, origin ~head center.
"""

import json
import os

import mne
import numpy as np

from pipeline import config

mne.set_log_level("WARNING")

# Approximate scalp-target coordinates (meters, MNE head coords) for each ROI,
# chosen to sit near the conventional 10-20 sites named in parentheses.
ROI_TARGETS = {
    "occipital":  (0.000, -0.090, -0.010),   # visual cortex (Oz/O1/O2)
    "vwfa":       (-0.060, -0.060, -0.030),  # left occipito-temporal (~PO7/P7)
    "pstg":       (-0.075, -0.020, -0.030),  # left posterior superior temporal (~T5/TP7)
    "wernicke":   (-0.065,  0.000,  0.000),  # left temporo-parietal (~CP5/TP7)
    "broca":      (-0.055,  0.060, -0.020),  # left inferior frontal (~F7/IFG)
    "parietal":   (-0.050, -0.030,  0.050),  # left angular/supramarginal (~P3/CP5)
}

MAX_ROI_RADIUS_M = 0.035

# Per-ROI radius overrides: "wernicke" is unusually sparse (2 channels) at the
# default radius, which is too thin to serve as an independent confirmation
# signal for word_detection.py's VWFA->Wernicke check. Widen just this one.
ROI_RADIUS_OVERRIDES = {
    "wernicke": 0.045,
}

# Channels ZuCo's own pipeline drops (outer face/neck ring + periocular);
# excluded from ROI assignment since they carry mostly EOG/muscle, not cortex.
PERIOCULAR_OUTER_RING = {
    "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E48", "E49", "E56",
    "E63", "E68", "E73", "E81", "E88", "E94", "E99", "E107", "E113",
    "E119", "E125", "E126", "E127", "E128",
}


def _montage_positions():
    montage = mne.channels.make_standard_montage(config.MONTAGE_NAME)
    pos = montage.get_positions()["ch_pos"]
    return {name: np.asarray(xyz) for name, xyz in pos.items() if name not in PERIOCULAR_OUTER_RING}


def compute_roi_channels(ch_names=None, save=True):
    """Assign each scalp channel to its nearest ROI target (within that ROI's
    radius — MAX_ROI_RADIUS_M by default, overridden per-ROI by
    ROI_RADIUS_OVERRIDES), returning {roi_name: [channel_labels]}.
    """
    positions = _montage_positions()
    if ch_names is not None:
        positions = {n: p for n, p in positions.items() if n in ch_names}

    roi_names = list(ROI_TARGETS.keys())
    targets = np.array([ROI_TARGETS[r] for r in roi_names])
    radii = np.array([ROI_RADIUS_OVERRIDES.get(r, MAX_ROI_RADIUS_M) for r in roi_names])

    roi_channels = {r: [] for r in roi_names}
    for name, xyz in positions.items():
        dists = np.linalg.norm(targets - xyz, axis=1)
        best = np.argmin(dists)
        if dists[best] <= radii[best]:
            roi_channels[roi_names[best]].append(name)

    for r in roi_names:
        roi_channels[r] = sorted(roi_channels[r], key=lambda n: int(n[1:]))

    if save:
        os.makedirs(config.DIRS["rois"], exist_ok=True)
        out_path = os.path.join(config.DIRS["rois"], "roi_channels.json")
        with open(out_path, "w") as f:
            json.dump(roi_channels, f, indent=2)
        print(f"[rois] wrote {out_path}")

    for r, chs in roi_channels.items():
        print(f"[rois] {r:10s} ({len(chs):2d} channels): {chs}")

    return roi_channels
