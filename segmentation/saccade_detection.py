import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

PERIOCULAR_IDX = {0, 7, 16, 24, 31, 125, 126, 127}
BG      = '#1a1a1a'
ROW_BG  = '#242424'
COL_HIT = '#2ecc71'
COL_MISS= '#e67e22'
COL_FA  = '#e74c3c'
COL_OFF = '#f39c12'


def load_eeg(filepath):
    mat   = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg   = mat['EEG']
    data  = np.array(eeg.data, dtype=np.float64)
    srate = int(eeg.srate)
    return data, srate


def _butter(sig, srate, cutoff, btype):
    Wn  = [c/(srate/2) for c in cutoff] if isinstance(cutoff,(list,tuple)) else cutoff/(srate/2)
    b,a = butter(4, Wn, btype=btype)
    return filtfilt(b, a, sig, axis=-1)


def extract_eog(data, srate):
    E1, E8, E25, E32, E126, E127 = (
        data[0], data[7], data[24], data[31], data[125], data[126]
    )
    hEOG = _butter(_butter(E1 - E32,                      srate, 100., 'low'), srate, 0.2, 'high')
    vEOG = _butter(_butter((E8+E25)/2 - (E126+E127)/2,   srate, 100., 'low'), srate, 0.2, 'high')
    return hEOG, vEOG


def detect_horizontal_saccades(hEOG, srate, threshold_factor=5.0,
                                interval_ms=50, min_amplitude_uv=10.0):
    # Stage A: 50ms windows (paper uses 200ms; 50ms chosen to avoid spanning two
    # saccades at the ~250ms ISI typical in natural reading)
    interval   = int(interval_ms / 1000.0 * srate)
    refractory = int(0.15 * srate)   # 150 ms
    backdate   = int(0.05 * srate)   # 50 ms
    n_look     = int(0.5 * srate / interval)  # windows in 500ms pre-saccade look-back
    n = len(hEOG)
    onsets = []
    last   = -refractory  # tracks raw threshold-crossing sample of last accepted saccade
    i = interval
    while i < n - interval:
        # Stage A: flag window if SD rose compared to previous window
        sd_prev = np.std(hEOG[i - interval:i])
        sd_curr = np.std(hEOG[i:i + interval])
        if sd_curr > sd_prev and sd_prev > 0:
            for j in range(i, i + interval):
                # Stage D: refractory — measure from last raw detection sample so
                # backdated inter-onset gap = j_gap − 0 = refractory = 150ms
                if j - last < refractory:
                    continue

                # Stage B: compute RASD = mean of 10 lowest SDs from 500ms look-back
                sds_pre = []
                for k in range(n_look):
                    w_end   = j - k * interval
                    w_start = j - (k + 1) * interval
                    if w_start < 0:
                        break
                    if w_end - w_start >= 5:
                        sds_pre.append(np.std(hEOG[w_start:w_end]))
                if not sds_pre:
                    continue
                sd_rasd = np.mean(sorted(sds_pre)[:10])
                if sd_rasd == 0:
                    continue

                pre_mean = np.mean(hEOG[max(0, j - interval):j])
                if abs(hEOG[j] - pre_mean) > threshold_factor * sd_rasd:
                    # Stage C: backdate by 50ms to estimate true saccade onset
                    c = max(0, j - backdate)
                    # Stage E: amplitude gate at backdated onset (paper §2.3)
                    if abs(hEOG[c]) > min_amplitude_uv:
                        onsets.append(c)
                        last = j  # track raw j so refractory enforces 150ms between onsets
                        break
        i += interval
    return np.array(onsets, dtype=int)


def detect_fixations_velocity(eog_signal, srate,
                               smooth_ms=20, vel_pct=80,
                               min_fix_ms=80, max_fix_ms=1500):
    sm  = gaussian_filter1d(eog_signal, sigma=srate * smooth_ms / 1000.0 / 2)
    vel = np.abs(np.gradient(sm)) * srate
    thr = np.percentile(vel, vel_pct)
    mn  = int(min_fix_ms  / 1000.0 * srate)
    mx  = int(max_fix_ms  / 1000.0 * srate)
    is_fix = vel < thr
    fixations = []
    in_fix    = False
    st        = 0
    for i, f in enumerate(is_fix):
        if f and not in_fix:
            st = i; in_fix = True
        elif not f and in_fix:
            d = i - st
            if mn <= d <= mx:
                fixations.append((st, i))
            in_fix = False
    if in_fix:
        d = len(is_fix) - st
        if mn <= d <= mx:
            fixations.append((st, len(is_fix)))
    return fixations


def match_fixations(detected, ground_truth, srate, tolerance_ms=150):
    if not detected or not ground_truth:
        return set(), set(), {}
    tol      = int(tolerance_ms / 1000.0 * srate)
    centers_d = np.array([(s+e)//2 for s,e in detected])
    centers_g = np.array([(s+e)//2 for s,e in ground_truth])
    matched_d, matched_g, pairs = set(), set(), {}
    for d_idx, dc in enumerate(centers_d):
        dists = np.abs(centers_g - dc)
        order = np.argsort(dists)
        for g_idx in order:
            if g_idx in matched_g:
                continue
            if dists[g_idx] <= tol:
                matched_g.add(g_idx)
                matched_d.add(d_idx)
                pairs[d_idx] = g_idx
                break
    return matched_d, matched_g, pairs


def segment_eeg(data, saccade_onsets, srate, pre_ms=400, post_ms=400,
                artifact_uv=500.0):
    pre, post = int(pre_ms/1000*srate), int(post_ms/1000*srate)
    n_ch, n_t = data.shape
    eval_idx  = [i for i in range(n_ch) if i not in PERIOCULAR_IDX]
    epochs, valid = [], []
    for onset in saccade_onsets:
        s, e = onset - pre, onset + post
        if s < 0 or e > n_t:
            continue
        ep = data[:, s:e].copy()
        ep -= ep[:, :pre].mean(axis=1, keepdims=True)
        if np.max(np.abs(ep[eval_idx])) < artifact_uv:
            epochs.append(ep)
            valid.append(onset)
    if not epochs:
        return np.empty((0, n_ch, pre+post)), np.array([], dtype=int)
    return np.stack(epochs), np.array(valid, dtype=int)


def visualize_fixations_vs_gt(detected_fix, gt_fix, srate,
                               hEOG=None, tolerance_ms=150,
                               time_window_s=None, output_path=None):
    if time_window_s is not None:
        t0 = int(time_window_s[0] * srate)
        t1 = int(time_window_s[1] * srate)
        detected_fix = [(s,e) for s,e in detected_fix if s >= t0 and e <= t1]
        gt_fix       = [(s,e) for s,e in gt_fix       if s >= t0 and e <= t1]
    else:
        all_ends = ([e for _,e in detected_fix] if detected_fix else [0]) + \
                   ([e for _,e in gt_fix]       if gt_fix       else [0])
        t0, t1 = 0, max(all_ends)

    matched_d, matched_g, pairs = match_fixations(detected_fix, gt_fix, srate, tolerance_ms)
    g_to_d = {v: k for k, v in pairs.items()}

    n_hits = len(matched_g)
    n_gt   = len(gt_fix)
    n_det  = len(detected_fix)
    n_fa   = n_det - len(matched_d)
    acc    = n_hits / n_gt * 100 if n_gt > 0 else 0.0

    n_rows = 4 if hEOG is not None else 3
    h_rat  = [1.8, 1.8, 1.2, 2.5] if hEOG is not None else [1.8, 1.8, 1.2]
    fw     = max(18, (t1 - t0) / srate * 1.5)
    fh     = 6 if hEOG is not None else 4.5

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(fw, fh),
        gridspec_kw={'height_ratios': h_rat, 'hspace': 0.06},
        facecolor=BG
    )
    for ax in axes:
        ax.set_facecolor(ROW_BG)
        ax.set_xlim(t0/srate, t1/srate)
        for sp in ax.spines.values():
            sp.set_color('#444')
        ax.tick_params(colors='#999', labelsize=7)

    def _boxes(ax, fixations, matched_set, prefix, row_label):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(row_label, color='#ccc', fontsize=8.5, labelpad=6,
                      rotation=0, ha='right', va='center')
        for idx, (s, e) in enumerate(fixations):
            is_fa   = (prefix == 'F') and (idx not in matched_set)
            is_miss = (prefix == 'G') and (idx not in matched_set)
            col = COL_FA if is_fa else (COL_MISS if is_miss else COL_HIT)
            xs, xe = s/srate, e/srate
            rect = mpatches.FancyBboxPatch(
                (xs+0.002, 0.05), max(xe-xs-0.004, 0.001), 0.90,
                boxstyle='round,pad=0.002', lw=0, facecolor=col, alpha=0.93, clip_on=True
            )
            ax.add_patch(rect)
            label = f'{prefix}{idx+1}'
            cx = (xs + xe) / 2
            fs = min(7.5, max(5.5, (xe-xs)*60))
            ax.text(cx, 0.5, label, ha='center', va='center',
                    fontsize=fs, color='white', fontweight='bold', clip_on=True)
        ax.set_xticks([])

    ax_det, ax_gt, ax_off = axes[0], axes[1], axes[2]
    _boxes(ax_det, detected_fix, matched_d, 'F', 'Detected')
    _boxes(ax_gt,  gt_fix,       matched_g, 'G', 'Ground\ntruth')

    ax_off.set_ylim(-1.5, 1.5)
    ax_off.set_yticks([])
    ax_off.set_ylabel('Timing\noffset', color='#ccc', fontsize=8.5, labelpad=6,
                       rotation=0, ha='right', va='center')
    ax_off.axhline(0, color='#555', lw=0.5, ls='--')

    y_top, y_bot = 0.9, -0.9
    for d_idx, g_idx in pairs.items():
        dc  = sum(detected_fix[d_idx]) / 2 / srate
        gc  = sum(gt_fix[g_idx])       / 2 / srate
        off = int((sum(detected_fix[d_idx])/2 - sum(gt_fix[g_idx])/2) / srate * 1000)
        hw  = 0.008
        ax_off.plot([dc-hw, dc+hw], [y_top, y_top],    color=COL_OFF, lw=1.4)
        ax_off.plot([dc,    dc   ], [y_top, 0.2 ],     color=COL_OFF, lw=1.0)
        ax_off.plot([gc,    gc   ], [-0.2,  y_bot],    color=COL_OFF, lw=1.0)
        ax_off.plot([gc-hw, gc+hw], [y_bot, y_bot],    color=COL_OFF, lw=1.4)
        ax_off.plot([dc,    gc   ], [0.2,   -0.2 ],    color=COL_OFF, lw=0.8, ls=':')
        ax_off.text((dc+gc)/2, 0, f'{off:+d}ms', ha='center', va='center',
                    fontsize=5.5, color=COL_OFF,
                    bbox=dict(facecolor=ROW_BG, edgecolor='none', pad=0.8))
    ax_off.set_xlabel('Time (s)', color='#999', fontsize=8)

    if hEOG is not None:
        ax_eog = axes[3]
        t_ax   = np.arange(t0, t1) / srate
        ax_eog.plot(t_ax, hEOG[t0:t1], color='#5dade2', lw=0.7, alpha=0.85)
        for s, e in detected_fix:
            ax_eog.axvspan(s/srate, e/srate, alpha=0.12, color=COL_HIT, lw=0)
        prev = t0
        for s, _ in sorted(detected_fix):
            if s > prev:
                ax_eog.axvspan(prev/srate, s/srate, alpha=0.18, color=COL_FA, lw=0)
            prev = _
        ax_eog.set_ylabel('hEOG (µV)', color='#ccc', fontsize=8, labelpad=4)
        ax_eog.set_xlabel('Time (s)', color='#999', fontsize=8)
        lim = max(abs(hEOG[t0:t1].min()), abs(hEOG[t0:t1].max())) * 1.15
        ax_eog.set_ylim(-lim, lim)
        ax_eog.tick_params(colors='#999', labelsize=7)
        ax_off.set_xlabel('')

    leg_handles = [
        mpatches.Patch(color=COL_HIT,  label='hit'),
        mpatches.Patch(color=COL_MISS, label='missed GT'),
        mpatches.Patch(color=COL_FA,   label='false alarm'),
    ]
    axes[0].legend(handles=leg_handles, loc='upper right', fontsize=7.5,
                   facecolor='#333', edgecolor='#555', labelcolor='white',
                   framealpha=0.8)

    title = (f'Ground-truth vs detected fixations  |  accuracy ±{tolerance_ms}ms: '
             f'{acc:.1f}%  ({n_hits}/{n_gt} GT matched, {n_fa} false alarms)')
    fig.suptitle(title, color='#ddd', fontsize=9.5, y=1.015)

    plt.tight_layout(rect=[0.06, 0, 1, 1])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {output_path}')
    return fig


def run(filepath, pre_ms=400, post_ms=400,
        threshold_factor=4.5, min_amplitude_uv=10.0,
        artifact_uv=500.0, vis_window_s=(10, 22)):

    data, srate   = load_eeg(filepath)
    data_hp       = _butter(data, srate, 0.5, 'high')
    hEOG, vEOG   = extract_eog(data, srate)
    combined_eog  = np.sqrt(hEOG**2 + vEOG**2)

    h_onsets = detect_horizontal_saccades(
        hEOG, srate, threshold_factor=threshold_factor,
        min_amplitude_uv=min_amplitude_uv
    )

    det_fix = detect_fixations_velocity(hEOG,         srate, smooth_ms=20, vel_pct=80)
    gt_fix  = detect_fixations_velocity(combined_eog, srate, smooth_ms=20, vel_pct=80)

    epochs, valid_onsets = segment_eeg(data_hp, h_onsets, srate, pre_ms, post_ms, artifact_uv)
    times_ms = np.linspace(-pre_ms, post_ms, epochs.shape[2]) if epochs.shape[0] > 0 else np.array([])

    print(f'Sampling rate                : {srate} Hz')
    print(f'Horizontal saccades (RASD)   : {len(h_onsets)}')
    print(f'EEG epochs accepted          : {epochs.shape[0]}  shape={epochs.shape}')
    print(f'Detected fixations (hEOG)    : {len(det_fix)}')
    print(f'GT fixations (combined EOG)  : {len(gt_fix)}')

    fig = visualize_fixations_vs_gt(
        det_fix, gt_fix, srate,
        hEOG=hEOG,
        tolerance_ms=150,
        time_window_s=vis_window_s,
        output_path='./fixation_comparison2.png'
    )
    plt.close(fig)

    return (epochs, valid_onsets, times_ms, srate,
            hEOG, vEOG, det_fix, gt_fix, h_onsets)


if __name__ == '__main__':
    filepath = '/media/kpdubey/8.0 TB Volume/Muskaan/dataset/ZuCo/zuco_1/NR/raw_data/ZDM/ZDM_NR1_EEG.mat'

    (epochs, onsets, times_ms, srate,
     hEOG, vEOG, det_fix, gt_fix, h_onsets) = run(filepath)

    np.save('./saccade_epochs.npy',  epochs)
    np.save('./saccade_onsets.npy',  onsets)
    np.save('./epoch_times_ms.npy',  times_ms)

    print('\nOutputs saved to /mnt/user-data/outputs/')