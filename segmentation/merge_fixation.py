import numpy as np
from numpy.linalg import norm
import sys
sys.path.insert(0, '/media/kpdubey/8.0 TB Volume/Muskaan/new_study')
from segmentation.saccade_detection import load_eeg, extract_eog, _butter, detect_fixations_velocity
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PERIOCULAR_IDX = {0, 7, 16, 24, 31, 125, 126, 127}
SCALP_IDX = [i for i in range(128) if i not in PERIOCULAR_IDX]


def _extract_erp(data, onset, srate, post_ms=400):
    n = int(post_ms / 1000.0 * srate)
    if onset < 0 or onset + n > data.shape[1]:
        return None
    ep = data[SCALP_IDX, onset:onset + n].copy()
    ep -= ep[:, :25].mean(axis=1, keepdims=True)  # 50ms baseline at 500 Hz
    return ep


def _cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    d = norm(a) * norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _gap_ms(fixations, i, j, srate):
    if i < j:
        return (fixations[j][0] - fixations[i][1]) / srate * 1000.0
    return (fixations[i][0] - fixations[j][1]) / srate * 1000.0


def merge_short_fixations(fixations, data, srate,
                           min_fix_ms=100, gap_ms=100, post_ms=400):
    fixations = [list(f) for f in fixations]
    i = 0
    while i < len(fixations):
        s, e = fixations[i]
        dur_ms = (e - s) / srate * 1000.0

        if dur_ms < min_fix_ms:
            left_close  = i > 0                  and _gap_ms(fixations, i-1, i, srate) < gap_ms
            right_close = i < len(fixations) - 1 and _gap_ms(fixations, i, i+1, srate) < gap_ms

            if left_close and right_close:
                erp_c = _extract_erp(data, s,                 srate, post_ms)
                erp_l = _extract_erp(data, fixations[i-1][0], srate, post_ms)
                erp_r = _extract_erp(data, fixations[i+1][0], srate, post_ms)

                if erp_c is not None and erp_l is not None and erp_r is not None:
                    if _cosine_sim(erp_c, erp_l) >= _cosine_sim(erp_c, erp_r):
                        fixations[i - 1][1] = e
                    else:
                        fixations[i + 1][0] = s
                    fixations.pop(i)
                    continue

            elif left_close:
                fixations[i - 1][1] = e
                fixations.pop(i)
                continue

            elif right_close:
                fixations[i + 1][0] = s
                fixations.pop(i)
                continue

        i += 1

    return [tuple(f) for f in fixations]


def plot_merge_result(fixations_before, fixations_after, srate,
                      gt_fixations=None, hEOG=None,
                      time_window_s=(10, 30), min_fix_ms=100,
                      output_path=None):

    t0 = int(time_window_s[0] * srate)
    t1 = int(time_window_s[1] * srate)

    before = [(s, e) for s, e in fixations_before if s >= t0 and e <= t1]
    after  = [(s, e) for s, e in fixations_after  if s >= t0 and e <= t1]
    gt     = [(s, e) for s, e in gt_fixations      if s >= t0 and e <= t1] \
             if gt_fixations is not None else []

    has_gt  = len(gt) > 0
    has_eog = hEOG is not None

    n_rows = 2 + has_gt + has_eog
    h_rat  = [1, 1] + ([1] if has_gt else []) + ([2] if has_eog else [])
    fig_h  = 4 + 1.5 * has_gt + 2.5 * has_eog

    fig, axes = plt.subplots(n_rows, 1, figsize=(20, fig_h),
                             gridspec_kw={'height_ratios': h_rat, 'hspace': 0.08},
                             facecolor='#1a1a1a')

    for ax in axes:
        ax.set_facecolor('#242424')
        ax.set_xlim(t0 / srate, t1 / srate)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#444')
        ax.tick_params(colors='#999', labelsize=7)

    def _draw(ax, fixations, label, short_col='#e74c3c', normal_col='#2ecc71'):
        ax.set_ylim(0, 1)
        ax.set_ylabel(label, color='#ccc', fontsize=8, rotation=0,
                      ha='right', va='center', labelpad=6)
        for s, e in fixations:
            dur = (e - s) / srate * 1000
            col = short_col if dur < min_fix_ms else normal_col
            xs, xe = s / srate, e / srate
            rect = mpatches.FancyBboxPatch(
                (xs + 0.002, 0.08), max(xe - xs - 0.004, 0.002), 0.84,
                boxstyle='round,pad=0.002', lw=0, facecolor=col, alpha=0.88)
            ax.add_patch(rect)
            ax.text((xs + xe) / 2, 0.5, f'{dur:.0f}ms',
                    ha='center', va='center', fontsize=5.5,
                    color='white', fontweight='bold', clip_on=True)
        ax.set_xticks([])

    row = 0
    _draw(axes[row], before, 'Before\nmerge');  row += 1
    _draw(axes[row], after,  'After\nmerge');   row += 1

    if has_gt:
        _draw(axes[row], gt, 'Ground\ntruth',
              short_col='#e74c3c', normal_col='#3498db')
        row += 1

    if has_eog:
        ax = axes[row]
        t_ax = np.arange(t0, t1) / srate
        ax.plot(t_ax, hEOG[t0:t1], color='#5dade2', lw=0.7, alpha=0.85)
        for s, e in after:
            ax.axvspan(s / srate, e / srate, alpha=0.12, color='#2ecc71', lw=0)
        if has_gt:
            for s, e in gt:
                ax.axvspan(s / srate, e / srate, alpha=0.10, color='#3498db', lw=0)
        ax.set_ylabel('hEOG (µV)', color='#ccc', fontsize=8)
        ax.set_xlabel('Time (s)', color='#999', fontsize=8)
        lim = max(abs(hEOG[t0:t1].min()), abs(hEOG[t0:t1].max())) * 1.15
        ax.set_ylim(-lim, lim)

    n_short = sum(1 for s, e in before if (e - s) / srate * 1000 < min_fix_ms)
    fig.suptitle(
        f'Fixation merging (gap < {min_fix_ms}ms, dur < {min_fix_ms}ms)  |  '
        f'window {time_window_s[0]}–{time_window_s[1]}s  |  '
        f'before: {len(before)} ({n_short} short)  →  after: {len(after)}'
        + (f'  |  GT: {len(gt)}' if has_gt else ''),
        color='#ddd', fontsize=9, y=1.01)

    handles = [mpatches.Patch(color='#2ecc71', label='detected (≥100ms)'),
               mpatches.Patch(color='#e74c3c', label='short (<100ms)')]
    if has_gt:
        handles.append(mpatches.Patch(color='#3498db', label='ground truth'))
    axes[0].legend(handles=handles, loc='upper right', fontsize=7,
                   facecolor='#333', edgecolor='#555', labelcolor='white')

    plt.tight_layout(rect=[0.06, 0, 1, 1])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        print(f'Saved → {output_path}')
    return fig


if __name__ == '__main__':

    filepath = '/media/kpdubey/8.0 TB Volume/Muskaan/dataset/ZuCo/zuco_1/NR/raw_data/ZDM/ZDM_NR1_EEG.mat'
    data, srate = load_eeg(filepath)
    data_hp     = _butter(data, srate, 0.5, 'high')
    hEOG, vEOG  = extract_eog(data, srate)
    combined    = np.sqrt(hEOG**2 + vEOG**2)

    fixations = detect_fixations_velocity(hEOG,     srate, smooth_ms=20, vel_pct=80)
    gt_fix    = detect_fixations_velocity(combined, srate, smooth_ms=20, vel_pct=80)

    short_before = sum(1 for s, e in fixations if (e - s) / srate * 1000 < 100)
    print(f'Total fixations before merging : {len(fixations)}')
    print(f'Short fixations (<100ms)       : {short_before}')

    merged = merge_short_fixations(fixations, data_hp, srate,
                                    min_fix_ms=100, gap_ms=100)

    short_after = sum(1 for s, e in merged if (e - s) / srate * 1000 < 100)
    print(f'Total fixations after merging  : {len(merged)}')
    print(f'Short fixations remaining      : {short_after}')
    print(f'Fixations removed by merging   : {len(fixations) - len(merged)}')

    durations_before = [(e - s) / srate * 1000 for s, e in fixations]
    durations_after  = [(e - s) / srate * 1000 for s, e in merged]
    print(f'\nMean duration before : {np.mean(durations_before):.1f} ms')
    print(f'Mean duration after  : {np.mean(durations_after):.1f} ms')

    np.save('./merged_fixations.npy', np.array(merged))
    print('\nSaved → merged_fixations.npy')

    fig = plot_merge_result(
        fixations, merged, srate,
        gt_fixations  = gt_fix,
        hEOG          = hEOG,
        time_window_s = (10, 30),
        min_fix_ms    = 100,
        output_path   = './fixation_merge_plot.png'
    )
    plt.close(fig)