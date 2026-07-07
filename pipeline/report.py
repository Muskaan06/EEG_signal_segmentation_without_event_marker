"""Assemble a markdown report summarizing dataset stats, preprocessing QA,
feature significance, model comparison, and the final interpretation.
"""

import os

import pandas as pd

from pipeline import config


def build_report(dataset_stats, preprocessing_summary, sig_summary, model_results,
                  deep_results, roi_channels, out_path=None):
    out_path = out_path or os.path.join(config.DIRS["report"], "report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append("# EEG Reading vs. Non-Reading Classification — Report\n")

    lines.append("## Dataset\n")
    lines.append(f"- Task: {dataset_stats.get('task')}")
    lines.append(f"- Subjects: {dataset_stats.get('n_subjects')} {dataset_stats.get('subjects')}")
    lines.append(f"- Sentences: {dataset_stats.get('n_sentences')}")
    lines.append(f"- Word (reading) events recovered: {dataset_stats.get('n_word_events')}")
    lines.append(f"- Sampling rate: {dataset_stats.get('sampling_rate_hz')} Hz")
    lines.append(f"- Channels (raw): {dataset_stats.get('n_channels')}\n")

    lines.append("## Preprocessing\n")
    for block, info in preprocessing_summary.items():
        lines.append(f"- **{block}**: line freq={info['line_freq']}Hz, "
                      f"{len(info['bads'])} bad channels interpolated, "
                      f"ICA excluded {len(info['ica_excluded'].get('eog', []))} EOG + "
                      f"{len(info['ica_excluded'].get('muscle', []))} muscle components")
    lines.append("")

    lines.append("## ROIs (scalp-location approximations)\n")
    for roi, chans in roi_channels.items():
        lines.append(f"- **{roi}**: {chans}")
    lines.append("")

    lines.append("## Most discriminative features (READING vs NON-READING)\n")
    lines.append("Ranked by |Cohen's d|, with paired t-test (across subjects) and "
                  "label-permutation p-values:\n")
    top = sig_summary.head(15).reset_index().rename(columns={"index": "feature"})
    lines.append(top.to_markdown(index=False))
    lines.append("")

    lines.append("## Model comparison\n")
    summary = model_results.groupby(["protocol", "model"])[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().reset_index()
    lines.append(summary.to_markdown(index=False))
    lines.append("")

    if deep_results is not None and len(deep_results):
        lines.append("## Deep learning models (raw epochs, subject-independent)\n")
        deep_summary = deep_results.groupby("model")[
            ["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().reset_index()
        lines.append(deep_summary.to_markdown(index=False))
        lines.append("")

    best_loso = summary[summary["protocol"] == "subject_independent"].sort_values("roc_auc", ascending=False)
    best_row = best_loso.iloc[0] if len(best_loso) else None

    lines.append("## Interpretation\n")
    top_rois = pd.Series([f.split("_")[0] for f in top["feature"]]).value_counts()
    lines.append(f"- Most informative ROIs (by frequency in top-15 discriminative features): "
                  f"{dict(top_rois.head(3))}")
    top_components = [f for f in top["feature"] if any(c in f for c in config.ERP_COMPONENTS)]
    lines.append(f"- ERP-component features in the top 15: {top_components}")
    top_bands = [f for f in top["feature"] if any(b in f for b in config.BANDS)]
    lines.append(f"- Frequency-band features in the top 15: {top_bands}")
    lines.append("")
    if best_row is not None:
        lines.append(
            f"**Can EEG reliably tell READING from NON-READING without decoding word "
            f"identity?** On this validation subset, the best subject-independent "
            f"(Leave-One-Subject-Out) model was **{best_row['model']}** with "
            f"accuracy={best_row['accuracy']:.2f}, F1={best_row['f1']:.2f}, "
            f"ROC-AUC={best_row['roc_auc']:.2f}. "
        )
    lines.append(
        "This is a binary reading-state classification (not word-identity decoding): "
        "labels only distinguish whether a word was being fixated/read vs. genuinely "
        "non-reading intervals (pre-block, inter-trial, inter-word gaps) from the same "
        "recordings, so any above-chance performance reflects general reading-state "
        "signatures (e.g. posterior/temporal ERP negativity, alpha/theta power changes) "
        "rather than which word was read."
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out_path}")
    return out_path
