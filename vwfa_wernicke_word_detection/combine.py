"""Union combination of VWFA-detected and Wernicke-ERP-detected word events.

Not a confirmation/gating rule (that was tried, in the old
vwfa_word_detection.word_detection.confirm_with_wernicke, and removed -- see
EEG_research/README.md). This is additive: VWFA's markers are the base set,
and a Wernicke-ERP marker is added only if it doesn't already overlap a VWFA
marker -- so Wernicke-ERP can only ever catch words VWFA missed, never
duplicate or override one VWFA already found.
"""
import pandas as pd


def combine_detections(vwfa_markers, wernicke_erp_markers):
    """vwfa_markers, wernicke_erp_markers: DataFrames with marker_start_sample/
    marker_end_sample (word_detection.build_marker_windows's output shape).
    Returns one DataFrame, sorted by marker_start_sample, with a `source`
    column ("vwfa" or "wernicke_erp") and the original columns preserved
    (missing values NaN where a column only exists on one side).
    """
    vwfa = vwfa_markers.copy()
    vwfa["source"] = "vwfa"

    if wernicke_erp_markers.empty:
        return vwfa.sort_values("marker_start_sample").reset_index(drop=True)

    vwfa_intervals = list(zip(vwfa["marker_start_sample"], vwfa["marker_end_sample"]))

    def overlaps_any_vwfa(start, end):
        return any(start < b and a < end for a, b in vwfa_intervals)

    added_rows = []
    for _, row in wernicke_erp_markers.iterrows():
        if not overlaps_any_vwfa(row["marker_start_sample"], row["marker_end_sample"]):
            r = row.copy()
            r["source"] = "wernicke_erp"
            added_rows.append(r)

    if not added_rows:
        return vwfa.sort_values("marker_start_sample").reset_index(drop=True)

    added = pd.DataFrame(added_rows)
    combined = pd.concat([vwfa, added], ignore_index=True, sort=False)
    return combined.sort_values("marker_start_sample").reset_index(drop=True)
