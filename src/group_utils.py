"""
group_utils.py — Country grouping / aggregation for EU trade flow maps.

Groups merge multiple EU countries into a single virtual node for both
the flow matrices and the centroid dict, so the spiral tree and arrow
renderers can treat the group as one source or terminal.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

# ── predefined groups ──────────────────────────────────────────────────────
DEFAULT_GROUPS: Dict[str, List[str]] = {
    'Benelux':  ['Belgium', 'Netherlands', 'Luxembourg'],
    'Nordics':  ['Sweden', 'Denmark', 'Finland'],
    'Baltics':  ['Estonia', 'Latvia', 'Lithuania'],
    'Visegrád': ['Poland', 'Czechia', 'Slovakia', 'Hungary'],
}


def apply_groups(
    export_matrix: pd.DataFrame,
    net_matrix: pd.DataFrame,
    centroids: Dict[str, Tuple[float, float]],
    active_groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Merge active groups into virtual nodes.

    Returns new (export_matrix, net_matrix, centroids) with member countries
    replaced by their group label. Originals are not modified.

    Parameters
    ----------
    active_groups : {group_name: [member_short_names]}

    Raises
    ------
    ValueError if two active groups share a member country.
    """
    # Validate no overlap
    seen: set = set()
    for gname, members in active_groups.items():
        overlap = seen & set(members)
        if overlap:
            raise ValueError(
                f"Countries {overlap} appear in more than one active group."
            )
        seen.update(members)

    exp = export_matrix.copy()
    net = net_matrix.copy()
    cents = dict(centroids)

    for gname, members in active_groups.items():
        present = [m for m in members if m in exp.index]
        if len(present) < 2:
            continue

        others = [c for c in exp.columns if c not in present]

        # ── aggregate export rows and columns ────────────────────────────
        # New group → other: sum of member exports to that other country
        # Other → new group: sum of that country's exports to all members
        for col in others:
            exp.loc[gname, col] = exp.loc[present, col].sum()
            exp.loc[col, gname] = exp.loc[col, present].sum()
        exp.loc[gname, gname] = 0.0

        # ── aggregate net matrix ─────────────────────────────────────────
        for col in others:
            net.loc[gname, col] = net.loc[present, col].sum()
            net.loc[col, gname] = -net.loc[gname, col]
        net.loc[gname, gname] = 0.0

        # ── drop member rows / columns ────────────────────────────────────
        to_drop = [m for m in present if m in exp.index]
        exp.drop(index=to_drop, columns=to_drop, errors='ignore', inplace=True)
        net.drop(index=to_drop, columns=to_drop, errors='ignore', inplace=True)

        # ── group centroid: simple mean of member centroids ───────────────
        valid = [m for m in present if m in cents]
        if valid:
            cx = sum(cents[m][0] for m in valid) / len(valid)
            cy = sum(cents[m][1] for m in valid) / len(valid)
            cents[gname] = (cx, cy)
        for m in present:
            cents.pop(m, None)

    return exp, net, cents