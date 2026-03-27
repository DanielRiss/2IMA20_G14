"""Audit compute_f_overlap: check whether raw ~160M is physically reasonable."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import spiral_tree as st
from data_loader import load_trade_data
from map_utils import load_eu_map, get_centroids

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)
DATA  = os.path.join(ROOT, 'data', 'data-18886936.csv')
LABEL = os.path.join(ROOT, 'data', 'label-18886936.csv')

export_mx, net_mx, countries = load_trade_data(DATA, LABEL, year=2024)
eu_gdf    = load_eu_map()
centroids = get_centroids(eu_gdf)
SOURCES   = ['Germany', 'France']

trees = []
for src_name in SOURCES:
    if src_name not in net_mx.index:
        continue
    row = net_mx.loc[src_name]
    net_flows = {d: float(v) for d, v in row.items()
                 if d != src_name and float(v) > 0}
    trees.append(st.compute_spiral_tree(
        source_name=src_name,
        terminal_names=list(net_flows.keys()),
        net_flows=net_flows,
        centroids=centroids,
        obstacle_names=[s for s in SOURCES if s != src_name],
        alpha_deg=25.0,
    ))

B         = st._OPT_OVERLAP_B          # 1.5 deg
n_samples = 6
total_viz_flow = sum(t.total_flow for t in trees) or 1.0

# ── replicate compute_f_overlap with full instrumentation ────────────────────

tree_segs = []
for tree in trees:
    a_list, b_list = [], []
    for pts in tree.edge_polylines:
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        a_list.append(arr[:-1])
        b_list.append(arr[1:])
    tree_segs.append(
        (np.vstack(a_list), np.vstack(b_list)) if a_list else None
    )

n_deep   = 0     # D < t
n_buff   = 0     # t <= D <= t+B
n_zero   = 0     # D > t+B
n_total  = 0
t_vals   = []    # all t values seen (one per edge)
D_all    = []    # every scalar D value
cost_per_eval = []  # every non-zero kernel result
cost_total = 0.0

# For a concrete example: first deep-zone hit
example_printed = False

for ti, tree in enumerate(trees):
    nodes = tree.tree_nodes
    if tree.total_flow <= 0:
        continue
    for (child_id, _), pts in zip(tree.edges, tree.edge_polylines):
        if len(pts) < 2:
            continue
        c_node = nodes[child_id]
        if c_node.width <= 0:
            continue
        t = (c_node.width / 2.0) / st._OPT_M_PER_DEG

        arr    = np.asarray(pts, dtype=float)
        idx    = np.round(np.linspace(0, len(arr) - 1, n_samples)).astype(int)
        samples = arr[idx]

        t_vals.append(t)

        for tj, segs_j in enumerate(tree_segs):
            if ti == tj or segs_j is None:
                continue
            seg_a_j, seg_b_j = segs_j
            D_vec = st._pts_to_segs_min_dist(samples, seg_a_j, seg_b_j)

            for d_scalar in D_vec:
                n_total += 1
                D_all.append(float(d_scalar))
                if d_scalar < t:
                    n_deep += 1
                    k = st._f_obs_kernel(float(d_scalar), t, B)
                    cost_per_eval.append(k)
                    cost_total += k
                    if not example_printed:
                        d_s = max(d_scalar, 1e-9)
                        c1  = (t / (B * d_s)) * (B / 2 + t)
                        c2  = (d_scalar / (B * t)) * (B / 2 - t)
                        print("=" * 60)
                        print("FINDING 2 — concrete deep-zone example")
                        print("=" * 60)
                        print(f"  t (half-width deg) = {t:.6f} deg")
                        print(f"  B (buffer deg)     = {B:.4f} deg")
                        print(f"  D (dist deg)       = {d_scalar:.8f} deg")
                        print(f"  D in metres        = {d_scalar * st._OPT_M_PER_DEG:,.0f} m")
                        print(f"  t in metres        = {t * st._OPT_M_PER_DEG:,.0f} m")
                        print(f"  t/B ratio          = {t/B:.4f}")
                        print(f"  D/t ratio          = {d_scalar/t:.6f}")
                        print(f"  c1 = (t/(B*D))*(B/2+t) = {c1:.4f}")
                        print(f"  c2 = (D/(B*t))*(B/2-t) = {c2:.4f}")
                        print(f"  kernel value       = {c1+c2:.4f}")
                        example_printed = True
                elif d_scalar <= t + B:
                    n_buff += 1
                    k = st._f_obs_kernel(float(d_scalar), t, B)
                    cost_per_eval.append(k)
                    cost_total += k
                else:
                    n_zero += 1

# ── Finding 1: statistics ────────────────────────────────────────────────────
print()
print("=" * 60)
print("FINDING 1 — Distribution of D values")
print("=" * 60)
print(f"  Total (point, tree_j) evaluations : {n_total:,}")
print(f"  D < t   (deep overlap, high cost)  : {n_deep:,}   ({100*n_deep/max(n_total,1):.1f}%)")
print(f"  t<=D<=t+B (buffer zone, some cost) : {n_buff:,}   ({100*n_buff/max(n_total,1):.1f}%)")
print(f"  D > t+B  (zero cost)               : {n_zero:,}   ({100*n_zero/max(n_total,1):.1f}%)")
print()
t_arr = np.array(t_vals)
print(f"  t (half-width) — mean: {t_arr.mean():.6f} deg  "
      f"({t_arr.mean()*st._OPT_M_PER_DEG:,.0f} m)   "
      f"max: {t_arr.max():.6f} deg  ({t_arr.max()*st._OPT_M_PER_DEG:,.0f} m)")
print(f"  B (buffer)            : {B:.4f} deg  ({B*st._OPT_M_PER_DEG:,.0f} m)")
print(f"  t_mean / B            : {t_arr.mean()/B:.4f}   "
      f"({'t >> B — formula in wrong regime' if t_arr.mean() > B else 't < B — OK'})")
print(f"  t_max  / B            : {t_arr.max()/B:.4f}")
D_arr = np.array(D_all)
print()
print(f"  D — mean: {D_arr.mean():.6f} deg  ({D_arr.mean()*st._OPT_M_PER_DEG:,.0f} m)")
print(f"  D — min : {D_arr.min():.8f} deg  ({D_arr.min()*st._OPT_M_PER_DEG:,.0f} m)")
print(f"  D — max : {D_arr.max():.6f} deg  ({D_arr.max()*st._OPT_M_PER_DEG:,.0f} m)")

# ── Finding 3: t for root edge ───────────────────────────────────────────────
print()
print("=" * 60)
print("FINDING 3 — t for a root edge (width = MAX_DISPLAY_W_M)")
print("=" * 60)
t_root = (st.MAX_DISPLAY_W_M / 2.0) / st._OPT_M_PER_DEG
print(f"  MAX_DISPLAY_W_M = {st.MAX_DISPLAY_W_M:,.0f} m")
print(f"  half-width      = {st.MAX_DISPLAY_W_M/2:,.0f} m")
print(f"  t (root edge)   = {t_root:.4f} deg")
print(f"  B               = {B:.4f} deg")
print(f"  t/B             = {t_root/B:.4f}   "
      f"({'t >> B' if t_root > B else 't < B'})")
print(f"  When t >> B the formula is dominated by the (t/(B*D))*(B/2+t) term,")
print(f"  which diverges as D -> 0 and scales as t^2/B for D ~ t.")
print(f"  For D = t/2: kernel ~ (t/(B*t/2))*(B/2+t) = (2/B)*(B/2+t) = 1 + 2t/B = {1 + 2*t_root/B:.1f}")

# ── Finding 4: is 160M reasonable? ──────────────────────────────────────────
print()
print("=" * 60)
print("FINDING 4 — Is raw F_overlap ~160M physically reasonable?")
print("=" * 60)
print(f"  Raw F_overlap from compute_f_total: ~321,924,065 / 2.0 weight = ~160,962,032")
print(f"  n_total evaluations                : {n_total:,}")
print(f"  n_deep (D < t)                     : {n_deep:,}  ({100*n_deep/max(n_total,1):.1f}%)")
if n_deep > 0:
    cp = np.array(cost_per_eval[:n_deep])
    print(f"  kernel values in deep zone — mean  : {cp.mean():.2f}   max: {cp.max():.2f}")
    print(f"  Sum kernel (deep + buff) from audit: {cost_total:,.1f}")
    naive_per_eval = cost_total / max(n_deep + n_buff, 1)
    print(f"  Mean cost per non-zero eval        : {naive_per_eval:.4f}")
    print()
    # Expected: if t >> B, a typical deep hit gives ~(t/B)^2 ~ large
    # For t ~ 0.2 deg, B = 1.5: t/B ~ 0.13 -> kernel ~ 1-2; reasonable?
    # For t ~ 3 deg, B = 1.5: t/B ~ 2   -> kernel ~ 5-30; very large
    print(f"  t_max / B = {t_arr.max()/B:.2f}")
    print(f"  At t_max, typical deep-zone kernel value ~ {(1 + 2*t_arr.max()/B):.1f}")
    print(f"  With {n_total:,} evals and {100*n_deep/n_total:.0f}% deep, "
          f"expected total ~ {n_deep * (1 + 2*t_arr.max()/B):,.0f}")
    print()
    if cost_total > 1_000_000:
        print("  VERDICT: Raw value is ARTIFICIALLY INFLATED.")
        print("  Root cause: t >> B for thick edges (root/near-root edges).")
        print("  The formula (t/(B*D))*(B/2+t) with t>>B and D<<t gives values")
        print("  proportional to t^2/B, not a bounded [0,1] cost as intended.")
    else:
        print("  VERDICT: Raw value appears physically reasonable.")
else:
    print("  No deep-zone hits — all evaluations are in buffer or zero zone.")
    print(f"  Sum kernel from buffer zone: {cost_total:,.1f}")
    print(f"  VERDICT: Cost is driven entirely by buffer-zone evaluations.")
