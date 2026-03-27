"""Full verification after shared-leaf exclusion fix in compute_f_overlap."""
import os, sys, math, copy, random as _rnd
import numpy as np
from collections import Counter
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

def build_trees():
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
    return trees

trees = build_trees()
w     = dict(st.DEFAULT_F_TOTAL_WEIGHTS)
B     = st._OPT_OVERLAP_B
n_s   = 6
total_viz_flow = sum(t.total_flow for t in trees) or 1.0

def full_cost(state):
    return st.compute_f_total(state, centroids, 25.0, w, st._OPT_OVERLAP_B, n_s)

# ── Finding 1 & 2: new F_overlap value and D distribution ───────────────────
print("=" * 60)
print("Finding 1 & 2: F_overlap and D distribution after fix")
print("=" * 60)

# Replicate compute_f_overlap internals with instrumentation
country_count = Counter(
    node.country
    for tree in trees
    for node in tree.tree_nodes.values()
    if node.is_leaf and node.country
)
shared_lonlat = np.array(
    [centroids[c] for c, cnt in country_count.items() if cnt >= 2 and c in centroids],
    dtype=float,
)
print(f"  Shared leaf countries: {sum(1 for c,n in country_count.items() if n>=2)} "
      f"({[c for c,n in country_count.items() if n>=2]})")
print(f"  shared_lonlat shape: {shared_lonlat.shape}")

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

n_deep = n_buff = n_zero = n_total = n_excluded = 0
cost_overlap = 0.0

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

        arr     = np.asarray(pts, dtype=float)
        idx     = np.round(np.linspace(0, len(arr)-1, n_s)).astype(int)
        samples = arr[idx]

        if shared_lonlat.shape[0] > 0:
            dts = np.min(np.hypot(
                samples[:, 0:1] - shared_lonlat[:, 0],
                samples[:, 1:2] - shared_lonlat[:, 1]), axis=1)
            excl = (dts < t)
            n_excluded += int(excl.sum())
            samples = samples[~excl]
        if len(samples) == 0:
            continue

        for tj, segs_j in enumerate(tree_segs):
            if ti == tj or segs_j is None:
                continue
            seg_a_j, seg_b_j = segs_j
            D_vec = st._pts_to_segs_min_dist(samples, seg_a_j, seg_b_j)

            for d in D_vec:
                n_total += 1
                if d < t:
                    n_deep += 1
                    cost_overlap += float(st._f_obs_kernel(float(d), t, B))
                elif d <= t + B:
                    n_buff += 1
                    cost_overlap += float(st._f_obs_kernel(float(d), t, B))
                else:
                    n_zero += 1

print(f"\n  Sample points excluded (within t of shared leaf): {n_excluded}")
print(f"  Remaining evaluations                            : {n_total}")
print(f"  D < t   (deep, high cost) : {n_deep:>4d}  ({100*n_deep/max(n_total,1):.1f}%)")
print(f"  t<=D<=t+B (buffer)        : {n_buff:>4d}  ({100*n_buff/max(n_total,1):.1f}%)")
print(f"  D > t+B  (zero)           : {n_zero:>4d}  ({100*n_zero/max(n_total,1):.1f}%)")
print(f"\n  Raw F_overlap (instrumented): {cost_overlap:,.2f}")
official = st.compute_f_overlap(trees, total_viz_flow, centroids, B=B, n_samples=n_s)
print(f"  Raw F_overlap (official fn) : {official:,.2f}")

# ── Finding 3: new F_total breakdown ────────────────────────────────────────
print()
print("=" * 60)
print("Finding 3: F_total term breakdown after fix")
print("=" * 60)
TERM_KEYS = ['c_obs', 'c_S', 'c_AR', 'c_str', 'c_cross', 'c_overlap']
f_total = full_cost(trees)
print(f"  F_total = {f_total:,.2f}")
for key in TERM_KEYS:
    wz = {k: 0.0 for k in w}; wz[key] = 1.0
    raw = st.compute_f_total(trees, centroids, 25.0, wz, st._OPT_OVERLAP_B, n_s)
    weighted = raw * w[key]
    pct = 100 * weighted / f_total if f_total > 0 else 0
    print(f"  {key:<12s}: raw={raw:>12,.2f}  weighted={weighted:>12,.2f}  ({pct:.2f}%)")

# ── Finding 4: sensitivity test — move one join node 10,000m ────────────────
print()
print("=" * 60)
print("Finding 4: Sensitivity — move one join node 10,000m in X")
print("=" * 60)

state4 = copy.deepcopy(trees)
germany_tree = state4[0]
join_node_id = None
for nid, node in germany_tree.tree_nodes.items():
    if node.is_steiner and len(node.children) >= 2:
        join_node_id = nid; join_node = node; break

if join_node_id is None:
    print("  No join node found.")
else:
    src_x, src_y = st._to3035(*centroids['Germany'])
    cost_before = full_cost(state4)

    orig_pos   = (join_node.x, join_node.y, join_node.R, join_node.phi)
    orig_edges = {i: germany_tree.edge_polylines[i]
                  for i, (c, p) in enumerate(germany_tree.edges)
                  if c == join_node_id or p == join_node_id}

    new_x = join_node.x + 10_000.0
    new_y = join_node.y
    new_R, new_phi = st._polar(new_x, new_y)
    join_node.x, join_node.y, join_node.R, join_node.phi = new_x, new_y, new_R, new_phi
    for i in orig_edges:
        c2, p2 = germany_tree.edges[i]
        cn, pn = germany_tree.tree_nodes[c2], germany_tree.tree_nodes[p2]
        new_pts = st._sample_spiral(cn.R, cn.phi, pn.R, pn.phi, src_x, src_y)
        germany_tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

    cost_after = full_cost(state4)
    print(f"  F_total before: {cost_before:,.2f}")
    print(f"  F_total after : {cost_after:,.2f}")
    print(f"  Total delta   : {cost_after - cost_before:+,.2f}")
    print()
    print(f"  Per-term breakdown:")
    for key in TERM_KEYS:
        wz = {k: 0.0 for k in w}; wz[key] = 1.0
        r_before = st.compute_f_total(trees,  centroids, 25.0, wz, st._OPT_OVERLAP_B, n_s)
        r_after  = st.compute_f_total(state4, centroids, 25.0, wz, st._OPT_OVERLAP_B, n_s)
        delta    = (r_after - r_before) * w[key]
        print(f"    {key:<12s}: delta={delta:+,.4f}")

    # revert
    join_node.x, join_node.y, join_node.R, join_node.phi = orig_pos
    for i, pts in orig_edges.items():
        germany_tree.edge_polylines[i] = pts

# ── Finding 5: 2000-iteration optimizer run ──────────────────────────────────
print()
print("=" * 60)
print("Finding 5: 2000-iteration optimizer cost progression")
print("=" * 60)

state5 = copy.deepcopy(trees)
src_xy5 = {}
for ti, tree in enumerate(state5):
    lon, lat = centroids[tree.source_name]
    src_xy5[ti] = st._to3035(lon, lat)

steiner_pool = [(ti, nid)
    for ti, tree in enumerate(state5)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner]

def _fc5():
    return st.compute_f_total(state5, centroids, 25.0, w, st._OPT_OVERLAP_B, n_s)

current_cost = _fc5()
T_init = (0.01 * current_cost) / math.log(1.0 / 0.25)
T_min  = 1e-3
MAX_IT = 2000
alpha  = (T_min / T_init) ** (1.0 / MAX_IT)
T      = T_init
rng    = _rnd.Random(42)

print(f"  T_init = {T_init:,.2f}  (F_total initial = {current_cost:,.2f})")

CHECKPOINTS = {1, 500, 1000, 1500, 2000}
accepted_down = accepted_up = rejected = 0

for it in range(1, MAX_IT + 1):
    ti, nid = rng.choice(steiner_pool)
    tree    = state5[ti]
    node    = tree.tree_nodes[nid]
    sx, sy  = src_xy5[ti]

    is_subdivision = (len(node.children) == 1)
    t_frac = T / T_init

    if is_subdivision:
        step_angle   = max(0.10 * t_frac, 0.001)
        new_phi      = node.phi + rng.gauss(0.0, step_angle)
        new_x, new_y = st._cart(node.R, new_phi)
        new_R        = node.R
    else:
        step         = max(node.R * 0.10 * t_frac, 500.0)
        new_x        = node.x + rng.gauss(0.0, step)
        new_y        = node.y + rng.gauss(0.0, step)
        new_R, new_phi = st._polar(new_x, new_y)

    if new_R < 10_000.0:
        rejected += 1; T *= alpha; continue

    if not is_subdivision:
        R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
        par            = tree.tree_nodes[node.parent] if node.parent is not None else None
        R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
        if R_parent_min >= R_children_min:
            rejected += 1; T *= alpha; continue
        if new_R < R_parent_min or new_R > R_children_min:
            new_R        = min(max(new_R, R_parent_min), R_children_min)
            new_x, new_y = st._cart(new_R, new_phi)

    orig_pos   = (node.x, node.y, node.R, node.phi)
    orig_edges = {i: tree.edge_polylines[i]
                  for i, (c2, p2) in enumerate(tree.edges)
                  if c2 == nid or p2 == nid}

    node.x, node.y, node.R, node.phi = new_x, new_y, new_R, new_phi
    for i in orig_edges:
        c2, p2 = tree.edges[i]
        cn, pn = tree.tree_nodes[c2], tree.tree_nodes[p2]
        new_pts = st._sample_spiral(cn.R, cn.phi, pn.R, pn.phi, sx, sy)
        tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

    new_cost = _fc5()
    delta    = new_cost - current_cost

    if delta < 0:
        accepted_down += 1; current_cost = new_cost
    elif rng.random() < math.exp(-delta / max(T, 1e-12)):
        accepted_up += 1; current_cost = new_cost
    else:
        node.x, node.y, node.R, node.phi = orig_pos
        for i, pts in orig_edges.items():
            tree.edge_polylines[i] = pts

    T *= alpha

    if it in CHECKPOINTS:
        print(f"  iter {it:>4d}: F_total = {current_cost:,.2f}")

print(f"\n  Accepted downhill : {accepted_down}")
print(f"  Accepted uphill   : {accepted_up}")
print(f"  Pre-rejected      : {rejected}")
f0 = st.compute_f_total(trees, centroids, 25.0, w, st._OPT_OVERLAP_B, n_s)
print(f"\n  F_total start : {f0:,.2f}")
print(f"  F_total end   : {current_cost:,.2f}")
print(f"  Reduction     : {f0 - current_cost:+,.2f}  ({100*(f0-current_cost)/f0:.3f}%)")
