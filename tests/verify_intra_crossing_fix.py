"""Verify intra-tree crossing guard in optimize_multi_tree."""
import os, sys, math, copy, random as _rnd, time, threading
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

# ── per-tree intra crossing count via count_crossings ───────────────────────
def intra_per_tree(trees):
    counts = []
    for tree in trees:
        _, intra = st.count_crossings([tree])
        counts.append(intra)
    return counts

# ── before ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Before optimization")
print("=" * 60)
before_intra = intra_per_tree(trees)
inter_before, intra_before_total = st.count_crossings(trees)
for ti, tree in enumerate(trees):
    print(f"  {tree.source_name}: {before_intra[ti]} intra-tree crossings")
print(f"  Total intra : {intra_before_total}")
print(f"  Total inter : {inter_before}")

# ── run optimizer via optimize_multi_tree (uses production code) ─────────────
print()
print("Running optimize_multi_tree for 2000 iterations...")

MAX_IT     = 2000
results    = {}
stop       = threading.Event()
cost_log   = {}
CHECKPOINTS = {1, 500, 1000, 2000}

intra_reverts   = [0]   # mutable counter — we instrument via on_update only for cost
accepted_total  = [0]

# Monkey-patch to count intra-crossing reverts we need a separate instrumented run.
# Instead: run the production optimizer for timing/cost, then re-run instrumented
# loop separately for revert count (same seed).

t0 = time.perf_counter()

def on_update(it, cost, snapshot):
    if it in CHECKPOINTS:
        cost_log[it] = cost
        print(f"  iter {it:>4d}: F_total = {cost:,.2f}")
    if it == -1:
        results['final_cost'] = cost
        results['final_trees'] = snapshot

st.optimize_multi_tree(
    trees, centroids, stop,
    on_update=on_update,
    max_iter=MAX_IT,
    update_every=1,   # capture every iteration for cost logging
)
wall_time = time.perf_counter() - t0

final_trees = results['final_trees']

# ── after ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("After optimization")
print("=" * 60)
after_intra = intra_per_tree(final_trees)
inter_after, intra_after_total = st.count_crossings(final_trees)
for ti, tree in enumerate(final_trees):
    delta = after_intra[ti] - before_intra[ti]
    tag   = f"(+{delta})" if delta > 0 else f"({delta})" if delta < 0 else "(unchanged)"
    print(f"  {tree.source_name}: {after_intra[ti]} intra-tree crossings  {tag}")
print(f"  Total intra : {intra_after_total}")
print(f"  Total inter : {inter_after}")

# ── convergence ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Cost progression")
print("=" * 60)
f0 = st.compute_f_total(trees, centroids, 25.0,
                         st.DEFAULT_F_TOTAL_WEIGHTS, st._OPT_OVERLAP_B, 6)
ff = results['final_cost']
print(f"  F_total start : {f0:,.2f}")
print(f"  F_total end   : {ff:,.2f}")
print(f"  Reduction     : {f0-ff:+,.2f}  ({100*(f0-ff)/f0:.2f}%)")

# ── timing ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Timing")
print("=" * 60)
print(f"  Wall time for {MAX_IT} iterations: {wall_time:.1f}s")
print(f"  Per iteration: {1000*wall_time/MAX_IT:.1f}ms")

# ── revert count — instrumented re-run with same seed ────────────────────────
# Re-run a copy of the loop with explicit intra-crossing revert tracking.
print()
print("=" * 60)
print("Intra-crossing revert count (instrumented re-run, seed=42)")
print("=" * 60)

import numpy as np

state2 = copy.deepcopy(trees)
w      = dict(st.DEFAULT_F_TOTAL_WEIGHTS)
src_xy = {}
for ti, tree in enumerate(state2):
    lon, lat   = centroids[tree.source_name]
    src_xy[ti] = st._to3035(lon, lat)

steiner_pool = [(ti, nid)
    for ti, tree in enumerate(state2)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner]

def _fc2():
    return st.compute_f_total(state2, centroids, 25.0, w, st._OPT_OVERLAP_B, 6)

current_cost = _fc2()
T_init = (0.01 * current_cost) / math.log(1.0 / 0.25)
T_min  = 1e-3
alpha  = (T_min / T_init) ** (1.0 / MAX_IT)
T      = T_init
rng    = _rnd.Random(42)

n_metropolis_accepted = 0
n_intra_reverted      = 0

for it in range(1, MAX_IT + 1):
    ti, nid = rng.choice(steiner_pool)
    tree    = state2[ti]
    node    = tree.tree_nodes[nid]
    sx, sy  = src_xy[ti]

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
        T *= alpha; continue

    if not is_subdivision:
        R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
        par            = tree.tree_nodes[node.parent] if node.parent is not None else None
        R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
        if R_parent_min >= R_children_min:
            T *= alpha; continue
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

    new_cost = _fc2()
    delta    = new_cost - current_cost

    if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
        n_metropolis_accepted += 1
        # Intra-crossing guard (same logic as production code)
        incident_idx = set(orig_edges.keys())
        non_incident = [
            np.asarray(tree.edge_polylines[i], dtype=float)
            for i in range(len(tree.edges))
            if i not in incident_idx and len(tree.edge_polylines[i]) >= 2
        ]
        new_intra = False
        for ai in incident_idx:
            arr_a = np.asarray(tree.edge_polylines[ai], dtype=float)
            if len(arr_a) < 2:
                continue
            for arr_b in non_incident:
                rows, _ = st._seg_cross_batch(arr_a, arr_b)
                if len(rows):
                    orig_arr_a = np.asarray(orig_edges[ai], dtype=float)
                    rows_before, _ = st._seg_cross_batch(orig_arr_a, arr_b)
                    if len(rows) > len(rows_before):
                        new_intra = True
                        break
            if new_intra:
                break
        if new_intra:
            n_intra_reverted += 1
            node.x, node.y, node.R, node.phi = orig_pos
            for i, pts in orig_edges.items():
                tree.edge_polylines[i] = pts
        else:
            current_cost = new_cost
    else:
        node.x, node.y, node.R, node.phi = orig_pos
        for i, pts in orig_edges.items():
            tree.edge_polylines[i] = pts

    T *= alpha

pct = 100 * n_intra_reverted / max(n_metropolis_accepted, 1)
print(f"  Metropolis-accepted moves       : {n_metropolis_accepted}")
print(f"  Reverted by intra-crossing guard: {n_intra_reverted}  ({pct:.1f}% of accepted)")
print(f"  Committed moves                 : {n_metropolis_accepted - n_intra_reverted}")
