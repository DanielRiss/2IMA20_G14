"""Diagnose upward cost drift in the optimizer.

Finding 2: controlled single-node move test.
Finding 3: isolated per-term cost breakdown at start and end of 2000 iterations.
"""
import os, sys, math, copy, random as _rnd
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
        result = st.compute_spiral_tree(
            source_name=src_name,
            terminal_names=list(net_flows.keys()),
            net_flows=net_flows,
            centroids=centroids,
            obstacle_names=[s for s in SOURCES if s != src_name],
            alpha_deg=25.0,
        )
        trees.append(result)
    return trees

trees = build_trees()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: isolated cost for one term
# ─────────────────────────────────────────────────────────────────────────────
TERM_KEYS = ['c_obs', 'c_S', 'c_AR', 'c_str', 'c_cross', 'c_overlap']

def cost_breakdown(state):
    results = {}
    for key in TERM_KEYS:
        w = {k: 0.0 for k in st.DEFAULT_F_TOTAL_WEIGHTS}
        w[key] = 1.0
        results[key] = st.compute_f_total(state, centroids, 25.0, w,
                                          st._OPT_OVERLAP_B, 6)
    results['total'] = st.compute_f_total(state, centroids, 25.0,
                                          st.DEFAULT_F_TOTAL_WEIGHTS,
                                          st._OPT_OVERLAP_B, 6)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# FINDING 2: controlled single-node move
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Finding 2: Controlled single-node move")
print("=" * 60)

state2 = copy.deepcopy(trees)

# Find first join node in Germany tree (ti=0)
germany_tree = state2[0]
join_node_id = None
join_node    = None
for nid, node in germany_tree.tree_nodes.items():
    if node.is_steiner and len(node.children) >= 2:
        join_node_id = nid
        join_node    = node
        break

if join_node is None:
    print("  No join node found in Germany tree.")
else:
    src_x, src_y = st._to3035(*centroids['Germany'])
    cost_before = st.compute_f_total(state2, centroids, 25.0,
                                     st.DEFAULT_F_TOTAL_WEIGHTS,
                                     st._OPT_OVERLAP_B, 6)

    print(f"  Node {join_node_id}: x={join_node.x:.0f}  y={join_node.y:.0f}  "
          f"R={join_node.R:.0f}  phi={join_node.phi:.4f}")
    print(f"  F_total BEFORE move: {cost_before:,.0f}")

    # Move +10,000m in x (absolute, not relative to source)
    orig_pos   = (join_node.x, join_node.y, join_node.R, join_node.phi)
    orig_edges = {
        i: germany_tree.edge_polylines[i]
        for i, (c, p) in enumerate(germany_tree.edges)
        if c == join_node_id or p == join_node_id
    }
    new_x = join_node.x + 10_000.0
    new_y = join_node.y
    new_R, new_phi = st._polar(new_x, new_y)

    join_node.x, join_node.y, join_node.R, join_node.phi = new_x, new_y, new_R, new_phi
    for i in orig_edges:
        c2, p2 = germany_tree.edges[i]
        cn, pn = germany_tree.tree_nodes[c2], germany_tree.tree_nodes[p2]
        new_pts = st._sample_spiral(cn.R, cn.phi, pn.R, pn.phi, src_x, src_y)
        germany_tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

    cost_after = st.compute_f_total(state2, centroids, 25.0,
                                    st.DEFAULT_F_TOTAL_WEIGHTS,
                                    st._OPT_OVERLAP_B, 6)

    delta = cost_after - cost_before
    print(f"  F_total AFTER  move: {cost_after:,.0f}")
    print(f"  Delta: {delta:+,.0f}  ({'up' if delta > 0 else 'DOWN'})")

    # Per-term breakdown of the delta
    print("\n  Per-term breakdown (after - before):")
    bd_before = cost_breakdown(copy.deepcopy(trees))  # use fresh copy for before
    bd_after  = cost_breakdown(state2)
    for key in TERM_KEYS + ['total']:
        d = bd_after[key] - bd_before[key]
        print(f"    {key:<12s}: before={bd_before[key]:>15,.1f}  "
              f"after={bd_after[key]:>15,.1f}  delta={d:+,.1f}")

    # Revert
    join_node.x, join_node.y, join_node.R, join_node.phi = orig_pos
    for i, pts in orig_edges.items():
        germany_tree.edge_polylines[i] = pts

# ─────────────────────────────────────────────────────────────────────────────
# FINDING 3: per-term drift over 2000 iterations
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Finding 3: Per-term cost at iteration 0, 1000, and 2000")
print("=" * 60)

state3 = copy.deepcopy(trees)
w      = dict(st.DEFAULT_F_TOTAL_WEIGHTS)

src_xy = {}
for ti, tree in enumerate(state3):
    lon, lat   = centroids[tree.source_name]
    src_xy[ti] = st._to3035(lon, lat)

steiner_pool = [
    (ti, nid)
    for ti, tree in enumerate(state3)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner
]

def _full_cost3():
    return st.compute_f_total(state3, centroids, 25.0, w, st._OPT_OVERLAP_B, 6)

current_cost = _full_cost3()
T_init  = (0.01 * current_cost) / math.log(1.0 / 0.25)
T_min   = 1e-3
MAX_IT  = 2000
alpha   = (T_min / T_init) ** (1.0 / MAX_IT)
T       = T_init
rng     = _rnd.Random(42)

snapshots = {}   # iter -> deepcopy of state3

def record(label):
    bd = cost_breakdown(copy.deepcopy(state3))
    snapshots[label] = bd
    print(f"\n  [{label}]  F_total = {bd['total']:,.0f}")
    for key in TERM_KEYS:
        wt = w.get(key, 0.0)
        print(f"    {key:<12s}: raw={bd[key]:>15,.1f}   weighted={bd[key]*wt:>15,.1f}")

record('iter=0')

for it in range(1, MAX_IT + 1):
    ti, nid = rng.choice(steiner_pool)
    tree    = state3[ti]
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
        T *= alpha
        continue

    if not is_subdivision:
        R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
        par            = tree.tree_nodes[node.parent] if node.parent is not None else None
        R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
        if R_parent_min >= R_children_min:
            T *= alpha
            continue
        if new_R < R_parent_min or new_R > R_children_min:
            new_R        = min(max(new_R, R_parent_min), R_children_min)
            new_x, new_y = st._cart(new_R, new_phi)

    orig_pos   = (node.x, node.y, node.R, node.phi)
    orig_edges = {
        i: tree.edge_polylines[i]
        for i, (c2, p2) in enumerate(tree.edges)
        if c2 == nid or p2 == nid
    }
    node.x, node.y, node.R, node.phi = new_x, new_y, new_R, new_phi
    for i in orig_edges:
        c2, p2 = tree.edges[i]
        cn, pn = tree.tree_nodes[c2], tree.tree_nodes[p2]
        new_pts = st._sample_spiral(cn.R, cn.phi, pn.R, pn.phi, sx, sy)
        tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

    new_cost = _full_cost3()
    delta    = new_cost - current_cost

    if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
        current_cost = new_cost
    else:
        node.x, node.y, node.R, node.phi = orig_pos
        for i, pts in orig_edges.items():
            tree.edge_polylines[i] = pts

    T *= alpha

    if it == 1000:
        record('iter=1000')

record('iter=2000')

# Summary of drift per term
print()
print("=" * 60)
print("Term-level drift (iter=2000 raw value - iter=0 raw value):")
print("=" * 60)
for key in TERM_KEYS + ['total']:
    d = snapshots['iter=2000'][key] - snapshots['iter=0'][key]
    sign = "UP  " if d > 0 else "down"
    print(f"  {key:<12s}: {d:>+15,.1f}   {sign}")
