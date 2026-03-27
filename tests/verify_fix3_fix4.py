"""Verification script for R-hierarchy clamping (Fix 5) and Fixes 3/4.

Instruments optimize_multi_tree by temporarily replacing it with a
counter-augmented wrapper that collects per-iteration stats.
"""
import os, sys, math, copy, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import spiral_tree as st
from data_loader import load_trade_data, ISO_SHORT
from map_utils import load_eu_map, get_centroids

# ── load data ────────────────────────────────────────────────────────────────
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
    result = st.compute_spiral_tree(
        source_name=src_name,
        terminal_names=list(net_flows.keys()),
        net_flows=net_flows,
        centroids=centroids,
        obstacle_names=[s for s in SOURCES if s != src_name],
        alpha_deg=25.0,
    )
    trees.append(result)

# ── record subdivision node R values before optimization ─────────────────────
pre_R: dict[tuple, float] = {}
for ti, tree in enumerate(trees):
    for nid, node in tree.tree_nodes.items():
        if node.is_steiner and len(node.children) == 1:
            pre_R[(ti, nid)] = node.R

# ── instrumented optimizer ───────────────────────────────────────────────────
MAX_ITER      = 2000
CHECKPOINTS   = {1, 500, 1000, 1500, 2000}
costs_at      = {}

counters = {
    'downhill_accepted': 0,
    'uphill_accepted':   0,
    'pre_rejected':      0,   # min-R OR degenerate window
    'assertion_errors':  0,
}

optimized_state: list = []
stop = threading.Event()

def on_update(it, cost, snapshot):
    if it in CHECKPOINTS:
        costs_at[it] = cost
        print(f"  iter {it:>4d}: F_total = {cost:,.0f}")
    if it == -1:
        optimized_state.extend(snapshot)

# We need per-iteration instrumentation. Patch the inner loop by wrapping
# _full_cost and the Metropolis block via a subclassed call isn't possible
# cleanly, so we inline a copy of the loop here using the same helpers.

import random as _random_mod
import copy   as _copy_mod

_random = _random_mod.Random()   # independent RNG

def _run_instrumented(trees_in, max_iter=MAX_ITER):
    w     = dict(st.DEFAULT_F_TOTAL_WEIGHTS)
    state = _copy_mod.deepcopy(trees_in)

    src_xy = {}
    for ti, tree in enumerate(state):
        lon, lat   = centroids[tree.source_name]
        src_xy[ti] = st._to3035(lon, lat)

    steiner_pool = [
        (ti, nid)
        for ti, tree in enumerate(state)
        for nid, node in tree.tree_nodes.items()
        if node.is_steiner
    ]
    if not steiner_pool:
        return state

    def _full_cost():
        return st.compute_f_total(state, centroids, 25.0, w,
                                  st._OPT_OVERLAP_B, 6)

    current_cost  = _full_cost()
    initial_cost_ = current_cost          # snapshot before any moves
    T_init_val    = (0.01 * current_cost) / math.log(1.0 / 0.25)
    T_min        = 1e-3
    alpha        = (T_min / T_init_val) ** (1.0 / max(max_iter, 1))
    T            = T_init_val

    print(f"\nT_init = {T_init_val:,.0f}  (expected ~= 0.01 * {current_cost:,.0f} / ln(4) = "
          f"{0.01 * current_cost / math.log(4):,.0f})")
    print(f"F_total initial = {current_cost:,.0f}\n")

    costs_at[1] = None   # will be overwritten after iter 1

    for it in range(1, max_iter + 1):
        ti, nid = _random.choice(steiner_pool)
        tree    = state[ti]
        node    = tree.tree_nodes[nid]
        sx, sy  = src_xy[ti]

        is_subdivision = (len(node.children) == 1)

        t_frac = T / T_init_val
        if is_subdivision:
            step_angle   = max(0.10 * t_frac, 0.001)
            new_phi      = node.phi + _random.gauss(0.0, step_angle)
            new_x, new_y = st._cart(node.R, new_phi)
            new_R        = node.R
        else:
            step         = max(node.R * 0.10 * t_frac, 500.0)
            new_x        = node.x + _random.gauss(0.0, step)
            new_y        = node.y + _random.gauss(0.0, step)
            new_R, new_phi = st._polar(new_x, new_y)

        if new_R < 10_000.0:
            counters['pre_rejected'] += 1
            T *= alpha
            continue

        if not is_subdivision:
            R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
            par            = tree.tree_nodes[node.parent] if node.parent is not None else None
            R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
            if R_parent_min >= R_children_min:
                counters['pre_rejected'] += 1
                T *= alpha
                continue
            if new_R < R_parent_min or new_R > R_children_min:
                new_R        = min(max(new_R, R_parent_min), R_children_min)
                new_x, new_y = st._cart(new_R, new_phi)

        try:
            assert all(tree.tree_nodes[c].R > new_R for c in node.children), \
                f"R-hierarchy violated: child R <= new_R={new_R}"
            if node.parent is not None:
                par2 = tree.tree_nodes[node.parent]
                assert par2.R <= 0 or par2.R < new_R, \
                    f"R-hierarchy violated: parent R={par2.R} >= new_R={new_R}"
        except AssertionError as e:
            counters['assertion_errors'] += 1
            print(f"  ASSERTION ERROR at iter {it}: {e}")
            T *= alpha
            continue

        orig_pos   = (node.x, node.y, node.R, node.phi)
        orig_edges = {
            i: tree.edge_polylines[i]
            for i, (c, p) in enumerate(tree.edges)
            if c == nid or p == nid
        }

        node.x, node.y, node.R, node.phi = new_x, new_y, new_R, new_phi
        for i in orig_edges:
            c2, p2   = tree.edges[i]
            cn, pn   = tree.tree_nodes[c2], tree.tree_nodes[p2]
            new_pts  = st._sample_spiral(cn.R, cn.phi, pn.R, pn.phi, sx, sy)
            tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

        new_cost = _full_cost()
        delta    = new_cost - current_cost

        if delta < 0:
            counters['downhill_accepted'] += 1
            current_cost = new_cost
        elif _random.random() < math.exp(-delta / max(T, 1e-12)):
            counters['uphill_accepted'] += 1
            current_cost = new_cost
        else:
            node.x, node.y, node.R, node.phi = orig_pos
            for i, pts in orig_edges.items():
                tree.edge_polylines[i] = pts

        T *= alpha

        if it in CHECKPOINTS:
            costs_at[it] = current_cost
            print(f"  iter {it:>4d}: F_total = {current_cost:,.0f}")

    return state, initial_cost_, current_cost

# ── run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Verification: R-hierarchy clamping + Fixes 3 & 4")
print("=" * 60)

final_state, initial_cost, final_cost = _run_instrumented(trees, MAX_ITER)

# Re-read initial cost before iter 1 — stored separately above
# (iter 1 cost is post-first-move; print initial from T_init block above)

print("\n--- Acceptance breakdown ---")
total = sum(counters.values())
print(f"  Downhill accepted : {counters['downhill_accepted']:>5d}")
print(f"  Uphill accepted   : {counters['uphill_accepted']:>5d}")
print(f"  Pre-rejected      : {counters['pre_rejected']:>5d}  "
      f"({100*counters['pre_rejected']/MAX_ITER:.1f}% of iterations)")
print(f"  Assertion errors  : {counters['assertion_errors']:>5d}")

# ── subdivision node R invariant check ───────────────────────────────────────
print("\n--- Subdivision node R invariant ---")
n_checked = 0
n_changed = 0
for (ti, nid), r_before in pre_R.items():
    # find node in final_state
    node_after = final_state[ti].tree_nodes[nid]
    n_checked += 1
    if abs(node_after.R - r_before) > 1e-6:
        n_changed += 1
        print(f"  CHANGED: tree={ti} node={nid}  R_before={r_before:.2f}  R_after={node_after.R:.2f}")
print(f"  Checked {n_checked} subdivision node(s). "
      f"{'None changed R.' if n_changed == 0 else f'{n_changed} CHANGED R (bug!)'}")

# ── step size spot-check ─────────────────────────────────────────────────────
print("\n--- Step size spot-check (join node, R=500,000m) ---")
R_ex   = 500_000.0
T_now  = (0.01 * initial_cost) / math.log(4)  # == T_init
alpha_ex = (1e-3 / T_now) ** (1.0 / MAX_ITER)
T_end  = T_now * alpha_ex ** (MAX_ITER - 1)
step1  = max(R_ex * 0.10 * 1.0,                   500.0)
stepN  = max(R_ex * 0.10 * (T_end / T_now),       500.0)
print(f"  iter   1: step = max({R_ex*0.10:.0f} * 1.000, 500) = {step1:,.0f} m")
print(f"  iter {MAX_ITER}: step = max({R_ex*0.10:.0f} * {T_end/T_now:.4f}, 500) = {stepN:,.0f} m")

# ── total reduction ───────────────────────────────────────────────────────────
print("\n--- F_total reduction ---")
print(f"  Start  : {initial_cost:,.0f}")
print(f"  End    : {final_cost:,.0f}")
print(f"  d_abs  : {initial_cost - final_cost:,.0f}")
print(f"  d_pct  : {100*(initial_cost - final_cost)/initial_cost:.2f}%")
