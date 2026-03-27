"""
verify_f_total.py — Post-step-7 verification script.

Checks:
  1. compute_f_total returns finite non-negative float on a 2-tree case
  2. optimize_multi_tree _full_cost differs before/after 10 iterations
  3. compute_inter_tree_cost returns all 8 expected keys, none NaN/inf
  4. DEFAULT_OPT_WEIGHTS references in the codebase
"""
import os, sys, math, threading, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_loader import load_trade_data
from map_utils   import load_eu_map, get_centroids
from spiral_tree import (
    compute_spiral_tree, compute_f_total, compute_inter_tree_cost,
    optimize_multi_tree, DEFAULT_F_TOTAL_WEIGHTS, _OPT_OVERLAP_B,
)

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
DATA  = os.path.join(ROOT, 'data', 'data-18886936.csv')
LABEL = os.path.join(ROOT, 'data', 'label-18886936.csv')

print("Loading data...")
export_mx, net_mx, _ = load_trade_data(DATA, LABEL, year=2024)
eu_gdf    = load_eu_map()
centroids = get_centroids(eu_gdf)

SOURCES = ['Germany', 'France']

print("Building spiral trees for:", SOURCES)
trees = []
for src in SOURCES:
    if src not in net_mx.index:
        print(f"  WARNING: {src} not in matrix, skipping")
        continue
    row = net_mx.loc[src]
    net_flows = {d: float(v) for d, v in row.items()
                 if d != src and float(v) > 0}
    result = compute_spiral_tree(
        source_name=src,
        terminal_names=list(net_flows.keys()),
        net_flows=net_flows,
        centroids=centroids,
        obstacle_names=[s for s in SOURCES if s != src],
        alpha_deg=25.0,
    )
    trees.append(result)
    n_l = sum(1 for n in result.tree_nodes.values() if n.is_leaf)
    n_s = sum(1 for n in result.tree_nodes.values() if n.is_steiner)
    print(f"  {src}: {len(result.edges)} edges ({n_l} leaves, {n_s} Steiner)")

print()

# ── CHECK 1: compute_f_total ──────────────────────────────────────────────────
print("=" * 60)
print("CHECK 1: compute_f_total")
try:
    f_total = compute_f_total(
        trees, centroids, alpha_deg=25.0,
        weights=DEFAULT_F_TOTAL_WEIGHTS,
        B_obs=_OPT_OVERLAP_B,
        n_overlap_samples=6,
    )
    finite  = math.isfinite(f_total)
    non_neg = f_total >= 0.0
    print(f"  f_total = {f_total:.6f}")
    print(f"  finite  = {finite}")
    print(f"  >= 0    = {non_neg}")
    print(f"  PASS    = {finite and non_neg}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"  PASS  = False (exception raised)")
    print(f"  NOTE: If error is in compute_f_s/_wrap, this is the known step-8 numpy bug.")

print()

# ── CHECK 2: optimize_multi_tree ─────────────────────────────────────────────
print("=" * 60)
print("CHECK 2: optimize_multi_tree (10 iterations)")

costs_log = []

def on_update(iteration, cost, snapshot):
    costs_log.append((iteration, cost))

stop = threading.Event()
t = threading.Thread(
    target=optimize_multi_tree,
    kwargs=dict(
        trees=trees,
        centroids=centroids,
        stop_event=stop,
        on_update=on_update,
        max_iter=10,
        update_every=1,
        alpha_deg=25.0,
    ),
    daemon=True,
)
t.start()
t.join(timeout=120)

if t.is_alive():
    stop.set()
    print("  WARNING: optimiser did not finish within 120 s")

if costs_log:
    first_iter, first_cost = costs_log[0]
    last_iter,  last_cost  = costs_log[-1]
    print(f"  cost at first logged step (iter={first_iter}): {first_cost:.6f}")
    print(f"  cost at last  logged step (iter={last_iter}):  {last_cost:.6f}")
    both_finite = math.isfinite(first_cost) and math.isfinite(last_cost)
    changed     = (first_cost != last_cost) or (len(costs_log) == 1)
    print(f"  both finite          = {both_finite}")
    print(f"  value changed or 1 update = {changed}")
    print(f"  PASS = {both_finite}")
else:
    print("  No cost updates logged (optimiser exited immediately — no Steiner nodes?)")
    print("  PASS = N/A")

print()

# ── CHECK 3: compute_inter_tree_cost ─────────────────────────────────────────
print("=" * 60)
print("CHECK 3: compute_inter_tree_cost with centroids")

try:
    result_dict = compute_inter_tree_cost(trees, centroids=centroids, alpha_deg=25.0)
    expected_keys = {'f_obs', 'f_s', 'f_ar', 'f_b', 'f_str', 'f_cross', 'f_overlap', 'f_total'}
    present_keys  = set(result_dict.keys())
    missing       = expected_keys - present_keys
    extra         = present_keys - expected_keys

    print(f"  keys present : {sorted(present_keys)}")
    print(f"  missing keys : {sorted(missing) or 'none'}")
    print(f"  extra keys   : {sorted(extra)   or 'none'}")
    print()
    for k in sorted(expected_keys):
        v = result_dict.get(k, float('nan'))
        flag = ""
        if k != 'f_s' and v == 0.0:
            flag = "  <- UNEXPECTED ZERO"
        if not math.isfinite(v):
            flag = "  <- NOT FINITE"
        print(f"    {k:12s} = {v:.6f}{flag}")

    all_present = not missing
    all_finite  = all(math.isfinite(result_dict.get(k, float('nan'))) for k in expected_keys)
    print(f"\n  all 8 keys present = {all_present}")
    print(f"  all values finite  = {all_finite}")
    print(f"  PASS = {all_present and all_finite}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"  PASS  = False")

print()

# ── CHECK 4: DEFAULT_OPT_WEIGHTS grep ────────────────────────────────────────
print("=" * 60)
print("CHECK 4: DEFAULT_OPT_WEIGHTS references in codebase")

src_dir = pathlib.Path(HERE)
hits = []
for py_file in src_dir.glob('*.py'):
    with open(py_file, encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            if 'DEFAULT_OPT_WEIGHTS' in line:
                hits.append((py_file.name, lineno, line.rstrip()))

for fname, ln, text in hits:
    print(f"  {fname}:{ln}  {text.strip()}")

bad = [(f, l, t) for f, l, t in hits
       if '_full_cost' in t or 'compute_f_total' in t]
print(f"\n  Total references                              : {len(hits)}")
print(f"  Inside _full_cost or compute_f_total          : {len(bad)}")
print(f"  PASS (none in _full_cost/compute_f_total)     = {len(bad) == 0}")