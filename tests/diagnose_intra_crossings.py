"""Diagnose intra-tree crossings before and after optimization.

Findings:
  1. Per-tree intra-crossing count before and after 2000-iteration optimizer run.
  2. (Code audit — answered inline, no runtime needed.)
  3. Feasibility check for post-acceptance revert on intra-crossing.
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
        trees.append(st.compute_spiral_tree(
            source_name=src_name,
            terminal_names=list(net_flows.keys()),
            net_flows=net_flows,
            centroids=centroids,
            obstacle_names=[s for s in SOURCES if s != src_name],
            alpha_deg=25.0,
        ))
    return trees

# ── per-tree intra-crossing counter (does not require shapely for pairs) ────
try:
    from shapely.geometry import LineString
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("WARNING: shapely not available; using segment-intersection fallback")

def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def _segments_cross(p1, p2, p3, p4):
    """True if open segment p1-p2 crosses open segment p3-p4 (ignores shared endpoints)."""
    ax, ay = p1; bx, by = p2; cx, cy = p3; dx, dy = p4
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    A, B, C, D = (ax,ay),(bx,by),(cx,cy),(dx,dy)
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def count_intra_per_tree(trees):
    """Return list of intra-crossing counts, one per tree."""
    results = []
    for tree in trees:
        polys = [pts for pts in tree.edge_polylines if len(pts) >= 2]
        crossings = 0
        if HAS_SHAPELY:
            lines = [LineString(p) for p in polys]
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    try:
                        if lines[i].crosses(lines[j]):
                            crossings += 1
                    except Exception:
                        pass
        else:
            # fallback: check all consecutive segment pairs across polylines
            segs = []
            for pts in polys:
                for k in range(len(pts)-1):
                    segs.append((pts[k], pts[k+1]))
            for i in range(len(segs)):
                for j in range(i+1, len(segs)):
                    if _segments_cross(segs[i][0], segs[i][1],
                                       segs[j][0], segs[j][1]):
                        crossings += 1
        results.append(crossings)
    return results

# ── Finding 1: before optimization ──────────────────────────────────────────
print("=" * 60)
print("Finding 1: Intra-tree crossings before optimization")
print("=" * 60)

trees_orig = build_trees()
before_counts = count_intra_per_tree(trees_orig)
inter_before, intra_before_total = st.count_crossings(trees_orig)
for ti, tree in enumerate(trees_orig):
    print(f"  {tree.source_name}: {before_counts[ti]} intra-tree crossings")
print(f"  Total intra (count_crossings): {intra_before_total}")
print(f"  Total inter (count_crossings): {inter_before}")

# ── Run 2000-iteration optimizer ─────────────────────────────────────────────
print()
print("Running 2000-iteration optimizer (seed=42)...")

state = copy.deepcopy(trees_orig)
w     = dict(st.DEFAULT_F_TOTAL_WEIGHTS)
src_xy = {}
for ti, tree in enumerate(state):
    lon, lat   = centroids[tree.source_name]
    src_xy[ti] = st._to3035(lon, lat)

steiner_pool = [(ti, nid)
    for ti, tree in enumerate(state)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner]

def _fc():
    return st.compute_f_total(state, centroids, 25.0, w, st._OPT_OVERLAP_B, 6)

current_cost = _fc()
T_init = (0.01 * current_cost) / math.log(1.0 / 0.25)
T_min  = 1e-3
MAX_IT = 2000
alpha  = (T_min / T_init) ** (1.0 / MAX_IT)
T      = T_init
rng    = _rnd.Random(42)

# Track how many times a newly accepted move introduced a new intra crossing
intra_introduced = 0
intra_checks     = 0

for it in range(1, MAX_IT + 1):
    ti, nid = rng.choice(steiner_pool)
    tree    = state[ti]
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

    new_cost = _fc()
    delta    = new_cost - current_cost

    accepted = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12))

    if accepted:
        # Check whether this accepted move introduced intra-tree crossings
        # in the affected tree by comparing the affected edges against all
        # other edges of the same tree.
        intra_checks += 1
        affected_set = set(orig_edges.keys())
        other_polys  = [tree.edge_polylines[i]
                        for i in range(len(tree.edges))
                        if i not in affected_set and len(tree.edge_polylines[i]) >= 2]
        new_crossing = False
        if HAS_SHAPELY:
            other_lines = [LineString(p) for p in other_polys]
            for ai in affected_set:
                if len(tree.edge_polylines[ai]) < 2:
                    continue
                aline = LineString(tree.edge_polylines[ai])
                for oline in other_lines:
                    try:
                        if aline.crosses(oline):
                            new_crossing = True
                            break
                    except Exception:
                        pass
                if new_crossing:
                    break
        if new_crossing:
            intra_introduced += 1
        current_cost = new_cost
    else:
        node.x, node.y, node.R, node.phi = orig_pos
        for i, pts in orig_edges.items():
            tree.edge_polylines[i] = pts

    T *= alpha

# ── Finding 1 continued: after optimization ──────────────────────────────────
print()
print("=" * 60)
print("Finding 1: Intra-tree crossings AFTER optimization")
print("=" * 60)
after_counts = count_intra_per_tree(state)
inter_after, intra_after_total = st.count_crossings(state)
for ti, tree in enumerate(state):
    delta = after_counts[ti] - before_counts[ti]
    sign  = f"(+{delta})" if delta > 0 else f"({delta})" if delta < 0 else "(unchanged)"
    print(f"  {tree.source_name}: {after_counts[ti]} intra-tree crossings  {sign}")
print(f"  Total intra (count_crossings): {intra_after_total}")
print(f"  Total inter (count_crossings): {inter_after}")

# ── Finding 2 (code audit) ───────────────────────────────────────────────────
print()
print("=" * 60)
print("Finding 2: Is there any intra-crossing check in the accept path?")
print("=" * 60)
print("  Lines 1393-1400 (Metropolis accept/revert block):")
print("    if delta < 0 or random() < exp(-delta/T):")
print("      current_cost = new_cost          <- accepted, NO crossing check")
print("    else:")
print("      revert node and edges            <- rejected on cost only")
print()
print("  There is NO intra-tree crossing check anywhere in the accept path.")
print("  The only geometric checks are:")
print("    - new_R < 10,000m  (minimum radius)")
print("    - R-hierarchy clamping (radial ordering)")
print("  Neither of these prevents a moved node from causing edge-edge crossings")
print("  within the same tree.")

# ── Finding 3: revert-on-crossing feasibility ────────────────────────────────
print()
print("=" * 60)
print("Finding 3: Revert-on-intra-crossing feasibility")
print("=" * 60)
print(f"  Accepted moves checked for intra crossings: {intra_checks}")
print(f"  Of those, moves that introduced new intra crossings: {intra_introduced}")
if intra_checks > 0:
    pct = 100 * intra_introduced / intra_checks
    print(f"  Rate: {pct:.1f}% of accepted moves create intra crossings")
print()
print("  Cost of one intra-crossing check per accepted move:")
print("  - Only the perturbed node's incident edges (len(orig_edges), typically 2-3)")
print("    need to be tested against all OTHER edges of the same tree.")
print("  - A single tree has O(N) edges. With N~25 edges, checking 2-3 affected")
print("    edges against ~22 others = ~50-70 segment-pair tests per accepted move.")
print("  - This is negligible vs. _full_cost() which evaluates all trees.")
print()
print("  Proposed check placement (after accept decision, before current_cost update):")
print("    if accepted:")
print("      if _has_new_intra_crossing(tree, affected_edge_indices, all_other_edges):")
print("          revert  <- treat as rejected")
print("      else:")
print("          current_cost = new_cost")
print()
print("  This is O(k * N) per accepted move where k = incident edges of perturbed node.")
print("  It is strictly cheaper than _full_cost() and can be run before updating cost.")
