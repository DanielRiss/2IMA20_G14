import sys, os, math, copy, random
sys.path.insert(0, 'src')

from data_loader import load_trade_data
from map_utils import load_eu_map, get_centroids
from flow_renderer import disparity_filter
from spiral_tree import (
    compute_spiral_tree, compute_f_total, DEFAULT_F_TOTAL_WEIGHTS,
    DEFAULT_OPT_WEIGHTS, PRIOR_TREE_OBSTACLE_M, _to3035, _polar, _cart,
    _sample_spiral, _smooth_subdivision_nodes, _OPT_OVERLAP_B,
    MAX_DISPLAY_W_M,
)
import matplotlib
matplotlib.use('Agg')

DATA_DIR = 'data'
export_matrix, net_matrix, countries = load_trade_data(
    os.path.join(DATA_DIR, 'data-18886936.csv'),
    os.path.join(DATA_DIR, 'label-18886936.csv'),
)
eu_gdf    = load_eu_map()
centroids = get_centroids(eu_gdf)

sources = ['Germany', 'France']
matrix  = export_matrix
threshold = 500.0
disparity_alpha = 0.15
alpha_deg = 25.0
exponent  = 0.5
width_scale = 1.0

sources_sorted = sorted(
    [s for s in sources if s in matrix.index],
    key=lambda s: float(matrix.loc[s].drop(s, errors='ignore').clip(lower=0).sum()),
    reverse=True,
)

built_trees = []
trees = []
for src in sources_sorted:
    row = matrix.loc[src]
    net_flows = {dst: float(v) for dst, v in row.items()
                 if dst != src and float(v) > threshold}
    net_flows_d = disparity_filter(net_flows, alpha=disparity_alpha)
    if not net_flows_d:
        continue
    src_lon, src_lat = centroids[src]
    new_sx, new_sy   = _to3035(src_lon, src_lat)
    prior_obstacles  = []
    for prev_tree, prev_sx, prev_sy in built_trees:
        for node in prev_tree.tree_nodes.values():
            if node.is_steiner or node.is_leaf:
                local_x = (prev_sx + node.x) - new_sx
                local_y = (prev_sy + node.y) - new_sy
                prior_obstacles.append((local_x, local_y, PRIOR_TREE_OBSTACLE_M))
    result = compute_spiral_tree(
        source_name=src, terminal_names=list(net_flows_d.keys()),
        net_flows=net_flows_d, centroids=centroids,
        obstacle_names=[s for s in sources_sorted if s != src],
        alpha_deg=alpha_deg, exponent=exponent,
        prior_obstacles=prior_obstacles, width_scale=width_scale,
    )
    built_trees.append((result, new_sx, new_sy))
    trees.append(result)

# Diagnostics 4: root width
for t in trees:
    root = t.tree_nodes[0]
    print(f"[Tree: {t.source_name}] nodes[0].width = {root.width:.1f} m  "
          f"(MAX_DISPLAY_W_M = {MAX_DISPLAY_W_M:.0f} m)")

try:
    from shapely.geometry import LineString as _LS
except ImportError:
    _LS = None

w        = DEFAULT_F_TOTAL_WEIGHTS.copy()
max_iter = 100
gd_iters = 0
T_min    = 1e-3
n_samp   = 6

state = copy.deepcopy(trees)
src_xy = {ti: _to3035(*centroids[tree.source_name]) for ti, tree in enumerate(state)}

steiner_pool = [
    (ti, nid)
    for ti, tree in enumerate(state)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner and len(node.children) >= 2
]

orig_positions = {
    (ti, nid): (node.x, node.y)
    for ti, tree in enumerate(state)
    for nid, node in tree.tree_nodes.items()
    if node.is_steiner
}

def full_cost():
    return compute_f_total(state, centroids, alpha_deg, w, _OPT_OVERLAP_B, n_samp)

f_before = full_cost()
T_init_dyn = (0.07 * f_before) / math.log(1.0 / 0.25)
alpha_cool  = (T_min / T_init_dyn) ** (1.0 / max(max_iter, 1))
T = T_init_dyn
current_cost = f_before

sa_accepted = 0
rev_cross   = 0
rev_disp    = 0

for it in range(max_iter):
    ti, nid = random.choice(steiner_pool)
    tree    = state[ti]
    node    = tree.tree_nodes[nid]
    sx, sy  = src_xy[ti]

    t_frac  = T / T_init_dyn
    step    = max(node.R * 0.10 * t_frac, 500.0)
    new_x   = node.x + random.gauss(0.0, step)
    new_y   = node.y + random.gauss(0.0, step)
    new_R, new_phi = _polar(new_x, new_y)

    if new_R < 10_000.0:
        T *= alpha_cool
        continue

    R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
    par            = tree.tree_nodes[node.parent] if node.parent is not None else None
    R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
    if R_parent_min >= R_children_min:
        T *= alpha_cool
        continue
    if new_R < R_parent_min or new_R > R_children_min:
        new_R        = min(max(new_R, R_parent_min), R_children_min)
        new_x, new_y = _cart(new_R, new_phi)

    orig_pos   = (node.x, node.y, node.R, node.phi)
    orig_edges = {i: tree.edge_polylines[i]
                  for i, (c, p) in enumerate(tree.edges) if c == nid or p == nid}

    node.x, node.y, node.R, node.phi = new_x, new_y, new_R, new_phi
    for i in orig_edges:
        c, p   = tree.edges[i]
        cn, pn = tree.tree_nodes[c], tree.tree_nodes[p]
        new_pts = _sample_spiral(cn.R, cn.phi, pn.R, pn.phi, sx, sy)
        tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

    new_cost = full_cost()
    delta    = new_cost - current_cost

    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
        incident_idx     = set(orig_edges.keys())
        non_incident_idx = [i for i in range(len(tree.edges))
                            if i not in incident_idx and len(tree.edge_polylines[i]) >= 2]
        new_intra = False
        if _LS is not None:
            for ai in incident_idx:
                if len(tree.edge_polylines[ai]) < 2: continue
                la = _LS(tree.edge_polylines[ai])
                oa = _LS(orig_edges[ai])
                for bi in non_incident_idx:
                    lb = _LS(tree.edge_polylines[bi])
                    try:
                        if la.crosses(lb) and not oa.crosses(lb):
                            new_intra = True; break
                    except Exception: continue
                if new_intra: break
        if new_intra:
            node.x, node.y, node.R, node.phi = orig_pos
            for i, pts in orig_edges.items(): tree.edge_polylines[i] = pts
            T *= alpha_cool
            rev_cross += 1
            continue

        ox, oy     = orig_positions[(ti, nid)]
        cumul_disp = math.hypot(node.x - ox, node.y - oy)
        if cumul_disp > node.R * 1.0:
            node.x, node.y, node.R, node.phi = orig_pos
            for i, pts in orig_edges.items(): tree.edge_polylines[i] = pts
            T *= alpha_cool
            rev_disp += 1
            continue

        current_cost = new_cost
        _smooth_subdivision_nodes(tree, *src_xy[ti])
        sa_accepted += 1
    else:
        node.x, node.y, node.R, node.phi = orig_pos
        for i, pts in orig_edges.items(): tree.edge_polylines[i] = pts

    T *= alpha_cool

f_after = current_cost

print()
print('=' * 60)
print('OPTIMIZER DIAGNOSTIC — Germany + France, 100 iterations')
print('=' * 60)
print(f' 1. F_opt before optimization:        {f_before:.6f}')
print(f' 2. F_opt after 100 iterations:       {f_after:.6f}')
print(f' 3. T_init (0.07 * F / ln4):          {T_init_dyn:.6f}')
print(f' 4. nodes[0].width — see above')
print(f' 5. Displacement limit:               node.R * 1.0')
print(f' 6. Accepted moves / 100:             {sa_accepted}')
print(f' 7. Reverted by intra-cross guard:    {rev_cross}')
print(f' 8. Reverted by displacement guard:   {rev_disp}')
print()
print(f' 9. DEFAULT_F_TOTAL_WEIGHTS:')
for k, v in DEFAULT_F_TOTAL_WEIGHTS.items():
    print(f'       {k}: {v}')
print(f'10. DEFAULT_OPT_WEIGHTS:')
for k, v in DEFAULT_OPT_WEIGHTS.items():
    print(f'       {k}: {v}')
print()
print('11. UI weights dict (interactive.py:556):')
print('       weights = {k: v.get() for k, v in self._opt_weight_vars.items()}')
print('       _opt_weight_vars keys: c_cross, c_overlap')
print(f'       Default values from DEFAULT_OPT_WEIGHTS: '
      f"c_cross={DEFAULT_OPT_WEIGHTS['c_cross']}, "
      f"c_overlap={DEFAULT_OPT_WEIGHTS['c_overlap']}")
print()
print('12. T_init line (spiral_tree.py:1543):')
print('       T_init = (0.07 * current_cost) / math.log(1.0 / 0.25)')
print()
print('13. Displacement limit (spiral_tree.py:1651):')
print('       if cumul_disp > node.R * 1.0:')
print()
print('14. steiner_pool condition (spiral_tree.py:1519):')
print('       if node.is_steiner and len(node.children) >= 2')
print(f'       Pool size for this run: {len(steiner_pool)}')
print()
print('15. _smooth_subdivision_nodes call (spiral_tree.py:1660):')
print('       _smooth_subdivision_nodes(tree, *src_xy[ti])  <- after each accepted move')
