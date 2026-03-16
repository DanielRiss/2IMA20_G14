"""
spiral_tree.py — Spiral Tree layout for multi-source flow maps.

Based on: Verbeek, Buchin & Speckmann (2011). "Flow Map Layout via Spiral Trees."
IEEE TVCG 17(12), 2536–2544.
And: Buchin, Speckmann & Verbeek (2011). "Angle-Restricted Steiner Arborescences
for Flow Map Layout." EuroCG 2011.

Public API
----------
compute_spiral_tree(source, terminal_list, net_flows, centroids, obstacle_list, alpha_deg)
    -> SpiralTreeResult

draw_spiral_trees(ax, trees, centroids, color_map)

compute_tree_stats(tree, centroids) -> dict

All centroid keys use the same short-name format as map_utils (e.g. 'Germany').
"""

from __future__ import annotations

import math
import heapq
import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from pyproj import Transformer

# ── CRS transformers ────────────────────────────────────────────────────────
_FWD = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_INV = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

EU_EXTENT_M        = 4_000_000.0
MAX_DISPLAY_W_M    = EU_EXTENT_M * 0.012   # ≈48 000 m
OBSTACLE_RADIUS_M  = 150_000.0
LEAF_OBSTACLE_M    =  30_000.0
N_SPIRAL_SAMPLES   = 40
MIN_W_FRAC         = 0.005


# ── coordinate helpers ───────────────────────────────────────────────────────

def _to3035(lon: float, lat: float) -> Tuple[float, float]:
    return _FWD.transform(lon, lat)

def _to_lonlat(x: float, y: float) -> Tuple[float, float]:
    return _INV.transform(x, y)

def _polar(dx: float, dy: float) -> Tuple[float, float]:
    return math.hypot(dx, dy), math.atan2(dy, dx)

def _cart(R: float, phi: float) -> Tuple[float, float]:
    return R * math.cos(phi), R * math.sin(phi)

def _wrap(phi: float) -> float:
    while phi >  math.pi: phi -= 2 * math.pi
    while phi <= -math.pi: phi += 2 * math.pi
    return phi


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    node_id:    int
    R:          float
    phi:        float     # UNROTATED angle in local space
    x:          float     # local Cartesian x in EPSG:3035 offset metres
    y:          float
    parent:     Optional[int] = None
    children:   List[int]     = field(default_factory=list)
    is_steiner: bool  = False
    is_leaf:    bool  = False
    weight:     float = 0.0
    width:      float = 0.0   # display width (metres); set by _assign_widths
    country:    Optional[str] = None


@dataclass
class SpiralTreeResult:
    source_name:   str
    tree_nodes:    Dict[int, TreeNode]
    edges:         List[Tuple[int, int]]               # (child_id, parent_id)
    total_flow:    float
    # Parallel to edges — sampled lon/lat polyline per edge (for drawing & stats)
    edge_polylines: List[List[Tuple[float, float]]]    = field(default_factory=list)
    poly_patches:   List[object]                       = field(default_factory=list)


# ── spiral region test ────────────────────────────────────────────────────────

def _in_spiral_region(p_R: float, p_phi: float,
                      q_R: float, q_phi: float, tan_a: float) -> bool:
    if q_R >= p_R:
        return False
    dphi = abs(q_phi - p_phi)
    if dphi > math.pi:
        dphi = 2 * math.pi - dphi
    return dphi <= tan_a * math.log(p_R / q_R)


# ── join-point computation ─────────────────────────────────────────────────────

def _join_point(
    u_R: float, u_phi: float,
    v_R: float, v_phi: float,
    tan_a: float,
    obstacles: List[Tuple[float, float, float]],
    phi_offset: float,
) -> Optional[Tuple[float, float]]:
    """u is left of v (u_phi < v_phi). Returns (R_j, phi_j_rotated) or None."""
    if tan_a < 1e-12:
        return None
    dphi = u_phi - v_phi
    ln_Rj = (0.5 * (math.log(max(u_R, 1e-9)) + math.log(max(v_R, 1e-9)))
             + dphi / (2.0 * tan_a))
    R_j = math.exp(ln_Rj)
    if R_j <= 0 or R_j >= min(u_R, v_R):
        return None
    phi_j_rot  = u_phi + tan_a * math.log(u_R / R_j)
    phi_j_orig = _wrap(phi_j_rot + phi_offset)
    xj, yj = _cart(R_j, phi_j_orig)
    for ox, oy, rad in obstacles:
        if math.hypot(xj - ox, yj - oy) < rad:
            return None
    return R_j, phi_j_rot


# ── wavefront ─────────────────────────────────────────────────────────────────

class _Wavefront:
    def __init__(self):
        self._keys: List[float] = []
        self._ids:  List[int]   = []

    def insert(self, phi: float, node_id: int):
        i = bisect.bisect_left(self._keys, phi)
        self._keys.insert(i, phi)
        self._ids.insert(i, node_id)

    def remove(self, node_id: int):
        try:
            i = self._ids.index(node_id)
            self._keys.pop(i); self._ids.pop(i)
        except ValueError:
            pass

    def left_neighbor(self, node_id: int) -> Optional[int]:
        try:
            i = self._ids.index(node_id)
            return self._ids[i - 1] if i > 0 else None
        except ValueError:
            return None

    def right_neighbor(self, node_id: int) -> Optional[int]:
        try:
            i = self._ids.index(node_id)
            return self._ids[i + 1] if i < len(self._ids) - 1 else None
        except ValueError:
            return None

    def active_ids(self) -> List[int]:
        return list(self._ids)


# ── greedy inward sweep ───────────────────────────────────────────────────────

def _build_tree(
    source_id: int,
    leaf_nodes: Dict[int, TreeNode],
    rotated_phi: Dict[int, float],
    tan_a: float,
    obstacles: List[Tuple[float, float, float]],
    phi_offset: float,
) -> Dict[int, TreeNode]:
    nodes: Dict[int, TreeNode] = {}
    next_id = [max(leaf_nodes.keys()) + 1]

    src = TreeNode(node_id=source_id, R=0.0, phi=0.0, x=0.0, y=0.0)
    nodes[source_id] = src
    for nid, n in leaf_nodes.items():
        nodes[nid] = n

    wavefront = _Wavefront()
    active: set = set(leaf_nodes.keys())
    heap: list = []
    for nid, n in leaf_nodes.items():
        heapq.heappush(heap, (-n.R, 0, nid))

    def rphi(nid: int) -> float:
        return rotated_phi[nid]

    def try_join(a_id: int, b_id: int):
        ra, rb = rphi(a_id), rphi(b_id)
        if ra > rb:
            a_id, b_id, ra, rb = b_id, a_id, rb, ra
        na, nb = nodes[a_id], nodes[b_id]
        result = _join_point(na.R, ra, nb.R, rb, tan_a, obstacles, phi_offset)
        if result is not None:
            R_j, phi_j_rot = result
            heapq.heappush(heap, (-R_j, 1, (a_id, b_id, phi_j_rot)))

    while heap:
        neg_R, etype, payload = heapq.heappop(heap)

        if etype == 0:
            t_id = payload
            if t_id not in active:
                continue
            t = nodes[t_id]
            rt = rphi(t_id)
            wavefront.insert(rt, t_id)

            for nb_id in (wavefront.left_neighbor(t_id),
                          wavefront.right_neighbor(t_id)):
                if nb_id is None or nb_id not in active:
                    continue
                nb = nodes[nb_id]
                if _in_spiral_region(nb.R, rphi(nb_id), t.R, rt, tan_a):
                    # t is in nb's spiral region: t becomes parent of nb
                    wavefront.remove(nb_id)
                    active.discard(nb_id)
                    nb.parent = t_id
                    t.children.append(nb_id)
                    new_l = wavefront.left_neighbor(t_id)
                    new_r = wavefront.right_neighbor(t_id)
                    for new_nb in (new_l, new_r):
                        if new_nb is not None and new_nb in active:
                            try_join(t_id, new_nb)
                else:
                    try_join(t_id, nb_id)

        else:
            u_id, v_id, phi_j_rot = payload
            if u_id not in active or v_id not in active:
                continue
            u_l = wavefront.left_neighbor(u_id)
            u_r = wavefront.right_neighbor(u_id)
            if v_id not in (u_l, u_r):
                continue

            ru, rv = rphi(u_id), rphi(v_id)
            if ru > rv:
                u_id, v_id, ru, rv = v_id, u_id, rv, ru
            nu, nv = nodes[u_id], nodes[v_id]
            result = _join_point(nu.R, ru, nv.R, rv, tan_a, obstacles, phi_offset)
            if result is None:
                continue
            R_j, phi_j_rot = result

            s_id = next_id[0]; next_id[0] += 1
            phi_j_orig = _wrap(phi_j_rot + phi_offset)
            xj, yj = _cart(R_j, phi_j_orig)
            s = TreeNode(node_id=s_id, R=R_j, phi=phi_j_orig,
                         x=xj, y=yj, is_steiner=True)
            nodes[s_id] = s
            rotated_phi[s_id] = phi_j_rot
            active.add(s_id)

            wavefront.remove(u_id); wavefront.remove(v_id)
            active.discard(u_id); active.discard(v_id)
            nu.parent = s_id; nv.parent = s_id
            s.children = [u_id, v_id]

            wavefront.insert(phi_j_rot, s_id)
            for nb_id in (wavefront.left_neighbor(s_id),
                          wavefront.right_neighbor(s_id)):
                if nb_id is not None and nb_id in active:
                    try_join(s_id, nb_id)

    for rem_id in wavefront.active_ids():
        if rem_id in active:
            nodes[rem_id].parent = source_id
            src.children.append(rem_id)

    # Unrotate phi for leaf nodes
    for nid, n in nodes.items():
        if n.is_leaf:
            n.phi = _wrap(rotated_phi[nid] + phi_offset)

    return nodes


# ── width thickening (Fix 3) ──────────────────────────────────────────────────

def _postorder(nodes: Dict[int, TreeNode], root_id: int) -> List[int]:
    order, stack = [], [root_id]
    while stack:
        nid = stack.pop()
        order.append(nid)
        stack.extend(nodes[nid].children)
    return list(reversed(order))


def _assign_widths(nodes: Dict[int, TreeNode], root_id: int) -> None:
    """
    Three-phase bottom-up width assignment with monotonicity assertion.

    Phase 1 — pure post-order: leaf.width = leaf.weight (raw flow);
               internal.width = sum(children.width)
    Phase 2 — assert conservation invariant at every internal node
    Phase 3 — scale all widths to display units: root maps to MAX_DISPLAY_W_M
    """
    # Phase 1: raw weights
    for nid in _postorder(nodes, root_id):
        n = nodes[nid]
        if n.is_leaf:
            n.width = n.weight
        else:
            n.width = sum(nodes[c].width for c in n.children)

    # Phase 2: verify invariant
    for nid, n in nodes.items():
        if not n.is_leaf and n.children:
            expected = sum(nodes[c].width for c in n.children)
            tol = 1e-6 * max(abs(n.width), 1.0)
            if abs(n.width - expected) > tol:
                raise AssertionError(
                    f"Width monotonicity violated at node {nid} "
                    f"(is_steiner={n.is_steiner}): "
                    f"node.width={n.width:.8g} != "
                    f"sum(children)={expected:.8g}  diff={n.width - expected:.3g}"
                )

    # Phase 3: scale to display units
    root_w = nodes[root_id].width
    if root_w <= 0:
        return
    scale = MAX_DISPLAY_W_M / root_w
    for n in nodes.values():
        n.width *= scale


# ── spiral edge sampling ──────────────────────────────────────────────────────

def _sample_spiral(
    c_R: float, c_phi: float,
    p_R: float, p_phi: float,
    src_x: float, src_y: float,
    n: int = N_SPIRAL_SAMPLES,
) -> List[Tuple[float, float]]:
    """Sample a log-spiral from child to parent; returns (lon, lat) list."""
    if c_R <= 0:
        return []

    if p_R < 1e-3:
        t_end   = math.log(c_R / 1.0)
        phi_end = p_phi
    else:
        t_end   = math.log(c_R / p_R)
        phi_end = p_phi

    if abs(t_end) < 1e-12:
        pts = []
        for k in range(n):
            t   = k / max(n - 1, 1)
            R   = c_R + t * (p_R - c_R)
            phi = c_phi + t * (phi_end - c_phi)
            lx, ly = _cart(R, phi)
            pts.append(_to_lonlat(src_x + lx, src_y + ly))
        return pts

    tan_beta = (phi_end - c_phi) / t_end
    pts = []
    for k in range(n):
        t     = k / max(n - 1, 1) * t_end
        R_t   = c_R * math.exp(-t)
        phi_t = c_phi + tan_beta * t
        if p_R < 1e-3 and k == n - 1:
            pts.append(_to_lonlat(src_x, src_y))
        else:
            lx, ly = _cart(R_t, phi_t)
            pts.append(_to_lonlat(src_x + lx, src_y + ly))
    return pts


# ── thick edge polygon ────────────────────────────────────────────────────────

def _edge_polygon(
    pts: List[Tuple[float, float]],
    half_w: float,
    color: str,
    alpha: float = 0.75,
) -> Optional[mpatches.PathPatch]:
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=float)

    tangents = np.zeros_like(arr)
    tangents[1:-1] = arr[2:] - arr[:-2]
    tangents[0]    = arr[1]  - arr[0]
    tangents[-1]   = arr[-1] - arr[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    tangents /= norms

    perp  = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    left  = arr + perp * half_w
    right = arr - perp * half_w

    poly  = np.vstack([left, right[::-1], left[[0]]])
    codes = ([Path.MOVETO]
             + [Path.LINETO] * (len(left) - 1)
             + [Path.LINETO] * len(right)
             + [Path.CLOSEPOLY])
    return mpatches.PathPatch(Path(poly, codes),
                              facecolor=color, edgecolor='none',
                              alpha=alpha, zorder=3)


# ── public: compute ───────────────────────────────────────────────────────────

def compute_spiral_tree(
    source_name: str,
    terminal_names: List[str],
    net_flows: Dict[str, float],
    centroids: Dict[str, Tuple[float, float]],
    obstacle_names: List[str],
    alpha_deg: float = 25.0,
) -> SpiralTreeResult:
    """Compute a spiral tree for one source country."""
    tan_a = math.tan(math.radians(alpha_deg))

    src_lon, src_lat = centroids[source_name]
    sx, sy = _to3035(src_lon, src_lat)

    leaf_nodes:  Dict[int, TreeNode] = {}
    rotated_phi: Dict[int, float]    = {}
    nid = 1

    raw_terminals = []
    for name in terminal_names:
        if name not in centroids or name == source_name:
            continue
        flow = net_flows.get(name, 0.0)
        if flow <= 0:
            continue
        lon, lat = centroids[name]
        tx, ty   = _to3035(lon, lat)
        dx, dy   = tx - sx, ty - sy
        R, phi   = _polar(dx, dy)
        if R < 500:
            continue
        raw_terminals.append((name, flow, R, phi, dx, dy))

    if not raw_terminals:
        src_node = TreeNode(node_id=0, R=0.0, phi=0.0, x=0.0, y=0.0)
        return SpiralTreeResult(source_name=source_name,
                                tree_nodes={0: src_node},
                                edges=[], total_flow=0.0)

    all_phis   = [t[3] for t in raw_terminals]
    phi_offset = (min(all_phis) + max(all_phis)) / 2.0

    for name, flow, R, phi, dx, dy in raw_terminals:
        rphi = _wrap(phi - phi_offset)
        node = TreeNode(node_id=nid, R=R, phi=phi, x=dx, y=dy,
                        is_leaf=True, weight=flow, country=name)
        leaf_nodes[nid]  = node
        rotated_phi[nid] = rphi
        nid += 1

    obstacles: List[Tuple[float, float, float]] = []
    for obs_name in obstacle_names:
        if obs_name == source_name or obs_name not in centroids:
            continue
        olon, olat = centroids[obs_name]
        ox, oy     = _to3035(olon, olat)
        obstacles.append((ox - sx, oy - sy, OBSTACLE_RADIUS_M))
    for name, flow, R, phi, dx, dy in raw_terminals:
        obstacles.append((dx, dy, LEAF_OBSTACLE_M))

    tree_nodes = _build_tree(
        source_id=0, leaf_nodes=leaf_nodes, rotated_phi=rotated_phi,
        tan_a=tan_a, obstacles=obstacles, phi_offset=phi_offset,
    )

    _assign_widths(tree_nodes, root_id=0)

    edges = [(nid, n.parent)
             for nid, n in tree_nodes.items() if n.parent is not None]

    # Pre-compute edge polylines (used for drawing and statistics)
    edge_polylines = []
    for child_id, parent_id in edges:
        c_node = tree_nodes[child_id]
        p_node = tree_nodes[parent_id]
        pts = _sample_spiral(c_node.R, c_node.phi,
                             p_node.R, p_node.phi, sx, sy)
        edge_polylines.append(pts)

    total_flow = sum(n.weight for n in tree_nodes.values() if n.is_leaf)

    return SpiralTreeResult(source_name=source_name,
                            tree_nodes=tree_nodes,
                            edges=edges,
                            edge_polylines=edge_polylines,
                            total_flow=total_flow)


# ── public: draw ──────────────────────────────────────────────────────────────

def draw_spiral_trees(
    ax: matplotlib.axes.Axes,
    trees: List[SpiralTreeResult],
    centroids: Dict[str, Tuple[float, float]],
    color_map: Dict[str, str],
    alpha: float = 0.75,
) -> None:
    """Draw all spiral trees onto ax (lon/lat coordinate space)."""
    max_total = max((t.total_flow for t in trees), default=1.0)

    for tree in trees:
        if not tree.edges:
            continue
        color  = color_map.get(tree.source_name, '#333333')
        nodes  = tree.tree_nodes
        src_lon, src_lat = centroids[tree.source_name]

        max_w = max((nodes[c].width for c, _ in tree.edges), default=0.0)
        if max_w <= 0:
            continue

        m_per_deg = 111_000.0

        for (child_id, parent_id), pts in zip(tree.edges, tree.edge_polylines):
            c_node = nodes[child_id]
            if c_node.width <= 0 or len(pts) < 2:
                continue

            half_w = (c_node.width / 2.0) / m_per_deg
            frac   = c_node.width / max_w

            if frac >= MIN_W_FRAC:
                patch = _edge_polygon(pts, half_w, color, alpha=alpha)
                if patch is not None:
                    ax.add_patch(patch)
            else:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, lw=0.6, alpha=alpha * 0.7, zorder=3)

        # Source marker
        sz = 10 + 6 * min(tree.total_flow / max(max_total, 1.0), 1.0)
        ax.plot(src_lon, src_lat, 's', color=color,
                markersize=sz, markeredgecolor='white',
                markeredgewidth=1.2, zorder=6)

        # Leaf markers
        for nid, node in nodes.items():
            if node.is_leaf and node.country and node.country in centroids:
                lon, lat = centroids[node.country]
                r = max(2.5, 3 + 5 * node.width / max_w)
                ax.plot(lon, lat, 'o', color=color,
                        markersize=r, markeredgecolor='white',
                        markeredgewidth=0.7, alpha=0.85, zorder=5)

        ax.text(src_lon, src_lat + 1.5, tree.source_name,
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                zorder=7, color=color,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', alpha=0.7, lw=0))


# ── statistics helpers ────────────────────────────────────────────────────────

def compute_tree_stats(
    tree: SpiralTreeResult,
    centroids: Dict[str, Tuple[float, float]],
    total_eu_flow: float,
) -> dict:
    """
    Compute per-tree statistics.

    Returns dict with keys:
      n_terminals, n_steiner, coverage_pct, F_str, F_sm, F_cost
    """
    nodes = tree.tree_nodes
    n_leaves  = sum(1 for n in nodes.values() if n.is_leaf)
    n_steiner = sum(1 for n in nodes.values() if n.is_steiner)
    coverage  = (tree.total_flow / total_eu_flow * 100.0
                 if total_eu_flow > 0 else 0.0)

    src_lon, src_lat = centroids.get(tree.source_name, (0.0, 0.0))

    # F_str: average (path_length / straight_dist) over leaves
    edge_lookup = {e: pts for e, pts in zip(tree.edges, tree.edge_polylines)}
    # Build parent map for traversal
    parent_map = {n.node_id: n.parent for n in nodes.values() if n.parent is not None}

    def path_to_root(leaf_id):
        path_edges = []
        cur = leaf_id
        while parent_map.get(cur) is not None:
            p = parent_map[cur]
            path_edges.append((cur, p))
            cur = p
        return path_edges

    def polyline_length(pts):
        if len(pts) < 2:
            return 0.0
        arr = np.asarray(pts)
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    f_str_vals = []
    for nid, node in nodes.items():
        if not node.is_leaf or node.country not in centroids:
            continue
        leaf_lon, leaf_lat = centroids[node.country]
        straight = math.hypot(leaf_lon - src_lon, leaf_lat - src_lat)
        if straight < 1e-9:
            continue
        path_len = sum(
            polyline_length(edge_lookup.get(e, []))
            for e in path_to_root(nid)
        )
        f_str_vals.append(path_len / straight)

    F_str = float(np.mean(f_str_vals)) if f_str_vals else 1.0

    # F_sm: average absolute angle change per edge
    angle_devs = []
    for pts in tree.edge_polylines:
        if len(pts) < 3:
            continue
        arr = np.asarray(pts)
        v1 = arr[1:-1] - arr[:-2]
        v2 = arr[2:]  - arr[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        mask = (n1 > 1e-12) & (n2 > 1e-12)
        if mask.any():
            cos_a = np.clip(
                np.sum(v1[mask] * v2[mask], axis=1) / (n1[mask] * n2[mask]),
                -1, 1
            )
            angle_devs.extend(np.arccos(cos_a).tolist())

    F_sm = float(np.mean(angle_devs)) if angle_devs else 0.0

    # Paper cost: c_obs=2.0, c_sm=0.4, c_ar=0.077, c_str=0.4
    F_cost = 0.4 * F_sm + 0.4 * F_str

    return {
        'n_terminals': n_leaves,
        'n_steiner':   n_steiner,
        'coverage_pct': coverage,
        'F_str':       F_str,
        'F_sm':        F_sm,
        'F_cost':      F_cost,
    }


def count_crossings(trees: List[SpiralTreeResult]):
    """
    Count inter-tree and intra-tree edge crossings using shapely.
    Returns (inter_crossings, intra_crossings).
    """
    try:
        from shapely.geometry import LineString
    except ImportError:
        return 0, 0

    segments = []   # (tree_idx, LineString)
    for i, tree in enumerate(trees):
        for pts in tree.edge_polylines:
            if len(pts) >= 2:
                segments.append((i, LineString(pts)))

    inter_cross = 0
    intra_cross = 0
    for a in range(len(segments)):
        for b in range(a + 1, len(segments)):
            ti, la = segments[a]
            tj, lb = segments[b]
            try:
                if la.crosses(lb):
                    if ti == tj:
                        intra_cross += 1
                    else:
                        inter_cross += 1
            except Exception:
                pass

    return inter_cross, intra_cross


# ── __main__ standalone test ─────────────────────────────────────────────────

if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_trade_data, ISO_SHORT
    from map_utils import load_eu_map, get_centroids

    _HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT  = os.path.dirname(_HERE)
    DATA  = os.path.join(ROOT, 'data', 'data-18886936.csv')
    LABEL = os.path.join(ROOT, 'data', 'label-18886936.csv')

    export_mx, net_mx, countries = load_trade_data(DATA, LABEL, year=2024)
    eu_gdf    = load_eu_map()
    centroids = get_centroids(eu_gdf)

    SOURCES    = ['Germany', 'France']
    COLOR_MAP  = {'Germany': '#e41a1c', 'France': '#377eb8'}

    trees = []
    for src_name in SOURCES:
        if src_name not in net_mx.index:
            continue
        row = net_mx.loc[src_name]
        net_flows = {d: float(v) for d, v in row.items()
                     if d != src_name and float(v) > 0}
        result = compute_spiral_tree(
            source_name=src_name,
            terminal_names=list(net_flows.keys()),
            net_flows=net_flows,
            centroids=centroids,
            obstacle_names=[s for s in SOURCES if s != src_name],
            alpha_deg=25.0,
        )
        trees.append(result)
        n_l = sum(1 for n in result.tree_nodes.values() if n.is_leaf)
        n_s = sum(1 for n in result.tree_nodes.values() if n.is_steiner)
        print(f"{src_name}: {len(result.edges)} edges ({n_l} leaves, {n_s} Steiner)")

    inter, intra = count_crossings(trees)
    print(f"Crossings: {inter} inter-tree, {intra} intra-tree")

    fig, ax = plt.subplots(figsize=(14, 10))
    eu_gdf.plot(ax=ax, color='#f5f5f0', edgecolor='#888888', linewidth=0.5)
    ax.set_xlim(-25, 45); ax.set_ylim(34, 72)
    ax.set_aspect('equal'); ax.axis('off')
    draw_spiral_trees(ax, trees, centroids, COLOR_MAP)
    plt.tight_layout()
    plt.show()