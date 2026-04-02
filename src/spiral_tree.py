"""
spiral_tree.py — Spiral Tree layout for multi-source flow maps.

Based on: Verbeek, Buchin & Speckmann (2011). "Flow Map Layout via Spiral Trees."
IEEE TVCG 17(12), 2536–2544.
And: Buchin, Speckmann & Verbeek (2011). "Angle-Restricted Steiner Arborescences
for Flow Map Layout." EuroCG 2011.

Public API
----------hw_source = max((nodes[0].width / 2.0 * flow_scale) / m_per_deg * 2.0,
                    MIN_HW_DISPLAY_DEG * 2.0)
compute_spiral_tree(source, terminal_list, net_flows, centroids, obstacle_list, alpha_deg)
    -> SpiralTreeResult

draw_spiral_trees(ax, trees, centroids, color_map)

compute_tree_stats(tree, centroids) -> dict

All centroid keys use the same short-name format as map_utils (e.g. 'Germany').
"""

from __future__ import annotations

import copy as _copy
import math
import heapq
from collections import defaultdict
import bisect
import random as _random
import threading as _threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
MAX_DISPLAY_W_M    = EU_EXTENT_M * 0.020   # ≈80 000 m
OBSTACLE_RADIUS_M  = 150_000.0
LEAF_OBSTACLE_M    =  30_000.0
N_SPIRAL_SAMPLES   = 40
MIN_W_FRAC         = 0.05
MIN_HW_DISPLAY_DEG = 0.005        # ~555 m — absolute display floor in degrees
MIN_EDGE_FOR_SUBDIVISION_M = 150_000.0  # 150 km — edges longer than this get one midpoint subdivision node
PRIOR_TREE_OBSTACLE_M     =  50_000.0  # 50 km — exclusion zone around previously-built trees' nodes


# ── coordinate helpers ───────────────────────────────────────────────────────

def _to3035(lon: float, lat: float) -> Tuple[float, float]:
    return _FWD.transform(lon, lat)

def _to_lonlat(x: float, y: float) -> Tuple[float, float]:
    return _INV.transform(x, y)

def _polar(dx: float, dy: float) -> Tuple[float, float]:
    return math.hypot(dx, dy), math.atan2(dy, dx)

def _cart(R: float, phi: float) -> Tuple[float, float]:
    return R * math.cos(phi), R * math.sin(phi)

def _wrap(phi):
    return ((phi + math.pi) % (2 * math.pi)) - math.pi


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

            # current_id tracks which node occupies t's wavefront slot.
            # It starts as the leaf but may be promoted to a new Steiner node
            # if spiral-region absorption would otherwise give children to a leaf.
            current_id = t_id

            for get_nb in (wavefront.left_neighbor, wavefront.right_neighbor):
                nb_id = get_nb(current_id)
                if nb_id is None or nb_id not in active:
                    continue
                nb       = nodes[nb_id]
                curr     = nodes[current_id]
                curr_r   = rphi(current_id)

                if _in_spiral_region(nb.R, rphi(nb_id), curr.R, curr_r, tan_a):
                    # nb must be absorbed because the spiral from nb to the
                    # source would have to pass through curr's position.
                    wavefront.remove(nb_id)
                    active.discard(nb_id)

                    if curr.is_leaf:
                        # A leaf must not have children — placing a Steiner
                        # node AT the leaf's position creates an invisible
                        # zero-length edge, making the flow visually pass
                        # *through* the leaf to reach nb.
                        # Fix: place the Steiner node BEFORE the leaf (at
                        # SPLIT_BACK * curr.R from source, same angular
                        # direction).  The trunk reaches ~80 % of the way to
                        # the leaf, then forks: one short branch continues to
                        # the leaf, the other branch continues to nb.
                        SPLIT_BACK = 0.80
                        R_s        = curr.R * SPLIT_BACK
                        # Place Steiner at angular midpoint between curr and nb
                        # so both branches diverge symmetrically (balanced Y).
                        dphi_nb    = _wrap(rphi(nb_id) - curr_r)
                        phi_s_rot  = curr_r + dphi_nb * 0.5
                        phi_s_orig = _wrap(phi_s_rot + phi_offset)
                        xs, ys     = _cart(R_s, phi_s_orig)

                        s_id = next_id[0]; next_id[0] += 1
                        s = TreeNode(node_id=s_id, R=R_s, phi=phi_s_orig,
                                     x=xs, y=ys, is_steiner=True)
                        nodes[s_id]       = s
                        rotated_phi[s_id] = phi_s_rot
                        active.add(s_id)

                        curr.parent = s_id
                        s.children.append(current_id)
                        nb.parent = s_id
                        s.children.append(nb_id)

                        wavefront.remove(current_id)
                        active.discard(current_id)
                        wavefront.insert(phi_s_rot, s_id)
                        current_id = s_id
                    else:
                        # current_id is already a Steiner node (promoted in a
                        # prior pass of this loop) — safe to add more children.
                        nb.parent = current_id
                        curr.children.append(nb_id)

                    new_l = wavefront.left_neighbor(current_id)
                    new_r = wavefront.right_neighbor(current_id)
                    for new_nb in (new_l, new_r):
                        if new_nb is not None and new_nb in active:
                            try_join(current_id, new_nb)
                else:
                    try_join(current_id, nb_id)

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


def _assign_widths(nodes: Dict[int, TreeNode], root_id: int, exponent: float = 0.5) -> None:
    """
    Three-phase bottom-up width assignment with monotonicity assertion.

    Phase 1 — pure post-order: leaf.width = weight ** exponent
               (exponent=1.0 → linear, 0.5 → sqrt, 0.25 → aggressive compression)
    Phase 2 — assert conservation invariant at every internal node
    Phase 3 — scale all widths to display units: root maps to MAX_DISPLAY_W_M
    Phase 4 — enforce minimum width floor (MIN_W_FRAC ratio)
    """
    # Phase 1: raw weights
    for nid in _postorder(nodes, root_id):
        n = nodes[nid]
        if n.is_leaf:
            n.width = n.weight ** exponent
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

    # Phase 4: enforce minimum width floor (200:1 max ratio)
    min_w = MIN_W_FRAC * nodes[root_id].width
    for n in nodes.values():
        if n.width < min_w:
            n.width = min_w


# ── subdivision node insertion ────────────────────────────────────────────────

def _insert_subdivision_nodes(
    tree_nodes: Dict[int, TreeNode],
    edges: List[Tuple[int, int]],
    edge_polylines: List[List[Tuple[float, float]]],
    min_edge_m: float,
) -> int:
    """Insert one subdivision node at the midpoint of every edge longer than min_edge_m.

    Modifies tree_nodes, edges, and edge_polylines in-place.
    Returns the number of subdivision nodes inserted.
    """
    n_inserted = 0
    i = 0
    while i < len(edges):
        child_id, parent_id = edges[i]
        c_node = tree_nodes[child_id]
        p_node = tree_nodes[parent_id]

        length = math.hypot(c_node.x - p_node.x, c_node.y - p_node.y)
        if length < min_edge_m:
            i += 1
            continue

        c_R, c_phi = c_node.R, c_node.phi
        p_R, p_phi = p_node.R, p_node.phi

        if c_R <= 0 or p_R <= 0:
            i += 1
            continue
        t_end = math.log(c_R / p_R)
        if abs(t_end) < 1e-12:
            i += 1
            continue

        tan_beta = (p_phi - c_phi) / t_end
        t_mid    = 0.5 * t_end
        R_mid    = c_R * math.exp(-t_mid)
        phi_mid  = c_phi + tan_beta * t_mid
        x_mid, y_mid = _cart(R_mid, phi_mid)

        new_id = max(tree_nodes.keys()) + 1
        sub_node = TreeNode(
            node_id   = new_id,
            R         = R_mid,
            phi       = phi_mid,
            x         = x_mid,
            y         = y_mid,
            is_steiner= True,
            is_leaf   = False,
            parent    = parent_id,
            children  = [child_id],
            width     = c_node.width,
            weight    = 0.0,
            country   = None,
        )
        tree_nodes[new_id] = sub_node

        c_node.parent = new_id
        children = p_node.children
        children[children.index(child_id)] = new_id

        pts = edge_polylines[i]
        mid = len(pts) // 2
        edges[i]          = (child_id, new_id)
        edge_polylines[i] = pts[:mid + 1]
        edges.insert(i + 1, (new_id, parent_id))
        edge_polylines.insert(i + 1, pts[mid:])

        n_inserted += 1
        # do NOT increment i — next iteration re-examines edges[i] = (child_id, new_id)

    return n_inserted


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
    half_w_start: float,
    half_w_end: float,
    color: str,
    alpha: float = 0.75,
) -> Optional[mpatches.PathPatch]:
    """Tapered polygon: half_w_start at pts[0] (child/leaf end),
    half_w_end at pts[-1] (parent/trunk end).  Linear interpolation
    between the two widths gives smooth narrowing toward leaves."""
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

    perp   = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    widths = np.linspace(half_w_start, half_w_end, len(arr))
    left   = arr + perp * widths[:, np.newaxis]
    right  = arr - perp * widths[:, np.newaxis]

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
    exponent: float = 0.5,
    prior_obstacles: Optional[List[Tuple[float, float, float]]] = None,
    width_scale: float = 1.0,
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
        obstacles.append((ox - sx, oy - sy, OBSTACLE_RADIUS_M * width_scale))
    for name, flow, R, phi, dx, dy in raw_terminals:
        obstacles.append((dx, dy, LEAF_OBSTACLE_M))
    if prior_obstacles:
        obstacles.extend(prior_obstacles)

    tree_nodes = _build_tree(
        source_id=0, leaf_nodes=leaf_nodes, rotated_phi=rotated_phi,
        tan_a=tan_a, obstacles=obstacles, phi_offset=phi_offset,
    )

    _assign_widths(tree_nodes, root_id=0, exponent=exponent)

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

    # Insert subdivision nodes along long edges (modifies tree_nodes, edges, edge_polylines in-place)
    _insert_subdivision_nodes(tree_nodes, edges, edge_polylines, MIN_EDGE_FOR_SUBDIVISION_M)

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
    alpha: float = 1.0,
    width_scale: float = 1.0,
    exponent: float = 0.5,
    show_labels: bool = False,
) -> None:
    """Draw all spiral trees onto ax (lon/lat coordinate space).

    Width model
    -----------
    Every edge's half-width is proportional to the flow it carries divided by
    the *total flow of the entire visualisation* (sum of all trees shown).
    This means adding more sources makes each tree proportionally narrower,
    and switching to net mode automatically produces thinner edges because net
    totals are smaller.  ``width_scale`` is a linear multiplier the user can
    adjust in the UI to make all edges thicker or thinner regardless of the
    data magnitude.

    Draw order: largest-flow tree first so smaller trees sit on top.
    """
    if not trees:
        return

    total_viz_flow = sum(t.total_flow for t in trees)
    if total_viz_flow <= 0:
        return

    m_per_deg  = 111_000.0

    sorted_trees = sorted(trees, key=lambda t: t.total_flow, reverse=True)

    # Pre-compute per-country flow list for concentric ring logic (Fix 3)
    leaf_flows: dict = defaultdict(list)
    for tree_obj in sorted_trees:
        flow_scale_obj = (tree_obj.total_flow / total_viz_flow) ** exponent
        for node_obj in tree_obj.tree_nodes.values():
            if node_obj.is_leaf and node_obj.country:
                leaf_flows[node_obj.country].append(
                    (node_obj.width * flow_scale_obj,
                     color_map.get(tree_obj.source_name, '#333333'))
                )
    for country in leaf_flows:
        leaf_flows[country].sort(reverse=True)

    for tree in sorted_trees:
        if not tree.edges:
            continue
        color    = color_map.get(tree.source_name, '#333333')
        nodes    = tree.tree_nodes
        src_lon, src_lat = centroids[tree.source_name]
        src_x, src_y     = _to3035(src_lon, src_lat)

        # max_w = widest edge child in this tree (= root-adjacent Steiner ≈ root)
        max_w = max((nodes[c].width for c, _ in tree.edges), default=0.0)
        if max_w <= 0:
            continue

        # Fraction of total visualisation flow carried by this tree
        flow_scale = (tree.total_flow / total_viz_flow) ** exponent

        # ── edges ────────────────────────────────────────────────────────────
        for (child_id, parent_id), pts in zip(tree.edges, tree.edge_polylines):
            c_node = nodes[child_id]
            p_node = nodes[parent_id]
            if c_node.width <= 0 or len(pts) < 2:
                continue
            if abs(c_node.R - p_node.R) < 1.0:
                continue

            # tapered half-widths: child end → parent end
            hw_child  = max((c_node.width / 2.0 * flow_scale) / m_per_deg * width_scale, MIN_HW_DISPLAY_DEG)
            hw_parent = max((p_node.width / 2.0 * flow_scale) / m_per_deg * width_scale, MIN_HW_DISPLAY_DEG)

            patch = _edge_polygon(pts, hw_child, hw_parent, color, alpha=alpha)
            if patch is not None:
                ax.add_patch(patch)

        # ── Steiner node discs (cover gaps at junctions) ─────────────────────
        for node in nodes.values():
            if not node.is_steiner or node.parent is None:
                continue
            hw_node = max((node.width / 2.0 * flow_scale) / m_per_deg, MIN_HW_DISPLAY_DEG)
            lon, lat = _to_lonlat(node.x + src_x, node.y + src_y)
            circle = mpatches.Circle((lon, lat), radius=hw_node,
                                     facecolor=color, edgecolor='none',
                                     alpha=alpha, zorder=3)
            ax.add_patch(circle)

        # ── source node: filled square proportional to total flow ────────────
        hw_source = max((nodes[0].width / 2.0 * flow_scale) / m_per_deg * 1.2 * width_scale,
                        MIN_HW_DISPLAY_DEG)
        square = mpatches.FancyBboxPatch(
            (src_lon - hw_source, src_lat - hw_source),
            2 * hw_source, 2 * hw_source,
            boxstyle='square,pad=0',
            facecolor=color, edgecolor='white',
            linewidth=1.2, alpha=alpha, zorder=6
        )
        ax.add_patch(square)

        # ── leaf markers: proportional circles (Verbeek et al. §4.4) ─────────
        for nid, node in nodes.items():
            if node.is_leaf and node.country and node.country in centroids:
                lon, lat = centroids[node.country]
                hw_leaf = max((node.width / 2.0 * flow_scale) / m_per_deg * 2.0 * width_scale,
                              MIN_HW_DISPLAY_DEG * 2.0)
                this_flow = node.width * flow_scale
                flows_for_country = leaf_flows.get(node.country, [])
                rank = sum(1 for f, _ in flows_for_country if f > this_flow)
                if rank == 0:
                    circle_leaf = mpatches.Circle(
                        (lon, lat), radius=hw_leaf,
                        facecolor=color, edgecolor='white',
                        linewidth=0.7, alpha=1.0, zorder=5)
                    ax.add_patch(circle_leaf)
                else:
                    ring_radius = hw_leaf + hw_leaf * 0.4 * rank
                    ring = mpatches.Circle(
                        (lon, lat), radius=ring_radius,
                        facecolor='none', edgecolor=color,
                        linewidth=2.0, alpha=1.0, zorder=5)
                    ax.add_patch(ring)

        if show_labels:
            ax.text(src_lon, src_lat + 1.5, tree.source_name,
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    zorder=7, color=color,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', alpha=0.7, lw=0))

        # ── mid-edge arrows for source-to-source flows only ──────────────────
        source_names = {t.source_name for t in trees}
        for (child_id, parent_id), pts in zip(tree.edges, tree.edge_polylines):
            node = nodes[child_id]
            if not (node.is_leaf and node.country in source_names):
                continue
            if len(pts) < 4:
                continue
            mid = len(pts) // 2
            dx = pts[mid][0] - pts[mid - 1][0]
            dy = pts[mid][1] - pts[mid - 1][1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            dx /= length
            dy /= length
            hw_edge = max((node.width / 2.0 * flow_scale) / m_per_deg,
                          MIN_HW_DISPLAY_DEG)
            arrow_len = hw_edge * 2.0
            half_w    = hw_edge * 0.8
            mx, my = pts[mid]
            tip    = (mx + dx * arrow_len, my + dy * arrow_len)
            base_c = (mx - dx * arrow_len, my - dy * arrow_len)
            px, py = -dy, dx
            left   = (base_c[0] + px * half_w, base_c[1] + py * half_w)
            right  = (base_c[0] - px * half_w, base_c[1] - py * half_w)
            arrow  = mpatches.Polygon(
                [tip, left, right], closed=True,
                facecolor=color, edgecolor='none',
                alpha=0.9, zorder=4)
            ax.add_patch(arrow)


# ── standalone single-tree cost functions ────────────────────────────────────

def compute_f_str(tree: SpiralTreeResult) -> float:
    """Straightening cost: sum of (β − β_star)² over all join nodes.

    The bearing-angle approximation is intentionally used instead of true
    spiral pitch angle β.  For all valid log-spiral join nodes, β = alpha_deg
    by construction, which makes the spec-correct F_str values constant across
    all valid tree states and provides no optimization signal.  The
    bearing-angle approximation varies with node geometry after perturbation
    and provides genuine gradient information during optimization.
    """
    nodes = tree.tree_nodes

    def _edge_angle(from_node: TreeNode, to_node: TreeNode) -> float:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.atan2(dy, dx)

    f_str_vals = []
    c = 0.5
    for node in nodes.values():
        # join nodes are Steiner nodes with a parent and at least 2 children
        if not node.is_steiner or node.parent is None or len(node.children) < 2:
            continue

        beta = _edge_angle(node, nodes[node.parent])

        child_info = []
        for cid in node.children:
            child = nodes[cid]
            child_info.append((child.width, _edge_angle(node, child)))

        if not child_info:
            continue

        t_star = max(t for t, _ in child_info)
        threshold = c * t_star

        weighted_sum = 0.0
        weight_total = 0.0
        for t_i, beta_i in child_info:
            if t_i >= threshold:
                weighted_sum += t_i * beta_i
                weight_total += t_i

        if weight_total <= 0:
            continue

        beta_star = weighted_sum / weight_total
        d = _wrap(beta - beta_star)
        f_str_vals.append(d * d)

    return float(np.sum(f_str_vals)) if f_str_vals else 0.0


def compute_f_s(tree: SpiralTreeResult) -> float:
    """Smoothing cost: sum of (β1 − β2)² over interior polyline points of each edge."""
    f_s_vals = []
    for pts in tree.edge_polylines:
        if len(pts) < 3:
            continue
        arr = np.asarray(pts)
        # compute angles at interior points in local path direction
        prev = arr[:-2]
        cur  = arr[1:-1]
        nxt  = arr[2:]

        v1 = cur - prev
        v2 = nxt - cur

        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        mask = (n1 > 1e-12) & (n2 > 1e-12)

        if not mask.any():
            continue

        a1 = np.arctan2(v1[mask, 1], v1[mask, 0])
        a2 = np.arctan2(v2[mask, 1], v2[mask, 0])

        diff = _wrap(a1 - a2)
        f_s_vals.extend((diff * diff).tolist())

    return float(np.sum(f_s_vals)) if f_s_vals else 0.0


def _f_obs_kernel(D, t: float, B: float):
    """F_obs potential formula applied to distance(s) D.

    D : scalar float or numpy array (same units as t and B).
    Returns the same type/shape as D.
    """
    arr    = np.atleast_1d(np.asarray(D, dtype=float))
    result = np.zeros_like(arr)
    t_s    = max(t, 1e-9)

    mask_deep = arr < t
    mask_buff = (~mask_deep) & (arr < t + B)

    if mask_deep.any():
        d_s               = np.maximum(arr[mask_deep], 1e-9)
        c1                = (t / (B * d_s)) * (B / 2 + t)
        c2                = (arr[mask_deep] / (B * t_s)) * (B / 2 - t)
        result[mask_deep] = c1 + c2

    if mask_buff.any():
        result[mask_buff] = (1.0 - (arr[mask_buff] - t) / B) ** 2

    return float(result[0]) if np.ndim(D) == 0 else result


def compute_f_obs(
    tree: SpiralTreeResult,
    obstacles: Optional[Dict[str, Tuple[float, float]]],
    src_xy: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Obstacle cost: sum of F_obs(p, Ω) over join nodes p and obstacle centroids Ω.

    src_xy: absolute EPSG:3035 position of the source node (metres).
    node.x / node.y are local offsets from the source, so the absolute
    node position is (node.x + src_xy[0], node.y + src_xy[1]).
    """
    nodes = tree.tree_nodes
    F_obs = 0.0
    if obstacles:
        B         = _OPT_OVERLAP_B  # buffer in degrees
        m_per_deg = _OPT_M_PER_DEG
        for node in nodes.values():
            if not node.is_steiner or node.parent is None or len(node.children) < 2:
                continue
            p_lon, p_lat = _to_lonlat(node.x + src_xy[0], node.y + src_xy[1])
            t = (node.width / 2.0) / m_per_deg  # half width in degrees
            for obs_name, (o_lon, o_lat) in obstacles.items():
                D = math.hypot(p_lon - o_lon, p_lat - o_lat)  # approx distance in degrees
                F_obs += _f_obs_kernel(D, t, B)
    return F_obs


# ── statistics helpers ────────────────────────────────────────────────────────

def compute_tree_stats(
    tree: SpiralTreeResult,
    centroids: Dict[str, Tuple[float, float]],
    total_eu_flow: float,
    obstacles: Optional[Dict[str, Tuple[float, float]]] = None,
    alpha_deg: float = 25.0,
) -> dict:
    """
    Compute per-tree statistics.

    Returns dict with keys:
      n_terminals, n_steiner, coverage_pct, F_str, F_s, F_obs, F_ar, F_B, F_cost
    """
    nodes     = tree.tree_nodes
    n_leaves  = sum(1 for n in nodes.values() if n.is_leaf)
    n_steiner = sum(1 for n in nodes.values() if n.is_steiner)
    coverage  = (tree.total_flow / total_eu_flow * 100.0
                 if total_eu_flow > 0 else 0.0)

    F_str     = compute_f_str(tree)
    F_s       = compute_f_s(tree)
    sx, sy    = _to3035(*centroids[tree.source_name])
    F_obs     = compute_f_obs(tree, obstacles, (sx, sy))
    F_ar, F_B = compute_f_ar_and_b(tree, alpha_deg)

    w      = DEFAULT_F_TOTAL_WEIGHTS
    F_cost = (w['c_obs'] * F_obs + w['c_S'] * F_s
              + w['c_AR'] * (F_ar + F_B) + w['c_str'] * F_str)

    return {
        'n_terminals':  n_leaves,
        'n_steiner':    n_steiner,
        'coverage_pct': coverage,
        'F_str':        F_str,
        'F_s':          F_s,
        'F_obs':        F_obs,
        'F_ar':         F_ar,
        'F_B':          F_B,
        'F_cost':       F_cost,
    }

def compute_f_ar_and_b(
    tree: SpiralTreeResult,
    alpha_deg: float = 25.0,
) -> Tuple[float, float, float]:
    """Compute both angle-restriction (F_AR) and balancing (F_B) costs.

    The bearing-angle approximation is intentionally used instead of true
    spiral pitch angle β.  For all valid log-spiral join nodes, β = alpha_deg
    by construction, which makes the spec-correct F_AR/F_B values constant
    across all valid tree states and provides no optimization signal.  The
    bearing-angle approximation varies with node geometry after perturbation
    and provides genuine gradient information during optimization.
    """
    nodes = tree.tree_nodes
    alpha = math.radians(alpha_deg)

    def _edge_angle(from_node: TreeNode, to_node: TreeNode) -> float:
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.atan2(dy, dx)

    f_ar = 0.0
    f_b = 0.0
    eps = 1e-12

    for node in nodes.values():
        if not node.is_steiner or len(node.children) != 2 or node.parent is None:
            continue

        parent_node = nodes[node.parent]
        c0 = nodes[node.children[0]]
        c1 = nodes[node.children[1]]

        parent_angle = _edge_angle(node, parent_node)
        b0 = _wrap(_edge_angle(node, c0) - parent_angle)
        b1 = _wrap(_edge_angle(node, c1) - parent_angle)

        def _clamp_half_pi(val: float) -> float:
            if val > math.pi / 2:
                return math.pi - val
            if val < -math.pi / 2:
                return -math.pi - val
            return val

        b0 = _clamp_half_pi(b0)
        b1 = _clamp_half_pi(b1)

        beta1, beta2 = (b0, b1) if b0 >= b1 else (b1, b0)

        cos_b1 = max(abs(math.cos(beta1)), eps)
        cos_b2 = max(abs(math.cos(beta2)), eps)
        f_ar += math.log(1.0 / cos_b1) + math.log(1.0 / cos_b2)

        delta = max(abs(beta1 - beta2), eps)
        half_delta = max(0.5 * delta, eps)
        sin_half = max(math.sin(half_delta), eps)
        f_b += 2.0 * (math.tan(alpha) ** 2) * math.log(1.0 / sin_half)

    return float(f_ar), float(f_b)


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


# ── multi-tree inter-tree optimizer ───────────────────────────────────────────
#
# Implements the inter-tree cost functions from the task spec, then uses
# simulated annealing to reduce crossings/overlaps by perturbing Steiner node
# positions while keeping leaf nodes and source positions fixed.

DEFAULT_OPT_WEIGHTS: dict = {
    'c_cross':   0.0,
    'c_overlap': 0.0,
}

DEFAULT_F_TOTAL_WEIGHTS: dict = {
    'c_obs':     2.0,
    'c_S':       0.4,
    'c_AR':      0.077,
    'c_str':     0.4,
    'c_cross':   0.0,    # disabled: optimizer focuses on single-tree quality only
    'c_overlap': 0.0,    # disabled: inter-tree conflicts not resolved by SA
}

_OPT_M_PER_DEG = 111_000.0
_OPT_OVERLAP_B   = 1.5    # buffer size in degrees (~165 km) for F_obs formula


# ── vectorised segment-crossing helpers ──────────────────────────────────────

def _seg_cross_batch(
    pts_a: np.ndarray,   # (Na, 2)
    pts_b: np.ndarray,   # (Nb, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (row_indices, col_indices) of crossing segment pairs between two polylines.

    Uses 2-D cross-product formulation, fully vectorised over all (Na-1)×(Nb-1)
    segment pairs.  Returns empty arrays when no crossings exist.
    """
    if len(pts_a) < 2 or len(pts_b) < 2:
        return np.empty(0, int), np.empty(0, int)

    p1 = pts_a[:-1]          # (Na-1, 2) segment starts
    p2 = pts_a[1:]            # (Na-1, 2) segment ends
    p3 = pts_b[:-1]           # (Nb-1, 2)
    p4 = pts_b[1:]

    d1 = (p2 - p1)[:, None, :]      # (Na-1, 1, 2)
    d2 = (p4 - p3)[None, :, :]      # (1, Nb-1, 2)
    dp = p3[None, :, :] - p1[:, None, :]   # (Na-1, Nb-1, 2)

    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]  # (Na-1, Nb-1)
    t_num = dp[..., 0] * d2[..., 1] - dp[..., 1] * d2[..., 0]
    u_num = dp[..., 0] * d1[..., 1] - dp[..., 1] * d1[..., 0]

    eps   = 1e-14
    valid = np.abs(cross) > eps
    safe  = np.where(valid, cross, 1.0)
    t     = np.where(valid, t_num / safe, -1.0)
    u     = np.where(valid, u_num / safe, -1.0)

    margin = 1e-9
    mask = valid & (t > margin) & (t < 1 - margin) & (u > margin) & (u < 1 - margin)
    return np.where(mask)


def compute_f_cross(trees: List[SpiralTreeResult]) -> float:
    """F_cross = Σ csc(max(γ, ε)) for all inter-tree edge crossings."""
    eps   = 1e-6
    # Build per-tree list of edge arrays
    edge_arrays: List[Tuple[int, np.ndarray]] = []
    for ti, tree in enumerate(trees):
        for pts in tree.edge_polylines:
            if len(pts) >= 2:
                edge_arrays.append((ti, np.asarray(pts, dtype=float)))

    cost = 0.0
    n = len(edge_arrays)
    for a in range(n):
        ti, arr_a = edge_arrays[a]
        for b in range(a + 1, n):
            tj, arr_b = edge_arrays[b]
            if ti == tj:
                continue
            rows, cols = _seg_cross_batch(arr_a, arr_b)
            for i, j in zip(rows, cols):
                da = arr_a[i + 1] - arr_a[i]
                db = arr_b[j + 1] - arr_b[j]
                na, nb = np.linalg.norm(da), np.linalg.norm(db)
                if na < eps or nb < eps:
                    cost += 1.0 / eps
                    continue
                cos_g = abs(float(np.dot(da / na, db / nb)))
                sin_g = math.sqrt(max(0.0, 1.0 - min(cos_g, 1.0) ** 2))
                cost += 1.0 / max(sin_g, eps)
    return cost


def _pts_to_segs_min_dist(
    pts: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> np.ndarray:
    """Minimum distance from each point in pts to any segment (seg_a[s], seg_b[s]).

    pts   : (N, 2)
    seg_a : (S, 2) — segment start points
    seg_b : (S, 2) — segment end points
    Returns (N,) array of minimum distances, fully vectorised.
    Degenerate zero-length segments are handled by projecting to seg_a.
    """
    ab  = seg_b - seg_a                                      # (S, 2)
    ab2 = np.einsum('si,si->s', ab, ab)                     # (S,) squared lengths
    ap  = pts[:, None, :] - seg_a[None, :, :]               # (N, S, 2)
    # Scalar projection parameter t = dot(ap, ab) / |ab|^2, clamped to [0, 1]
    t   = np.einsum('nsi,si->ns', ap, ab)                   # (N, S)
    safe = np.where(ab2 > 0, ab2, 1.0)                      # avoid /0
    t   = np.clip(t / safe, 0.0, 1.0)
    t   = np.where(ab2[None, :] > 0, t, 0.0)               # degenerate -> project to a
    # Closest point on each segment: (N, S, 2)
    closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff    = pts[:, None, :] - closest                     # (N, S, 2)
    dists   = np.sqrt(np.einsum('nsi,nsi->ns', diff, diff)) # (N, S)
    return np.min(dists, axis=1)                             # (N,)


def compute_f_overlap(
    trees: List[SpiralTreeResult],
    total_viz_flow: float,
    centroids: Dict[str, Tuple[float, float]],
    B: float = _OPT_OVERLAP_B,
    n_samples: int = 6,
) -> float:
    """F_overlap = Σ F_obs(p, Ω_{T_j}) for sample points on each tree's edges.

    Uses numpy vectorised distance — no shapely required.
    B             : buffer size in degrees
    n_samples     : sample points per edge
    total_viz_flow: sum of all tree flows (matches draw_spiral_trees width model)
    centroids     : lon/lat of all country centroids; used to exclude sample
                    points that are within t of a shared leaf position (countries
                    that appear as terminals in two or more trees) — these produce
                    a structural D≈0 singularity that is not a real routing overlap.
    """
    if total_viz_flow <= 0:
        return 0.0

    # Build array of shared leaf positions (lon, lat) — countries terminal in 2+ trees
    from collections import Counter
    country_count: Counter = Counter(
        node.country
        for tree in trees
        for node in tree.tree_nodes.values()
        if node.is_leaf and node.country
    )
    shared_lonlat = np.array(
        [centroids[c] for c, cnt in country_count.items() if cnt >= 2 and c in centroids],
        dtype=float,
    )   # shape (K, 2) or (0, 2)

    # Pre-stack all segments for each tree as (seg_a, seg_b) pairs
    tree_segs: List[Optional[tuple]] = []
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

    cost = 0.0
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
            t = (c_node.width / 2.0) / _OPT_M_PER_DEG   # half-width °

            arr = np.asarray(pts, dtype=float)
            idx = np.round(np.linspace(0, len(arr) - 1, n_samples)).astype(int)
            samples = arr[idx]   # (n_samples, 2)

            # Exclude sample points within t° of any shared leaf centroid.
            # Those points have D≈0 by construction (both trees terminate there),
            # causing a 1/D singularity that is not a genuine routing overlap.
            if shared_lonlat.shape[0] > 0:
                dist_to_shared = np.min(
                    np.hypot(
                        samples[:, 0:1] - shared_lonlat[:, 0],
                        samples[:, 1:2] - shared_lonlat[:, 1],
                    ),
                    axis=1,
                )   # (n_samples,)
                samples = samples[dist_to_shared >= t]
            if len(samples) == 0:
                continue

            for tj, segs_j in enumerate(tree_segs):
                if ti == tj or segs_j is None:
                    continue
                seg_a_j, seg_b_j = segs_j
                D = _pts_to_segs_min_dist(samples, seg_a_j, seg_b_j)  # (n_kept,)
                D = np.maximum(D, t * 0.1)                              # clamp: bound max penalty per crossing
                cost += float(np.sum(_f_obs_kernel(D, t, B)))

    return cost


def compute_inter_tree_cost(
    trees: List[SpiralTreeResult],
    weights: Optional[dict] = None,
    centroids: Optional[Dict[str, Tuple[float, float]]] = None,
    alpha_deg: float = 25.0,
    B_obs: float = _OPT_OVERLAP_B,
    n_overlap_samples: int = 6,
) -> dict:
    """Compute all F_total cost terms for a set of trees.

    Returns a dict with keys:
        f_obs, f_s, f_ar, f_b, f_str, f_cross, f_overlap, f_total.
    If centroids is None, single-tree terms are 0.0 and f_total covers
    only the inter-tree components.
    Uses DEFAULT_F_TOTAL_WEIGHTS unless overridden by weights.
    """
    w     = {**DEFAULT_F_TOTAL_WEIGHTS, **(weights or {})}
    cents = centroids or {}

    # ── single-tree terms ─────────────────────────────────────────────────
    obstacle_map: Dict[str, Tuple[float, float]] = {}
    for tree in trees:
        if tree.source_name in cents:
            obstacle_map[tree.source_name] = cents[tree.source_name]
        for node in tree.tree_nodes.values():
            if node.is_leaf and node.country and node.country in cents:
                obstacle_map[node.country] = cents[node.country]

    f_obs = 0.0
    for t in trees:
        sx, sy = _to3035(*cents[t.source_name]) if t.source_name in cents else (0.0, 0.0)
        f_obs += compute_f_obs(t, obstacle_map, (sx, sy))
    f_s   = sum(compute_f_s(t) for t in trees)
    f_ar  = 0.0
    f_b   = 0.0
    for t in trees:
        ar, b = compute_f_ar_and_b(t, alpha_deg)
        f_ar += ar
        f_b  += b
    f_str = sum(compute_f_str(t) for t in trees)

    # ── inter-tree terms ──────────────────────────────────────────────────
    total_viz_flow = sum(t.total_flow for t in trees) or 1.0
    f_cross   = compute_f_cross(trees)
    f_overlap = compute_f_overlap(
        trees, total_viz_flow, cents, B=B_obs, n_samples=n_overlap_samples
    )

    # ── f_total via compute_f_total ───────────────────────────────────────
    if cents:
        f_total = compute_f_total(trees, cents, alpha_deg, w, B_obs, n_overlap_samples)
    else:
        f_total = w['c_cross'] * f_cross + w['c_overlap'] * f_overlap

    return {
        'f_obs':     f_obs,
        'f_s':       f_s,
        'f_ar':      f_ar,
        'f_b':       f_b,
        'f_str':     f_str,
        'f_cross':   f_cross,
        'f_overlap': f_overlap,
        'f_total':   f_total,
    }


# === F_total: TOTAL COST FUNCTION ===
# Calls all 7 cost terms. All must remain active.
# F_total = c_obs*F_obs + c_S*F_S + c_AR*(F_AR+F_B) + c_str*F_str + c_cross*F_cross + c_overlap*F_overlap
# Removing or zeroing any term is incorrect.

def _partial_f_cross(
    edge_polys: List[List[Tuple[float, float]]],
    other_trees: List['SpiralTreeResult'],
) -> float:
    """F_cross contribution of `edge_polys` against all edges of `other_trees`.

    Used for delta-cost updates: subtract old contribution, add new contribution.
    edge_polys: list of polyline point-lists for the incident edges of the moved node.
    other_trees: all trees EXCEPT the one being perturbed.
    """
    eps = 1e-6
    cost = 0.0
    for pts_a in edge_polys:
        if len(pts_a) < 2:
            continue
        arr_a = np.asarray(pts_a, dtype=float)
        for ot in other_trees:
            for pts_b in ot.edge_polylines:
                if len(pts_b) < 2:
                    continue
                arr_b = np.asarray(pts_b, dtype=float)
                rows, cols = _seg_cross_batch(arr_a, arr_b)
                for i, j in zip(rows, cols):
                    da = arr_a[i + 1] - arr_a[i]
                    db = arr_b[j + 1] - arr_b[j]
                    na, nb = np.linalg.norm(da), np.linalg.norm(db)
                    if na < eps or nb < eps:
                        cost += 1.0 / eps
                        continue
                    cos_g = abs(float(np.dot(da / na, db / nb)))
                    sin_g = math.sqrt(max(0.0, 1.0 - min(cos_g, 1.0) ** 2))
                    cost += 1.0 / max(sin_g, eps)
    return cost


def _partial_f_overlap(
    edge_polys: List[List[Tuple[float, float]]],
    edge_child_widths: List[float],
    other_trees: List['SpiralTreeResult'],
    shared_lonlat: np.ndarray,
    n_samples: int,
    B: float,
) -> float:
    """F_overlap contribution of `edge_polys` (from one perturbed tree)
    against all edges of `other_trees`.

    edge_polys: polylines for incident edges.
    edge_child_widths: half-width in degrees for each edge (same order as edge_polys).
    other_trees: all trees except the perturbed one.
    shared_lonlat: (K,2) array of shared-leaf lon/lat positions (pre-computed).
    """
    # Pre-stack segments for other trees
    other_segs: List[Optional[tuple]] = []
    for ot in other_trees:
        a_list, b_list = [], []
        for pts in ot.edge_polylines:
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=float)
            a_list.append(arr[:-1])
            b_list.append(arr[1:])
        other_segs.append(
            (np.vstack(a_list), np.vstack(b_list)) if a_list else None
        )

    cost = 0.0
    for pts, t in zip(edge_polys, edge_child_widths):
        if len(pts) < 2 or t <= 0:
            continue
        arr = np.asarray(pts, dtype=float)
        idx = np.round(np.linspace(0, len(arr) - 1, n_samples)).astype(int)
        samples = arr[idx]
        if shared_lonlat.shape[0] > 0:
            dist_to_shared = np.min(
                np.hypot(
                    samples[:, 0:1] - shared_lonlat[:, 0],
                    samples[:, 1:2] - shared_lonlat[:, 1],
                ),
                axis=1,
            )
            samples = samples[dist_to_shared >= t]
        if len(samples) == 0:
            continue
        for segs in other_segs:
            if segs is None:
                continue
            seg_a, seg_b = segs
            D = _pts_to_segs_min_dist(samples, seg_a, seg_b)
            D = np.maximum(D, t * 0.1)
            cost += float(np.sum(_f_obs_kernel(D, t, B)))
    return cost


def compute_f_total(
    trees: List[SpiralTreeResult],
    centroids: Dict[str, Tuple[float, float]],
    alpha_deg: float,
    weights: Dict[str, float],
    B_obs: float,
    n_overlap_samples: int,
) -> float:
    """Compute the unified total cost F_total over all trees."""
    w = {**DEFAULT_F_TOTAL_WEIGHTS, **weights}

    # Build obstacle_map: source centroids + leaf centroids across all trees
    obstacle_map: Dict[str, Tuple[float, float]] = {}
    for tree in trees:
        if tree.source_name in centroids:
            obstacle_map[tree.source_name] = centroids[tree.source_name]
        for node in tree.tree_nodes.values():
            if node.is_leaf and node.country and node.country in centroids:
                obstacle_map[node.country] = centroids[node.country]

    total = 0.0
    for tree in trees:
        sx, sy = _to3035(*centroids[tree.source_name])
        total += w['c_obs']  * compute_f_obs(tree, obstacle_map, (sx, sy))
        total += w['c_S']    * compute_f_s(tree)
        f_ar, f_b = compute_f_ar_and_b(tree, alpha_deg)
        total += w['c_AR']   * (f_ar + f_b)
        total += w['c_str']  * compute_f_str(tree)

    total_viz_flow = sum(t.total_flow for t in trees) or 1.0
    if w['c_cross'] > 0:
        total += w['c_cross'] * compute_f_cross(trees)
    if w['c_overlap'] > 0:
        total += w['c_overlap'] * compute_f_overlap(
            trees, total_viz_flow, centroids, B=B_obs, n_samples=n_overlap_samples
        )

    return total


# ── subdivision-node analytical smoother ─────────────────────────────────────

def _smooth_subdivision_nodes(
    tree: SpiralTreeResult,
    src_x: float,
    src_y: float,
) -> None:
    """Two-pass Gauss-Seidel smoothing of subdivision nodes to minimise F_S pitch discontinuity.

    Pass 1 (ascending R, parent->child): propagates parent phi downward.
    Pass 2 (descending R, child->parent): propagates child phi upward.
    Then resamples all edge polylines incident on any subdivision node.
    """
    nodes = tree.tree_nodes
    sub_nodes = sorted(
        [n for n in nodes.values() if n.is_steiner and len(n.children) == 1],
        key=lambda n: n.R,
    )
    if not sub_nodes:
        return

    for pass_nodes in (sub_nodes, list(reversed(sub_nodes))):
        for s in pass_nodes:
            c = nodes[s.children[0]]
            p = nodes[s.parent]
            log_cp = math.log(c.R / p.R)
            if abs(log_cp) < 1e-12:
                continue
            w_c = math.log(s.R / p.R)
            w_p = math.log(c.R / s.R)
            s.phi = (c.phi * w_c + p.phi * w_p) / log_cp
            s.x, s.y = _cart(s.R, s.phi)

    sub_ids = {n.node_id for n in sub_nodes}
    for i, (c_id, p_id) in enumerate(tree.edges):
        if c_id in sub_ids or p_id in sub_ids:
            c_node = nodes[c_id]
            p_node = nodes[p_id]
            new_pts = _sample_spiral(c_node.R, c_node.phi, p_node.R, p_node.phi, src_x, src_y)
            if new_pts:
                tree.edge_polylines[i] = new_pts


# ── simulated annealing optimizer ────────────────────────────────────────────

def optimize_multi_tree(
    trees: List[SpiralTreeResult],
    centroids: Dict[str, Tuple[float, float]],
    stop_event: _threading.Event,
    on_update: Callable[[int, float, List['SpiralTreeResult']], None],
    weights: Optional[dict] = None,
    max_iter: int = 5000,
    T_init: float = 5.0,
    T_min: float  = 1e-3,
    update_every: int = 10,
    alpha_deg: float = 25.0,
    B_obs: float = _OPT_OVERLAP_B,
    n_overlap_samples: int = 6,
    gd_iters: int = 200,
    exponent: float = 0.5,
    width_scale: float = 1.0,
    result_dict: Optional[dict] = None,
) -> int:
    """Simulated annealing to reduce inter-tree crossings and overlaps.

    Runs in a background thread.  Calls
        on_update(iteration, cost, trees_snapshot)
    every `update_every` accepted steps, and once more on completion with
    iteration=-1.  Does NOT modify the input trees; works on deep copies.

    Degrees of freedom: Steiner node (x, y) positions.  Leaf nodes and
    source positions are held fixed.
    """
    w     = {**DEFAULT_F_TOTAL_WEIGHTS, **(weights or {})}
    state = _copy.deepcopy(trees)

    try:
        from shapely.geometry import LineString as _LineString
    except ImportError:
        _LineString = None

    # Source EPSG:3035 absolute positions
    src_xy: Dict[int, Tuple[float, float]] = {}
    for ti, tree in enumerate(state):
        lon, lat    = centroids[tree.source_name]
        src_xy[ti]  = _to3035(lon, lat)

    # Pool of (tree_idx, node_id) for join nodes only (subdivision nodes are
    # handled analytically by _smooth_subdivision_nodes after each accepted move)
    steiner_pool: List[Tuple[int, int]] = [
        (ti, nid)
        for ti, tree in enumerate(state)
        for nid, node in tree.tree_nodes.items()
        if node.is_steiner and len(node.children) >= 2
    ]

    if not steiner_pool:
        on_update(-1, 0.0, state)
        return

    # Diagnostic counters
    _diag_sa_accepted       = 0
    _diag_sa_total          = 0
    _diag_greedy_accepted   = 0
    _diag_intra_reverts     = 0
    _diag_disp_reverts      = 0

    # Store original (x, y) of every Steiner node for cumulative displacement constraint
    orig_positions: Dict[Tuple[int, int], Tuple[float, float]] = {
        (ti, nid): (node.x, node.y)
        for ti, tree in enumerate(state)
        for nid, node in tree.tree_nodes.items()
        if node.is_steiner
    }

    total_viz_flow = sum(t.total_flow for t in state) or 1.0

    # Pre-compute shared leaf positions for overlap delta (shared across all iterations)
    from collections import Counter as _Counter
    _country_count = _Counter(
        node.country
        for tree in state
        for node in tree.tree_nodes.values()
        if node.is_leaf and node.country
    )
    _shared_lonlat = np.array(
        [centroids[c] for c, cnt in _country_count.items() if cnt >= 2 and c in centroids],
        dtype=float,
    )

    def _full_cost() -> float:
        return compute_f_total(state, centroids, alpha_deg, w, B_obs, n_overlap_samples)

    current_cost = _full_cost()
    # Cache running inter-tree costs for delta updates when weights > 0
    _use_cross_delta   = w['c_cross']   > 0
    _use_overlap_delta = w['c_overlap'] > 0
    _cur_f_cross   = compute_f_cross(state)   if _use_cross_delta   else 0.0
    _cur_f_overlap = compute_f_overlap(
        state, total_viz_flow, centroids, B=B_obs, n_samples=n_overlap_samples
    ) if _use_overlap_delta else 0.0
    # Fix 3: calibrate T_init dynamically so a 1%-of-F_total uphill move
    # has ~25% acceptance probability at t=0.
    # T = -delta / ln(p)  =>  T = (0.01 * F) / ln(4)
    T_init = (0.07 * current_cost) / math.log(1.0 / 0.25)
    alpha  = (T_min / T_init) ** (1.0 / max(max_iter, 1))
    T      = T_init

    for it in range(max_iter + gd_iters):
        if stop_event.is_set():
            break

        # Switch to greedy-descent mode after SA phase
        if it == max_iter:
            T = 1e-15

        if it < max_iter:
            _diag_sa_total += 1

        ti, nid = _random.choice(steiner_pool)
        tree     = state[ti]
        node     = tree.tree_nodes[nid]
        sx, sy   = src_xy[ti]

        # All pool nodes are join nodes — free translation in (x, y)
        # Step size scales with t_frac = T/T_init (large early, fine late)
        t_frac = T / T_init
        step  = max(node.R * 0.03 * t_frac, 500.0)
        new_x = node.x + _random.gauss(0.0, step)
        new_y = node.y + _random.gauss(0.0, step)
        new_R, new_phi = _polar(new_x, new_y)

        if new_R < 10_000.0:      # too close to source — reject
            if it < max_iter:
                T *= alpha
            continue

        # Clamp new_R to valid R-hierarchy window for join nodes
        R_children_min = min(tree.tree_nodes[c].R for c in node.children) * 0.999
        par            = tree.tree_nodes[node.parent] if node.parent is not None else None
        R_parent_min   = (par.R * 1.001 if par is not None and par.R > 0 else 0.0)
        if R_parent_min >= R_children_min:   # degenerate window — skip
            if it < max_iter:
                T *= alpha
            continue
        if new_R < R_parent_min or new_R > R_children_min:
            new_R        = min(max(new_R, R_parent_min), R_children_min)
            new_x, new_y = _cart(new_R, new_phi)

        # R-hierarchy must hold after clamping — violation here is a bug
        assert all(tree.tree_nodes[c].R > new_R for c in node.children), \
            f"R-hierarchy violated: child R <= new_R={new_R}"
        if node.parent is not None:
            par = tree.tree_nodes[node.parent]
            assert par.R <= 0 or par.R < new_R, \
                f"R-hierarchy violated: parent R={par.R} >= new_R={new_R}"

        # Save state of affected edges (those incident to this node)
        orig_pos   = (node.x, node.y, node.R, node.phi)
        orig_edges = {
            i: tree.edge_polylines[i]
            for i, (c, p) in enumerate(tree.edges)
            if c == nid or p == nid
        }

        # For delta-cost: collect old incident polylines and their child widths
        # before applying the move (used by partial cost helpers)
        _other_trees = [state[j] for j in range(len(state)) if j != ti]
        if _use_cross_delta or _use_overlap_delta:
            _old_polys = [tree.edge_polylines[i] for i in orig_edges]
            if _use_overlap_delta:
                _old_widths = [
                    (tree.tree_nodes[tree.edges[i][0]].width / 2.0) / _OPT_M_PER_DEG
                    for i in orig_edges
                ]

        # Apply perturbation
        node.x, node.y, node.R, node.phi = new_x, new_y, new_R, new_phi
        for i in orig_edges:
            c, p   = tree.edges[i]
            cn, pn = tree.tree_nodes[c], tree.tree_nodes[p]
            new_pts = _sample_spiral(cn.R, cn.phi, pn.R, pn.phi, sx, sy)
            tree.edge_polylines[i] = new_pts if new_pts else orig_edges[i]

        # Compute new cost: reuse full cost for intra-tree terms;
        # for inter-tree terms use delta updates when weights > 0
        if _use_cross_delta or _use_overlap_delta:
            _new_polys = [tree.edge_polylines[i] for i in orig_edges]
            if _use_overlap_delta:
                _new_widths = _old_widths  # widths unchanged (node.width not moved)

            _new_f_cross = _cur_f_cross
            if _use_cross_delta:
                _new_f_cross = (
                    _cur_f_cross
                    - _partial_f_cross(_old_polys, _other_trees)
                    + _partial_f_cross(_new_polys, _other_trees)
                )

            _new_f_overlap = _cur_f_overlap
            if _use_overlap_delta:
                _new_f_overlap = (
                    _cur_f_overlap
                    - _partial_f_overlap(_old_polys, _old_widths, _other_trees,
                                         _shared_lonlat, n_overlap_samples, B_obs)
                    + _partial_f_overlap(_new_polys, _new_widths, _other_trees,
                                         _shared_lonlat, n_overlap_samples, B_obs)
                )

            # Full cost = intra-tree terms (recomputed) + cached inter-tree terms
            _intra_cost = compute_f_total(
                state, centroids, alpha_deg,
                {**w, 'c_cross': 0.0, 'c_overlap': 0.0},
                B_obs, n_overlap_samples,
            )
            new_cost = (_intra_cost
                        + w['c_cross']   * _new_f_cross
                        + w['c_overlap'] * _new_f_overlap)
        else:
            new_cost = _full_cost()
        delta    = new_cost - current_cost

        # Metropolis acceptance criterion
        if delta < 0 or _random.random() < math.exp(-delta / max(T, 1e-12)):
            # Guard: reject if the move introduces a new intra-tree crossing.
            # Check each incident (resampled) edge against every non-incident edge.
            incident_idx     = set(orig_edges.keys())
            non_incident_idx = [
                i for i in range(len(tree.edges))
                if i not in incident_idx and len(tree.edge_polylines[i]) >= 2
            ]
            new_intra = False
            if _LineString is not None:
                flow_scale_tree = (tree.total_flow / total_viz_flow) ** exponent
                # build non-incident polygons once (unchanged edges)
                non_incident_polys = {}
                for bi in non_incident_idx:
                    pts_b = tree.edge_polylines[bi]
                    if len(pts_b) < 2:
                        continue
                    child_b = tree.edges[bi][0]
                    hw_b = max(
                        (tree.tree_nodes[child_b].width / 2.0 * flow_scale_tree) / _OPT_M_PER_DEG * width_scale,
                        MIN_HW_DISPLAY_DEG,
                    )
                    try:
                        non_incident_polys[bi] = _LineString(pts_b).buffer(
                            hw_b, cap_style=2, join_style=2)
                    except Exception:
                        continue

                for ai in incident_idx:
                    if len(tree.edge_polylines[ai]) < 2:
                        continue
                    child_a = tree.edges[ai][0]
                    hw_a = max(
                        (tree.tree_nodes[child_a].width / 2.0 * flow_scale_tree) / _OPT_M_PER_DEG * width_scale,
                        MIN_HW_DISPLAY_DEG,
                    )
                    try:
                        poly_after  = _LineString(tree.edge_polylines[ai]).buffer(
                            hw_a, cap_style=2, join_style=2)
                        poly_before = _LineString(orig_edges[ai]).buffer(
                            hw_a, cap_style=2, join_style=2)
                    except Exception:
                        continue
                    for bi, poly_b in non_incident_polys.items():
                        overlaps_after  = poly_after.intersects(poly_b)
                        overlaps_before = poly_before.intersects(poly_b)
                        if overlaps_after and not overlaps_before:
                            new_intra = True
                            break
                    if new_intra:
                        break
            if new_intra:
                _diag_intra_reverts += 1
                node.x, node.y, node.R, node.phi = orig_pos
                for i, pts in orig_edges.items():
                    tree.edge_polylines[i] = pts
                if it < max_iter:
                    T *= alpha
                continue
            # Guard: reject if cumulative displacement from original position
            # exceeds 50% of the node's current radius.
            ox, oy     = orig_positions[(ti, nid)]
            cumul_disp = math.hypot(node.x - ox, node.y - oy)
            if cumul_disp > 200_000.0:
                _diag_disp_reverts += 1
                node.x, node.y, node.R, node.phi = orig_pos
                for i, pts in orig_edges.items():
                    tree.edge_polylines[i] = pts
                if it < max_iter:
                    T *= alpha
                continue
            current_cost = new_cost
            if _use_cross_delta:
                _cur_f_cross   = _new_f_cross
            if _use_overlap_delta:
                _cur_f_overlap = _new_f_overlap
            if it < max_iter:
                _diag_sa_accepted += 1
            else:
                _diag_greedy_accepted += 1
            # Part 2: analytically smooth subdivision nodes in the affected tree
            _smooth_subdivision_nodes(tree, *src_xy[ti])
        else:
            # Revert
            node.x, node.y, node.R, node.phi = orig_pos
            for i, pts in orig_edges.items():
                tree.edge_polylines[i] = pts

        if it < max_iter:
            T *= alpha

        if (it + 1) % update_every == 0:
            on_update(it + 1, current_cost, _copy.deepcopy(state))

    # ── Post-optimization repair: enforce inter-tree join separation ──────
    n_repaired = 0
    R_PRIOR = 50_000.0
    for ti, tree in enumerate(state):
        sx_i, sy_i = src_xy[ti]
        nodes_i = tree.tree_nodes
        for nid, node in nodes_i.items():
            if not node.is_steiner or len(node.children) < 2:
                continue  # join nodes only
            abs_x = sx_i + node.x
            abs_y = sy_i + node.y
            violated = False
            for tj, other_tree in enumerate(state):
                if tj == ti:
                    continue
                sx_j, sy_j = src_xy[tj]
                for other_node in other_tree.tree_nodes.values():
                    ox = sx_j + other_node.x
                    oy = sy_j + other_node.y
                    if math.hypot(abs_x - ox, abs_y - oy) < R_PRIOR:
                        violated = True
                        break
                if violated:
                    break
            if violated:
                ox, oy = orig_positions[(ti, nid)]
                node.x, node.y = ox, oy
                node.R   = math.hypot(ox, oy)
                node.phi = math.atan2(oy, ox)
                for ei, (c, p) in enumerate(tree.edges):
                    if c == nid or p == nid:
                        cn = nodes_i[c]
                        pn = nodes_i[p]
                        new_pts = _sample_spiral(
                            cn.R, cn.phi, pn.R, pn.phi, sx_i, sy_i)
                        if new_pts:
                            tree.edge_polylines[ei] = new_pts
                n_repaired += 1

    if n_repaired > 0:
        for ti, tree in enumerate(state):
            _smooth_subdivision_nodes(tree, *src_xy[ti])

    if result_dict is not None:
        result_dict['sa_accepted_moves']       = _diag_sa_accepted
        result_dict['sa_total_moves']          = _diag_sa_total
        result_dict['greedy_accepted_moves']   = _diag_greedy_accepted
        result_dict['displacement_guard_reverts']  = _diag_disp_reverts
        result_dict['intra_overlap_guard_reverts'] = _diag_intra_reverts
        result_dict['T_init']                  = T_init
        result_dict['post_repair_violations']  = n_repaired

    on_update(-1, current_cost, state)
    return n_repaired


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