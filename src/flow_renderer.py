"""
flow_renderer.py — Draw multi-source EU trade flow maps.

Data modes  : 'gross' (raw exports) or 'net' (exports minus imports)
Style modes : straight arrows or spiral trees
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from map_utils import render_basemap
from spiral_tree import compute_spiral_tree, draw_spiral_trees

# Color palette for up to 10 source countries
SOURCE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
]

# Spiral tree cache: (frozenset(sources), alpha_deg, threshold_meur) → trees
_SPIRAL_CACHE: dict = {}


def render_to_axes(
    ax,
    eu_gdf,
    centroids,
    export_matrix,
    sources,
    threshold_meur=0.0,
    net_mode=False,
    net_matrix=None,
    title="EU Trade Flow Map",
    spiral_mode=False,
    alpha_deg=25.0,
    world_gdf=None,
    xlim=(-25, 45),
    ylim=(34, 72),
    highlighted_countries=None,
) -> list:
    """
    Draw a multi-source flow map onto an existing matplotlib Axes.

    Returns
    -------
    list of SpiralTreeResult when spiral_mode=True; empty list otherwise.
    """
    ax.clear()
    matrix = net_matrix if net_mode else export_matrix

    # ── basemap ────────────────────────────────────────────────────────────
    render_basemap(ax, eu_gdf, world_gdf=world_gdf,
                   highlighted_countries=highlighted_countries or list(sources),
                   xlim=xlim, ylim=ylim)

    if spiral_mode:
        mode_label = f"Spiral Tree (α={alpha_deg:.0f}°)"
    elif net_mode:
        mode_label = "Net Exports"
    else:
        mode_label = "Gross Exports"

    if title:
        ax.set_title(
            f"{title}\n({mode_label}, threshold ≥ {threshold_meur:.0f} M EUR)",
            fontsize=11, pad=8
        )

    if not sources:
        return []

    color_map = {src: SOURCE_COLORS[i % len(SOURCE_COLORS)]
                 for i, src in enumerate(sources)}

    # ── Spiral Tree mode ──────────────────────────────────────────────────
    if spiral_mode:
        cache_key = (frozenset(sources), alpha_deg, threshold_meur, net_mode)
        if cache_key not in _SPIRAL_CACHE:
            trees = []
            for src in sources:
                if src not in matrix.index:
                    continue
                row = matrix.loc[src]
                net_flows = {
                    dst: float(v) for dst, v in row.items()
                    if dst != src and float(v) > threshold_meur
                }
                if not net_flows:
                    continue
                result = compute_spiral_tree(
                    source_name=src,
                    terminal_names=list(net_flows.keys()),
                    net_flows=net_flows,
                    centroids=centroids,
                    obstacle_names=[s for s in sources if s != src],
                    alpha_deg=alpha_deg,
                )
                trees.append(result)
            _SPIRAL_CACHE[cache_key] = trees
        else:
            trees = _SPIRAL_CACHE[cache_key]

        # Anchor display scale to gross totals so that net flows always
        # appear proportionally thinner than the corresponding gross flows.
        gross_max = max(
            (float(export_matrix.loc[src].sum())
             for src in sources if src in export_matrix.index),
            default=None,
        )
        draw_spiral_trees(ax, trees, centroids, color_map,
                          global_max_flow=gross_max)

        legend_elements = [
            Line2D([0], [0], color=color_map[src], linewidth=2.5, label=src)
            for src in sources if src in color_map
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
                      title='Source Country', title_fontsize=8,
                      framealpha=0.85, edgecolor='#cccccc')
        return trees

    # ── Straight-arrow mode ────────────────────────────────────────────────
    # Always normalise to the gross maximum so net arrows are proportionally
    # thinner than gross ones (net ≤ gross in absolute value for any pair).
    max_val = 0.0
    for src in sources:
        if src not in export_matrix.index:
            continue
        for dst in export_matrix.columns:
            if dst != src:
                v = float(export_matrix.loc[src, dst])
                if v > 0:
                    max_val = max(max_val, v)

    legend_elements = []
    for i, src in enumerate(sources):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=src))

        if src not in centroids:
            continue
        src_xy = centroids[src]

        if max_val > 0:
            for dst in matrix.columns:
                if dst == src or dst not in centroids:
                    continue
                val = matrix.loc[src, dst] if src in matrix.index else 0.0
                if val <= threshold_meur:
                    continue
                dst_xy = centroids[dst]
                norm  = val / max_val
                lw    = 0.4 + 7.0 * norm
                alpha = 0.35 + 0.55 * norm
                ax.annotate(
                    "",
                    xy=dst_xy, xytext=src_xy,
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                        mutation_scale=10, connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=3,
                )

    # Source markers
    for i, src in enumerate(sources):
        if src not in centroids:
            continue
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        x, y = centroids[src]
        ax.plot(x, y, 'o', color=color, markersize=9,
                markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        ax.text(x, y + 1.5, src, ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
                  title='Source Country', title_fontsize=8,
                  framealpha=0.85, edgecolor='#cccccc')

    if max_val > 0:
        ax.text(0.01, 0.01,
                f"Line width ∝ flow  |  max: {max_val:,.0f} M EUR",
                transform=ax.transAxes, fontsize=7.5, color='#666666',
                va='bottom')

    return []


def draw_flow_map(eu_gdf, centroids, export_matrix, sources,
                  threshold_meur=0.0, net_mode=False, net_matrix=None,
                  title="EU Trade Flow Map", output_path=None):
    """
    Render a multi-source flow map and save or display it (CLI entry point).

    Parameters
    ----------
    eu_gdf         : GeoDataFrame from map_utils.load_eu_map()
    centroids      : dict {country: (lon, lat)} from map_utils.get_centroids()
    export_matrix  : pd.DataFrame, gross export flows in million EUR
    sources        : list[str], source country names to highlight
    threshold_meur : float, minimum flow value to draw (million EUR)
    net_mode       : bool, if True use net_matrix instead of export_matrix
    net_matrix     : pd.DataFrame, net export flows (required if net_mode=True)
    title          : str, map title
    output_path    : str or None, if given save PNG here; else display
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    render_to_axes(ax, eu_gdf, centroids, export_matrix, sources,
                   threshold_meur=threshold_meur, net_mode=net_mode,
                   net_matrix=net_matrix, title=title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def invalidate_spiral_cache():
    """Clear the spiral tree cache (call after data reload or threshold change)."""
    _SPIRAL_CACHE.clear()