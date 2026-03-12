"""
flow_renderer.py — Draw multi-source EU trade flow maps.

Flows are drawn as curved arrows; line width and opacity scale with flow value.
Each source country gets a distinct color.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Color palette for up to 10 source countries
SOURCE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
]


def draw_flow_map(eu_gdf, centroids, export_matrix, sources,
                  threshold_meur=0.0, net_mode=False, net_matrix=None,
                  title="EU Trade Flow Map", output_path=None):
    """
    Render a multi-source flow map and save or display it.

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
    matrix = net_matrix if net_mode else export_matrix

    # Determine max flow value across all drawn flows for scaling
    max_val = 0.0
    for src in sources:
        if src not in matrix.index:
            continue
        for dst in matrix.columns:
            if dst == src:
                continue
            val = matrix.loc[src, dst]
            if net_mode:
                val = val  # can be negative; we draw only positive (net exporter)
            if val > threshold_meur:
                max_val = max(max_val, val)

    if max_val == 0:
        print("No flows above threshold — nothing to draw.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Draw EU country polygons
    eu_gdf.plot(ax=ax, color='#f5f5f0', edgecolor='#888888', linewidth=0.5)

    # Map extent: Europe
    ax.set_xlim(-25, 45)
    ax.set_ylim(34, 72)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_elements = []

    for i, src in enumerate(sources):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=src)
        )

        if src not in centroids:
            print(f"Warning: centroid not found for '{src}'")
            continue

        src_xy = centroids[src]

        for dst in matrix.columns:
            if dst == src or dst not in centroids:
                continue

            val = matrix.loc[src, dst] if src in matrix.index else 0.0

            # In net mode draw only net-positive flows (src is net exporter to dst)
            if val <= threshold_meur:
                continue

            dst_xy = centroids[dst]
            norm = val / max_val
            lw = 0.4 + 7.0 * norm
            alpha = 0.35 + 0.55 * norm

            ax.annotate(
                "",
                xy=dst_xy,
                xytext=src_xy,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    mutation_scale=10,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=3,
            )

    # Draw source country markers on top
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

    # Legend
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              title='Source Country', title_fontsize=9,
              framealpha=0.85, edgecolor='#cccccc')

    mode_label = "Net Exports" if net_mode else "Gross Exports"
    ax.set_title(f"{title}\n({mode_label}, threshold ≥ {threshold_meur:.0f} M EUR)",
                 fontsize=13, pad=12)

    ax.text(0.01, 0.01,
            f"Line width ∝ flow value  |  max shown: {max_val:,.0f} M EUR",
            transform=ax.transAxes, fontsize=8, color='#666666', va='bottom')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()
