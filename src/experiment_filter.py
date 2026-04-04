"""
This experiment want to
"""
import matplotlib.pyplot as plt
import numpy as np
from flow_renderer import render_to_axes
from map_utils import get_centroids, load_eu_map
from data_loader import load_trade_data
from interactive import DATA_FILE, LABEL_FILE
from spiral_tree import _OPT_OVERLAP_B, compute_inter_tree_cost, DEFAULT_F_TOTAL_WEIGHTS

eu_gdf = load_eu_map()
eu_gdf = eu_gdf[eu_gdf.geometry.notna() & ~eu_gdf.geometry.is_empty].copy()
centroids = get_centroids(eu_gdf)
export_matrix, net_matrix, _ = load_trade_data(label_path=LABEL_FILE, data_path=DATA_FILE, year=2024)
ax = plt.subplot()

def compute_information_loss(sources, threshold_meur, disparity_alpha, alpha_deg = 25, width_scale = 1):
    # Keep alpha and width_scale fixed, vary threshold_meur and alpha_deg to see how it affects the mean number of nodes per source in the rendered trees. The information loss can be approximated by the reduction in mean nodes per source compared to a baseline with no thresholding (threshold_meur=0) and a standard alpha_deg (e.g., 25 degrees).


    baseline_trees = render_to_axes(ax=ax,
                                    eu_gdf=eu_gdf,
                                    centroids=centroids,
                                    export_matrix=export_matrix,
                                    sources=sources,
                                    threshold_meur=0,
                                    spiral_mode=True,
                                    alpha_deg=25,
                                    width_scale=1,
                                    disparity_alpha=1)
    

    def compute_mean_nodes_per_source(trees):
        nr_of_sources = len(sources)
        total_nodes = sum(len(tree.tree_nodes) for tree in trees)
        mean_nodes_per_source_baseline = total_nodes / nr_of_sources if nr_of_sources > 0 else 0
        return mean_nodes_per_source_baseline

    baseline_mean_nodes_per_source = compute_mean_nodes_per_source(baseline_trees)

    comparing_trees = render_to_axes(ax=ax,
                                     eu_gdf=eu_gdf,
                                     centroids=centroids,
                                     export_matrix=export_matrix,
                                     sources=sources,
                                     threshold_meur=threshold_meur,
                                     spiral_mode=True,
                                     alpha_deg=alpha_deg,
                                     width_scale=width_scale,
                                     disparity_alpha=disparity_alpha)
    
    comparing_mean_nodes_per_source = compute_mean_nodes_per_source(comparing_trees)
    
    return baseline_mean_nodes_per_source - comparing_mean_nodes_per_source

def get_rendered_tree_stats(sources, threshold_meur, disparity_alpha, alpha_deg = 25, width_scale = 1):
    trees = render_to_axes(ax=ax,
                            eu_gdf=eu_gdf,
                            centroids=centroids,
                            export_matrix=export_matrix,
                            sources=sources,
                            threshold_meur=threshold_meur,
                            spiral_mode=True,
                            alpha_deg=alpha_deg,
                            width_scale=width_scale,
                            disparity_alpha=disparity_alpha)
    
    total_nodes = sum(len(tree.tree_nodes) for tree in trees)
    
    total_eu = float(net_matrix[net_matrix > threshold_meur].sum().sum())
    stats = compute_inter_tree_cost(trees, DEFAULT_F_TOTAL_WEIGHTS, centroids, alpha_deg = 25, B_obs=_OPT_OVERLAP_B, n_overlap_samples=6)
    
    total_leaves = sum(
        sum(1 for node in tree.tree_nodes.values() if node.is_leaf) 
        for tree in trees
    )
    mean_leaves = total_leaves / len(sources) if sources else 0

    # at least 3 destination countries per source on average
    return stats['f_total'] 


def get_total_flow(sources, threshold_meur, disparity_alpha, alpha_deg = 25, width_scale = 1):
    baseline_trees = render_to_axes(ax=ax,
                                    eu_gdf=eu_gdf,
                                    centroids=centroids,
                                    export_matrix=export_matrix,
                                    sources=sources,
                                    threshold_meur=0,
                                    spiral_mode=True,
                                    alpha_deg=25,
                                    width_scale=1,
                                    disparity_alpha=1)
    
    trees = render_to_axes(ax=ax,
                            eu_gdf=eu_gdf,
                            centroids=centroids,
                            export_matrix=export_matrix,
                            sources=sources,
                            threshold_meur=threshold_meur,
                            spiral_mode=True,
                            alpha_deg=alpha_deg,
                            width_scale=width_scale,
                            disparity_alpha=disparity_alpha)
    
    baseline_total_flow = sum(node.weight for tree in baseline_trees for node in tree.tree_nodes.values() if node.is_leaf)
    total_flow = sum(node.weight for tree in trees for node in tree.tree_nodes.values() if node.is_leaf)

    
    return total_flow / baseline_total_flow if baseline_total_flow > 0 else 0  

def experiment_flow(min_meur, max_meur, step_meur, min_disparity_alpha, max_disparity_alpha, step_disparity_alpha, sources):
    values_meur = np.array(range(min_meur, max_meur + step_meur, step_meur))
    values_alpha = np.array(np.arange(min_disparity_alpha, max_disparity_alpha, step_disparity_alpha))

    flow_grid = np.zeros((len(values_meur), len(values_alpha)))

    for i, value_meur in enumerate(values_meur):
        for j, value_alpha in enumerate(values_alpha):
            flow = get_total_flow(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            flow_grid[i, j] = flow
            print(f"Threshold MEUR: {value_meur}, Disparity Alpha: {value_alpha}, Total Flow: {flow}")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    im1 = ax1.contourf(values_alpha, values_meur, flow_grid, levels=20, cmap='YlOrRd')
    ax1.set_title("Flow Retention Heatmap")
    fig.colorbar(im1, ax=ax1)

    plt.show()

def experiment_2(min_meur, max_meur, step_meur, min_disparity_alpha, max_disparity_alpha, step_disparity_alpha, sources, weight = 0.5):
    values_meur = np.array(range(min_meur, max_meur + step_meur, step_meur))
    values_alpha = np.array(np.arange(min_disparity_alpha, max_disparity_alpha + step_disparity_alpha, step_disparity_alpha))

    cost_grid = np.zeros((len(values_meur), len(values_alpha)))
    loss_grid = np.zeros((len(values_meur), len(values_alpha)))
    flow_grid = np.zeros((len(values_meur), len(values_alpha)))

    for i, value_meur in enumerate(values_meur):
        for j, value_alpha in enumerate(values_alpha):
            loss = compute_information_loss(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            cost = get_rendered_tree_stats(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            flow = get_total_flow(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            loss_grid[i, j] = loss
            cost_grid[i, j] = cost
            flow_grid[i, j] = flow
            print(f"Threshold MEUR: {value_meur}, Disparity Alpha: {value_alpha}, Information Loss: {loss}, Tree Cost: {cost}, Total Flow: {flow}")

    def normalize(grid):
        print(grid)
        finite_mask = np.isfinite(grid)
        grid_min = np.min(grid[finite_mask])
        grid_max = np.max(grid[finite_mask])
        
        normalized = np.where(
            finite_mask,
            (grid - grid_min) / (grid_max - grid_min) if grid_max != grid_min else np.zeros_like(grid),
            np.inf  # keep inf values as inf
        )
        print(normalized)
        return normalized

    # --- 2. Calculate Combined Loss ---
    # Adjust 'weight' to favor Cost (closer to 1) or Info Loss (closer to 0)
    
    norm_cost = normalize(cost_grid)
    norm_loss = normalize(loss_grid)
    combined_loss = (weight * norm_cost) + ((1 - weight) * norm_loss)

    # --- 3. Find the Optimum ---
    # This finds the (i, j) index of the lowest combined value
    min_idx = np.unravel_index(np.argmin(combined_loss), combined_loss.shape)
    best_meur = values_meur[min_idx[0]]
    best_alpha = values_alpha[min_idx[1]]

    # --- 4. Plotting ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 5))

    # Plot 1: Cost Heatmap
    im1 = ax1.contourf(values_alpha, values_meur, cost_grid, levels=20, cmap='YlOrRd')
    ax1.set_title("Tree Cost")
    fig.colorbar(im1, ax=ax1)

    # Plot 2: Info Loss Heatmap
    im2 = ax2.contourf(values_alpha, values_meur, loss_grid, levels=20, cmap='YlGnBu')
    ax2.set_title("Information Loss")
    fig.colorbar(im2, ax=ax2)

    # Plot 3: The Tradeoff (Combined)
    # This is your "Continuous Heatmap"
    im3 = ax3.contourf(values_alpha, values_meur, combined_loss, levels=50, cmap='viridis')
    ax3.scatter(best_alpha, best_meur, color='red', marker='*', s=200, label='Optimal')
    ax3.set_title(f"Optimal Choice")
    ax3.legend()
    fig.colorbar(im3, ax=ax3)

    # Plot 4: Pareto Scatter (Cost vs Loss)
    # Every point represents a combination of Alpha and MEUR
    ax4.scatter(norm_loss.flatten(), norm_cost.flatten(), alpha=0.5, c=combined_loss.flatten(), cmap='viridis')
    ax4.legend(*ax4.get_legend_handles_labels(), title="Combined Loss")

    sort_idx = np.argsort(norm_loss.flatten())
    plt.plot(norm_loss.flatten()[sort_idx], norm_cost.flatten()[sort_idx], color='black', linestyle='--', alpha=0.3, label='Pareto Boundary')
    ax4.set_xlabel("Information Loss")
    ax4.set_ylabel("Tree Cost")
    ax4.set_title("Pareto Front View")

    # General Labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Disparity Alpha")
        ax.set_ylabel("Threshold MEUR")


    plt.tight_layout()
    plt.show()

    print(f"Optimal Parameters: MEUR={best_meur}, Alpha={best_alpha:.2f}")


def experiment_final(min_meur, max_meur, step_meur, min_disparity_alpha, max_disparity_alpha, step_disparity_alpha, sources, weight = 0.5):
    values_meur = np.array(range(min_meur, max_meur + step_meur, step_meur))
    values_alpha = np.array(np.arange(min_disparity_alpha, max_disparity_alpha, step_disparity_alpha))

    cost_grid = np.zeros((len(values_meur), len(values_alpha)))
    flow_grid = np.zeros((len(values_meur), len(values_alpha)))

    for i, value_meur in enumerate(values_meur):
        for j, value_alpha in enumerate(values_alpha):
            cost = get_rendered_tree_stats(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            flow = get_total_flow(sources=sources, threshold_meur=value_meur, disparity_alpha=value_alpha)
            cost_grid[i, j] = cost
            flow_grid[i, j] = flow
            print(f"Threshold MEUR: {value_meur}, Disparity Alpha: {value_alpha}, Tree Cost: {cost}, Total Flow: {flow}")

    def normalize(grid):
        print(grid)
        finite_mask = np.isfinite(grid)
        grid_min = np.min(grid[finite_mask])
        grid_max = np.max(grid[finite_mask])
        
        normalized = np.where(
            finite_mask,
            (grid - grid_min) / (grid_max - grid_min) if grid_max != grid_min else np.zeros_like(grid),
            np.inf  # keep inf values as inf
        )
        print(normalized)
        return normalized

    # --- 2. Calculate Combined Loss ---
    # Adjust 'weight' to favor Cost (closer to 1) or Info Loss (closer to 0)
    
    norm_cost = normalize(cost_grid)

    combined_loss = (weight * (1 - norm_cost)) + ((1 - weight) * flow_grid)

        # --- 4. Plotting ---
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(21, 6))

    # Plot 1: Cost Heatmap
    im1 = ax1.contourf(values_alpha, values_meur, cost_grid, levels=20, cmap='viridis_r')
    ax1.set_title("Tree Cost")
    fig.colorbar(im1, ax=ax1)

    # Plot 2: Info Loss Heatmap
    im2 = ax2.contourf(values_alpha, values_meur, flow_grid, levels=20, cmap='viridis')
    ax2.set_title("Percentage of flow retained")
    fig.colorbar(im2, ax=ax2)

    # Plot 3: The Tradeoff (Combined)
    # This is your "Continuous Heatmap"
    im3 = ax3.contourf(values_alpha, values_meur, combined_loss, levels=50, cmap='viridis')
    ax3.set_title("Combined normalized cost and flow retention")
    #ax3.legend()
    fig.colorbar(im3, ax=ax3)

    # General Labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Disparity Alpha")
        ax.set_ylabel("Threshold MEUR")

    plt.tight_layout()
    plt.show()



min_meur = 0
max_meur = 100000
step_meur = 20000

min_disparity_alpha = 0.1
max_disparity_alpha = 1
step_disparity_alpha = 0.2

sources = ["France", "Germany", "Greece", "Lithuania"] # check verslag ff voor welke landen

#info_meur, cost_meur = experiment_meur(min=min_meur, max=max_meur, step=step_meur, sources=sources)
#info_alpha, cost_alpha = experiment_alpha(min=min_disparity_alpha, max=max_disparity_alpha, step=step_disparity_alpha, sources=sources)

experiment_final(min_meur=min_meur, max_meur=max_meur, step_meur=step_meur, min_disparity_alpha=min_disparity_alpha, max_disparity_alpha=max_disparity_alpha, step_disparity_alpha=step_disparity_alpha, sources=sources, weight=0.5)

#experiment_flow(min_meur=min_meur, max_meur=max_meur, step_meur=step_meur, min_disparity_alpha=min_disparity_alpha, max_disparity_alpha=max_disparity_alpha, step_disparity_alpha=step_disparity_alpha, sources=sources)    