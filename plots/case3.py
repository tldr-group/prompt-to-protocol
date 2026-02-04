import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ============= 1. Data Loading & Processing =============
# Set up paths
base_dir = Path(__file__).parent.parent / "data" / "case3_p2o"
p2p_dir = Path(__file__).parent.parent / "data" / "case3_p2p"
cccv_dir = Path(__file__).parent.parent / "data" / "case3_cccv"
p2o_dir = base_dir

def get_best_loss_from_folder(folder_path):
    """Get best loss from a folder, trying best_loss.txt first, then optimization_results.csv"""
    # Try best_loss.txt first
    best_loss_path = folder_path / "best_loss.txt"
    if best_loss_path.exists():
        with open(best_loss_path, 'r') as f:
            return float(f.read().strip())
    
    # Try optimization_results.csv
    csv_path = folder_path / "optimization_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'total_loss' in df.columns and len(df) > 0:
            return float(df['total_loss'].min())
    
    return None

print("Processing results...")
p2o_folder_count = len([f for f in os.listdir(p2o_dir) if os.path.isdir(p2o_dir / f) and f.startswith('20')])
print(f"Total P2O folders found: {p2o_folder_count}")

# --- P2P ---
p2p_folders = sorted([f for f in os.listdir(p2p_dir) if f.startswith("Control_simulation_results")])
p2p_best_losses = []
print("\n=== P2P Best Loss Details ===")
for idx, folder in enumerate(p2p_folders[:10], 1):
    csv_path = p2p_dir / folder / "generation_log.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # First truncate to 210 data points, then find best
        df_truncated = df.head(210)
        gen_best_losses = df_truncated.groupby('Generation')['Child_Loss'].min()
        best_loss = np.min(gen_best_losses.values)
        p2p_best_losses.append(best_loss)
        
        # Find the row with the best loss in truncated data
        best_row = df_truncated[df_truncated['Child_Loss'] == best_loss].iloc[0]
        print(f"\nRun {idx} ({folder}):")
        print(f"  Best Loss: {best_loss:.6f}")
        print(f"  Generation: {best_row['Generation']}")
        print(f"  Data points used: {len(df_truncated)} (max generation: {df_truncated['Generation'].max()})")
        print(f"  Full row data:")
        print(f"  {best_row.to_dict()}")
print("=" * 50 + "\n")

# --- CCCV ---
cccv_folders = sorted([f for f in os.listdir(cccv_dir) if f.startswith("CCCVResults")])
cccv_losses = []
for folder in cccv_folders[:10]:
    best_loss_path = cccv_dir / folder / "best_loss.txt"
    if best_loss_path.exists():
        with open(best_loss_path, 'r') as f:
            cccv_losses.append(float(f.read().strip()))

# --- P2O ---
p2o_folders = sorted([f for f in os.listdir(p2o_dir) if os.path.isdir(p2o_dir / f) and f.startswith("20")])
# Initial
p2o_initial_losses = []
for folder in p2o_folders[:10]:
    loss = get_best_loss_from_folder(p2o_dir / folder)
    if loss is not None:
        p2o_initial_losses.append(loss)

# Iterations (Fixed)
remaining_folders = p2o_folders[10:]
p2o_iteration_all_losses = []
max_iterations = 10
# Ensure we try to read exactly 10 iterations (30 folders)
for iter_idx in range(max_iterations):
    i = iter_idx * 3
    if i >= len(remaining_folders):
        # If we run out of folders, use empty list for missing iterations
        p2o_iteration_all_losses.append([])
        continue
    iteration_folders = remaining_folders[i:i+3]
    losses = []
    for folder in iteration_folders:
        loss = get_best_loss_from_folder(p2o_dir / folder)
        if loss is not None:
            losses.append(loss)
    # Always append losses (even if empty) to maintain iteration count
    p2o_iteration_all_losses.append(losses)

print(f"P2O Initial networks: {len(p2o_initial_losses)}")
print(f"P2O Iterations read: {len(p2o_iteration_all_losses)}")
for i, losses in enumerate(p2o_iteration_all_losses):
    print(f"  Iteration {i+1}: {len(losses)} networks")

# Best so far
p2o_best_so_far = []
if p2o_initial_losses:
    current_best = min(p2o_initial_losses)
    p2o_best_so_far.append(current_best)
    for iter_losses in p2o_iteration_all_losses:
        if iter_losses:  # Only update if there are losses in this iteration
            current_best = min(current_best, min(iter_losses))
        p2o_best_so_far.append(current_best)
else:
    # Fallback if no P2O data found to avoid errors
    p2o_best_so_far = [0]
    p2o_initial_losses = [0] 

# Pool Range Data Preparation
# num_iters should always be max_iterations (10) now
num_iters = len(p2o_iteration_all_losses)
iteration_pool_ranges_loss = {}
for current_iter in range(0, num_iters + 1):
    losses_upto = list(p2o_initial_losses)
    for it in range(num_iters):
        if it + 1 <= current_iter and p2o_iteration_all_losses[it]:  # Check if iteration has losses
            losses_upto.extend(p2o_iteration_all_losses[it])
    if losses_upto:
        pool_losses = sorted(losses_upto)[:10]
        iteration_pool_ranges_loss[current_iter] = {
            'min': float(np.min(pool_losses)),
            'max': float(np.max(pool_losses))
        }

p2o_best_for_plot = p2o_best_so_far[:num_iters + 1]


# ============= 2. Plotting =============
print("\nCreating optimized plot...")

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.family': 'sans-serif',
})

# Colors
color_p2o = '#4EABC8'       
color_cccv = '#F4856D'      
color_p2p = '#7777B3'       
color_dots = '#555555'      
color_best_line = '#2A8CA8' 
color_pool_range = '#A8D5E2'

fig, ax = plt.subplots(figsize=(9, 6))

x_pos = 1.0
positions = []
labels = []
scatter_size = 15 

# --- 1. CCCV ---
cccv_pos = x_pos
positions.append(x_pos)
labels.append('CCCV\n(10 runs)')
if cccv_losses:
    ax.boxplot([cccv_losses], positions=[x_pos], widths=0.5, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color_cccv, alpha=0.3, linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.2))
    ax.scatter([x_pos]*len(cccv_losses), cccv_losses, s=scatter_size, 
            color=color_cccv, alpha=0.8, zorder=3)
x_pos += 2.0

# --- 2. P2P ---
positions.append(x_pos)
labels.append('P2P\n(10 runs)')
if p2p_best_losses:
    ax.boxplot([p2p_best_losses], positions=[x_pos], widths=0.5, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color_p2p, alpha=0.3, linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.2))
    ax.scatter([x_pos]*len(p2p_best_losses), p2p_best_losses, s=scatter_size, 
            color=color_p2p, alpha=0.8, zorder=3)
x_pos += 2.0

# --- 3. P2O Initial ---
p2o_initial_pos = x_pos
positions.append(x_pos)
labels.append('P2O\nInitial\n(10 nets)')
if p2o_initial_losses:
    ax.boxplot([p2o_initial_losses], positions=[x_pos], widths=0.5, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color_p2o, alpha=0.3, linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.2))
    ax.scatter([x_pos]*len(p2o_initial_losses), p2o_initial_losses, s=scatter_size, 
            color=color_p2o, alpha=0.8, zorder=3)
x_pos += 1.6

# --- 4. P2O Iterations ---
p2o_iter_positions = []
best_so_far_x = [p2o_initial_pos]

for i in range(num_iters):
    iter_losses = p2o_iteration_all_losses[i]
    iter_x = x_pos
    positions.append(iter_x)
    labels.append(f'Iter {i+1}')
    p2o_iter_positions.append(iter_x)
    best_so_far_x.append(iter_x)
    
    # No jitter - align all dots vertically
    if iter_losses:  # Only plot if there are losses
        for loss in iter_losses:
            ax.scatter(iter_x, loss, s=scatter_size, color='gray', alpha=0.6, zorder=2)
    
    x_pos += 1.0

# --- 5. Pool Range Shading ---
if iteration_pool_ranges_loss:
    pool_x = [p2o_initial_pos] + p2o_iter_positions
    pool_min = [iteration_pool_ranges_loss[i]['min'] for i in range(num_iters + 1)]
    pool_max = [iteration_pool_ranges_loss[i]['max'] for i in range(num_iters + 1)]
    
    ax.fill_between(pool_x, pool_min, pool_max, color=color_pool_range, alpha=0.2, zorder=0)

# --- 6. Best So Far Line ---
ax.plot(best_so_far_x, p2o_best_for_plot, color=color_best_line, linewidth=2, 
        marker='s', markersize=4, zorder=4, label='_nolegend_')

# ============= 3. Styling & Annotations =============

all_data_points = (cccv_losses + p2p_best_losses + p2o_initial_losses + 
                   [x for sublist in p2o_iteration_all_losses for x in sublist])

if all_data_points:
    y_max_val = max(all_data_points)
    y_min_val = min(all_data_points)
    data_range = y_max_val - y_min_val
    # Add head room
    ax.set_ylim(bottom=y_min_val - data_range*0.05, top=y_max_val + data_range * 0.2)
    
    # Phase text positioning
    text_y_pos = y_max_val + data_range * 0.1

    # Background shading
    ax.axvspan(0, p2o_initial_pos + 0.8, color='#F0F0F0', alpha=0.5, zorder=-5)

    # Init Phase Text
    ax.text((cccv_pos + p2o_initial_pos)/2, text_y_pos, 
            "Initialization Phase", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#444')
    ax.text((cccv_pos + p2o_initial_pos)/2, text_y_pos - data_range*0.1, 
            "(10 Runs per Method\nEqual Trials per Run)", 
            ha='center', va='center', fontsize=12, color='#666')

    # Iter Phase Text
    if p2o_iter_positions:
        iter_center = (p2o_iter_positions[0] + p2o_iter_positions[-1]) / 2
        ax.text(iter_center, text_y_pos, 
                "P2O Iterative Phase", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#444')

    # Stats Box
    stats_str = (
        r"$\bf{CCCV}$" + f": {np.mean(cccv_losses):.4f} ± {np.std(cccv_losses):.4f}\n"
        f"      Best: {np.min(cccv_losses):.4f}\n\n"
        r"$\bf{P2P}$" + f": {np.mean(p2p_best_losses):.4f} ± {np.std(p2p_best_losses):.4f}\n"
        f"      Best: {np.min(p2p_best_losses):.4f}\n\n"
        r"$\bf{P2O\ Init}$" + f": {np.mean(p2o_initial_losses):.4f} ± {np.std(p2o_initial_losses):.4f}\n"
        f"      P2O Iter 1-10 Best: {p2o_best_for_plot[-1]:.4f}"
    )

    ax.text(0.46, 0.80, stats_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='#CCCCCC'))

# Axes
ax.set_ylabel('Best Loss', fontsize=15)
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor=color_cccv, alpha=0.4, label='CCCV Distribution'),
    Patch(facecolor=color_p2p, alpha=0.4, label='P2P Distribution'),
    Patch(facecolor=color_p2o, alpha=0.4, label='P2O Initial Dist.'),
    Patch(facecolor=color_pool_range, alpha=0.3, label='P2O Pool Range (Best 10)'),
    Line2D([0], [0], color=color_best_line, marker='s', markersize=5, 
           label='P2O Best So Far'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=5, alpha=0.7, label='Individual Run'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
          edgecolor='#DDDDDD', fontsize=10, bbox_to_anchor=(1.0, 0.82))

plt.tight_layout()
output_path = Path(__file__).parent.parent / "results" / "case3.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
