#------------------------------------------------------------------------------------------------------------------
#   Simulated Annealing - Mars Crater Descent
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import plotly.graph_objects as go

#------------------------------------------------------------------------------------------------------------------
#   Load map
#------------------------------------------------------------------------------------------------------------------
crater_map = np.load("crater_map.npy")
scale = 10.045  # meters per pixel after downscaling

print(f"Map shape: {crater_map.shape}")
print(f"Scale: {scale} m/pixel")

#------------------------------------------------------------------------------------------------------------------
#   Simulated Annealing algorithm
#------------------------------------------------------------------------------------------------------------------
def simulated_annealing(crater_map, start_row, start_col, max_height_diff=2.0,
                        T_init=500.0, T_min=0.0001, alpha=0.9995, max_iter=100000):
    """
    Simulated annealing descent into the crater.
    - Picks a random valid neighbor each iteration.
    - Accepts better moves always; worse moves with probability exp(-delta/T).
    - Temperature decreases geometrically by alpha each iteration.
    """
    n_rows, n_cols = crater_map.shape
    current_row, current_col = start_row, start_col
    current_height = crater_map[current_row, current_col]

    best_row, best_col = current_row, current_col
    best_height = current_height

    path = [(current_row, current_col)]

    # 8-neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    T = T_init
    iteration = 0

    while T > T_min and iteration < max_iter:
        # Get all valid neighbors (within bounds, valid pixel, height diff <= 2.0)
        valid_neighbors = []
        for dr, dc in neighbors:
            nr, nc = current_row + dr, current_col + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                nh = crater_map[nr, nc]
                if nh >= 0 and abs(current_height - nh) <= max_height_diff:
                    valid_neighbors.append((nr, nc, nh))

        if not valid_neighbors:
            break  # Stuck, no valid moves

        # Pick a random valid neighbor
        idx = np.random.randint(len(valid_neighbors))
        nr, nc, nh = valid_neighbors[idx]

        # delta > 0 means neighbor is higher (worse for descent)
        delta = nh - current_height

        if delta <= 0:
            # Better or equal: always accept
            current_row, current_col = nr, nc
            current_height = nh
            path.append((current_row, current_col))
        else:
            # Worse: accept with probability exp(-delta / T)
            prob = np.exp(-delta / T)
            if np.random.random() < prob:
                current_row, current_col = nr, nc
                current_height = nh
                path.append((current_row, current_col))

        # Track best position found
        if current_height < best_height:
            best_height = current_height
            best_row, best_col = current_row, current_col

        T *= alpha
        iteration += 1

    return path, (best_row, best_col, best_height), iteration

#------------------------------------------------------------------------------------------------------------------
#   Convert meters to pixel indices
#------------------------------------------------------------------------------------------------------------------
def meters_to_pixel(x_meters, y_meters, scale):
    col = int(round(x_meters / scale))
    row = int(round(y_meters / scale))
    return row, col

#------------------------------------------------------------------------------------------------------------------
#   Run tests
#------------------------------------------------------------------------------------------------------------------
np.random.seed(42)

test_positions = [
    (3350, 5800),
    (3500, 5500),
    (2700, 4800),
]

print("\n" + "="*80)
print("SIMULATED ANNEALING RESULTS")
print("="*80)
print("Parameters: T_init=500, T_min=0.0001, alpha=0.9995, max_iter=100000")

# Find the global minimum for reference
valid_mask = crater_map >= 0
min_val = crater_map[valid_mask].min()
min_pos = np.unravel_index(np.argmin(np.where(valid_mask, crater_map, np.inf)), crater_map.shape)
print(f"\nGlobal minimum depth: {min_val:.2f} m at pixel ({min_pos[0]}, {min_pos[1]})")
print(f"Global minimum location: x={min_pos[1]*scale:.1f} m, y={min_pos[0]*scale:.1f} m")

all_paths = []
for x_m, y_m in test_positions:
    row, col = meters_to_pixel(x_m, y_m, scale)
    row = max(0, min(row, crater_map.shape[0] - 1))
    col = max(0, min(col, crater_map.shape[1] - 1))

    start_height = crater_map[row, col]
    path, (best_r, best_c, best_h), iters = simulated_annealing(crater_map, row, col)
    end_row, end_col = path[-1]
    end_height = crater_map[end_row, end_col]

    print(f"\nStart: x={x_m}m, y={y_m}m (pixel [{row},{col}], height={start_height:.2f}m)")
    print(f"  Best found: pixel [{best_r},{best_c}] (x={best_c*scale:.1f}m, y={best_r*scale:.1f}m)")
    print(f"  Best height: {best_h:.2f}m | Iterations: {iters} | Path length: {len(path)}")
    print(f"  Descent: {start_height - best_h:.2f}m")
    print(f"  Distance to global min: {np.sqrt((best_r-min_pos[0])**2 + (best_c-min_pos[1])**2)*scale:.1f}m")

    all_paths.append((x_m, y_m, path, (best_r, best_c)))

#------------------------------------------------------------------------------------------------------------------
#   3D Visualization with Plotly (full crater)
#------------------------------------------------------------------------------------------------------------------
n_rows, n_cols = crater_map.shape
colors_list = ['cyan', 'lime', 'blue', 'magenta', 'white', 'yellow']

zona_display = np.where(crater_map == -1, np.nan, crater_map)

x_coords = scale * np.arange(n_cols)
y_coords = scale * (n_rows - np.arange(n_rows))
X, Y = np.meshgrid(x_coords, y_coords)

# --- Combined 3D plot with all paths ---
fig_all = go.Figure()

fig_all.add_trace(go.Surface(
    x=X, y=Y, z=zona_display,
    colorscale='hot', cmin=0, showscale=True,
    colorbar=dict(title='Height (m)'),
    lighting=dict(ambient=0.3, diffuse=0.8, roughness=0.4, specular=0.2),
    lightposition=dict(x=0, y=0, z=5000),
    name='Terrain'
))

for i, (x_m, y_m, path, (best_r, best_c)) in enumerate(all_paths):
    path_xs = [p[1] * scale for p in path]
    path_ys = [(n_rows - p[0]) * scale for p in path]
    path_zs = [crater_map[p[0], p[1]] + 1.5 for p in path]
    fig_all.add_trace(go.Scatter3d(
        x=path_xs, y=path_ys, z=path_zs,
        mode='lines',
        line=dict(color=colors_list[i % len(colors_list)], width=5),
        name=f'Start ({x_m},{y_m})m'
    ))
    fig_all.add_trace(go.Scatter3d(
        x=[path_xs[0]], y=[path_ys[0]], z=[path_zs[0] + 2],
        mode='markers', marker=dict(size=5, color='green', symbol='diamond'),
        name=f'Start ({x_m},{y_m})', showlegend=False
    ))
    best_h = crater_map[best_r, best_c]
    fig_all.add_trace(go.Scatter3d(
        x=[best_c * scale], y=[(n_rows - best_r) * scale], z=[best_h + 3],
        mode='markers', marker=dict(size=6, color='red', symbol='x'),
        name=f'Best ({x_m},{y_m})', showlegend=False
    ))

fig_all.update_layout(
    title='Simulated Annealing - Mars Crater Descent (3D)',
    scene=dict(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        zaxis_title='Height (m)',
        aspectmode='manual',
        aspectratio=dict(x=1, y=n_rows/n_cols, z=0.4)
    ),
    legend=dict(x=0.01, y=0.99)
)
fig_all.show()

# --- Individual 3D plot per starting position ---
for i, (x_m, y_m, path, (best_r, best_c)) in enumerate(all_paths):
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=zona_display,
        colorscale='hot', cmin=0, showscale=True,
        colorbar=dict(title='Height (m)'),
        lighting=dict(ambient=0.3, diffuse=0.8, roughness=0.4, specular=0.2),
        lightposition=dict(x=0, y=0, z=5000),
        name='Terrain'
    ))

    path_xs = [p[1] * scale for p in path]
    path_ys = [(n_rows - p[0]) * scale for p in path]
    path_zs = [crater_map[p[0], p[1]] + 1.5 for p in path]
    fig.add_trace(go.Scatter3d(
        x=path_xs, y=path_ys, z=path_zs,
        mode='lines',
        line=dict(color=colors_list[i % len(colors_list)], width=6),
        name=f'SA: {len(path)} points'
    ))

    fig.add_trace(go.Scatter3d(
        x=[path_xs[0]], y=[path_ys[0]], z=[path_zs[0] + 2],
        mode='markers+text', text=['Start'], textposition='top center',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))

    best_h = crater_map[best_r, best_c]
    fig.add_trace(go.Scatter3d(
        x=[best_c * scale], y=[(n_rows - best_r) * scale], z=[best_h + 3],
        mode='markers+text', text=['Best'], textposition='top center',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='Best'
    ))

    fig.update_layout(
        title=f'Simulated Annealing - Start ({x_m},{y_m})m -> Best: {best_h:.2f}m, {len(path)} path points',
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='Height (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=n_rows/n_cols, z=0.4)
        )
    )
    fig.show()

print("\n3D visualizations generated in browser.")
