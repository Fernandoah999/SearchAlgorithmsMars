import numpy as np
import heapq
from collections import deque
import time

# ----------------------------------------------------------------------------------------------------------------------
#   Load map
# ----------------------------------------------------------------------------------------------------------------------
mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape
scale = 10.0174  # meters per pixel

# ----------------------------------------------------------------------------------------------------------------------
#   Parameters
# ----------------------------------------------------------------------------------------------------------------------
MAX_HEIGHT_DIFF = 0.25  # maximum allowed height difference (meters)

# Start and end coordinates (in meters)
x_start, y_start = 2850, 6400
x_end, y_end = 3150, 6800

# Convert (x, y) coordinates in meters to (row, column) matrix indices
def coords_to_rc(x, y):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

start = coords_to_rc(x_start, y_start)
end = coords_to_rc(x_end, y_end)

print(f"Map: {nr} rows x {nc} columns")
print(f"Start (r,c): {start} -> height: {mars_map[start]:.2f} m")
print(f"End   (r,c): {end} -> height: {mars_map[end]:.2f} m")
print(f"Max allowed height difference: {MAX_HEIGHT_DIFF} m")
print()

# ----------------------------------------------------------------------------------------------------------------------
#   Movements: 8 directions (including diagonals)
# ----------------------------------------------------------------------------------------------------------------------
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Distance in meters for each movement
MOVE_COST = {
    (-1, 0): scale, (1, 0): scale, (0, -1): scale, (0, 1): scale,
    (-1, -1): scale * np.sqrt(2), (-1, 1): scale * np.sqrt(2),
    (1, -1): scale * np.sqrt(2), (1, 1): scale * np.sqrt(2)
}

def get_neighbors(r, c):
    """Returns valid neighbors: within map bounds, height != -1, height difference < MAX_HEIGHT_DIFF."""
    neighbors = []
    current_h = mars_map[r, c]
    for dr, dc in MOVES:
        nr2, nc2 = r + dr, c + dc
        if 0 <= nr2 < nr and 0 <= nc2 < nc:
            h = mars_map[nr2, nc2]
            if h != -1 and abs(h - current_h) < MAX_HEIGHT_DIFF:
                neighbors.append((nr2, nc2, MOVE_COST[(dr, dc)]))
    return neighbors

def reconstruct_path(came_from, current):
    """Reconstructs the path from the end node back to the start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def path_distance(path):
    """Calculates the total path distance in meters."""
    dist = 0.0
    for i in range(len(path) - 1):
        dr = path[i+1][0] - path[i][0]
        dc = path[i+1][1] - path[i][1]
        dist += MOVE_COST[(dr, dc)]
    return dist

# ----------------------------------------------------------------------------------------------------------------------
#   A* Search
# ----------------------------------------------------------------------------------------------------------------------
def heuristic(node, goal):
    """Euclidean distance in meters as heuristic."""
    return scale * np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current], len(visited)

        for nr2, nc2, cost in get_neighbors(*current):
            neighbor = (nr2, nc2)
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f, neighbor))

    return None, 0, len(visited)

# ----------------------------------------------------------------------------------------------------------------------
#   BFS (Breadth-First Search)
# ----------------------------------------------------------------------------------------------------------------------
def bfs(start, goal):
    queue = deque([start])
    came_from = {}
    visited = {start}

    while queue:
        current = queue.popleft()

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_distance(path), len(visited)

        for nr2, nc2, _ in get_neighbors(*current):
            neighbor = (nr2, nc2)
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return None, 0, len(visited)

# ----------------------------------------------------------------------------------------------------------------------
#   DFS (Depth-First Search)
# ----------------------------------------------------------------------------------------------------------------------
def dfs(start, goal, max_depth=50000):
    stack = [(start, 0)]
    came_from = {}
    visited = {start}

    while stack:
        current, depth = stack.pop()

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_distance(path), len(visited)

        if depth >= max_depth:
            continue

        for nr2, nc2, _ in get_neighbors(*current):
            neighbor = (nr2, nc2)
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append((neighbor, depth + 1))

    return None, 0, len(visited)

# ----------------------------------------------------------------------------------------------------------------------
#   UCS (Uniform Cost Search)
# ----------------------------------------------------------------------------------------------------------------------
def ucs(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, current_cost, len(visited)

        for nr2, nc2, cost in get_neighbors(*current):
            neighbor = (nr2, nc2)
            new_cost = current_cost + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_cost, neighbor))

    return None, 0, len(visited)

# ----------------------------------------------------------------------------------------------------------------------
#   Run algorithms
# ----------------------------------------------------------------------------------------------------------------------
algorithms = [
    ("A*", a_star),
    ("BFS", bfs),
    ("DFS", dfs),
    ("UCS", ucs),
]

results = {}

for name, func in algorithms:
    print(f"Running {name}...")
    t0 = time.time()
    path, dist, nodes_visited = func(start, end)
    elapsed = time.time() - t0

    results[name] = (path, dist, nodes_visited, elapsed)

    if path:
        print(f"  Route found!")
        print(f"  Distance: {dist:.2f} meters")
        print(f"  Steps: {len(path)}")
        print(f"  Nodes visited: {nodes_visited}")
        print(f"  Time: {elapsed:.2f} s")
    else:
        print(f"  No route found.")
        print(f"  Nodes visited: {nodes_visited}")
        print(f"  Time: {elapsed:.2f} s")
    print()

# ----------------------------------------------------------------------------------------------------------------------
#   Comparative summary
# ----------------------------------------------------------------------------------------------------------------------
print("=" * 70)
print(f"{'Algorithm':<12} {'Route?':<8} {'Distance (m)':<16} {'Steps':<10} {'Nodes':<12} {'Time (s)':<10}")
print("=" * 70)
for name in ["A*", "BFS", "DFS", "UCS"]:
    path, dist, nodes, elapsed = results[name]
    found = "Yes" if path else "No"
    steps = len(path) if path else 0
    print(f"{name:<12} {found:<8} {dist:<16.2f} {steps:<10} {nodes:<12} {elapsed:<10.2f}")
print("=" * 70)

# ----------------------------------------------------------------------------------------------------------------------
#   3D route visualization on the map
# ----------------------------------------------------------------------------------------------------------------------
import plotly.graph_objects as go

colors_map = {"A*": "cyan", "BFS": "lime", "DFS": "magenta", "UCS": "yellow"}

# Crop the area of interest based on ALL found routes
all_rows = [start[0], end[0]]
all_cols = [start[1], end[1]]
for alg_name in ["A*", "BFS", "DFS", "UCS"]:
    path = results[alg_name][0]
    if path:
        all_rows.extend([p[0] for p in path])
        all_cols.extend([p[1] for p in path])

margin_px = 20  # additional margin in pixels
r_min_crop = max(0, min(all_rows) - margin_px)
r_max_crop = min(nr, max(all_rows) + margin_px)
c_min_crop = max(0, min(all_cols) - margin_px)
c_max_crop = min(nc, max(all_cols) + margin_px)

zona = mars_map[r_min_crop:r_max_crop, c_min_crop:c_max_crop]
zona_display = np.where(zona == -1, np.nan, zona)

# x, y coordinates in meters for the cropped area
x_coords = scale * np.arange(c_min_crop, c_max_crop)
y_coords = scale * (nr - np.arange(r_min_crop, r_max_crop))
X_crop, Y_crop = np.meshgrid(x_coords, y_coords)

# --- 3D plot with all routes together ---
fig_all = go.Figure()

fig_all.add_trace(go.Surface(
    x=X_crop, y=Y_crop, z=zona_display,
    colorscale='hot', cmin=0, showscale=True,
    colorbar=dict(title='Height (m)'),
    lighting=dict(ambient=0.3, diffuse=0.8, roughness=0.4, specular=0.2),
    lightposition=dict(x=0, y=0, z=5000),
    name='Terrain'
))

for alg_name in ["A*", "BFS", "DFS", "UCS"]:
    path, dist, nodes, elapsed = results[alg_name]
    if path:
        path_xs = [p[1] * scale for p in path]
        path_ys = [(nr - p[0]) * scale for p in path]
        path_zs = [mars_map[p[0], p[1]] + 1.5 for p in path]  # raise slightly above surface
        fig_all.add_trace(go.Scatter3d(
            x=path_xs, y=path_ys, z=path_zs,
            mode='lines',
            line=dict(color=colors_map[alg_name], width=5),
            name=f'{alg_name}: {dist:.0f} m'
        ))

# Start and end markers
z_start = mars_map[start] + 3
z_end = mars_map[end] + 3
fig_all.add_trace(go.Scatter3d(
    x=[x_start], y=[y_start], z=[z_start],
    mode='markers+text', text=['Start'], textposition='top center',
    marker=dict(size=8, color='green', symbol='diamond'),
    name='Start'
))
fig_all.add_trace(go.Scatter3d(
    x=[x_end], y=[y_end], z=[z_end],
    mode='markers+text', text=['End'], textposition='top center',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='End'
))

fig_all.update_layout(
    title='Route comparison - 3D View',
    scene=dict(
        xaxis_title='x (m)',
        yaxis_title='y (m)',
        zaxis_title='Height (m)',
        aspectmode='manual',
        aspectratio=dict(
            x=1,
            y=(r_max_crop - r_min_crop) / (c_max_crop - c_min_crop),
            z=0.4
        )
    ),
    legend=dict(x=0.01, y=0.99)
)
fig_all.show()

# --- Individual 3D plot for each algorithm ---
for alg_name in ["A*", "BFS", "DFS", "UCS"]:
    path, dist, nodes, elapsed = results[alg_name]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X_crop, y=Y_crop, z=zona_display,
        colorscale='hot', cmin=0, showscale=True,
        colorbar=dict(title='Height (m)'),
        lighting=dict(ambient=0.3, diffuse=0.8, roughness=0.4, specular=0.2),
        lightposition=dict(x=0, y=0, z=5000),
        name='Terrain'
    ))

    if path:
        path_xs = [p[1] * scale for p in path]
        path_ys = [(nr - p[0]) * scale for p in path]
        path_zs = [mars_map[p[0], p[1]] + 1.5 for p in path]
        fig.add_trace(go.Scatter3d(
            x=path_xs, y=path_ys, z=path_zs,
            mode='lines',
            line=dict(color=colors_map[alg_name], width=6),
            name=f'{alg_name}: {dist:.0f} m'
        ))

    fig.add_trace(go.Scatter3d(
        x=[x_start], y=[y_start], z=[z_start],
        mode='markers+text', text=['Start'], textposition='top center',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_end], y=[y_end], z=[z_end],
        mode='markers+text', text=['End'], textposition='top center',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='End'
    ))

    status = f"Route: {dist:.0f} m, {len(path)} steps" if path else "No route"
    fig.update_layout(
        title=f'{alg_name} - {status}',
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='Height (m)',
            aspectmode='manual',
            aspectratio=dict(
                x=1,
                y=(r_max_crop - r_min_crop) / (c_max_crop - c_min_crop),
                z=0.4
            )
        )
    )
    fig.show()

print("\n3D visualizations generated in browser.")
