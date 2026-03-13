import numpy as np
import heapq
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
TIMEOUT = 120  # maximum time per execution in seconds

# Movements: 8 directions
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]

MOVE_COST = {
    (-1, 0): scale, (1, 0): scale, (0, -1): scale, (0, 1): scale,
    (-1, -1): scale * np.sqrt(2), (-1, 1): scale * np.sqrt(2),
    (1, -1): scale * np.sqrt(2), (1, 1): scale * np.sqrt(2)
}

# ----------------------------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------------------------
def coords_to_rc(x, y):
    r = nr - round(y / scale)
    c = round(x / scale)
    return r, c

def get_neighbors(r, c):
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
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# ----------------------------------------------------------------------------------------------------------------------
#   A* Search (with timeout)
# ----------------------------------------------------------------------------------------------------------------------
def heuristic(node, goal):
    return scale * np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def a_star(start, goal, timeout=TIMEOUT):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    t0 = time.time()

    while open_set:
        if time.time() - t0 > timeout:
            return None, 0, len(visited), time.time() - t0, "TIMEOUT"

        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_score[current], len(visited), time.time() - t0, "OK"

        for nr2, nc2, cost in get_neighbors(*current):
            neighbor = (nr2, nc2)
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f, neighbor))

    return None, 0, len(visited), time.time() - t0, "NO ROUTE"

# ----------------------------------------------------------------------------------------------------------------------
#   Coordinate pairs to test
# ----------------------------------------------------------------------------------------------------------------------
# All pairs are within the same connected component of the map (the largest, 224328 pixels)
# to guarantee a valid path exists between them.
test_cases = {
    "Short routes (< 500 m)": [
        ("Short 1 (~314 m)", (3296, 11230), (3586, 11109)),
        ("Short 2 (~417 m)", (3296, 11230), (3666, 11039)),
    ],
    "Medium routes (1000 - 5000 m)": [
        ("Medium 1 (~1445 m)", (3296, 11230), (4658, 11710)),
        ("Medium 2 (~2975 m)", (3296, 11230), (1933, 13874)),
        ("Medium 3 (~4438 m)", (3296, 11230), (3376, 6792)),
    ],
    "Long routes (> 10000 m)": [
        ("Long 1 (~10848 m)", (3296, 11230), (4328, 431)),
        ("Long 2 (~13042 m)", (571, 13564), (3777, 922)),
    ],
}

# ----------------------------------------------------------------------------------------------------------------------
#   Run tests
# ----------------------------------------------------------------------------------------------------------------------
all_results = []

for category, pairs in test_cases.items():
    print(f"\n{'='*70}")
    print(f"  {category}")
    print(f"{'='*70}")

    for name, (x1, y1), (x2, y2) in pairs:
        linear_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        start = coords_to_rc(x1, y1)
        goal = coords_to_rc(x2, y2)

        print(f"\n  {name}")
        print(f"  Start: ({x1}, {y1}) -> r={start[0]}, c={start[1]}, h={mars_map[start]:.1f} m")
        print(f"  End:   ({x2}, {y2}) -> r={goal[0]}, c={goal[1]}, h={mars_map[goal]:.1f} m")
        print(f"  Linear distance: {linear_dist:.0f} m")
        print(f"  Running A*...", end=" ", flush=True)

        path, dist, nodes, elapsed, status = a_star(start, goal)

        if status == "OK":
            print(f"Route found!")
            print(f"    Distance traveled: {dist:.2f} m")
            print(f"    Steps: {len(path)}")
            print(f"    Nodes visited: {nodes}")
            print(f"    Time: {elapsed:.2f} s")
        elif status == "TIMEOUT":
            print(f"TIMEOUT ({TIMEOUT}s)")
            print(f"    Nodes visited before timeout: {nodes}")
        else:
            print(f"No route found.")
            print(f"    Nodes visited: {nodes}")
            print(f"    Time: {elapsed:.2f} s")

        all_results.append({
            "name": name,
            "category": category,
            "start_xy": (x1, y1),
            "end_xy": (x2, y2),
            "start_rc": start,
            "end_rc": goal,
            "linear_dist": linear_dist,
            "path": path,
            "dist": dist,
            "nodes": nodes,
            "time": elapsed,
            "status": status,
        })

# ----------------------------------------------------------------------------------------------------------------------
#   Summary table
# ----------------------------------------------------------------------------------------------------------------------
print(f"\n\n{'='*90}")
print(f"{'Case':<22} {'Linear dist':>12} {'Status':>10} {'Route dist':>12} {'Nodes':>10} {'Time':>10}")
print(f"{'='*90}")
for r in all_results:
    dist_str = f"{r['dist']:.0f} m" if r['status'] == 'OK' else '-'
    print(f"{r['name']:<22} {r['linear_dist']:>10.0f} m {r['status']:>10} {dist_str:>12} {r['nodes']:>10} {r['time']:>9.2f}s")
print(f"{'='*90}")

# ----------------------------------------------------------------------------------------------------------------------
#   3D visualization for each route
# ----------------------------------------------------------------------------------------------------------------------
print("\nGenerating 3D visualizations...")

for r in all_results:
    # Calculate crop based on the route or endpoints
    all_rows = [r["start_rc"][0], r["end_rc"][0]]
    all_cols = [r["start_rc"][1], r["end_rc"][1]]
    if r["path"]:
        all_rows.extend([p[0] for p in r["path"]])
        all_cols.extend([p[1] for p in r["path"]])

    margin_px = 20
    r_min = max(0, min(all_rows) - margin_px)
    r_max = min(nr, max(all_rows) + margin_px)
    c_min = max(0, min(all_cols) - margin_px)
    c_max = min(nc, max(all_cols) + margin_px)

    zona = mars_map[r_min:r_max, c_min:c_max]
    zona_display = np.where(zona == -1, np.nan, zona)

    x_coords = scale * np.arange(c_min, c_max)
    y_coords = scale * (nr - np.arange(r_min, r_max))
    X_crop, Y_crop = np.meshgrid(x_coords, y_coords)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_crop, y=Y_crop, z=zona_display,
        colorscale='hot', cmin=0, showscale=True,
        colorbar=dict(title='Height (m)'),
        lighting=dict(ambient=0.3, diffuse=0.8, roughness=0.4, specular=0.2),
        lightposition=dict(x=0, y=0, z=5000),
        name='Terrain'
    ))

    x1, y1 = r["start_xy"]
    x2, y2 = r["end_xy"]
    z1 = mars_map[r["start_rc"]] + 2
    z2 = mars_map[r["end_rc"]] + 2

    if r["path"]:
        path_xs = [p[1] * scale for p in r["path"]]
        path_ys = [(nr - p[0]) * scale for p in r["path"]]
        path_zs = [mars_map[p[0], p[1]] + 1.5 for p in r["path"]]
        fig.add_trace(go.Scatter3d(
            x=path_xs, y=path_ys, z=path_zs,
            mode='lines',
            line=dict(color='cyan', width=5),
            name=f'Route: {r["dist"]:.0f} m'
        ))

    fig.add_trace(go.Scatter3d(
        x=[x1], y=[y1], z=[z1],
        mode='markers+text', text=['Start'], textposition='top center',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x2], y=[y2], z=[z2],
        mode='markers+text', text=['End'], textposition='top center',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='End'
    ))

    status_txt = f"Route: {r['dist']:.0f} m in {r['time']:.2f}s" if r["status"] == "OK" else r["status"]
    fig.update_layout(
        title=f"{r['name']} - {status_txt}",
        scene=dict(
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            zaxis_title='Height (m)',
            aspectmode='manual',
            aspectratio=dict(
                x=1,
                y=(r_max - r_min) / max(c_max - c_min, 1),
                z=0.4
            )
        )
    )
    fig.show()

print("\n3D visualizations generated.")

# ----------------------------------------------------------------------------------------------------------------------
#   Bar chart: time and nodes per case
# ----------------------------------------------------------------------------------------------------------------------
import plotly.graph_objects as go

names = [r["name"] for r in all_results]
times = [r["time"] for r in all_results]
nodes = [r["nodes"] for r in all_results]
statuses = [r["status"] for r in all_results]
bar_colors = ['green' if s == 'OK' else 'orange' if s == 'TIMEOUT' else 'red' for s in statuses]

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Execution time (s)", "Nodes visited"))

fig_bar.add_trace(go.Bar(
    x=names, y=times, marker_color=bar_colors, name='Time (s)',
    text=[f"{t:.2f}s" for t in times], textposition='auto'
), row=1, col=1)

fig_bar.add_trace(go.Bar(
    x=names, y=nodes, marker_color=bar_colors, name='Nodes',
    text=[f"{n:,}" for n in nodes], textposition='auto'
), row=1, col=2)

fig_bar.update_layout(
    title='A* performance by route distance',
    showlegend=False,
    height=500
)
fig_bar.show()
