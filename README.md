# Mars Project - Search and Exploration Algorithms

Data analysis project implementing search algorithms to plan navigation routes for a Rover on Mars surface elevation maps.

## Project Structure

```
proyecto_mars/
├── height_map_preprocessing.py  # Preprocessing of .IMG → .npy files (both maps)
├── algoritmos.py                # Comparison of 4 search algorithms (A*, BFS, DFS, UCS)
├── performance.py               # A* performance analysis at various distances
├── mars_map.IMG                 # Surface map (not included due to size)
├── mars_map.npy                 # Processed surface map
├── class 7/
│   ├── greedy_search.py         # Greedy algorithm for crater descent
│   ├── simulated_annealing.py   # Simulated Annealing algorithm for crater descent
│   ├── crater_map.IMG           # Crater map (not included due to size)
│   └── crater_map.npy           # Processed crater map
└── README.md
```

## Part 1: Route Search (surface)

The Mars elevation map has dimensions of **756 x 1814 pixels**, where each pixel represents an area of **10.0174 x 10.0174 meters**. The Rover can move in 8 directions (including diagonals) with the following constraints:

- The height difference between the current and next position must be **less than 0.25 meters**
- Pixels with height **-1** are invalid and the Rover cannot traverse them

### Coordinate Conversion

```
r = nr - round(y / scale)
c = round(x / scale)
```

Where `nr` = number of rows, `scale` = 10.0174 m/pixel.

### Scripts

#### `algoritmos.py`
Compares 4 search algorithms to find a route from **(2850, 6400)** to **(3150, 6800)**:

| Algorithm | Type | Optimal | Description |
|-----------|------|---------|-------------|
| **A*** | Informed | Yes | Uses Euclidean heuristic to prioritize nodes closer to the destination |
| **BFS** | Uninformed | In steps | Explores by levels, finds the route with the fewest steps |
| **DFS** | Uninformed | No | Explores depth-first, may find very long routes |
| **UCS** | Uninformed | Yes | Always expands the node with the lowest accumulated cost |

#### `performance.py`
Evaluates A* performance with coordinate pairs at various distances (short, medium, and long).

## Part 2: Crater Descent (`class 7/`)

The crater map has original dimensions of **7163 x 10770 pixels** (~1.0045 m/pixel). After preprocessing, each pixel represents **10.045 x 10.045 meters** (1077 x 717 pixels).

The explorer robot:
- Only knows the depth of its **8 neighboring pixels**
- Cannot move to pixels with a height difference **greater than 2.0 meters**
- Goal: descend to the bottom of the crater from various starting positions

### Scripts

#### `greedy_search.py`
Greedy search: always moves to the neighbor with the lowest elevation. Stops when no neighbor is better than the current position.

- Gets stuck quickly at **local minima** (2-7 typical steps)
- Cannot climb to escape plateaus or terrain irregularities

#### `simulated_annealing.py`
Simulated annealing: accepts moves to worse positions with probability `e^(-Δ/T)`, where `T` is the temperature that gradually decreases.

- Parameters: `T_init=500`, `T_min=0.0001`, `α=0.9995`, `max_iter=100000`
- Capable of **escaping local minima** and reaching the crater bottom
- Significantly more effective than Greedy for this problem

### Test Positions

| Position (x, y) m | Greedy (descent) | SA (descent) |
|--------------------|------------------|--------------|
| (3350, 5800) | ~7 m | ~48 m |
| (3500, 5500) | ~9 m | ~88 m |
| (2700, 4800) | ~2 m | variable |

## Preprocessing

`height_map_preprocessing.py` processes both `.IMG` files:
- `mars_map.IMG` → `mars_map.npy` (surface map for route search)
- `class 7/crater_map.IMG` → `class 7/crater_map.npy` (crater map for descent)

If any of the `.IMG` files are not present, they are simply skipped.

## Requirements

```
numpy
scikit-image
matplotlib
plotly
```

Installation:
```bash
pip install numpy scikit-image matplotlib plotly
```

## Running

```bash
# 1. Preprocess maps (first time only, requires .IMG files)
python height_map_preprocessing.py

# 2. Route search (Part 1)
python algoritmos.py
python performance.py

# 3. Crater descent (Part 2)
cd "class 7"
python greedy_search.py
python simulated_annealing.py
```

## Key Findings

### Part 1 (Route search)
- With the 0.25m height difference constraint, the map fragments into **355 connected components**
- **A*** is the most efficient algorithm: finds the optimal route while visiting fewer nodes than the others

### Part 2 (Crater descent)
- **Greedy Search** is useless on irregular terrain: gets stuck at local minima after a few steps
- **Simulated Annealing** is much more effective thanks to the probabilistic acceptance of "bad" moves, allowing it to explore and escape local minima
- Neither algorithm guarantees reaching the bottom from any position
