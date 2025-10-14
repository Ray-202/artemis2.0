import rasterio
from pyproj import CRS, Transformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import heapq
from math import sqrt

# Configuration
path = '/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'
cell_size_m = 5.0
max_safe_slope = 12.0   # degrees; > this considered impassable
visualize_search_default = True
vis_interval_default = 200
pause_time_default = 0.01
max_iters_default = 300000

# Open raster once
src = rasterio.open(path)
slope = src.read(1).astype(float)
nodata = src.nodata
crs = src.crs

if nodata is not None:
    slope[slope == nodata] = np.nan

n_rows, n_cols = slope.shape

# CRS transformers (lon/lat on Moon <-> raster)
moon_lonlat = CRS.from_proj4('+proj=longlat +R=1737400 +no_defs')
to_raster = Transformer.from_crs(moon_lonlat, crs, always_xy=True)
from_raster = Transformer.from_crs(crs, moon_lonlat, always_xy=True)

# Simple slope-based cost: 1 + slope/max_safe_slope for slope <= threshold; else inf
slope_term = np.where(np.isnan(slope), np.nan, slope / max_safe_slope)
cost_grid = np.where(np.isnan(slope), np.nan,
                     np.where(slope <= max_safe_slope, 1.0 + slope_term, np.inf))

safe_mask = (slope <= max_safe_slope)

# Normalize extents for plotting (robust percentiles)
s_vmin, s_vmax = np.nanpercentile(slope, 1), np.nanpercentile(slope, 99)
c_display = cost_grid.copy()
c_display[np.isinf(c_display)] = np.nan
c_vmin, c_vmax = np.nanpercentile(c_display, 1), np.nanpercentile(c_display, 99)

norm_s = Normalize(vmin=s_vmin, vmax=s_vmax)
norm_c = Normalize(vmin=c_vmin, vmax=c_vmax)

# Figure: two panels (slope, cost)
fig, (ax_slope, ax_cost) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

im_s = ax_slope.imshow(slope, cmap='terrain', norm=norm_s, interpolation='nearest')
ax_slope.set_title(f'Slope (cells = {cell_size_m:.0f}×{cell_size_m:.0f} m)')
ax_slope.set_xlabel('Column')
ax_slope.set_ylabel('Row')
fig.colorbar(im_s, ax=ax_slope, fraction=0.046, pad=0.04, label='Slope (°)')

im_c = ax_cost.imshow(c_display, cmap='viridis', norm=norm_c, interpolation='nearest')
ax_cost.set_title('Traversal Cost (slope-only)')
ax_cost.set_xlabel('Column')
ax_cost.set_ylabel('Row')
fig.colorbar(im_c, ax=ax_cost, fraction=0.046, pad=0.04, label='Cost')

# Interactive state
pointA = None
pointB = None
markerA = None
markerB = None
path_line = None
visual_overlay = None
visualize_search = visualize_search_default
running = True

# neighbors (8-connected grid)
def neighbors8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                yield nr, nc, sqrt(dr*dr + dc*dc)

# A* search (visualization throttled)
def astar(start, goal, visualize=True, pause_time=0.0, vis_interval=200, max_iters=300000, slope_diff_weight=0.02):
    sr, sc = start
    gr, gc = goal

    if np.isnan(slope[sr, sc]) or np.isnan(slope[gr, gc]):
        print("Start or goal is invalid (NaN).")
        return None, np.inf

    if np.isinf(cost_grid[sr, sc]) or np.isinf(cost_grid[gr, gc]):
        print("Start or goal is impassable.")
        return None, np.inf

    # admissible heuristic: Euclidean distance * minimum finite per-unit traversal cost
    finite_mask = np.isfinite(cost_grid)
    if np.any(finite_mask):
        min_traversal = float(np.nanmin(cost_grid[finite_mask]))
        min_traversal = max(min_traversal, 1e-8)
    else:
        min_traversal = 1.0

    def heuristic(a, b):
        return min_traversal * sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    open_heap = []
    heapq.heappush(open_heap, (heuristic((sr, sc), (gr, gc)), (sr, sc)))
    g_score = np.full((n_rows, n_cols), np.inf)
    g_score[sr, sc] = 0.0
    came_from = {}
    in_closed = np.zeros((n_rows, n_cols), dtype=bool)

    overlay = np.zeros((n_rows, n_cols), dtype=np.uint8)
    iter_count = 0

    while open_heap:
        iter_count += 1
        if iter_count > max_iters:
            print(f"A* aborted after {iter_count} iterations (max_iters={max_iters}).")
            return None, np.inf

        _, (r, c) = heapq.heappop(open_heap)
        if in_closed[r, c]:
            continue
        in_closed[r, c] = True
        overlay[r, c] = 2

        if (r, c) == (gr, gc):
            # reconstruct
            path = [(gr, gc)]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            print(f"A* finished: {len(path)} steps, g={g_score[gr, gc]:.3f}, iterations={iter_count}")
            return path, g_score[gr, gc]

        for nr, nc, dist in neighbors8(r, c):
            if not np.isfinite(cost_grid[nr, nc]) or not np.isfinite(cost_grid[r, c]):
                continue
            avg_trav = 0.5 * (cost_grid[r, c] + cost_grid[nr, nc])
            slope_pen = slope_diff_weight * abs(slope[nr, nc] - slope[r, c])
            tentative_g = g_score[r, c] + dist * avg_trav + slope_pen

            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f = tentative_g + heuristic((nr, nc), (gr, gc))
                heapq.heappush(open_heap, (f, (nr, nc)))
                overlay[nr, nc] = 1

        # visualization update (throttled)
        if visualize and (iter_count % vis_interval == 0):
            _visualize_overlay(overlay, came_from, (r, c))
            plt.pause(pause_time)

    print("A* failed: no path found.")
    return None, np.inf

def _visualize_overlay(overlay, came_from, current):
    global visual_overlay, path_line
    if visual_overlay is None:
        cmap = plt.get_cmap('plasma')
        visual_overlay = ax_slope.imshow(overlay, cmap=cmap, alpha=0.45, interpolation='nearest', vmin=0, vmax=2)
    else:
        visual_overlay.set_data(overlay)

    # draw current best partial path (to current)
    path = []
    node = current
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path = path[::-1]
    xs = [c for (r,c) in path]
    ys = [r for (r,c) in path]
    if path_line is None:
        if len(xs) > 1:
            path_line, = ax_slope.plot(xs, ys, color='yellow', linewidth=1.2)
    else:
        if len(xs) > 1:
            path_line.set_data(xs, ys)
        else:
            path_line.set_data([], [])
    fig.canvas.draw_idle()

# Click handlers: set A then B to run A*
click_state = {'count': 0}
def onclick_set_points(event):
    global pointA, pointB, markerA, markerB, path_line, visual_overlay
    if event.inaxes is not ax_slope:
        return
    if event.xdata is None or event.ydata is None:
        return
    col = int(event.xdata)
    row = int(event.ydata)
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        return

    if click_state['count'] % 2 == 0:
        # set A
        pointA = (row, col)
        if markerA:
            try: markerA.remove()
            except Exception: pass
        markerA = ax_slope.plot(col, row, marker='o', color='lime', markersize=8)[0]
        print(f"Point A set at row={row}, col={col}")
    else:
        # set B and run A*
        pointB = (row, col)
        if markerB:
            try: markerB.remove()
            except Exception: pass
        markerB = ax_slope.plot(col, row, marker='o', color='red', markersize=8)[0]
        print(f"Point B set at row={row}, col={col}")

        # clear previous overlays/paths
        if visual_overlay:
            try: visual_overlay.remove()
            except Exception: pass
            visual_overlay = None
        if path_line:
            try: path_line.remove()
            except Exception: pass
            path_line = None

        # Run A*
        path, total_cost = astar(pointA, pointB, visualize=visualize_search,
                                 pause_time=pause_time_default, vis_interval=vis_interval_default,
                                 max_iters=max_iters_default)
        if path is not None:
            cols = [c for (r,c) in path]
            rows = [r for (r,c) in path]
            ax_slope.plot(cols, rows, color='yellow', linewidth=2)
            fig.canvas.draw_idle()
            print(f"Final path length {len(path)}  total cost {total_cost:.3f}")
        else:
            print("No path found.")

    click_state['count'] += 1
    fig.canvas.draw_idle()

# Keyboard handler: r reset, v toggle visualize, q quit
def on_key(event):
    global pointA, pointB, markerA, markerB, path_line, visual_overlay, visualize_search, running, click_state
    key = getattr(event, 'key', None)
    if key == 'r':
        # reset everything
        pointA = pointB = None
        click_state['count'] = 0
        for obj in (markerA, markerB, path_line, visual_overlay):
            if obj:
                try: obj.remove()
                except Exception: pass
        markerA = markerB = path_line = visual_overlay = None
        fig.canvas.draw_idle()
        print("Reset points and overlays.")
    elif key == 'v':
        visualize_search = not visualize_search
        print(f"Visualization toggled -> {visualize_search}")
    elif key == 'q':
        running = False
        plt.close('all')
        print("Quitting and closing windows.")

# Connect handlers
cid_points = fig.canvas.mpl_connect('button_press_event', onclick_set_points)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

# Show once and enter responsive loop
plt.ion()
plt.show(block=False)

try:
    while running and plt.get_fignums():
        plt.pause(0.1)
except KeyboardInterrupt:
    pass
finally:
    src.close()
