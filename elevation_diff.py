import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ====== CONFIG ======
DEM_PATH = "ldem_map.tif"   # your aligned elevation (meters)
CELL_SIZE = 5.0             # meters per pixel (cardinal moves). Diagonals = 5*sqrt(2)

# ====== LOAD DEM ======
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(np.float64)
    profile = src.profile

# mask invalids
dem[~np.isfinite(dem)] = np.nan

# ====== 8-NEIGHBOR OFFSETS & DISTANCES ======
# (dr, dc, distance_in_meters)
import math
NEIGHBORS = [
    (-1,  0, CELL_SIZE),               # N
    ( 1,  0, CELL_SIZE),               # S
    ( 0, -1, CELL_SIZE),               # W
    ( 0,  1, CELL_SIZE),               # E
    (-1, -1, CELL_SIZE*math.sqrt(2)),  # NW
    (-1,  1, CELL_SIZE*math.sqrt(2)),  # NE
    ( 1, -1, CELL_SIZE*math.sqrt(2)),  # SW
    ( 1,  1, CELL_SIZE*math.sqrt(2)),  # SE
]

H, W = dem.shape

# To hold per-direction Δh and slope (grade) if you want it
delta_h_stack = []      # list of arrays, one per neighbor
grade_stack = []        # Δh / run

for dr, dc, run in NEIGHBORS:
    # shift DEM to get neighbor values
    shifted = np.full_like(dem, np.nan)
    # compute valid slices
    r_src = slice(max(0, -dr), min(H, H - dr))
    c_src = slice(max(0, -dc), min(W, W - dc))
    r_dst = slice(max(0, dr), min(H, H + dr))
    c_dst = slice(max(0, dc), min(W, W + dc))

    shifted[r_dst, c_dst] = dem[r_src, c_src]

    dH = shifted - dem               # Δh toward that neighbor (meters)
    delta_h_stack.append(dH)

    grade = dH / run                 # rise/run (unitless). multiply by 100 for %
    grade_stack.append(grade)

delta_h_stack = np.stack(delta_h_stack, axis=0)   # shape (8, H, W)
grade_stack   = np.stack(grade_stack, axis=0)     # shape (8, H, W)

# ====== PER-PIXEL STEEPEST UPHILL / DOWNHILL ======
# max positive Δh to any neighbor (meters)
max_uphill_dh   = np.nanmax(np.where(delta_h_stack > 0, delta_h_stack, np.nan), axis=0)
# most negative Δh (i.e., steepest downhill; still negative meters)
max_downhill_dh = np.nanmin(np.where(delta_h_stack < 0, delta_h_stack, np.nan), axis=0)

# A single "signed best" map:
# pick the one with larger magnitude between uphill and downhill at each pixel
abs_up   = np.abs(max_uphill_dh)
abs_down = np.abs(max_downhill_dh)
signed_best_dh = np.where(np.nan_to_num(abs_up, nan=-1) >= np.nan_to_num(abs_down, nan=-1),
                          max_uphill_dh, max_downhill_dh)

# ====== QUICK STATS ======
valid_signed = signed_best_dh[np.isfinite(signed_best_dh)]
print("\n--- Δh (per-step) summary ---")
print(f"Steepest per-pixel uphill Δh (median): {np.nanmedian(max_uphill_dh):.2f} m")
print(f"Steepest per-pixel downhill Δh (median): {np.nanmedian(max_downhill_dh):.2f} m")
print(f"Signed 'best' Δh | min: {np.nanmin(valid_signed):.2f} m, "
      f"max: {np.nanmax(valid_signed):.2f} m, "
      f"mean: {np.nanmean(valid_signed):.2f} m, "
      f"std: {np.nanstd(valid_signed):.2f} m")

# Robust color scaling (trim extreme outliers so the map is readable)
q = np.nanquantile(np.abs(valid_signed), 0.99)
vmax = max(q, 1e-6)

# ====== PLOTS ======
plt.figure(figsize=(18, 5))

# (1) Steepest Uphill Δh
ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(max_uphill_dh, cmap="Reds", vmin=0, vmax=vmax)
ax1.set_title("Steepest Uphill Δh to a Neighbor (m)")
ax1.axis("off")
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label("meters")

# (2) Steepest Downhill Δh (shown as positive magnitude)
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(-max_downhill_dh, cmap="Blues", vmin=0, vmax=vmax)
ax2.set_title("Steepest Downhill |Δh| to a Neighbor (m)")
ax2.axis("off")
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label("meters")

# (3) Signed best: red = uphill, blue = downhill
ax3 = plt.subplot(1, 3, 3)
im3 = ax3.imshow(signed_best_dh, cmap="coolwarm", vmin=-vmax, vmax=vmax)
ax3.set_title("Signed Elevation Change to Steepest Neighbor (m)\n(red=uphill, blue=downhill)")
ax3.axis("off")
cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cb3.set_label("meters")

plt.tight_layout()
plt.show()