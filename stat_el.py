import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import rasterio

DEM_PATH = "ldem_map.tif"

# --- Load DEM ---
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1)
    transform = src.transform

# --- Basic statistics ---
valid = dem[np.isfinite(dem)]  # remove NaN or nodata if any
print("\n--- ELEVATION DATA SUMMARY ---")
print(f"Shape: {dem.shape}")
print(f"Min elevation: {np.min(valid):.2f} m")
print(f"Max elevation: {np.max(valid):.2f} m")
print(f"Mean elevation: {np.mean(valid):.2f} m")
print(f"Std dev: {np.std(valid):.2f} m")
print(f"Range: {np.ptp(valid):.2f} m")

# --- Histogram of elevations ---
plt.figure(figsize=(7,4))
plt.hist(valid, bins=60, color='steelblue', edgecolor='black')
plt.title("Elevation Distribution (meters)")
plt.xlabel("Elevation (m)")
plt.ylabel("Pixel count")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# --- Elevation profile across a random row ---
row = dem.shape[0] // 2
plt.figure(figsize=(8,3))
plt.plot(dem[row, :], color='darkorange')
plt.title(f"Elevation Profile — Row {row}")
plt.xlabel("Column index")
plt.ylabel("Elevation (m)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- 3D terrain visualization (downsampled for speed) ---
sample = dem[::50, ::50]  # take every 50th pixel to reduce load
x = np.arange(sample.shape[1])
y = np.arange(sample.shape[0])
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, sample, cmap='terrain', linewidth=0, antialiased=False)
ax.set_title("3D Terrain Preview (Downsampled)")
ax.set_xlabel("X index")
ax.set_ylabel("Y index")
ax.set_zlabel("Elevation (m)")
plt.tight_layout()
plt.show()

# --- Random elevation samples ---
print("\nSample elevation values (10 random pixels):")
rows = np.random.randint(0, dem.shape[0], 10)
cols = np.random.randint(0, dem.shape[1], 10)
for r, c in zip(rows, cols):
    print(f"({r}, {c}) → {dem[r, c]:.2f} m")
    
    
    
    
H, W = dem.shape
target_pts = 400_000                 # ~400k points
stride = max(1, int(np.ceil(np.sqrt((H*W)/target_pts))))
print("Using stride:", stride)

sample = dem[::stride, ::stride]
y = np.arange(sample.shape[0]); x = np.arange(sample.shape[1])
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, sample, cmap='terrain', linewidth=0, antialiased=False)
ax.view_init(elev=50, azim=-70)
plt.tight_layout(); plt.show()