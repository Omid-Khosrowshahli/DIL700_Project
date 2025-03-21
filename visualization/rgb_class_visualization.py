import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# Define the mapping of class labels to names and colors
class_mapping = {
    1: ("Marine Debris", "red"),
    2: ("Dense Sargassum", "darkorange"),
    3: ("Sparse Sargassum", "yellow"),
    4: ("Natural Organic Material", "brown"),
    5: ("Ship", "green"),
    6: ("Clouds", "gray"),
    7: ("Marine Water", "blue"),
    8: ("Sediment-Laden Water", "peru"),
    9: ("Foam", "purple"),
    10: ("Turbid Water", "darkblue"),
    11: ("Shallow Water", "lightblue"),
    12: ("Waves", "gold"),
    13: ("Cloud Shadows", "black"),
    14: ("Wakes", "pink"),
    15: ("Mixed Water", "cyan"),
}

# Load the Sentinel-2 RGB image
with rasterio.open("S2_1-12-19_48MYU_0.tif") as src:
    rgb_image = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1)

# Load the classification map
with rasterio.open("S2_1-12-19_48MYU_0_cl.tif") as class_src:
    classification_map = class_src.read(1)

# Get unique class labels **excluding non-annotated pixels (0)
unique_labels = np.unique(classification_map)
unique_labels = unique_labels[unique_labels > 0]
print("Unique annotated class labels found:", unique_labels)

# Filter class mapping to include only present labels
filtered_class_mapping = {k: v for k, v in class_mapping.items() if k in unique_labels}

class_labels = list(filtered_class_mapping.keys())  # Only present labels
class_colors = [filtered_class_mapping[label][1] for label in class_labels]
cmap = mcolors.ListedColormap(class_colors)
norm = mcolors.BoundaryNorm(class_labels + [max(class_labels) + 1], cmap.N)

# Mask non-annotated pixels in classification map (set to NaN)
masked_classification_map = np.where(classification_map == 0, np.nan, classification_map)

# RGB on top, Classification below
fig, ax = plt.subplots(2, 1, figsize=(8, 12))  # 2 rows, 1 column

# Plot the RGB image
ax[0].imshow(rgb_image / np.max(rgb_image))  # Normalize RGB values
ax[0].set_title("Sentinel-2 RGB")
ax[0].axis("off")

# Plot the classification map
im = ax[1].imshow(masked_classification_map, cmap=cmap, norm=norm)

rect = patches.Rectangle((0, 0), 1, 1, transform=ax[1].transAxes,
                         linewidth=2, edgecolor='black', facecolor='none')
ax[1].add_patch(rect)

ax[1].set_title("Classification Map")
ax[1].axis("off")

legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=name)
                  for label, (name, color) in filtered_class_mapping.items()]
ax[1].legend(handles=legend_handles, loc="lower left")

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig("figure2.png", dpi=300, bbox_inches="tight")
plt.show()
