import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the shapefile
shapefile_path = "S2_4-9-16_16PCC"  # Adjust this to your file path
gdf = gpd.read_file(f"{shapefile_path}.shp")

print(gdf.head())

# Define color map for the 'conf' values
conf_colors = {1: 'green', 2: 'orange', 3: 'red'}  # High: green, Moderate: orange, Low: red

# Map the 'conf' values to colors
gdf['color'] = gdf['conf'].map(conf_colors)

# Plot the shapefile colored by the 'color' column
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color=gdf['color'], edgecolor="black")

import matplotlib.patches as mpatches
legend_labels = {
    'High': 'green',
    'Moderate': 'orange',
    'Low': 'red'
}

handles = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]

ax.legend(handles=handles, title="Confidence Levels")

plt.title("S2 tile shapefile visualization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig("shapefile_visualization.png", dpi=300, bbox_inches='tight')

plt.show()
