import matplotlib.pyplot as plt
import numpy as np

# Class names and corresponding pixel counts
classes = ['MD', 'DenS', 'SpS', 'NatM', 'Ship', 'Cloud', 'MWater', 'SLWater', 'Foam', 'TWater', 'SWater', 'Waves', 'CloudS', 'Wakes', 'MixWater']
pixel_counts = [3399, 2797, 2357, 864, 5803, 117400, 129159, 372937, 1225, 157612, 17369, 5827, 11728, 8490, 410]  # Example pixel counts

# Create the plot
plt.figure(figsize=(6, 6))
plt.bar(classes, pixel_counts, color='skyblue', width=0.3)

# Set the scale to logarithmic for the y-axis
plt.yscale('log')

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Pixel Count (Log Scale)')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.savefig('fig2.png', dpi=300)
plt.show()
