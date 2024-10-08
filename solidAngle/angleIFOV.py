import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

'''
This module calculates and visualizes how the Instantaneous Field of View (IFOV) changes
for both positive and negative obstacles at varying distances from a camera placed at a
certain height. It simulates the IFOV for obstacles of different sizes, showing how the
field of view and the number of pixels occupied by the obstacle change as the stopping
distance increases. Additionally, the code visualizes how the elevation angle decreases 
with increasing stopping distance for different camera heights. These visualizations are
helpful for understanding the relationship between obstacle size, distance, and camera
view in applications such as obstacle detection and machine vision systems
'''

# Define variables
stoppingDistance = np.arange(1, 1001)  # Stopping distance in meters
hc = 2  # Camera height in meters
hp = np.arange(0.1, 1.1, 0.1)  # Positive obstacle height in meters
wn = hp  # Negative obstacle width in meters

IFOVp = np.zeros((len(hp), len(stoppingDistance)))  # Positive obstacle field of view
IFOVn = np.zeros((len(wn), len(stoppingDistance)))  # Negative obstacle field of view

# Calculate IFOV for positive and negative obstacles
for i in range(len(hp)):
    IFOVp[i, :] = np.arctan(hc / stoppingDistance) - np.arctan((hc - hp[i]) / stoppingDistance)
    IFOVn[i, :] = np.arctan(hc / stoppingDistance) - np.arctan(hc / (stoppingDistance + wn[i]))

# Plot for Positive Obstacle IFOV
plt.figure()
colors = cm.get_cmap('viridis', len(hp))  # Use 'viridis' colormap
for i in range(len(hp)):
    plt.plot(stoppingDistance, IFOVp[i, :] * 10**3, color=colors(i), linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.title('Positive Obstacle IFOV')
plt.xlabel('Sensor distance to the scene [m]')
plt.ylabel('IFOV [milliradians]')
plt.ylim([10**-4, 10**3])
hp_labels = [f'{h:.1f}' for h in hp]
plt.legend(hp_labels, title="Object size [m]", loc='lower left')
plt.tight_layout()
plt.savefig('IFOV_Positive.png', dpi=300)
plt.show()

# Plot for Negative Obstacle IFOV
plt.figure()
colors = cm.get_cmap('plasma', len(wn))  # Use 'plasma' colormap
for i in range(len(wn)):
    plt.plot(stoppingDistance, IFOVn[i, :] * 10**3, color=colors(i), linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.title('Negative Obstacle IFOV')
plt.xlabel('Sensor distance to the scene [m]')
plt.ylabel('IFOV [milliradians]')
plt.ylim([10**-4, 10**3])
wn_labels = [f'{w:.1f}' for w in wn]
plt.legend(wn_labels, title="Object size [m]", loc='upper right')
plt.tight_layout()
plt.savefig('IFOV_Negative.png', dpi=300)
plt.show()

# Number of pixels
IFOVenvisioned = 0.43  # millirads

# Plot for Positive Obstacle Number of Pixels
plt.figure()
colors = cm.get_cmap('viridis', len(hp))
for i in range(len(hp)):
    plt.plot(stoppingDistance, IFOVp[i, :] * 10**3 / IFOVenvisioned, color=colors(i), linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.title('Positive Obstacle Number of Pixels')
plt.xlabel('Sensor distance to the scene [m]')
plt.ylabel('Number of pixels filled')
plt.ylim([10**-1, 10**3])
plt.xlim([min(stoppingDistance), max(stoppingDistance)])
plt.legend(hp_labels, title="Object size [m]", loc='lower left')
plt.tight_layout()
plt.savefig('IFOV_Positive_NumPixels.png', dpi=300)
plt.show()

# Plot for Negative Obstacle Number of Pixels
plt.figure()
colors = cm.get_cmap('plasma', len(wn))
for i in range(len(wn)):
    plt.plot(stoppingDistance, IFOVn[i, :] * 10**3 / IFOVenvisioned, color=colors(i), linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.title('Negative Obstacle Number of Pixels')
plt.xlabel('Sensor distance to the scene [m]')
plt.ylabel('Number of pixels filled')
plt.ylim([10**-1, 10**3])
plt.xlim([min(stoppingDistance), max(stoppingDistance)])
plt.legend(wn_labels, title="Object size [m]", loc='upper right')
plt.tight_layout()
plt.savefig('IFOV_Negative_NumPixels.png', dpi=300)
plt.show()

# How angle reduces with stopping distance
hc_values = np.arange(1, 4)  # Camera heights from 1 to 3 meters
s_values = np.arange(1, 101)  # Stopping distances from 1 to 100 meters

# Create the figure for angle vs stopping distance
plt.figure(figsize=(8, 6))
for hc in hc_values:
    angles = np.degrees(np.arctan(hc / s_values))  # Elevation angle in degrees
    #angles_approx = np.degrees(hc / s_values)  # Approximation
    plt.plot(s_values, angles, linewidth=2, label=f'{hc} m')
    #plt.plot(s_values, angles_approx, linewidth=2, linestyle='--')

plt.xscale('linear')
plt.yscale('log')  # Logarithmic scale for Y-axis
plt.xlabel('Stopping Distance [m]')
plt.ylabel('Elevation Angle θ_n [degrees]')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.title('Elevation Angle vs Stopping Distance')

# Add legend
plt.legend(title="Camera Height [m]", loc='upper right')




# Adjust layout and save
plt.tight_layout()
plt.savefig('Elevation_Angle_vs_StoppingDistance.png', dpi=300)
plt.show()
