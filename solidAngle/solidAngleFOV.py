import numpy as np
import matplotlib.pyplot as plt

"""
Solid Angle and Field of View (FOV) calculations for a camera system.
The code calculates and visualizes the solid angle, etendue, field of view (FOV),
instantaneous field of view (IFOV), depth of field (DOF), and ground sampling distance (GSD)
for a camera system. These parameters are essential for understanding the camera's.
"""

# Sensor parameters
A = (100e-6) ** 2  # Area (m^2) of cold f-stop (slit/aperture)
f_number = 2.29  # f-number of optical system

# Aperture diameter (D) and focal length (f)
D = 0.035  # Aperture diameter in meters
f = D * f_number  # Focal distance in meters

# Solid angle & etendue
NA = 1 / (2 * f_number)
Omega = np.pi * NA ** 2  # Solid angle
G = A * Omega  # Etendue [mW/(mm2*sr)]

# FOV, DoF, and GSD
pixSize = np.sqrt(A)  # Pixel size in meters
AOV = 2 * np.degrees(np.arctan(pixSize / (2 * f)))  # Angle of view in degrees
IFOV = pixSize / f  # Instantaneous FOV

confu = 2 * pixSize  # Circle of confusion
imageHeight = 1  # Image height in meters

# Initialize arrays for DoF and GSD
DOF = np.zeros(1000)
GSD = np.zeros(1000)

# Loop through distances (d) from 1 to 1000 meters
for d in range(1, 1001):
    DOF[d - 1] = 2 * f_number * confu * (d / f) ** 2  # Depth of field
    GSD[d - 1] = (d * pixSize) / (f * imageHeight)  # Ground sampling distance

# Plot Depth of Field (DOF)
plt.figure()
plt.subplot(121)
plt.plot(DOF, linewidth=2, label='DOF')
plt.title('DOF')
plt.xlabel('Distance [m]')
plt.ylabel('DOF [m]')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('DOF.png', dpi=300)
plt.show()

# Plot Scene patch size per pixel (similar to IFOV)
plt.plot(IFOV * np.arange(1, 1001) * 100, linewidth=2, label='Scene patch size per pixel')  
plt.xlabel('Distance [m]')
plt.ylabel('Scene patch size per pixel [cm]')
plt.grid(True, alpha=0.5, linestyle='--')
plt.title(f'AOV = {AOV:}°')
plt.legend()
plt.tight_layout()
plt.savefig('FOV_50mph.png', dpi=300)
plt.show()

# Plot Ground Sampling Distance (GSD)
plt.figure()
plt.plot(GSD * 100, linewidth=2, label= 'GSD')  # GSD in cm
plt.title('GSD')
plt.xlabel('Distance [m]')
plt.ylabel('GSD [cm]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('GSD.png', dpi=300)
plt.show()
