import numpy as np
import matplotlib.pyplot as plt
#import math

# Sensor parameters
A = (30e-6)**2 # pixel area in m^2
f_number = 2.29 # f-number dod optical system 

'''
Note: f-number is defined as N = f/D, where f:focal distance, and D: lens diamater
Thus if f = 50mm, then f/2.9 = 17.24mm aperture diameter
If f = f/2.0, then f/2.0 = 25mm aperture diameter
How does photon flux change w.r.t to f-number? Answer: It doesn't.
'''

D = 0.1; # lens diameter in m
f = D * f_number # focal distance in m

# Solid angle and etendue
'''
Etendue (G) describes the ability of a source to emit light or the ability 
of an optical system to accept light. For a monochromator, its etendue of accepting light
is a function of the entrance slit area (A) times the solid angle (omega) from
which light is accepted.

The etendue is a limiting factor for the throughput of the monochromator.
A monochromator with smaller A and ? will have smaller etendue.
'''
numericalAperture = 1 / (2 * f_number) # numerical aperture
omega = np.pi * (numericalAperture)**2 # solid angle in steradians
etendue = A * omega # etendue in m^2 sr

# Field of view (FOV), Depth of field (DOF), and Ground Sampling Distance (GSD) 
pixSize = np.sqrt(A) # pixel size; in meters
AoV = 2 * np.degrees(np.arctan(pixSize / (2 * f))) # angle of view in degrees (per pixel)
IFoV = pixSize / f # instantaneous field of view m/degrees

circleConfusion = 2 * pixSize # Circle of confusion
imageHeight = 1; # in meters

distances = np.arange(1, 1001)
DoF = np.zeros_like(distances, dtype=np.float64)
Gr_Sam_Dis = np.zeros_like(distances, dtype=np.float64)
IFoV_distance = np.zeros_like(distances, dtype=np.float64)  

for i, d in enumerate(distances):
    # Depth of field
    DoF[i] = 2 * f_number * circleConfusion * (d / f)**2
    # Ground Sampling Distance (GSD) in meters
    Gr_Sam_Dis[i] = (d * pixSize) / (f * imageHeight);
    # Instantaneous Field of View (IFoV) in meters
    IFoV_distance[i] = 2 * (d * np.tan(np.deg2rad(AoV / 2)))

print(AoV)

# Plotting
# Plotting DOF
plt.figure(figsize=(12, 4))

# Subplot 1: Depth of Field
plt.subplot(1, 3, 1)
plt.plot(distances, DoF, linewidth=2)
plt.title('Depth of Field')
plt.xlabel('Distance [m]')
plt.ylabel('DOF [m]')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.yscale('log')

# Subplot 2: Instantaneous Field of View (IFOV)
plt.subplot(1, 3, 2)
plt.plot(distances, IFoV_distance * 100, linewidth=2)
plt.xlabel('Distance (m)')
plt.ylabel('Object size (cm)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.title(f'AOV = {AoV:.2f}º')

# Subplot 3: Ground Sampling Distance (GSD)
plt.subplot(1, 3, 3)
plt.plot(distances, Gr_Sam_Dis * 100, linewidth=2)
plt.xlabel('Distance (m)')
plt.ylabel('GSD (cm)')
plt.grid(True)
plt.title('Ground Sampling Distance GSD (cm)')

plt.tight_layout()
plt.show()