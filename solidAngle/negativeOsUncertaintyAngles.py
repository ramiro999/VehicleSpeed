import numpy as np
import matplotlib.pyplot as plt

'''
The code calculates and visualizes the uncertainty in estimating the size of a negative
obstacle (width) at different elevation angles, considering a fixed camera height.
It also highlights the uncertainty range between two specific angles, filling the area
beetween these curves to emphasize the variation. The objective is to understand how the 
camera's elevation angle affects the uncertainty in estimating obstacle size.
'''

# Define variables
hc = 4  # Camera height in meters
wn = np.arange(0, 0.51, 0.01)  # Object width in meters (negative obstacle)
angles = np.arange(1, 46)  # Elevation angles in degrees

# Initialize arrays
U_neg = np.zeros((len(wn), len(angles)))  # Uncertainty array

# Loop through angles and object widths to calculate uncertainty
for i in range(len(wn)):
    hw = wn[i] * np.tan(np.radians(angles))  # Calculate height-width relationship
    U_neg[i, :] = (hw / hc) * 100  # Uncertainty in percentage

# Plot the results
plt.figure()
for i in range(len(angles)):
    plt.plot(wn * 100, U_neg[:, i], linewidth=2, label=f'{angles[i]}°')  # Plot uncertainty for each angle

plt.title('Negative Obstacle Uncertainty')
plt.xlabel('Negative Obstacle Width [cm]')
plt.ylabel('Uncertainty [%]')
plt.grid(True)
plt.legend(title='Elevation Angle', loc='upper left')
plt.tight_layout()

# Filling between two curves (angle 1 and angle 2)
curve1 = U_neg[:, 0]
curve2 = U_neg[:, 1]
x = wn * 100
x2 = np.concatenate([x, x[::-1]])
inBetween = np.concatenate([curve1, curve2[::-1]])
plt.fill(x2, inBetween, color='blue', alpha=0.5)
#plt.savefig('Negative_Obstacle_Uncertainty.png', dpi=300)
plt.show()

# Plot width vs heights
plt.figure()
for i in range(len(angles)):
    hw = wn * np.tan(np.radians(angles[i]))
    plt.plot(wn * 100, (hw / hc) * 100, linewidth=2)

plt.title('Negative Obstacle Height vs Width')
plt.xlabel('Negative Obstacle Height [cm]')
plt.ylabel('Uncertainty [%]')
plt.grid(True)
plt.tight_layout()
#plt.savefig('Negative_Obstacle_Height_vs_Width.png', dpi=300)
plt.show()