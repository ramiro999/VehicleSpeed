import numpy as np
import matplotlib.pyplot as plt

# Define variables
hc = np.arange(1, 4, 1) # Camera height [m]
ho = np.arange(0, 0.51, 0.01) # Object height [m]

# Initialize arrays
#U_pos = np.zeros(len(hc), len(ho))
#U_neg = np.zeros(len(hc), len(ho))

# Plot for Positive obstacle
plt.figure()

# Loop through camera heights and positive obstacle
for i in range(len(hc)):
    U_pos = ho[i] / (hc[i] - ho[i]) * 100 # Positive obstacle
    plt.plot(ho * 100, U_pos, linewidth=2)

# Set title and labels
plt.title('Positive obstacle')
plt.xlabel('Object height [cm]')
plt.ylabel('Required range precision [%]')
plt.legend([f'{h:} m' for h in hc], title='Camera height [m]', loc='upper right')

# Enable grid
plt.grid(True)
plt.ylim([0, 50])
plt.tight_layout()
#plt.savefig('Positive_Obstacle_Uncertainty_ObjectHeight.png', dpi=300)

# Plot for Negative obstacle
plt.figure()

# Loop through camera heights and negative obstacle
for j in range(len(hc)):
    U_neg = ho / hc[j] * 100 # Negative obstacle uncertainty
    plt.plot(ho * 100, U_neg, linewidth=2)

# Set title and labels
plt.title('Negative obstacle')
plt.xlabel('Object height [cm]')
plt.ylabel('Required range precision [%]')
plt.legend([f'{h:} m' for h in hc], title='Camera height [m]', loc='upper right')

# Enable grid
plt.grid(True)
plt.ylim([0, 50])
plt.tight_layout()
#plt.savefig('Negative_Obstacle_Uncertainty_ObjectHeight.png', dpi=300)
#plt.show()

# Uncomment the next section for plotting object height vs uncertainty

# Finding object height for a given uncertainty
plt.figure()
U = np.arange(1, 21) / 100 
ho = np.zeros_like(U)

for j in range(len(hc)):
    for i in range(len(U)):
        ho[i] = U[i] * hc[j] / (1 + U[i])
    plt.plot(U * 100, ho * 100, linewidth=2)

# Set title and labels
plt.title('Object height vs uncertainty')
plt.xlabel('Required range precision [%]')
plt.ylabel('Object height [cm]')
plt.legend([f'{h:} m' for h in hc], title='Camera height [m]', loc='upper right')

# Enable grid
plt.grid(True)

# Adjust layout for tight plots
plt.tight_layout()

# Save the figure
plt.savefig('Object_Height_vs_Uncertainty.png', dpi=300)

# Show the plot
plt.show()
