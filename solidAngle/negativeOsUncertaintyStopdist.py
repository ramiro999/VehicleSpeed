import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

'''

'''

# Define variables
wn = np.arange(0.1, 1.1, 0.1)
s = np.arange(1, 1000)

# Use parula colormap for color order
colors = cm.viridis(np.linspace(0, 1, len(wn)))

# Create the figure
fig, ax = plt.subplots()

# Initialize arrays
U_neg = np.zeros((len(wn), len(s)))

# Loop through stopping distances and object widths to calculate uncertainty
for i in range(len(wn)):
    U_neg[i, :] = (wn[i] / s) * 100
    ax.plot(s, U_neg[i, :], color=colors[i], linewidth=2)

# Set title and labels
ax.set_title('Negative obstacle')
ax.set_xlabel('Stopping distance [m]')
ax.set_ylabel('Required range precision [%]')

# Add legend
legend_labels = [f'{w:.1f} m' for w in wn]
lgd = ax.legend(legend_labels, title='Negative width [m]', loc='upper right')

# Set logarithmic scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Enable grid
ax.grid(True)

# Adjust layout for tight plots
plt.tight_layout()

# Save the figure
plt.savefig('Negative_Obstacle_Uncertainty_Stopdist.png', dpi=300)

# Show the plot
plt.show()