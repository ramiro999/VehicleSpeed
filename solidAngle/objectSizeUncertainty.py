import numpy as np
import matplotlib.pyplot as plt

# Define the variables 
hc = np.arange(1, 4)  # height camera [in m]
ho = np.arange(0, 0.51, 0.01)  # height object [in m]

# Creaate the figure for the positive obstacle
plt.figure()
for j in range(len(hc)):
    U_pos = np.zeros(len(ho))  # Inicialize U_pos
    U_neg = np.zeros(len(ho))  # Inicialize U_neg
    for i in range(len(ho)):
        if hc[j] != ho[i]:  # Evite dividing by zero
            U_pos[i] = ho[i] / (hc[j] - ho[i]) * 100  # Calculating positive uncertainty
        else:
            U_pos[i] = np.inf  # Asignate infinity to avoid division by zero
    plt.plot(ho * 100, U_pos, linewidth=2)  # Graficate U_pos

# Configurate the positive obstacle graph
plt.title('Positive obstacle')
plt.xlabel('Object height [cm]')
plt.ylabel('Required range precision [%]')
plt.legend([f'{h} m' for h in hc], title="Camera height", loc='upper left')
plt.grid(True)
plt.ylim([0, 50])
plt.tight_layout()
plt.savefig('Positive_obstacle_Distinguishability.png', dpi=300)
plt.show()

# Create the figure for the negative obstacle
plt.figure()
for j in range(len(hc)):
    U_neg = np.zeros(len(ho))  # Inicializate U_neg
    for i in range(len(ho)):
        U_neg[i] = ho[i] / hc[j] * 100  # Calculating negative uncertainty
    plt.plot(ho * 100, U_neg, linewidth=2)  # Graficate U_neg

# Configurarate the negative obstacle graph
plt.title('Negative obstacle')
plt.xlabel('Negative obstacle depth [cm]')
plt.ylabel('Required range precision [%]')
plt.legend([f'{h} m' for h in hc], title="Camera height", loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Negative_obstacle_Distinguishability.png', dpi=300)
plt.show()

# Find the height of object for a certain uncertainty
plt.figure()
U = np.arange(1, 21) / 100  # Uncertainty in percentage

for j in range(len(hc)):
    ho_values = np.zeros(len(U)) # Inicialize ho_values
    for i in range(len(U)):
        ho_values[i] = (U[i] * hc[j] / (1 + U[i]))
    plt.plot(U * 100, ho_values * 100, linewidth=2) # Graficate ho_values

# Configurate the graph
plt.xlabel('Uncertainty [%]')
plt.ylabel('Object height [cm]')
plt.legend([f'{h} m' for h in hc], title="Camera height", loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Object_Height_vs_Uncertainty.png', dpi=300)
plt.show()
