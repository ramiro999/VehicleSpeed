import numpy as np
import matplotlib.pyplot as plt

# Velocity rates
v_mph = np.arange(1, 151)   # speed in miles/hour
v_kph = v_mph * (1.609 / 1) # speed in km/hour
v_mtph = v_mph * (1609.34 / 1) # speed in meters/hour
v_mtps = v_mtph * (1 / 60**2) # speed in meters/second

# Parameters for the model SPIE
Tper = 0.1 # perception time; in seconds
Tact = 0.25 # actuaction time (latency); in seconds
a = -3 # deceleration; in m/s^2
mu = 0.55 # friction coefficient
g = 9.81 # gravity acceleration; in meters/second^2
d_offset = 2 # offset distance; in meters

w = 1.82; # width of wheelbase (1.82 m for HMMVW)
cog = 0.767; # height of center of gravity (0.767 m for HMMVW)
turningCar = 7.62; # HMMWV minimum turning radius is 7.62 m

# Variables for the distances and the model

d_per = np.zeros_like(v_mtps) 
d_act = np.zeros_like(v_mtps)
d_brake = np.zeros_like(v_mtps)
# Looakead distance for stopping
d_look_stop = np.zeros_like(v_mtps)

K_roll = np.zeros_like(v_mtps)
K_slip = np.zeros_like(v_mtps)
turning = np.zeros_like(v_mtps)
d_swerve = np.zeros_like(v_mtps)
# Lookahead distance for swerving
d_look_swerve = np.zeros_like(v_mtps)

# Compute the lookahead distance requirements
for v in range(1, 151):
    # distance traveled
    d_per[v-1] = v_mtps[v-1] * (2*Tper) # Perception distance (Looking twice
    d_act[v-1] = v_mtps[v-1] * Tact # Action distance (Latency)

    # Stopping distance (SPIE)
    d_brake[v-1] = - v_mtps[v-1]**2 / (2 * a) # (-2*mu*g) # in meters
    d_look_stop[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_brake[v-1] # in meters


    # Swerving distance (SPIE)
    K_roll[v-1] = (g * (w/2)) / (cog * v_mtps[v-1]**2) # in meters
    K_slip[v-1] = (mu * g) / (v_mtps[v-1]**2) # in meters

    turning[v-1] = max(1 / (min(K_roll[v-1], K_slip[v-1])), turningCar)

    d_swerve[v-1] = np.real(np.sqrt(turning[v-1]**2 - (turning[v-1] - w)**2)) # in meters

    d_look_swerve[v-1] =  d_offset + d_per[v-1] + d_act[v-1] + d_swerve[v-1] # in meters

# Plot
plt.plot(v_mph, d_look_stop, linewidth=2, label='Stopping Distance [m]')
plt.plot(v_mph, d_look_swerve, linewidth=2, label='Swerve distance [m]')
plt.grid(True)
plt.xlabel('Vehicle speed [mph]')
plt.ylabel('Lookahead distance [m]')
plt.ylim(0, 800)
plt.legend()
plt.tight_layout()
plt.savefig('Lookahead_Distance_For_Stopping_Distance(mph).png')
plt.show()

# Print the values in the txt file in the same directory (math-model)
with open('Lookahead_Distance_For_Swerve_Stop.txt','w') as f:
    f.write('Stopping Distance [m]' + '\n')
    for x, y in zip(v_mph, d_look_stop):
        f.write('SPIE: Vehicle speed [mph]: ' + str(x) + ' -- Lookahead distance [m]: ' + str(y) + '\n')
    f.write('Swerve distance [m]' + '\n')
    for x, y in zip(v_mph, d_look_swerve):
        f.write('vT: Vehicle speed [mph]: ' + str(x) + ' -- Lookahead distance [m]: ' + str(y) + '\n')



 
