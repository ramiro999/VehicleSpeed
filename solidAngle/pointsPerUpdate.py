import numpy as np
import matplotlib.pyplot as plt

# Velocity rates 
v_mph = np.arange(1, 101) # miles/hour
v_kph = v_mph * (1.609/1) # km/hour
v_mtph = v_mph * (1609.34/1) # meters/hour
v_mtps = v_mtph * (1 / 3600) # meters/second

# Parameters
Tper = 0.1 # perception time [in seconds]
Tact = 0.25 # planning/actuation time [in seconds]
a = -3 # deceleration in m/s^2 (-3 for HMMVW)
mu = 0.55 # friction coefficient (0.55 for gravel)
g = 9.81 # gravity in m/s^2
d_offset = 2 # offset distance due to car length [in meters]

w = 1.82 # width of wheelbase (1.82 m for HMMVW)
cog = 0.767 # height of center of gravity (0.767 m for HMMVW)
turningCar = 7.62 # turning radius of car (7.62 m for HMMVW)

# Initialize variables
d_per = np.zeros_like(v_mtps)
d_act = np.zeros_like(v_mtps)
d_brake = np.zeros_like(v_mtps)
lookahead_dist_stop = np.zeros_like(v_mtps)
Kroll = np.zeros_like(v_mtps)
Kslip = np.zeros_like(v_mtps)
turning = np.zeros_like(v_mtps)
d_swerve = np.zeros_like(v_mtps)
lookahead_dist_swerve = np.zeros_like(v_mtps)
HFOV = np.zeros_like(v_mtps)

# Compute HFOV
for v in range(1,101):
    d_per[v-1] = v_mtps[v-1] * (2*Tper) # Perception distance (Looking twice)
    d_act[v-1] = v_mtps[v-1] * Tact # Planning/Actuation distance

    # Stopping distance
    d_brake[v-1] = - v_mtps[v-1]**2 / (2*a) # Braking distance
    lookahead_dist_stop[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_brake[v-1] # Total stopping distance

    # Swerving distance
    Kroll[v-1] = (g *(w/2)) / (cog * v_mtps[v-1]**2) # Roll angle
    Kslip[v-1] = (mu * g) / (v_mtps[v-1]**2) # Slip angle
    turning[v-1] = max(1 / (min(Kroll[v-1], Kslip[v-1])), turningCar) # Turning radius
    d_swerve[v-1] = np.real(np.sqrt(turning[v-1]**2 - (turning[v-1] - w)**2)) # Swerving distance
    lookahead_dist_swerve[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_swerve[v-1] # Total swerving distance

    # HFOV
    HFOV[v-1] = lookahead_dist_stop[v-1] / turning[v-1] # Horizontal Field of View

#HFOV = np.kron(np.deg2rad(25), np.ones(100)) # Horizontal Field of View 

# Plot
plt.figure()
plt.plot(v_mph, HFOV * 10**3, linewidth=2)
plt.xlabel('Vehicle speed [mph]')
plt.ylabel('Horizontal Field of View [miliradians]')
plt.title('Horizontal Field of View vs Vehicle Speed')
plt.grid()
plt.show()

# Save data
np.save('HFOV.npy', HFOjson)


