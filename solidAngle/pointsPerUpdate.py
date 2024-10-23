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

# Compute VFOV (Vertical Field of View)
hc = 2 # height of camera (2 m)
thetaSlope = np.deg2rad(15) # maximum angle of slope (15 degrees)
thetaMin = np.arctan(hc / lookahead_dist_stop) # angle below horizon as determined by stopping distance
thetaMax = np.arctan(hc / d_offset) # angle below horizon as determined by baseline B (length of car)

VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax) # Vertical Field of View  # Pregunta aqui, como puso el VFOV

# VFOV = np.kron(np.deg2rad(25), np.ones(100)) # Vertical Field of View

# Plot FOV
plt.figure()
plt.plot(v_kph, HFOV * 10**3, linewidth=2, label= 'HAOV [miliradians]')
plt.plot(v_kph, VFOV * 10**3, linewidth=2, label= 'VAOV [miliradians]')
plt.xlabel('Vehicle speed [kph]')
plt.ylabel('Angle of View (AOV) [miliradians]')
plt.title('Angle of View (AOV) vs Vehicle Speed')
plt.grid()
plt.legend()
plt.savefig('HFOV_VFOV_Miliradians.png', dpi=300)
plt.show()

# Plot FOV 2 
plt.figure()
plt.plot(v_kph, np.rad2deg(HFOV), linewidth=2, label= 'HAOV [degrees]')
plt.plot(v_kph, np.rad2deg(VFOV), linewidth=2, label= 'VAOV [degrees]')
plt.xlabel('Vehicle speed [kph]')
plt.ylabel('Angle of View (AOV) [degrees]')
plt.title('Angle of View (AOV) vs Vehicle Speed')
plt.grid()
plt.legend()
plt.savefig('HAOV_VAOV_Degrees.png', dpi=300)
plt.show()

# --------------- Compute IFOV (Instantaneous Field of View) ----------------
# Parameters (esto hace que el punto de inicio sea diferente para cada obstaculo)
hp = 0.1 # Positive obstacle height (in meters)
wn = 0.95 # Negative obstacle width (in meters)

IFOVp = np.arctan(hc / lookahead_dist_stop) - np.arctan((hc-hp) / lookahead_dist_stop) # Positive IFOV
IFOVn = np.arctan(hc / lookahead_dist_stop) - np.arctan(hc / (lookahead_dist_stop + wn)) # Negative IFOV

# Plot IFOV positive and negative milidiarands
plt.figure()
plt.plot(v_kph, (IFOVp * 10**3), linewidth=2, label= 'IFOV Positive [miliradians]')
plt.plot(v_kph, (IFOVn * 10**3), linewidth=2, label= 'IFOV Negative [miliradians]')
plt.xlabel('Vehicle speed [kph]')
plt.ylabel('Instantaneous Field of View [miliradians]')
plt.title('Instantaneous Field of View vs Vehicle Speed')
plt.grid()
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig('IFOV_Positive_Negative.png', dpi=300)
plt.show()

# Plot IFOV positive and negative degrees
plt.figure()
plt.plot(v_kph, np.rad2deg(IFOVp), linewidth=2, label= 'IFOV Positive [degrees]')
plt.plot(v_kph, np.rad2deg(IFOVn), linewidth=2, label= 'IFOV Negative [degrees]')
plt.xlabel('Vehicle speed [kph]')
plt.ylabel('Instantaneous Field of View [degrees]')
plt.title('Instantaneous Field of View vs Vehicle Speed')
plt.grid()
plt.legend()
plt.xscale('log')
plt.yscale('log')
#plt.savefig('IFOV_Positive_Negative_Degrees.png', dpi=300)
plt.show()

"""
# Calculate the points per update (Points per Update)
ppuPh = HFOV / IFOVp
ppuPv = VFOV / IFOVp
ppuNh = HFOV / IFOVn
ppuNv = VFOV / IFOVn


# Plot points per update for positive obstacles
plt.figure()
plt.plot(v_kph, ppuPh, linewidth=2, label='HFOV / IFOVp')
plt.plot(v_kph, ppuPv, linewidth=2, label='VFOV / IFOVp')
plt.xlabel('Vehicle speed [mph]')
plt.ylabel('Points per update')
plt.grid(True)
plt.title('For positive obstacles')
plt.legend(loc='upper left')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('ppuPh_Positives.png', dpi=300)
plt.show()

# Plot points per update for negative obstacles
plt.figure()
plt.plot(v_kph, ppuNh, linewidth=2, label='HFOV / IFOVn')
plt.plot(v_kph, ppuNv, linewidth=2, label='VFOV / IFOVn')
plt.xlabel('Vehicle speed [mph]')
plt.ylabel('Points per update')
plt.grid(True)
plt.title('For negative obstacles')
plt.legend(loc='upper left')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('ppuNh_Negatives.png', dpi=300)
plt.show()
"""