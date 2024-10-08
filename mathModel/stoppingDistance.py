import numpy as np
import matplotlib.pyplot as plt

# Parameters

mu = 0.3 # friction coefficient
g = 9.81 # gravity; in meters/second^2

t = 2 * 0.1 # perception time; in seconds
l = 0.25 # latency; in seconds
a = - mu * g # deceleration; in m/s^2  (esto cambia dependiendo de la friccion)
B = 2 # buffer; in m (security distance)

# Stopping distance (Velocity rates)

v_mph = np.arange(1, 151) # speed in miles/hour
v_kph = v_mph * (1.609 / 1)  # speed in km/hour
v_mtph = v_mph * (1609.34 / 1) # speed in meters/hour
v_mtps = v_mtph * (1 / 60**2) # speed in meters/second

tdist = np.zeros_like(v_mtps)


for v in range(1, 151):
    #  distance traveled
    tdist[v-1] = v_mtps[v-1] * (t + l) - v_mtps[v-1]**2 / (2 * a) + B # in meters

# 1 mile = 1609.34 meters
# Plot
# plt.plot(v_kph, tdist, linewidth=2) 
# plt.xlabel('Speed (km/h)')
# plt.ylabel('Lookahead distance (m)')
# plt.legend()
# plt.show()

# Lookahead distance

Tper = 0.1 # perception time; in seconds
Tact = 0.25 # actuaction time (latency); in seconds
mu = 0.3 # friction coefficient
g = 9.81 # gravity; in meters/second^2

d_offset = 1; # offset distance; in meters

d_per = np.zeros_like(v_mtps)
d_act = np.zeros_like(v_mtps)
d_brake = np.zeros_like(v_mtps)
d_look = np.zeros_like(v_mtps)

for v in range(1, 151):
    # distance perception
    d_per[v-1] = v_mtps[v-1] * (2*Tper) # in meters
    d_act[v-1] = v_mtps[v-1] * (2*Tact) # in meters
    d_brake[v-1] = 1/2 * v_mtps[v-1]**2 / (2*mu*g) # in meters
    d_look[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_brake[v-1] # in meters

# Plot
plt.plot(v_kph, d_look, linewidth=2, label='Stopping Distance [m] vT')
plt.plot(v_kph, tdist, linewidth=2, label='Stopping distance [m] vSPIE')
plt.grid(True)
plt.xlabel('Vehicle speed [kph]')
plt.ylabel('Lookahead distance [m]')
plt.ylim(0, 800)
plt.legend()
plt.tight_layout()
plt.savefig('LookaheadDistanceForStoppingDistance(kph).png')
plt.show()

# Print the values of the stopping distance for a speed of 60 mph
#print('Distance for 60mph in vT: ', d_look[59])
#print('Distance for 60mph in SPIE: ', tdist[59])

# Print the values for the all x and y in the array for t_dist
#for x, y in zip(v_mph, tdist):
    #print('SPIE: Vehicle speed [mph]: ', x, '-- Lookahead distance [mph]: ', y)


# Print the values for the all x and y in the array for d_look
#for x, y in zip(v_mph, d_look):
    #print('vT: Vehicle speed [mph]: ', x, '-- Lookahead distance [mph]:', y)

# Print the values in the txt file in the same directory (math-model)
with open('Lookahead_Distance_For_Stopping.txt','w') as f:
    f.write('SPIE results' + '\n')
    for x, y in zip(v_kph, tdist):
        f.write('SPIE: Vehicle speed [kph]: ' + str(x) + ' -- Lookahead distance [m]: ' + str(y) + '\n')
    f.write('vT results' + '\n')
    for x, y in zip(v_kph, d_look):
        f.write('vT: Vehicle speed [kph]: ' + str(x) + ' -- Lookahead distance [m]: ' + str(y) + '\n')



# Explaining a few things
# 1. I used [v-1] because the range starts at 1 and not 0, so I used v-1 to start at 0, otherwise 
# will occur the error (IndexError: index 100 is out of bounds for axis 0 with size 100)
# 2. I used v_mtps[v-1] but I could plot any of the other speeds for example in the plot is v_mph, but I could use v_kph or v_mtph
# 3. I used v_mph because it is the most common unit for speed in the US



