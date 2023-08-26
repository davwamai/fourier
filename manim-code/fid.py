import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
M0 = 1
T2 = 0.1
f = 10
phi = 0
T1 = 0.15
t = np.linspace(0, 1, 1000)  # Time array

# Synthetic NMR FID signals
S_x = M0 * np.exp(-t/T2) * np.cos(2 * np.pi * f * t + phi)
S_y = M0 * np.exp(-t/T2) * np.cos(2 * np.pi * f * t + phi + np.pi/2)  # y is 90 degree phase-shifted
S_z = M0 * (1 - np.exp(-t/T1))  # z is increasing due to T1 relaxation

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(S_y, S_x, S_z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()