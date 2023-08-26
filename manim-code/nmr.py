from manim import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv('regression.csv', header=None)

# # Get data from each column and multiply by 10
# x_d = [10 * element for element in df[0]]
# y_d  = [10 * element for element in df[1]]

# Fs = 2000 #sampling freq
# tstep = 1 / Fs #sampling time interval 
# f0 = 100 #signal freq

# N = int(10 * Fs/f0) # num samples 

# t = np.linspace(0, (N-1)*tstep, N)
# fstep = Fs / N
# f = np.linspace(0, (N-1)*fstep, N)

# y = -2 * np.sin(2 * np.pi * f0 * t)

# X = np.fft.fft(y)
# X_mag = np.abs(X) / N

# f_plot = f[0:int(N/2+1)]
# x_mag_plot = 2 * X_mag[0:int(N/2+1)]
# x_mag_plot[0] = x_mag_plot[0] / 2 #DC omponent does not need to multiply by 2 


# fig, [ax1, ax2] = plt.subplots(nrows=2, ncols= 1)
# ax1.plot(t,y,'.-')
# ax2.plot(f_plot,x_mag_plot, '.-')
# plt.show()

######### Synthetic Signal ##########

M0 = 1
T2 = 0.15
f = 200
phi = 0
dt = 0.001 # time step size, e.g. 1 ms
t_pulse = np.arange(0, 1, dt) # 1 second long pulse
t = np.arange(0, 100*dt, dt) # time array for first 1000 points

# Single pulse FID signal equation
S_pulse = M0 * np.exp(-t_pulse/T2) * np.cos(2 * np.pi * f * t_pulse + phi)

# First 1000 points
S_1000 = M0 * np.exp(-t/T2) * np.cos(2 * np.pi * f * t + phi)

# Plot first 1000 points
plt.figure(figsize=(10,6))
plt.plot(t, S_1000)
plt.title('First 1000 points of FID signal')
plt.xlabel('Time (s)')
plt.ylabel('Signal amplitude')
plt.grid(True)
plt.show()

# Fast Fourier Transform
fft_result = np.fft.fft(S_1000)

# Generate frequency axis for the FFT results
n = S_1000.size
freq = np.fft.fftfreq(n, d=dt)

# Plot FFT
plt.figure(figsize=(10,6))
plt.plot(np.abs(freq), np.abs(fft_result))  # We only care about absolute value for spectrum
plt.title('FFT of first 1000 points of FID signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT amplitude')
plt.grid(True)
plt.show()

# Generate dataset by repeating the single pulse
n_pulses = 5000
dataset = np.tile(S_pulse, n_pulses)

# dataset now contains 5000 pulses, each of 1 second duration with 1 ms time step
print('Size of dataset:', len(dataset))

#############################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for the first signal
# M0_1, T2_1, f_1, phi_1 = 1.0, 2.0, 1.0, 0.0
# # Parameters for the second signal
# M0_2, T2_2, f_2, phi_2 = 1.0, 2.0, 1.0, 0.0

# # Time vector
# t = np.linspace(0, 10, 1000)

# # Compute the signals
# S1 = M0_1 * np.exp(-t/T2_1 - 1j * (2 * np.pi * f_1 * t + phi_1))
# S2 = M0_2 * np.exp(-t/T2_2 - 1j * (2 * np.pi * f_2 * t + phi_2))

# # Combined signal
# S = S1 + S2

# # Plot the signals
# plt.figure(figsize=(8, 8))
# plt.plot(S.real, S.imag)
# plt.xlabel('Real')
# plt.ylabel('Imaginary')
# plt.title('Combined NMR FID signal')
# plt.grid(True)
# plt.show()

#############################################################################################

# from mpl_toolkits.mplot3d import Axes3D

# # Parameters for the first signal
# M0_1, T2_1, f_1, phi_1 = 1.0, 2.0, 1.0, 0.0
# # Parameters for the second signal
# M0_2, T2_2, f_2, phi_2 = 1.0, 2.0, 1.0, 0.0

# # Time vector
# t = np.linspace(0, 10, 1000)

# # Compute the signals
# S1 = M0_1 * np.exp(-t/T2_1 - 1j * (2 * np.pi * f_1 * t + phi_1))
# S2 = M0_2 * np.exp(-t/T2_2 - 1j * (2 * np.pi * f_2 * t + phi_2))

# # Combined signal
# S = S1 + S2

# # Compute polar coordinates
# r = np.abs(S)  # length of the spin vector
# phi = np.angle(S)  # argument of the signal
# theta = np.linspace(np.pi, np.pi/2, len(t))  # from xy-plane to z-axis and back

# # Convert to Cartesian coordinates
# x = r * np.sin(theta) * np.cos(phi)
# y = r * np.sin(theta) * np.sin(phi)
# z = r * np.cos(theta)

# # Create the figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the 3D trajectory
# ax.plot3D(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D NMR FID signal')

# plt.show()