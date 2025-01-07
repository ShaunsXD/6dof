import numpy as np
import matplotlib.pyplot as plt
import math
import ussa1976
import pandas as pd

import numerical_integration_methods
import flat_earth_eom
import vehicles
from interpolators import fastInterp1


#Part 1: Init Simulation
#Atmospheric data
atmosphere = ussa1976.compute()
#Get essential gravity and atmosphereic data
alt_m = atmosphere["z"].values
rho_kgpm3 = atmosphere["rho"].values
c_mps = atmosphere["cs"].values #Speed of sound
g_mps2 = ussa1976.core.compute_gravity(alt_m)
#amod = atmospheric model
amod = {
    "alt_m": alt_m,
    "rho_kgpm3": rho_kgpm3,
    "c_mps": c_mps,
    "g_mps2": g_mps2}

vmod, params = vehicles.NASACase4()

#Set init conditions
u0_bf_mps = params["u0_bf_mps"]
v0_bf_mps = params["v0_bf_mps"]
w0_bf_mps = params["w0_bf_mps"]
p0_bf_rps = params["p0_bf_rps"]
q0_bf_rps = params["q0_bf_rps"]
r0_bf_rps = params["r0_bf_rps"]
phi0_rad = params["phi0_rad"]
theta0_rad = params["theta0_rad"]
psi0_rad = params["psi0_rad"]
p10_n_m = params["p10_n_m"]
p20_n_m = params["p20_n_m"]
p30_n_m = params["p30_n_m"]


#Assigin iniit condition to a state vector array
x0 = np.array([
    u0_bf_mps, #x-axis body-fixed vel (m/s)
    v0_bf_mps, #y-axis body-fixed vel (m/s)
    w0_bf_mps, #z-axis body-fixed vel (m/s)
    p0_bf_rps, #roll rate (rad/s)
    q0_bf_rps, #pitch rate (rad/s)
    r0_bf_rps, #yaw rate (rad/s)
    phi0_rad, #roll angle (rad)
    theta0_rad, #pitch angle (rad)
    psi0_rad, #yaw angle (rad)
    p10_n_m, #x-axis position (m)
    p20_n_m, #y-axis position (m)
    p30_n_m #z-axis position (m)
])

#Make the init condition array a col vector
nx0 = x0.size

#Set time conditions
t0_s = 0.0
tf_s = 30
h_s = 0.01

#Part 2 Approximate solutions to the governing equations

#Preallocate solution array
t_s = np.arange(t0_s, tf_s + h_s, h_s); nt_s = t_s.size
x = np.empty((nx0, nt_s), dtype=float)

#Assign the init condition x0 to the sol'n array x
x[:, 0] = x0

#Perform forward Euler integration
t_s, x = numerical_integration_methods.rk4(flat_earth_eom.flat_earth_eom, t_s, x, h_s, vmod, amod)

#Post processing Stuff

#Airspeed
true_airspeed_mps = np.zeros((nt_s, 1), dtype=float)
for i, element in enumerate(t_s):
    true_airspeed_mps[i, 0] = math.sqrt(x[0, i]**2 + x[1, i]**2 + x[2, i]**2)

#Altitude, speed of sound, and air density
altitude_m = np.zeros((nt_s, 1), dtype=float)
cs_mps = np.zeros((nt_s, 1), dtype=float)
rho_kgpm3 = np.zeros((nt_s, 1), dtype=float)
for i, element in enumerate(t_s):
    altitude_m[i, 0] = -x[11, i]
    cs_mps[i, 0] = fastInterp1(amod["alt_m"], amod["c_mps"], -x[11, i])
    rho_kgpm3[i, 0] = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], -x[11, i])

#Angle of attack
alpha_rad = np.zeros((nt_s, 1), dtype=float)
for i, element in enumerate(t_s):
    if x[0, i] == 0 and x[2, i] == 0:
        w_over_u = 0
    else:
        w_over_u = x[2, i] / x[0, i]
    alpha_rad[i, 0] = math.atan(w_over_u)

#Angle of side slip
beta_rad = np.zeros((nt_s, 1), dtype=float)
for i, element in enumerate(t_s):
    if x[1, i] == 0 and true_airspeed_mps[i, 0] == 0:
        v_over_VT = 0
    else:
        v_over_VT = x[1, i] / true_airspeed_mps[i, 0]
    beta_rad[i, 0] = math.asin(v_over_VT)

#Mach number
mach = np.zeros((nt_s, 1), dtype=float)
for i, element in enumerate(t_s):
    mach[i, 0] = true_airspeed_mps[i, 0] / cs_mps[i, 0]

print(f"Numerical terminal velocity is {x[0, -1]}: m/s")

#Part 3 Plot Results
# fig, axes = plt.subplots(2, 4, figsize=(10, 6)) #1 row, 2 cols
# fig.set_facecolor("black")
#
# #Axial velocity u^b_CM/n
# axes[0, 0].plot(t_s, x[0, :], color="yellow")
# axes[0, 0].set_xlabel("Time [s]", color="white")
# axes[0, 0].set_ylabel("u [m/s]", color="white")
# axes[0, 0].grid(True)
# axes[0, 0].set_facecolor("black")
# axes[0, 0].tick_params(colors="white")
#
# #Y-axis velocity v^b_CM/n
# axes[0, 1].plot(t_s, x[1, :], color="yellow")
# axes[0, 1].set_xlabel("Time [s]", color="white")
# axes[0, 1].set_ylabel("v [m/s]", color="white")
# axes[0, 1].grid(True)
# axes[0, 1].set_facecolor("black")
# axes[0, 1].tick_params(colors="white")
#
# #Z-axis velocity w^b_CM/n
# axes[0, 2].plot(t_s, x[2, :], color="yellow")
# axes[0, 2].set_xlabel("Time [s]", color="white")
# axes[0, 2].set_ylabel("w [m/s]", color="white")
# axes[0, 2].grid(True)
# axes[0, 2].set_facecolor("black")
# axes[0, 2].tick_params(colors="white")
#
# #Roll, angel, phi
# axes[0, 3].plot(t_s, x[6, :], color="yellow")
# axes[0, 3].set_xlabel("Time [s]", color="white")
# axes[0, 3].set_ylabel("phi [rad]", color="white")
# axes[0, 3].grid(True)
# axes[0, 3].set_facecolor("black")
# axes[0, 3].tick_params(colors="white")
#
# #Roll rate p^b_b/n
# axes[1, 0].plot(t_s, x[3, :], color="yellow")
# axes[1, 0].set_xlabel("Time [s]", color="white")
# axes[1, 0].set_ylabel("p [rad/s]", color="white")
# axes[1, 0].grid(True)
# axes[1, 0].set_facecolor("black")
# axes[1, 0].tick_params(colors="white")
#
# #Pitch rate q^b_b/n
# axes[1, 1].plot(t_s, x[4, :], color="yellow")
# axes[1, 1].set_xlabel("Time [s]", color="white")
# axes[1, 1].set_ylabel("q [rad/s]", color="white")
# axes[1, 1].grid(True)
# axes[1, 1].set_facecolor("black")
# axes[1, 1].tick_params(colors="white")
#
# #Yaw rate r^b_b/n
# axes[1, 2].plot(t_s, x[5, :], color="yellow")
# axes[1, 2].set_xlabel("Time [s]", color="white")
# axes[1, 2].set_ylabel("r [rad/s]", color="white")
# axes[1, 2].grid(True)
# axes[1, 2].set_facecolor("black")
# axes[1, 2].tick_params(colors="white")
#
# #Pitch angel theta
# axes[1, 3].plot(t_s, x[7, :], color="yellow")
# axes[1, 3].set_xlabel("Time [s]", color="white")
# axes[1, 3].set_ylabel("theta [rad]", color="white")
# axes[1, 3].grid(True)
# axes[1, 3].set_facecolor("black")
# axes[1, 3].tick_params(colors="white")
#
# plt.tight_layout()
# plt.show()
#
# #Plotting angle of attack, angle of side slip, and mach number
# fig, axes = plt.subplots(1, 3, figsize=(10, 6))
# fig.set_facecolor("black")
#
# #Angle of attack
# axes[0].plot(t_s, alpha_rad, color="yellow")
# axes[0].set_xlabel("Time [s]", color="white")
# axes[0].set_ylabel("alpha [rad]", color="white")
# axes[0].grid(True)
# axes[0].set_facecolor("black")
# axes[0].tick_params(colors="white")
#
# #Angle of side slip
# axes[1].plot(t_s, beta_rad, color="yellow")
# axes[1].set_xlabel("Time [s]", color="white")
# axes[1].set_ylabel("beta [rad]", color="white")
# axes[1].grid(True)
# axes[1].set_facecolor("black")
# axes[1].tick_params(colors="white")
#
# #Mach number
# axes[2].plot(t_s, mach, color="yellow")
# axes[2].set_xlabel("Time [s]", color="white")
# axes[2].set_ylabel("Mach", color="white")
# axes[2].grid(True)
# axes[2].set_facecolor("black")
# axes[2].tick_params(colors="white")


#Plotting Altitude, Mach Number, Air Density, Pitch Angle, Speed of Sound, True Airspeed, and Pitch Rate
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
fig.set_facecolor("black")

#Altitude
axes[0, 0].plot(t_s, altitude_m, color="yellow")
axes[0, 0].set_xlabel("Time [s]", color="white")
axes[0, 0].set_ylabel("Altitude [m]", color="white")
axes[0, 0].grid(True)
axes[0, 0].set_facecolor("black")
axes[0, 0].tick_params(colors="white")

#Mach Number
axes[0, 1].plot(t_s, mach, color="yellow")
axes[0, 1].set_xlabel("Time [s]", color="white")
axes[0, 1].set_ylabel("Mach", color="white")
axes[0, 1].grid(True)
axes[0, 1].set_facecolor("black")
axes[0, 1].tick_params(colors="white")

#Air Density
axes[0, 2].plot(t_s, rho_kgpm3, color="yellow")
axes[0, 2].set_xlabel("Time [s]", color="white")
axes[0, 2].set_ylabel("Air Density [kg/m^3]", color="white")
axes[0, 2].grid(True)
axes[0, 2].set_facecolor("black")
axes[0, 2].tick_params(colors="white")

#Pitch Angle
axes[0, 3].plot(t_s, x[7, :], color="yellow")
axes[0, 3].set_xlabel("Time [s]", color="white")
axes[0, 3].set_ylabel("Pitch Angle [rad]", color="white")
axes[0, 3].grid(True)
axes[0, 3].set_facecolor("black")
axes[0, 3].tick_params(colors="white")

#Speed of Sound
axes[1, 0].plot(t_s, cs_mps, color="yellow")
axes[1, 0].set_xlabel("Time [s]", color="white")
axes[1, 0].set_ylabel("Speed of Sound [m/s]", color="white")
axes[1, 0].grid(True)
axes[1, 0].set_facecolor("black")
axes[1, 0].tick_params(colors="white")

#True Airspeed
axes[1, 1].plot(t_s, true_airspeed_mps, color="yellow")
axes[1, 1].set_xlabel("Time [s]", color="white")
axes[1, 1].set_ylabel("True Airspeed [m/s]", color="white")
axes[1, 1].grid(True)
axes[1, 1].set_facecolor("black")
axes[1, 1].tick_params(colors="white")

#Pitch Rate
axes[1, 2].plot(t_s, x[4, :], color="yellow")
axes[1, 2].set_xlabel("Time [s]", color="white")
axes[1, 2].set_ylabel("Pitch Rate [rad/s]", color="white")
axes[1, 2].grid(True)
axes[1, 2].set_facecolor("black")
axes[1, 2].tick_params(colors="white")

#Angle of Attack
axes[1, 3].plot(t_s, alpha_rad, color="yellow")
axes[1, 3].set_xlabel("Time [s]", color="white")
axes[1, 3].set_ylabel("Angle of Attack [rad]", color="white")
axes[1, 3].grid(True)
axes[1, 3].set_facecolor("black")
axes[1, 3].tick_params(colors="white")




plt.tight_layout()
plt.show()

data = {
    "Altitude [m]": altitude_m.flatten(),
    "Mach Number": mach.flatten(),
    "Air Density [kg/m^3]": rho_kgpm3.flatten(),
    "Pitch Angle [rad]": x[7, :],
    "Speed of Sound [m/s]": cs_mps.flatten(),
    "True Airspeed [m/s]": true_airspeed_mps.flatten(),
    "Pitch Rate [rad/s]": x[4, :],
    "Roll Angle [rad]": x[6, :],
    "Roll Rate [rad/s]": x[3, :],
    "Yaw Angle [rad]": x[8, :],
    "Yaw Rate [rad/s]": x[5, :],
    "Angle of Attack [rad]": alpha_rad.flatten(),
}

df = pd.DataFrame(data)
df.to_csv("verify1.csv", index=False)
