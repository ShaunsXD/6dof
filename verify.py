import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# #Save all 7 of those parameters to a .csv file by making them a dataframe
# data = {
#     "Altitude [m]": altitude_m.flatten(),
#     "Mach Number": mach.flatten(),
#     "Air Density [kg/m^3]": rho_kgpm3.flatten(),
#     "Pitch Angle [rad]": x[7, :],
#     "Speed of Sound [m/s]": cs_mps.flatten(),
#     "True Airspeed [m/s]": true_airspeed_mps.flatten(),
#     "Pitch Rate [rad/s]": x[4, :]
# }
#
# df = pd.DataFrame(data)
# df.to_csv("NASA_Atmo_Case_1.csv", index=False)


#data_NASA = pd.read_csv('Atmos_01_DroppedSphere/Atmos_01_sim_02.csv')
#data_NASA = pd.read_csv('Atmos_02_TumblingBrickNoDamping/Atmos_02_sim_02.csv')
#data_NASA = pd.read_csv('Atmos_03_TumblingBrickDamping/Atmos_03_sim_02.csv')
data_NASA = pd.read_csv('Atmos_04_DroppedSphereRoundNonRotation/Atmos_04_sim_02.csv')



data_verify = pd.read_csv("verify1.csv")

nasa_alt = data_NASA['altitudeMsl_ft']
nasa_mach = data_NASA['mach']
nasa_air_density = data_NASA['airDensity_slug_ft3'] * 515.379
nasa_pitch = data_NASA['eulerAngle_deg_Pitch'] * np.pi / 180
nasa_cs = data_NASA['speedOfSound_ft_s'] * 0.3048
nasa_true_airspeed = data_NASA['trueAirspeed_nmi_h'] * 0.514444
nasa_pitch_rate = data_NASA['bodyAngularRateWrtEi_deg_s_Pitch'] * np.pi / 180
nasa_pitch_angle = data_NASA["eulerAngle_deg_Pitch"] * np.pi/180
nasa_roll_angle = data_NASA["eulerAngle_deg_Roll"] * np.pi/180
nasa_yaw_angle = data_NASA["eulerAngle_deg_Yaw"] * np.pi/180
nasa_yaw_rate = data_NASA["bodyAngularRateWrtEi_deg_s_Yaw"] * np.pi/180

verify_alt = data_verify['Altitude [m]']
verify_mach = data_verify['Mach Number']
verify_air_density = data_verify['Air Density [kg/m^3]']
verify_pitch = data_verify['Pitch Angle [rad]']
verify_cs = data_verify['Speed of Sound [m/s]']
verify_true_airspeed = data_verify['True Airspeed [m/s]']
verify_pitch_rate = data_verify['Pitch Rate [rad/s]']
verify_pitch_angle = data_verify['Pitch Angle [rad]']
verify_roll_angle = data_verify['Roll Angle [rad]']
verify_yaw_angle = data_verify['Yaw Angle [rad]']
verify_yaw_rate = data_verify['Yaw Rate [rad/s]']

#Mod verify_yaw_angle by 2pi
verify_yaw_angle = verify_yaw_angle % (2*np.pi)
#Make verify_yaw_angle negative if it is greater than pi
verify_yaw_angle = np.where(verify_yaw_angle > np.pi, verify_yaw_angle - 2*np.pi, verify_yaw_angle)



#Get time array
#Set time conditions
t0_s = 0.0
tf_s = 30
h_s = 0.01
time = np.arange(t0_s, tf_s, h_s)

plt.figure()
plt.plot(np.arange(t0_s, tf_s + h_s, h_s), verify_yaw_angle, label="Altitude [ft] (verify)", linestyle='dashed')
plt.plot(data_NASA['time'], nasa_yaw_angle, label="Altitude [ft] (NASA)")
plt.plot(np.arange(t0_s, tf_s + h_s, h_s), verify_pitch_angle, label="Pitch Angle [rad] (verify)", linestyle='dashed')
plt.plot(np.arange(t0_s, tf_s + h_s, h_s), verify_roll_angle, label="Roll Angle [rad] (verify)", linestyle='dashed')
plt.plot(data_NASA['time'], nasa_pitch_angle, label="Pitch Angle [rad] (NASA)")
plt.plot(data_NASA['time'], nasa_roll_angle, label="Roll Angle [rad] (NASA)")
plt.show()

plt.figure()
plt.plot(np.arange(t0_s, tf_s + h_s, h_s), verify_alt * 3.281, label="Altitude [m] (verify)", linestyle='dashed')
plt.plot(data_NASA['time'], nasa_alt, label="Altitude [m] (NASA)")

plt.grid(True)
plt.legend()
plt.show()