"""
Simple Flight Dynamics Model (FDM) example that makes the altitude increase and the plane roll in the air.
"""
import time
from flightgear_python.fg_if import FDMConnection
import pandas as pd
i = 0

def fdm_callback(fdm_data, event_pipe):
    global i

    data_verify = pd.read_csv("verify1.csv")
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

    fdm_data.alt_m = verify_alt[i]
    fdm_data.phi_rad = verify_roll_angle[i]
    fdm_data.theta_rad = verify_pitch_angle[i]
    fdm_data.psi_rad = verify_yaw_angle[i]

    print(verify_alt[i])
    i = i + 10 if i < 3000 else 0
    return fdm_data  # return the whole structure

"""
Start FlightGear with `--native-fdm=socket,out,30,localhost,5501,udp --native-fdm=socket,in,30,localhost,5502,udp`
(you probably also want `--fdm=null` and `--max-fps=30` to stop the simulation fighting with
these external commands)
"""
if __name__ == '__main__':  # NOTE: This is REQUIRED on Windows!
    fdm_conn = FDMConnection()
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    fdm_conn.start()  # Start the FDM RX/TX loop

    while True:
        # could also do `fdm_conn.event_pipe.parent_send` so you just need to pass around `fdm_conn`  # send tuple
        time.sleep(0.0005)