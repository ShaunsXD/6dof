import math
import numpy as np
import vehicles
from interpolators import fastInterp1

def flat_earth_eom(t, x, vmod, amod):
    """
    Allows for numerical approximations of solutions of the governing equations for an aircraft

    Naming convention follows Ben Dickinson's video guide <var_name>_<coord system>_<units>

    :param t: time [s], scalar
    :param x: state vector at time t, numpy array.
        x[0] = u_b_mps, axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[1] = v_b_mps, lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[2] = w_b_mps, vertical velocity of CM wrt inertial CS resolved in aircraft body fixed CS
        x[3] = p_b_rps, roll angular velocity of body fixed CS wrt inertial CS
        x[4] = q_b_rps, pitch angular velocity of body fixed CS wrt inertial CS
        x[5] = r_b_rps, yaw angular velocity of body fixed CS wrt inertial CS
        x[6] = phi_rad, roll angle
        x[7] = theta_rad, pitch angle
        x[8] = psi_rad, yaw angle
        x[9] = p1_n_m, x-axis position of aircraft resolved in NED CS
        x[10] = p2_n_m, y-axis position of aircraft resolved in NED CS
        x[11] = p3_n_m, z-axis position of aircraft resolved in NED CS
    :param vmod: vehicle model data stored as a dictionary containing various parameters
    :param amod: atmospheric model data stored as a dictionary containing various parameters
    :return: time derivative of each state in x (RHS of governing equations)
    """
    #Preallocate LHS of equations
    dx = np.empty((12,), dtype=float)

    #Extract state variables
    u_b_mps = x[0]
    v_b_mps = x[1]
    w_b_mps = x[2]
    p_b_rps = x[3]
    q_b_rps = x[4]
    r_b_rps = x[5]
    phi_rad = x[6]
    theta_rad = x[7]
    psi_rad = x[8]
    p1_n_m = x[9]
    p2_n_m = x[10]
    p3_n_m = x[11]

    #Precompute trig operations on Euler angles
    c_phi = math.cos(phi_rad)
    c_theta = math.cos(theta_rad)
    c_psi = math.cos(psi_rad)
    s_phi = math.sin(phi_rad)
    s_theta = math.sin(theta_rad)
    s_psi = math.sin(psi_rad)
    t_theta = math.tan(theta_rad)


    #Get vehicle mass and moments of inertia
    m_kg = vmod['m_kg']
    Jxz_b_kgm2 = vmod['Jxz_b_kgm2']
    Jxx_b_kgm2 = vmod['Jxx_b_kgm2']
    Jyy_b_kgm2 = vmod['Jyy_b_kgm2']
    Jzz_b_kgm2 = vmod['Jzz_b_kgm2']
    h_m = -p3_n_m #Get current altitude

    #Get vehicle damping coefficients
    Clp = vmod['Clp']
    Clr = vmod['Clr']
    Cmq = vmod['Cmq']
    Cnp = vmod['Cnp']
    Cnr = vmod['Cnr']
    b_m = vmod['b_m']
    c_m = vmod['c_m']
    A_ref_m2 = vmod['Aref_m2']

    #US std atmosphere
    rho_interp_kgpm3 = fastInterp1(amod['alt_m'], amod['rho_kgpm3'], h_m)
    #rho_interp_kgpm3 = 1.20

    c_interp_mp2 = fastInterp1(amod['alt_m'], amod['c_mps'], h_m)

    #Air data calculation
    true_airspeed_mps = math.sqrt(u_b_mps**2 + v_b_mps**2 + w_b_mps**2) #VT

    qbar_kgpm2 = 0.5 * rho_interp_kgpm3 * true_airspeed_mps**2

    #Angle of attack and angle of side slip calculations
    if u_b_mps == 0 and w_b_mps == 0:
        w_over_u = 0
    else:
        w_over_u = w_b_mps / u_b_mps

    if true_airspeed_mps == 0 and v_b_mps == 0:
        v_over_VT = 0
    else:
        v_over_VT = v_b_mps / true_airspeed_mps
    alpha_rad = math.atan(w_over_u)
    beta_rad = math.asin(v_over_VT)
    s_alpha = math.sin(alpha_rad)
    c_alpha = math.cos(alpha_rad)
    s_beta = math.sin(beta_rad)
    c_beta = math.cos(beta_rad)


    #Gravity that acts normal to earth tangent CS
    #gz_interp_n_mps2 = 9.81
    gz_interp_n_mps2 = fastInterp1(amod['alt_m'], amod['g_mps2'], h_m)

    #Resolve DCM in body coordinate system
    gx_b_mps2 = -math.sin(theta_rad) * gz_interp_n_mps2
    gy_b_mps2 = math.sin(phi_rad) * math.cos(theta_rad) * gz_interp_n_mps2
    gz_b_mps2 = math.cos(phi_rad) * math.cos(theta_rad) * gz_interp_n_mps2

    #Aerodynamic forces and external forces(Coming soon)
    drag_kgmps2 = vmod["CD_approx"] * qbar_kgpm2 * vmod["Aref_m2"]
    side_kgpms2 = 0
    lift_kgmps2 = 0
    #print(drag_kgmps2, qbar_kgpm2, true_airspeed_mps, rho_interp_kgpm3)
    #Resolve drag with DCM in body coordinate system
    #F = -DCM.T * [drag, side, lift]

    DCM = np.array([[math.cos(alpha_rad)*math.cos(beta_rad), math.sin(beta_rad), math.sin(alpha_rad)*math.cos(beta_rad)],
                    [-math.cos(alpha_rad)*math.sin(beta_rad), math.cos(beta_rad), math.sin(alpha_rad)*math.sin(beta_rad)],
                    [-math.sin(alpha_rad), 0, math.cos(alpha_rad)]])
    F = np.array([drag_kgmps2, side_kgpms2, lift_kgmps2])
    drag_force = np.dot(-DCM.T, F)
    Fx_b_kgmps2 = drag_force[0]
    Fy_b_kgmps2 = drag_force[1]
    Fz_b_kgmps2 = drag_force[2]
    # Fx_b_kgmps2 = -(drag_kgmps2*c_alpha*c_beta - side_kgpms2*c_alpha*s_beta - lift_kgmps2*s_alpha)
    # Fy_b_kgmps2 = -(drag_kgmps2*s_beta + side_kgpms2*c_beta)
    # Fz_b_kgmps2 = -(lift_kgmps2*c_alpha + drag_kgmps2*s_alpha*c_beta + side_kgpms2*s_alpha*s_beta)
    #print(Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2)
    #Allow constant thrust
    # Fx_b_kgmps2 -= 5000
    # Fy_b_kgmps2 -= 2000
    # Fz_b_kgmps2 += 1000

    #External moments
    l_b_kgm2ps2 = vehicles.Cl_brick(Clp, Clr, p_b_rps, r_b_rps, b_m, true_airspeed_mps)*qbar_kgpm2*A_ref_m2*b_m
    m_b_kgm2ps2 = vehicles.Cm_brick(Cmq, q_b_rps, c_m, true_airspeed_mps)*qbar_kgpm2*A_ref_m2*c_m
    n_b_kgm2ps2 = vehicles.Cn_brick(Cnp, Cnr, p_b_rps, r_b_rps, b_m, true_airspeed_mps)*qbar_kgpm2*A_ref_m2*b_m

    #Denominator in roll and yaw rate equations
    Den = Jxx_b_kgm2*Jzz_b_kgm2 - Jxz_b_kgm2**2

    #x-axis (roll axis) velocity equation
    #State: u_b_mps
    dx[0] = 1 / m_kg * Fx_b_kgmps2 + gx_b_mps2 - \
            w_b_mps * q_b_rps + v_b_mps * r_b_rps

    #y-axis (pitch axis) velocity equation
    #State: v_b_mps
    dx[1] = 1 / m_kg * Fy_b_kgmps2 + gy_b_mps2 - \
            u_b_mps * r_b_rps + w_b_mps * p_b_rps

    #z-axis (yaw axis) velocity equation
    #State: w_b_mps
    dx[2] = 1 / m_kg * Fz_b_kgmps2 + gz_b_mps2 - \
            v_b_mps * p_b_rps + u_b_mps * q_b_rps

    #Roll equation
    #State: p_b_rps

    #debug
    # if p_b_rps > 1:
    #     print()
    # test = (Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2) * p_b_rps * q_b_rps -
    #          (Jzz_b_kgm2 * (Jzz_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2 ** 2) + q_b_rps * r_b_rps +
    #          Jzz_b_kgm2 * l_b_kgm2ps2 +
    #          Jxz_b_kgm2 * n_b_kgm2ps2) / Den
    # a = (Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2) * p_b_rps * q_b_rps)
    # b = (Jzz_b_kgm2 * (Jzz_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2 ** 2) + q_b_rps * r_b_rps
    # c = Jzz_b_kgm2 * l_b_kgm2ps2
    # d = Jxz_b_kgm2 * n_b_kgm2ps2
    # e = Den

    #Testing by solving matrix equation directly?
    #Inertia matrix
    I = np.array([[Jxx_b_kgm2, 0, 0], [0, Jyy_b_kgm2, 0], [0, 0, Jzz_b_kgm2]])
    #Angular velocity vector
    omega_b_i = np.array([p_b_rps, q_b_rps, r_b_rps])
    #External moment vector
    M_b = np.array([l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2])
    omega_dot_cross = np.cross(omega_b_i, np.dot(I, omega_b_i))
    omega_dot_b_i = np.dot(np.linalg.inv(I), M_b - omega_dot_cross)
    dx[3] = omega_dot_b_i[0]
    dx[4] = omega_dot_b_i[1]
    dx[5] = omega_dot_b_i[2]


    #Old rotational equations, bug with singuarlity on denominator? Not sure, ended up converting to solving
    #matrix equation directly. Fixed bug with that? Not sure why when they should be equivalent

    # dx[3] = (Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2) * p_b_rps * q_b_rps -
    #          (Jzz_b_kgm2 * (Jzz_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2 ** 2) + q_b_rps * r_b_rps +
    #          Jzz_b_kgm2 * l_b_kgm2ps2 +
    #          Jxz_b_kgm2 * n_b_kgm2ps2) / Den
    #
    # #Pitch equation
    # #State: q_b_rps
    # dx[4] = ((Jzz_b_kgm2 - Jxx_b_kgm2) * p_b_rps * r_b_rps -
    #          Jxz_b_kgm2 * (p_b_rps ** 2 - r_b_rps ** 2) +
    #          m_b_kgm2ps2) / Jyy_b_kgm2
    #
    # #Yaw equation
    # #State: r_b_rps
    # dx[5] = ((Jxx_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2 ** 2) * p_b_rps * q_b_rps +
    #          Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2) * q_b_rps * r_b_rps +
    #          Jxz_b_kgm2 * l_b_kgm2ps2 +
    #          Jxz_b_kgm2 * n_b_kgm2ps2) / Den

    #debug
    # if phi_rad > 0.2:
    #     print()
    # test = p_b_rps + math.sin(phi_rad) * math.tan(theta_rad) * q_b_rps + math.cos(phi_rad) * math.tan(theta_rad) * r_b_rps
    # a = math.sin(phi_rad)
    # b = math.tan(theta_rad)
    # c = math.cos(phi_rad)

    rot = np.array([[1, math.sin(phi_rad)*math.tan(theta_rad), math.cos(phi_rad)*math.tan(theta_rad)],
                    [0, math.cos(phi_rad), -math.sin(phi_rad)],
                    [0, math.sin(phi_rad)/math.cos(theta_rad), math.cos(phi_rad)/math.cos(theta_rad)]])
    body_angular_rates = np.dot(rot, np.array([p_b_rps, q_b_rps, r_b_rps]))
    dx[6] = body_angular_rates[0]
    dx[7] = body_angular_rates[1]
    dx[8] = body_angular_rates[2]


    # #Kinematic equations (solved for each axis but not used) (just use matrix eqn above instead?)
    # dx[6] = p_b_rps + math.sin(phi_rad) * math.tan(theta_rad) * q_b_rps + \
    #         math.cos(phi_rad) * math.tan(theta_rad) * r_b_rps
    #
    # dx[7] = math.cos(phi_rad) * q_b_rps - \
    #         math.sin(phi_rad) * r_b_rps
    #
    # dx[8] = math.sin(phi_rad)/math.cos(theta_rad)*q_b_rps+\
    #         math.cos(phi_rad)/math.cos(theta_rad)*r_b_rps


    #Position (Navigation) equations
    dx[9] = c_theta*c_psi*u_b_mps + \
            (-c_phi*s_phi+s_psi*s_theta*c_psi)*v_b_mps + \
            (s_phi*s_psi + c_phi*s_theta*c_psi)*w_b_mps
    dx[10] = c_theta*s_psi*u_b_mps + \
             (c_phi*c_psi+s_phi*s_theta*s_psi)*v_b_mps + \
             (-s_phi*c_psi + c_phi*s_theta*s_psi)*w_b_mps

    # print(f"u velocity: {u_b_mps} v velocity: {v_b_mps} w velocity: {w_b_mps}")
    # print(f"p angular velocity: {p_b_rps} q angular velocity: {q_b_rps} r angular velocity: {r_b_rps}")
    # print(f"phi: {phi_rad * 180/math.pi } theta: {theta_rad * 180/math.pi} psi: {psi_rad * 180/math.pi}")
    # print()
    dx[11] = -s_theta*u_b_mps + \
              s_phi*c_theta*v_b_mps + \
              c_phi*c_theta*w_b_mps

    return dx




