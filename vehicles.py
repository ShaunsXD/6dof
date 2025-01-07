import math


def Cm_brick(Cmq, q_b_rps, c_m, true_airspeed_mps):
    return Cmq*q_b_rps*c_m/(2*true_airspeed_mps)


def Cl_brick(Clp, Clr, p_b_rps, r_b_rps, b_m, true_airspeed_mps):
    return Clp*p_b_rps*b_m/(2*true_airspeed_mps) + Clr*r_b_rps*b_m/(2*true_airspeed_mps)


def Cn_brick(Cnp, Cnr, p_b_rps, r_b_rps, b_m, true_airspeed_mps):
    return Cnp*p_b_rps*b_m/(2*true_airspeed_mps) + Cnr*r_b_rps*b_m/(2*true_airspeed_mps)


def bulletVehicle():
# Define a vehicle, vmod = vehicle data
# 50 Calliber bullet aproximation that I just googled sorta
    r_bowl_m = 0.0579
    m_bowl_kg = 2
    J_bowl_kgm2 = 0.4 * m_bowl_kg * r_bowl_m**2
    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": J_bowl_kgm2,
            "Jyy_b_kgm2": J_bowl_kgm2,
            "Jzz_b_kgm2": J_bowl_kgm2,
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0, #Temp for terminal because I don't know it yet
            "CD_approx": 0.5,
            "Aref_m2": math.pi * r_bowl_m**2,
            "Clp": 0,
            "Clr": 0,
            "Cmq": 0,
            "Cnp": 0,
            "Cnr": 0,
            "b_m": 0,
            "c_m": 0}

    return vmod


def bowlingBallVehicle():
    # Bowling ball
    r_bowl_m = 0.10795
    m_bowl_kg = 5.1
    J_bowl_kgm2 = 0.4 * m_bowl_kg * r_bowl_m**2
    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": J_bowl_kgm2,
            "Jyy_b_kgm2": J_bowl_kgm2,
            "Jzz_b_kgm2": J_bowl_kgm2,
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0, #Temp for terminal because I don't know it yet
            "CD_approx": 0.45,
            "Aref_m2": math.pi * r_bowl_m**2,
            "Clp": 0,
            "Clr": 0,
            "Cmq": 0,
            "Cnp": 0,
            "Cnr": 0,
            "b_m": 0,
            "c_m": 0}
    return vmod


def NASACase1():
# NASA Atmo Case 1
    r_bowl_m = 1
    m_bowl_kg = 14.5939
    J_bowl_kgm2 = 0.4 * m_bowl_kg * r_bowl_m**2
    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": 4.8809446628,
            "Jyy_b_kgm2": 4.8809446628,
            "Jzz_b_kgm2": 4.8809446628,
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0, #Temp for terminal because I don't know it yet
            "CD_approx": 0.0,
            "Aref_m2": math.pi * r_bowl_m**2,
            "Clp": 0,
            "Clr": 0,
            "Cmq": 0,
            "Cnp": 0,
            "Cnr": 0,
            "b_m": 0,
            "c_m": 0}

    u0_bf_mps = 1e-10
    v0_bf_mps = 0
    w0_bf_mps = 0
    p0_bf_rps = 0 * math.pi / 180
    q0_bf_rps = 0 * math.pi / 180
    r0_bf_rps = 0 * math.pi / 180
    phi0_rad = 0 * math.pi / 180
    theta0_rad = 0 * math.pi / 180
    psi0_rad = 0.0
    p10_n_m = 0.0
    p20_n_m = 0.0
    p30_n_m = -30000 / 3.281
    params = {
        "u0_bf_mps": u0_bf_mps,
        "v0_bf_mps": v0_bf_mps,
        "w0_bf_mps": w0_bf_mps,
        "p0_bf_rps": p0_bf_rps,
        "q0_bf_rps": q0_bf_rps,
        "r0_bf_rps": r0_bf_rps,
        "phi0_rad": phi0_rad,
        "theta0_rad": theta0_rad,
        "psi0_rad": psi0_rad,
        "p10_n_m": p10_n_m,
        "p20_n_m": p20_n_m,
        "p30_n_m": p30_n_m
    }

    return vmod, params


def NASACase2():
    # NASA Atmo Case 2 (Brick)
    r_bowl_m = 1
    m_bowl_kg = 2.2679619056149
    inch_to_m = 0.0254
    ft2m = 0.304878
    l = 8 * inch_to_m
    w = 4 * inch_to_m
    h = 2.25 * inch_to_m
    a = m_bowl_kg / 12 * (w**2 + h**2)
    b_m = 0.33333*ft2m
    c_m = 0.66667*ft2m
    #Unphysical constants for moments but just for test case purposes
    Clp = 0.0 #Roll damping from roll rate
    Clr = 0.0 #Roll damping from yaw rate
    Cmq = 0.0 #Pitch damping from pitch rate
    Cnp = 0.0 #Yaw damping from roll rate
    Cnr = 0.0 #Yaw damping from yaw rate

    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": m_bowl_kg / 12 * (w**2 + h**2),
            "Jyy_b_kgm2": m_bowl_kg / 12 * (l**2 + h**2),
            "Jzz_b_kgm2": m_bowl_kg / 12 * (l**2 + w**2),
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0, #Temp for terminal because I don't know it yet
            "CD_approx": 0.0,
            "Aref_m2": l*w*h,
            "Clp": Clp,
            "Clr": Clr,
            "Cmq": Cmq,
            "Cnp": Cnp,
            "Cnr": Cnr,
            "b_m": b_m,
            "c_m": c_m}

    u0_bf_mps = 1e-10
    v0_bf_mps = 0
    w0_bf_mps = 0
    p0_bf_rps = 10 * math.pi / 180
    q0_bf_rps = 20 * math.pi / 180
    r0_bf_rps = 30 * math.pi / 180
    phi0_rad = 0 * math.pi / 180
    theta0_rad = 0 * math.pi / 180
    psi0_rad = 0.0
    p10_n_m = 0.0
    p20_n_m = 0.0
    p30_n_m = -30000 / 3.281
    params = {
        "u0_bf_mps": u0_bf_mps,
        "v0_bf_mps": v0_bf_mps,
        "w0_bf_mps": w0_bf_mps,
        "p0_bf_rps": p0_bf_rps,
        "q0_bf_rps": q0_bf_rps,
        "r0_bf_rps": r0_bf_rps,
        "phi0_rad": phi0_rad,
        "theta0_rad": theta0_rad,
        "psi0_rad": psi0_rad,
        "p10_n_m": p10_n_m,
        "p20_n_m": p20_n_m,
        "p30_n_m": p30_n_m
    }

    return vmod, params


def NASACase3():
    # NASA Atmo Case 3 (Brick with damping)
    in2m = 0.0254
    slug2kg = 14.5939
    kg2slug = 1/slug2kg
    ft2m = 0.304878

    m_brick_slug = 0.1554048
    m_brick_kg = m_brick_slug * kg2slug

    Jxx_slugft2 = 0.00189422
    Jxx_kgm2 = slug2kg*(ft2m**2)*Jxx_slugft2

    Jyy_slugft2 = 0.00621102
    Jyy_kgm2 = slug2kg*(ft2m**2)*Jyy_slugft2

    Jzz_slugft2 = 0.00719467
    Jzz_kgm2 = slug2kg*(ft2m**2)*Jzz_slugft2

    Jzx_slugft2 = 0
    Jzx_kgm2 = slug2kg*(ft2m**2)*Jzx_slugft2

    length_brick_m = 8*in2m
    width_brick_m = 4*in2m
    A_ref_brick_m2 = length_brick_m*width_brick_m

    b_m = 0.33333*ft2m
    c_m = 0.66667*ft2m

    Clp = -1.0
    Clr = 0.0
    Cmq = -1.0
    Cnp = 0.0
    Cnr = -1.0



    vmod = {"m_kg": m_brick_kg,
            "Jxx_b_kgm2": Jxx_kgm2,
            "Jyy_b_kgm2": Jyy_kgm2,
            "Jzz_b_kgm2": Jzz_kgm2,
            "Jxz_b_kgm2": Jzx_kgm2,
            "Vterm_mps": 0, #Temp for terminal because I don't know it yet
            "CD_approx": 0.0,
            "Aref_m2": A_ref_brick_m2,
            "Clp": Clp,
            "Clr": Clr,
            "Cmq": Cmq,
            "Cnp": Cnp,
            "Cnr": Cnr,
            "b_m": b_m,
            "c_m": c_m}

    u0_bf_mps = 1e-10
    v0_bf_mps = 0
    w0_bf_mps = 0
    p0_bf_rps = 10 * math.pi / 180
    q0_bf_rps = 20 * math.pi / 180
    r0_bf_rps = 30 * math.pi / 180
    phi0_rad = 0 * math.pi / 180
    theta0_rad = 0 * math.pi / 180
    psi0_rad = 0.0
    p10_n_m = 0.0
    p20_n_m = 0.0
    p30_n_m = -30000 / 3.281
    params = {
        "u0_bf_mps": u0_bf_mps,
        "v0_bf_mps": v0_bf_mps,
        "w0_bf_mps": w0_bf_mps,
        "p0_bf_rps": p0_bf_rps,
        "q0_bf_rps": q0_bf_rps,
        "r0_bf_rps": r0_bf_rps,
        "phi0_rad": phi0_rad,
        "theta0_rad": theta0_rad,
        "psi0_rad": psi0_rad,
        "p10_n_m": p10_n_m,
        "p20_n_m": p20_n_m,
        "p30_n_m": p30_n_m
    }

    return vmod, params


def NASACase4():
    # r_bowl_m = 1
    # m_bowl_kg = 14.5939
    # J_bowl_kgm2 = 0.4 * m_bowl_kg * r_bowl_m ** 2
    # vmod = {"m_kg": m_bowl_kg,
    #         "Jxx_b_kgm2": 4.8809446628,
    #         "Jyy_b_kgm2": 4.8809446628,
    #         "Jzz_b_kgm2": 4.8809446628,
    #         "Jxz_b_kgm2": 0,
    #         "Vterm_mps": 0,  # Temp for terminal because I don't know it yet
    #         "CD_approx": 0.5,
    #         "Aref_m2": math.pi * r_bowl_m ** 2,
    #         "Clp": 0,
    #         "Clr": 0,
    #         "Cmq": 0,
    #         "Cnp": 0,
    #         "Cnr": 0,
    #         "b_m": 0,
    #         "c_m": 0}
    # return vmod

    # Bowling ball
    r_bowl_m = 0.0762
    m_bowl_kg = 14.5939
    J_bowl_kgm2 = 0.4 * m_bowl_kg * r_bowl_m ** 2
    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": J_bowl_kgm2,
            "Jyy_b_kgm2": J_bowl_kgm2,
            "Jzz_b_kgm2": J_bowl_kgm2,
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0,  # Temp for terminal because I don't know it yet
            "CD_approx": 0.1,
            "Aref_m2": math.pi * r_bowl_m ** 2,
            "Clp": 0,
            "Clr": 0,
            "Cmq": 0,
            "Cnp": 0,
            "Cnr": 0,
            "b_m": 0,
            "c_m": 0}

    u0_bf_mps = 1e-10
    v0_bf_mps = 0
    w0_bf_mps = 0
    p0_bf_rps = 10 * math.pi / 180
    q0_bf_rps = 20 * math.pi / 180
    r0_bf_rps = 30 * math.pi / 180
    phi0_rad = 0 * math.pi / 180
    theta0_rad = 0 * math.pi / 180
    psi0_rad = 0.0
    p10_n_m = 0.0
    p20_n_m = 0.0
    p30_n_m = -30000 / 3.281
    params = {
        "u0_bf_mps": u0_bf_mps,
        "v0_bf_mps": v0_bf_mps,
        "w0_bf_mps": w0_bf_mps,
        "p0_bf_rps": p0_bf_rps,
        "q0_bf_rps": q0_bf_rps,
        "r0_bf_rps": r0_bf_rps,
        "phi0_rad": phi0_rad,
        "theta0_rad": theta0_rad,
        "psi0_rad": psi0_rad,
        "p10_n_m": p10_n_m,
        "p20_n_m": p20_n_m,
        "p30_n_m": p30_n_m
    }

    return vmod, params

    return vmod

def cylinder():
    # Cylinder
    r_bowl_m = 1
    m_bowl_kg = 30
    l = 3
    vmod = {"m_kg": m_bowl_kg,
            "Jxx_b_kgm2": 2,
            "Jyy_b_kgm2": 1,
            "Jzz_b_kgm2": 3,
            "Jxz_b_kgm2": 0,
            "Vterm_mps": 0,  # Temp for terminal because I don't know it yet
            "CD_approx": 0.0,
            "Aref_m2": math.pi * r_bowl_m ** 2}
    return vmod