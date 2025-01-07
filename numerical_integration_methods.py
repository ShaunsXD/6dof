import numpy as np

def forward_euler(f, t_s, x, h_s, vmod, amod):
    """
    Forward Euler Integration Method

    :param f: Function representing RHS of diff eq
    :param t_s: A vector of points in time at which numerical sol'n will be approximated
    :param x: Numerically approximated solution data to the DE, f
    :param h_s: Step size in seconds
    :return: Vector of points in time at whcih numerical sol'ns was approximated and the numerically approximated solution to the DE< f
    """

    #Forward Euler Integration
    for i in range(1, len(t_s)):
        x[:, i] = x[:, i-1] + h_s * f(t_s[i-1], x[:, i-1], vmod, amod)

    return t_s, x


def rk4(f, t_s, x, h_s, vmod, amod):
    """
    Runge-Kutta 4 Integration Method

    :param f:
    :param t_s:
    :param x:
    :param h_s:
    :param vmod:
    :param amod:
    :return:
    """
    #Runge-Kutta 4 Integration
    for i in range(1, len(t_s)):
        k1 = h_s * f(t_s[i-1], x[:, i-1], vmod, amod)
        k2 = h_s * f(t_s[i-1] + h_s/2, x[:, i-1] + k1/2, vmod, amod)
        k3 = h_s * f(t_s[i-1] + h_s/2, x[:, i-1] + k2/2, vmod, amod)
        k4 = h_s * f(t_s[i-1] + h_s, x[:, i-1] + k3, vmod, amod)

        x[:, i] = x[:, i-1] + (k1 + 2*k2 + 2*k3 + k4)/6

    return t_s, x