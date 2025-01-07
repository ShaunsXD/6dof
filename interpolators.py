import math
import numpy as np


def fastInterp1(x, y, h):
    """
    Interpolation Function given altitudes, y, and current altitude h. Checks to see if h is in altitude. If it is,
    return the corresponding y. If not, interpolate the value of y at h. If h is above the range of the altitude, hold

    :param x: altitudes data
    :param y: related column vector
    :param h: current height
    :return: interpolated value from related column vector
    """
    #Check to see if h is in x
    idx = np.where(x == h)
    if len(idx[0]) == 1:
        return y[idx[0][0]]
    elif h < x[0]:
        return y[0]
    elif h > x[-1]:
        return y[-1]
    else:
        #Find the two points in x that h lies between
        idx = np.where(x < h)
        idx1 = idx[0][-1]
        idx2 = idx[0][-1] + 1

        #Linearly interpolate the value of y at h
        y_interp = y[idx1] + (y[idx2] - y[idx1]) * (h - x[idx1]) / (x[idx2] - x[idx1])

        return y_interp
