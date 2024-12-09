import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def log10_eps(x):
    return np.log10(x + np.finfo(float).tiny)


def interpolate(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                if_log_z: bool = False, bounds_error=False):
    """
    Interpolate the two-variable z function or its logarithm
    :param x: <np.ndarray> variable 1
    :param y: <np.ndarray> variable 2
    :param z: <np.ndarray> matrix of values to interpolate
    :param if_log_z: if True, interpolate log10(z)
    :param bounds_error: if True, give an error if (x2, y2) are outside the interpolation bounds
    :return: <RegularGridInterpolator>
    """
    if if_log_z:
        z1 = log10_eps(z)
    else:
        z1 = z

    return RegularGridInterpolator((x, y), z1, bounds_error=bounds_error, fill_value=None)


def save_interpolator(x: np.ndarray[float], y: np.ndarray[float], interpolator: RegularGridInterpolator,
                      folder: str, filename: str,
                      x_name: str = "redshift", y_name: str = "wavelength", interp_name="interp"):
    """
    Save the interpolator to a pickle file
    :param x: <np.ndarray> axis 1
    :param y: <np.ndarray> axis 2
    :param interpolator: <RegularGridInterpolator>
    :param folder: path to save the pickle file
    :param filename: name of the pickle file
    :param x_name: name of the x variable
    :param y_name: name of the y variable
    :param interp_name: name of the interpolator
    :return:
    """
    interp_dict = {x_name: x, y_name: y, interp_name: interpolator}

    path = os.path.join(folder, filename)

    with open(path, "wb") as pickle_file:
        pickle.dump(interp_dict, pickle_file)

    return


def plot_interpolated_values(x: np.ndarray[float], y: np.ndarray[float], interp: RegularGridInterpolator,
                             delog: bool = False):
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    data = interp((grid_x, grid_y))

    if delog:
        data = 10 ** data
        y = 10 ** y

    plt.pcolormesh(y, x, data)
    plt.colorbar()
    plt.xscale('log')
    return
