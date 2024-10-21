import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from config.settings import DATA_SL_DIR


def ebl_intensity_data_import(if_log: bool = False):
    redshifts = [0., 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2,
                 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.]  # [DL]
    n = len(redshifts)

    path = os.path.join(DATA_SL_DIR, 'ebl_saldana21_comoving.txt')

    with open(path, 'r') as open_file:
        all_data = open_file.readlines()

        m = len(all_data) - 7

        wavelength = np.zeros(m)  # [mkm]
        intensity = np.zeros([n, m])  # [nW m-2 sr-1], specific intensity

        for i in range(m):
            line_i = all_data[i + 7].split(' ')

            counter = 0
            for j, elem in enumerate(line_i):
                if elem == '':
                    continue
                if counter == 0:
                    wavelength[i] = float(elem)
                    counter += 1
                else:
                    intensity[counter - 1, i] = float(elem)
                    counter += 1

    if if_log:
        data = np.log10(intensity + 1e-30)
    else:
        data = intensity

    return redshifts, wavelength, data


def plot_interpolated_values(x: np.ndarray[float], y: np.ndarray[float], interp: RegularGridInterpolator,
                             delog: bool = False):
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    data = interp((grid_x, grid_y))

    if delog:
        data = 10 ** data

    plt.pcolormesh(y, x, data)
    plt.colorbar()
    plt.xscale('log')
    return


def plot_SL_data(x: np.ndarray[float], y: np.ndarray[float], data: np.ndarray[float],
                 delog: bool = False):
    if delog:
        data = 10 ** data

    plt.pcolormesh(y, x, data)
    plt.colorbar()
    plt.xscale('log')
    return


def save_interpolator(x: np.ndarray[float], y: np.ndarray[float], interpolator: RegularGridInterpolator,
                      x_name: str = "redshift", y_name: str = "wavelength", interp_name="interp"):
    interp_dict = {x_name: x, y_name: y, interp_name: interpolator}

    path = os.path.join(DATA_SL_DIR, "interpolated_intensity_SL.pck")

    with open(path, "wb") as pickle_file:
        pickle.dump(interp_dict, pickle_file)

    return


def plot_and_compare(rsh, wvl, inten, f, if_log):
    new_rsh = np.linspace(rsh[0], rsh[-1] + 1, 50)
    new_wvl = np.logspace(np.log10(wvl[0]) - 1, np.log10(wvl[-1]) + 1, 100, base=10)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plot_SL_data(rsh, wvl, inten, delog=if_log)
    plt.subplot(1, 2, 2)
    plot_interpolated_values(new_rsh, new_wvl, f, delog=if_log)

    plt.tight_layout()
    plt.show()
    return


def main():
    if_log = True

    rsh, wvl, inten = ebl_intensity_data_import(if_log=if_log)
    f = RegularGridInterpolator((rsh, wvl), inten,
                                bounds_error=False, fill_value=None)

    save_interpolator(rsh, wvl, f)

    plot_and_compare(rsh, wvl, inten, f, if_log)

    return


if __name__ == '__main__':
    main()
