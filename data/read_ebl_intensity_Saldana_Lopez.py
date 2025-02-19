import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from config.settings import DATA_SL_DIR
from src.interpolation_2D import plot_interpolated_values, save_interpolator, log10_eps, interpolate


def ebl_intensity_data_import(folder, filename,
                              if_log_data: bool = False, if_log_wvl: bool = True):
    redshifts = np.array([0., 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                          1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
                          3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0,
                          5.2, 5.4, 5.6, 5.8, 6.])  # [DL]
    n = redshifts.size

    path = os.path.join(folder, filename)

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

    if if_log_data:
        data = log10_eps(intensity)
    else:
        data = intensity

    if if_log_wvl:
        wvl = log10_eps(wavelength)
    else:
        wvl = wavelength

    return redshifts, wvl, data


def plot_SL_data(x: np.ndarray[float], y: np.ndarray[float], data: np.ndarray[float],
                 delog: bool = False):
    if delog:
        data = 10 ** data

    plt.pcolormesh(y, x, data)
    plt.colorbar()
    plt.xscale('log')
    return


def plot_and_compare(rsh, lg_wvl, inten, f, if_log):
    wvl = 10 ** lg_wvl
    new_rsh = np.linspace(rsh[0], rsh[-1] + 1, 50)
    new_lg_wvl = np.linspace(lg_wvl[0] - 1, lg_wvl[-1] + 1, 100)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plot_SL_data(rsh, wvl, inten, delog=if_log)

    plt.subplot(1, 2, 2)
    plot_interpolated_values(new_rsh, new_lg_wvl, f, delog=if_log)

    plt.tight_layout()
    plt.show()
    return


def main():
    redshift, lg_wvl, inten = ebl_intensity_data_import(folder=DATA_SL_DIR, filename='ebl_saldana21_comoving.txt',
                                                        if_log_data=False, if_log_wvl=True)

    _, _, delta_inten = ebl_intensity_data_import(folder=DATA_SL_DIR, filename='eblerr_saldana21_comoving.txt',
                                                  if_log_data=False, if_log_wvl=True)

    f = interpolate(x=redshift, y=lg_wvl, z=inten, if_log_z=True)
    f_plus = interpolate(x=redshift, y=lg_wvl, z=inten + delta_inten, if_log_z=True)
    f_minus = interpolate(x=redshift, y=lg_wvl, z=inten - delta_inten, if_log_z=True)

    save_interpolator(x=redshift, y=lg_wvl, interpolator=f,
                      folder=DATA_SL_DIR, filename="interpolated_intensity_SL.pck")
    save_interpolator(x=redshift, y=lg_wvl, interpolator=f_plus,
                      folder=DATA_SL_DIR, filename="interpolated_intensity_SL_upper.pck")
    save_interpolator(x=redshift, y=lg_wvl, interpolator=f_minus,
                      folder=DATA_SL_DIR, filename="interpolated_intensity_SL_lower.pck")

    plot_and_compare(redshift, lg_wvl, log10_eps(inten), f_plus, True)

    return


if __name__ == '__main__':
    main()
