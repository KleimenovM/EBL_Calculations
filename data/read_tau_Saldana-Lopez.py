import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

from data.read_ebl_intensity_Saldana_Lopez import interpolate, log10_eps, save_interpolator
from config.settings import DATA_SL_DIR


def ebl_data_import():
    redshifts = 0  # [DL], redshifts
    tau = 0  # [DL], optical density

    with open("Saldana-Lopez/tau_saldana-lopez21.out", "r") as file_name:
        data = file_name.readlines()

    m = len(data) - 4
    energies = np.zeros(m)  # [eV], background photon energies
    for i, line in enumerate(data):
        if i < 2 or i == 3:
            continue

        elif i == 2:
            line_i = line.strip().split(", ")
            n = len(line_i)
            redshifts = np.zeros(n)
            for j in range(n):
                if j == 0:
                    redshifts[j] = float(line_i[j][-5:])
                else:
                    redshifts[j] = float(line_i[j])
                tau = np.zeros([n, m])

        else:
            line_i = line.strip().split(" ")
            counter = 0
            for j in range(len(line_i)):
                if line_i[j] == '':
                    continue
                if j == 0:
                    energies[i-4] = float(line_i[j])
                    counter += 1
                else:
                    tau[counter-1, i-4] = float(line_i[j])
                    counter += 1

    lg_energies = np.log10(energies)  # [DL], lg(e/eV)

    f = interpolate(x=redshifts, y=lg_energies, z=tau, if_log_z=False)
    save_interpolator(redshifts, lg_energies, f,
                      folder=DATA_SL_DIR, filename='interpolated_optical_depth_SL.pck',
                      x_name='redshift', y_name='lg_energy', interp_name='interp')

    redshift = np.linspace(0, 6, 500)
    lg_energy = np.linspace(lg_energies[0], lg_energies[-1], 500)
    r_grid, lg_e_grid = np.meshgrid(redshift, lg_energy, indexing='ij')

    tau_int = f((r_grid, lg_e_grid))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(energies, redshifts, tau, vmin=0, vmax=10)
    plt.xscale('log')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.pcolormesh(10**lg_energy, redshift, tau_int, vmin=0, vmax=10)
    plt.xscale('log')
    plt.colorbar()

    plt.show()

    return


if __name__ == '__main__':
    ebl_data_import()
