import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from config.constants import MPC_M
from config.settings import PICS_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth


def plot_dl_dz():
    z = np.linspace(0, 1, 100)

    ebl = EBLSaldanaLopez(cmb_on=False, if_err=0)
    od = OpticalDepth(ebl=ebl)

    dl_dz = od.dist_element(z)

    fontsize = 14
    plt.figure(figsize=(8, 6))
    plt.plot(z, dl_dz / MPC_M * 1e-3, linewidth=3, color='#A04DA3')

    plt.xlim(0, 1)
    plt.ylim(0, 5)
    plt.xlabel("z (redshift)", fontsize=fontsize)
    plt.ylabel(r"$\partial L/\partial z$, Gpc", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(linestyle='--', color="lightgrey")
    plt.show()

    return


def plot_l_z():
    z = np.linspace(0, 1, 1000)

    ebl = EBLSaldanaLopez(cmb_on=False, if_err=0)
    od = OpticalDepth(ebl=ebl)

    dl_dz = od.dist_element(z)
    dist = cumulative_trapezoid(dl_dz, z)

    fontsize = 14
    plt.figure(figsize=(7, 5))
    plt.plot(z[:-1], dl_dz[0] * z[:-1] / MPC_M * 1e-3, linewidth=2, color='black', linestyle='--')
    plt.plot(z[:-1], dist / MPC_M * 1e-3, linewidth=3, color='#A04DA3')

    plt.xlim(0, 1)
    plt.ylim(0, 3)
    plt.xlabel("z (redshift)", fontsize=fontsize)
    plt.ylabel(r"$L$, Gpc", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(linestyle='--', color="lightgrey")

    plt.tight_layout()
    # plt.savefig(os.path.join(PICS_DIR, "l_of_z.png"), dpi=400)
    plt.show()


if __name__ == '__main__':
    # plot_dl_dz()
    plot_l_z()
