import numpy as np
from matplotlib import pyplot as plt

from config.constants import MPC_M
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


if __name__ == '__main__':
    plot_dl_dz()
