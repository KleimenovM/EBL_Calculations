import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson

from config.settings import PICS_DIR, DATA_SL_DIR
from src.ebl_photon_density import EBLSaldanaLopez, EBLSimple, EBLBasis
from src.functional_basis import FunctionalBasis, BSplineBasis
from src.optical_depth import OpticalDepth


def norm2(f, x):
    return simpson(f**2, x)


def fit_saldana_lopez(n: int = 10, if_plot: bool = False, if_show: bool = False):
    m: int = 20000

    fb: FunctionalBasis = BSplineBasis(n=n, m=m)
    lg_wvl, basis_intensities = fb.get_distribution_list()

    ebl_SL = EBLSaldanaLopez(cmb_on=False)
    SL_intensity = ebl_SL.intensity(wvl=10**lg_wvl, z=0)

    v = simpson(SL_intensity * basis_intensities, lg_wvl)

    m = np.zeros([n, n])
    for i in range(n):
        m[i, :] = simpson(basis_intensities[i] * basis_intensities, lg_wvl)

    t = np.linalg.inv(m)

    a = t @ v

    if if_plot:
        plt.plot(lg_wvl, SL_intensity, linewidth=3)
        plt.plot(lg_wvl, a @ basis_intensities, linewidth=3)
        plt.plot(lg_wvl, a * basis_intensities.T, alpha=.5)

    if if_show:
        plt.show()

    df = norm2(SL_intensity - a @ basis_intensities, lg_wvl)
    f = norm2(SL_intensity, lg_wvl)

    return a, df / f


def plot_differences():
    plt.figure(figsize=(12, 8))

    n = [7, 8, 15, 17]

    fs = 12

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(f"N = {n[i]}")

        fit_saldana_lopez(n[i], if_plot=True)

        plt.tick_params(labelsize=fs)
        plt.grid(linestyle='--', color='lightgray')
        plt.xlabel(r"$\lg(\lambda / mkm)$, wavelength", fontsize=fs)
        plt.ylabel(r"$\nu I_\nu,~W\ m^{-2}\ sr^{-1}$", fontsize=fs)

    plt.tight_layout()
    plt.show()
    return


def find_optimal_number_of_basis_functions():
    n = np.arange(1, 60, 1)
    df = np.zeros(n.size)
    for i, n_i in enumerate(n):
        a_i, df_i = fit_saldana_lopez(n=n_i, if_plot=False)
        df[i] = df_i

    plt.figure(figsize=(8, 6))
    plt.title("Relative error as a function of basis functions number")

    plt.scatter(n, df)
    plt.plot((8, 8), (1e-6, 2), color='red', linestyle='--', label='N=8')
    plt.plot((17, 17), (1e-6, 2), color='green', linestyle='--', label='N=17')

    plt.xlabel("N, number of basis functions", fontsize=14)
    plt.xlim(0, 60)

    plt.ylabel(r"$\varepsilon$, relative error", fontsize=14)
    plt.yscale('log')
    plt.ylim(1e-6, 1)

    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(linestyle='--', color='lightgray')

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    find_optimal_number_of_basis_functions()
    # plot_differences()

