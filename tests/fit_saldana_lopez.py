import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid as integrate
from scipy.optimize import minimize

from config.settings import PICS_DIR, DATA_SL_DIR
from src.ebl_photon_density import EBLSaldanaLopez, EBLSimple, EBLBasis
from src.functional_basis import FunctionalBasis, BSplineBasis, ExpParabolicBasis
from src.optical_depth import OpticalDepth


def norm2(f, x):
    return integrate(f**2, x)


def chi2_minimization(data_vector, basis_vectors, weights, x):
    n = basis_vectors.shape[0]
    v = integrate(weights * data_vector * basis_vectors, x)

    m = np.zeros([n, n])
    for i in range(n):
        m[i, :] = integrate(weights * basis_vectors[i] * basis_vectors, x)

    t = np.linalg.inv(m)
    return t @ v


def fit_saldana_lopez_vector(fb: FunctionalBasis = BSplineBasis(n=10, m=40000),
                             z_fit: float = 0.00, sl_type: int = 0,
                             if_plot: bool = False, if_show: bool = False):
    ebl_SL = EBLSaldanaLopez(cmb_on=False, if_err=sl_type)
    ebl_SL_upper = EBLSaldanaLopez(cmb_on=False, if_err=1)
    ebl_SL_lower = EBLSaldanaLopez(cmb_on=False, if_err=-1)

    lg_wvl_SL = ebl_SL.lg_wavelength

    d = 0.0
    fb.low_lg_wvl, fb.high_lg_wvl = lg_wvl_SL[0] - d, lg_wvl_SL[-1] + d
    lg_wvl, basis_intensities = fb.get_distribution_list()

    wvl = 10**lg_wvl
    energy = ebl_SL.wvl_to_e(wvl)
    lg_e = np.log10(energy)

    SL_intensity = ebl_SL.intensity(wvl=wvl, z=z_fit)
    SL_upper = ebl_SL_upper.intensity(wvl=wvl, z=z_fit)
    SL_lower = ebl_SL_lower.intensity(wvl=wvl, z=z_fit)

    weights = 1  # ((SL_upper - SL_lower) / energy)**(-2) / energy

    a = chi2_minimization(SL_intensity, basis_intensities, weights, lg_e)

    if if_plot:
        e = ebl_SL.wvl_to_e(wvl)
        colors = ['#A04DA3', '#53548A', '#438086']
        plt.fill_between(e, SL_lower * 1e9, SL_upper * 1e9, color=colors[2], alpha=.5)
        plt.plot(e, SL_intensity * 1e9, linewidth=4, color=colors[2], label='Saldana-Lopez')
        plt.plot(e, a @ basis_intensities * 1e9, linewidth=2, color='red', label='B-spline fit')
        plt.legend(loc=1)
        plt.plot(e, a * basis_intensities.T * 1e9, alpha=.1, color='black')
        plt.xscale('log')

    if if_show:
        plt.show()

    df = norm2(np.sqrt(weights * (SL_intensity - a @ basis_intensities)**2), lg_e)
    f = norm2(np.sqrt(weights) * SL_intensity, lg_e)

    return a, df / f


def fit_saldana_lopez_evolution():
    # TODO: write this function
    return


def plot_differences():
    plt.figure(figsize=(12, 8))

    n = [7, 8, 15, 17]

    fs = 12

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(f"N = {n[i]}")

        fb = BSplineBasis(n=n[i])
        fit_saldana_lopez_vector(fb, if_plot=True)

        plt.tick_params(labelsize=fs)
        plt.grid(linestyle='--', color='lightgray')
        plt.xlabel(r"$\lambda,~mkm$", fontsize=fs)
        plt.ylabel(r"$\lambda I_\lambda,~W\ m^{-2}\ sr^{-1}$", fontsize=fs)

    plt.tight_layout()
    plt.show()
    return


def check_densities(fb=BSplineBasis(n=20)):
    ebl_fb = EBLBasis(fb, v=fit_saldana_lopez_vector(fb)[0])

    ebl_sl = EBLSaldanaLopez()

    energy = ebl_fb.wvl_to_e(fb.get_wvl_range())
    plt.plot(energy, ebl_fb.density_e(energy, z=0), label='fit')
    plt.plot(energy, ebl_sl.density_e(energy, z=0), label='SL')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    return


def find_optimal_number_of_basis_functions():
    n = np.arange(2, 40, 1)
    df = np.zeros(n.size)
    for i, n_i in enumerate(n):
        fb = BSplineBasis(n=n_i)
        a_i, df_i = fit_saldana_lopez_vector(fb=fb, if_plot=False)
        df[i] = df_i

    plt.figure(figsize=(8, 6))
    # plt.title(r"$\chi^2$ as a function of basis dimension")

    plt.scatter(n, df, color='royalblue')
    plt.plot((8, 8), (1e-6, 2), color='red', linestyle='--', label='N=8')
    plt.plot((17, 17), (1e-6, 2), color='green', linestyle='--', label='N=17')
    # plt.plot(n, n**0, color='red', linestyle='--')

    plt.xlabel(r"$N$, number of basis functions", fontsize=14)
    plt.xlim(0, 40)

    plt.ylabel(r"$\varepsilon$, relative error", fontsize=14)
    # plt.ylabel(r"$\chi^2$ value", fontsize=14)
    plt.yscale('log')
    plt.ylim(1e-6, 1)

    plt.tick_params(labelsize=14)
    # plt.legend(fontsize=14)
    plt.grid(linestyle='--', color='lightgray')

    plt.tight_layout()
    plt.show()
    return


def a_single_plot():
    plt.figure(figsize=(4, 4))

    n = 17

    fb = BSplineBasis(n=n)
    fit_saldana_lopez_vector(fb, if_plot=True)

    plt.tick_params()
    plt.grid(linestyle='--', color='lightgray')
    plt.xlabel(r"energy, eV")
    plt.ylabel(r"$\nu I_\nu, \mathrm{nW\ m^{-2}\ sr^{-1}}$")

    plt.tight_layout()
    plt.savefig(os.path.join(PICS_DIR, "fit_SL_one_pic.png"), dpi=600)
    plt.show()
    return


if __name__ == '__main__':
    a_single_plot()
    # find_optimal_number_of_basis_functions()
    # plot_differences()
    # check_densities()

