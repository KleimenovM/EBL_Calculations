import os
import pickle as pck

import matplotlib.pyplot as plt
import numpy as np

from analysis.fit_saldana_lopez import fit_saldana_lopez_vector, fit_saldana_lopez_evolution
from config.plotting import STD_cmaps
from config.settings import MCMC_DIR, PICS_DIR
from src.ebl_photon_density import EBLBasis, EBLSaldanaLopez
from src.functional_basis import BSplineBasis
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.source_base import Source

colors = ['red', 'green']


def study_means_and_stds(filename: str, if_plot=False):

    vector = fit_saldana_lopez_vector(fb=BSplineBasis(n=8))[0]

    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources: list[Source] = data[0]
    results = data[1]

    m = (results[0][1]).shape[1] - 4
    n = len(sources)

    alphas = np.zeros([n, m])

    for i, s in enumerate(sources):
        alphas[i, :] = np.mean(results[i][1][:, :-4], axis=0)

    means = np.mean(alphas, axis=0)
    cov = np.cov(alphas, rowvar=False)
    corr = np.corrcoef(alphas, rowvar=False)

    if if_plot:
        plt.figure()
        plt.pcolormesh(corr, cmap=STD_cmaps[1])
        for j in range(m):
            line = r"$\alpha_{"f"{j+1}"r"}$"
            plt.text(-0.6, 0.3 + j, line, fontsize=12)
            plt.text(0.3 + j, -0.6, line, fontsize=12)
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(PICS_DIR, "corr"))
        plt.show()

    return means, cov


def compare_transparencies(filename, if_plot=False):
    means, var = study_means_and_stds(filename, if_plot=False)
    print("Done!")

    fb = BSplineBasis(n=8)
    vector = fit_saldana_lopez_vector(fb=fb)[0]
    ebl = fit_saldana_lopez_evolution(EBLBasis(fb, v=vector))

    od = OpticalDepth(ebl, series_expansion=True)
    odi = OpticalDepthInterpolator(od)
    lg_e, e = odi.lg_e0, odi.e0
    print("Done2!")
    for i, z in enumerate([0.01, 0.05, 0.1, 0.3, 0.5, 0.9]):
        res = odi.get(z0=z * np.ones_like(e), lg_e0=lg_e)[0]
        plt.plot(e, np.exp(-res), color='royalblue', alpha=1 - 0.17)

    ebl.vector = means
    od = OpticalDepth(ebl, series_expansion=True)
    odi = OpticalDepthInterpolator(od)
    for i, z in enumerate([0.01, 0.05, 0.1, 0.3, 0.5, 0.9]):
        res = odi.get(z * np.ones_like(e), lg_e)[0]
        plt.plot(e, np.exp(-res), color='orangered', alpha=1 - 0.17 * i)

    plt.xscale('log')
    plt.show()
    """"""
    return


def compare_distributions(filenames: list[str]):
    fb = BSplineBasis(n=8)
    vector = fit_saldana_lopez_vector(fb=fb)[0]
    ebl = fit_saldana_lopez_evolution(EBLBasis(fb, v=vector))
    wvl = 10 ** np.linspace(fb.low_lg_wvl, fb.high_lg_wvl, 1000)  # [mkm]
    eps = ebl.wvl_to_e(wvl)  # [eV]

    fbc = fb.get_distribution_list(np.log10(wvl))[1]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    linestyles = ['solid', 'dashed', 'dotted']
    for i, z in enumerate([0.03, 0.14, 0.6]):
        plt.loglog(eps, ebl.intensity(wvl, z), alpha=1 - 0.3 * i,
                   color='royalblue', label=f"SL, z={z}", linestyle=linestyles[i])

    # Create the secondary x-axis
    ax_top = ax.secondary_xaxis('top', functions=(ebl.e_to_wvl, ebl.wvl_to_e))
    ax_top.set_xlabel("wavelength, mkm")
    ax.set_xlabel("energy, eV")
    ax.set_ylabel(r"$[\lambda I_\lambda],~\mathrm{W\,m^{-2}\,s^{-1}}$")

    for k, fn in enumerate(filenames):
        means, cov = study_means_and_stds(fn, if_plot=False)
        print(cov.shape)

        ebl.vector = means
        for i, z in enumerate([0.03, 0.14, 0.6]):
            res = ebl.intensity(wvl, z)
            plt.loglog(eps, res, alpha=1-0.3*i,
                       color=colors[k], label=f"BS{k+1}, {z}", linestyle=linestyles[i])

    plt.loglog((eps[0], eps[-1]), (np.mean(vector), np.mean(vector)),
               linestyle='dashed', color='black')

    plt.grid(linestyle='dashed', color='lightgray')

    plt.xlabel(r"$\epsilon$, eV")
    plt.ylabel(r"$[\lambda I_\lambda],~\mathrm{nW\,s^{-1}\,cm^{-2}}$")

    plt.legend(ncol=3)
    plt.ylim(5e-10, )

    ax = plt.subplot(122)
    z = 0.03
    ebl_SL = EBLSaldanaLopez(if_err=0)
    ebl_SL1 = EBLSaldanaLopez(if_err=-1)
    ebl_SL2 = EBLSaldanaLopez(if_err=1)
    plt.loglog(eps, ebl_SL.intensity(wvl, z), alpha=1,
               color='royalblue', label=f"SL, z={z}", linestyle=linestyles[0])
    plt.fill_between(eps, ebl_SL1.intensity(wvl, z), ebl_SL2.intensity(wvl, z),
                     color='royalblue', alpha=0.2, label=r"SL $1\sigma$")

    means, cov = study_means_and_stds(fns[0], if_plot=False)
    print(cov.shape)
    ebl.vector = means
    err = np.sqrt(np.diag(fbc.T @ cov @ fbc))
    res = ebl.intensity(wvl, z)
    plt.loglog(eps, res, alpha=1,
               color=colors[0], label=f"BS{1}, z={z}", linestyle=linestyles[0])
    plt.fill_between(eps, res - err, res + err,
                     color=colors[0], alpha=0.2, label=r"BS1 $1\sigma$")

    plt.grid(linestyle='dashed', color='lightgray')

    ax_top = ax.secondary_xaxis('top', functions=(ebl.e_to_wvl, ebl.wvl_to_e))
    ax_top.set_xlabel("wavelength, mkm")
    ax.set_xlabel(r"$\epsilon$, eV")

    plt.legend(ncol=2, loc=8)
    plt.ylim(5e-10, )

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    fns = ["20250311_mrg28_258n_32w_40000st.pck",
           "20250310_mrg28_258n_32w_10000st.pck"]
    # study_means_and_stds(fns[0], if_plot=True)
    compare_distributions(fns)
