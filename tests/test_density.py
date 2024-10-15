import matplotlib.pyplot as plt
import numpy as np

from ebl_photon_density import EBL


def test_intensity():
    wvl = 10**np.linspace(-1.5, 2.7, 1000)  # [mkm]
    ebl = EBL()
    plt.plot(wvl, ebl.intensity(wvl))
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"$\lambda,~mkm$")
    plt.ylabel(r"$\nu\, I_\nu,~\mathrm{W~m^{-2}~sr^{-1}}$")
    return


def test_intensity2():
    energy = 10**np.linspace(-3, 1, 1000)  # [eV]
    ebl = EBL()
    plt.plot(energy, ebl.intensity(ebl.e_to_wvl(energy)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\epsilon,~eV$")
    plt.ylabel(r"$\epsilon\,I_\epsilon,~\mathrm{W~m^{-2}~sr^{-1}}$")
    return


def test_density():
    lg_e0 = np.linspace(-7, 2, 1000)
    e0 = 10 ** lg_e0  # [eV]

    ebl = EBL()

    plt.plot(e0, ebl.density_e(e=e0, z=1.0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\epsilon,~eV$")
    plt.ylabel(r"$\dfrac{\partial n}{\partial \epsilon},~\mathrm{m^{-3}~eV^{-1}}$")
    return


if __name__ == '__main__':
    plt.figure(figsize=(13, 6))
    plt.subplot(1, 3, 1)
    test_intensity()

    plt.subplot(1, 3, 2)
    test_intensity2()

    plt.subplot(1, 3, 3)
    test_density()

    plt.tight_layout()
    plt.show()
