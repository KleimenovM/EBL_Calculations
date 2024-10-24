import matplotlib.pyplot as plt
import numpy as np

from src.ebl_photon_density import EBLSimple, EBLSaldanaLopez


def test_intensity(ebl, z0):
    wvl = 10**np.linspace(-1, 3, 1000)  # [mkm]
    intensity = ebl.intensity(wvl, z=z0)
    plt.plot(wvl, intensity * 1e9)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"$\lambda,~mkm$")
    plt.ylabel(r"$\nu\, I_\nu,~\mathrm{W~m^{-2}~sr^{-1}}$")
    return


def test_intensity2(ebl, z0):
    energy = 10**np.linspace(-3, 1, 1000)  # [eV]
    plt.plot(energy, ebl.intensity(ebl.e_to_wvl(energy), z=z0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\epsilon,~eV$")
    plt.ylabel(r"$\epsilon\,I_\epsilon,~\mathrm{W~m^{-2}~sr^{-1}}$")
    return


def test_density(ebl, z0):
    lg_e0 = np.linspace(-3.5, 2, 1000)
    e0 = 10 ** lg_e0  # [eV]

    plt.plot(e0, ebl.density_e(e=e0, z=z0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\epsilon,~eV$")
    plt.ylabel(r"$\dfrac{\partial n}{\partial \epsilon},~\mathrm{m^{-3}~eV^{-1}}$")
    return


def main():
    ebl1 = EBLSimple()
    ebl2 = EBLSaldanaLopez()
    z0 = np.array([3])

    plt.figure(figsize=(13, 6))
    plt.subplot(1, 3, 1)
    test_intensity(ebl1, z0)
    test_intensity(ebl2, z0)
    plt.xlim(1e-1, 1e3)
    plt.ylim(1e-2, 1e2)
    plt.grid(linestyle='dashed', color='lightgray')

    plt.subplot(1, 3, 2)
    test_intensity2(ebl1, z0)
    test_intensity2(ebl2, z0)
    plt.ylim(1e-10, 1e-7)

    plt.subplot(1, 3, 3)
    test_density(ebl1, z0)
    test_density(ebl2, z0)
    plt.ylim(1e-5)

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    main()
