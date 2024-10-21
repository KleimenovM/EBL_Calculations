import matplotlib.pyplot as plt
import numpy as np

from src.cross_section import CrossSection
from src.ebl_photon_density import EBLSimple, EBLSaldanaLopez
from src.optical_depth_mod import OpticalDepthMod


def test_optical_depth():
    n_e0 = 10
    lg_e0_line = np.linspace(10, 14, n_e0)  # [DL], incident photon energy, lg(e/eV), 0.01 >>> 100 TeV
    e0_line = 10**lg_e0_line  # [eV]

    cs = CrossSection()
    ebl_model1 = EBLSaldanaLopez()
    ebl_model2 = EBLSimple()

    od1 = OpticalDepthMod(ebl=ebl_model1, cs=cs, e0=0.0, z0=0.0)
    od2 = OpticalDepthMod(ebl=ebl_model2, cs=cs, e0=0.0, z0=0.0)

    plt.figure(figsize=(10, 10))
    for i, z0 in enumerate([0.03, 0.14, 0.60]):
        res1 = np.zeros(n_e0)
        res2 = np.zeros(n_e0)
        print(i)
        for j, e0 in enumerate(e0_line):
            res1[j] = od1.get(e0=e0, z0=z0)
            res2[j] = od2.get(e0=e0, z0=z0)
        plt.subplot(2, 1, 1)
        plt.plot(e0_line * 1e-12, res1, label=f"$z_0$ = {z0}")
        plt.subplot(2, 1, 2)
        plt.plot(e0_line * 1e-12, res2, label=f"$z_0$ = {z0}")

    for i in range(2):
        plt.subplot(2, 1, 1+i)
        plt.title("Gamma ray optical depth")
        plt.legend()
        plt.xlabel('E, TeV')
        plt.xscale('log')
        plt.xlim(0.04, 30)

        plt.ylabel(r'$\tau_{\gamma\gamma}$')
        plt.ylim(0, 5)

        plt.grid(linestyle='dashed', color='lightgray')

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    test_optical_depth()
