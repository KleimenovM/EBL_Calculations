import os.path
import time

import matplotlib.pyplot as plt
import numpy as np

from config.settings import PICS_DIR
from src.cross_section import CrossSection
from src.ebl_photon_density import EBLSaldanaLopez, EBLSimple
from src.optical_depth import OpticalDepth


def test_optical_depth():
    n_e0: int = 100

    lg_e0_line = np.linspace(10.5, 13.5, n_e0)  # [DL], incident photon energy, lg(e/eV), 0.01 >>> 100 TeV
    e0_line = 10**lg_e0_line  # [eV]

    n_z = 100
    n_e = 200
    n_mu = 200

    ebl1 = EBLSaldanaLopez()
    ebl2 = EBLSimple()
    cs = CrossSection()

    od1 = OpticalDepth(ebl1, cs)
    od2 = OpticalDepth(ebl2, cs)

    colors = ['royalblue', 'red', 'green']

    plt.figure(figsize=(10, 6))
    for i, z0 in enumerate([0.03, 0.14, 0.60]):
        res = np.zeros((2, n_e0))
        t1 = time.time()
        for j, e0 in enumerate(e0_line):
            res[0, j] = od1.get2(e0, z0, n_z, n_e, n_mu)
            res[1, j] = od2.get2(e0, z0, n_z, n_e, n_mu)
        print(time.time() - t1)
        plt.plot(e0_line * 1e-12, res[0], label=f"$z_0$ = {z0}", linestyle="solid", color=colors[i])
        plt.plot(e0_line * 1e-12, res[1], label=f"$z_0$ = {z0}", linestyle="dashed", color=colors[i])

    plt.title("Gamma ray optical depth for $f_{evol}$")
    plt.legend()
    plt.xlabel('E, TeV')
    plt.xscale('log')
    plt.xlim(0.04, 30)

    plt.ylabel(r'$\tau_{\gamma\gamma}$')
    plt.ylim(-0.1, 5)

    plt.grid(linestyle='dashed', color='lightgray')
    plt.tight_layout()

    plt.savefig(os.path.join(PICS_DIR, "optical_depth2.png"))
    plt.savefig(os.path.join(PICS_DIR, "optical_depth2.pdf"))
    plt.show()
    return


if __name__ == '__main__':
    test_optical_depth()
