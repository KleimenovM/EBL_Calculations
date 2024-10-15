import matplotlib.pyplot as plt
import numpy as np

from optical_depth import OpticalDepth


def test_optical_depth():
    lg_e0_line = np.linspace(10, 14, 100)  # [DL], incident photon energy, lg(e/eV), 0.01 >>> 100 TeV
    e0_line = 10**lg_e0_line  # [eV]

    od = OpticalDepth()
    plt.figure(figsize=(10, 6))
    for z0 in [0.03, 0.14, 0.60]:
        res = []
        for e0 in e0_line:
            res.append(od.get(e0, z0, n_z=30, n_e=100, n_mu=40))
        plt.plot(e0_line * 1e-12, res, label=f"$z_0$ = {z0}")

    plt.title("Gamma ray optical depth for $f_{evol} = 1.7$")
    plt.legend()
    plt.xlabel('E, TeV')
    plt.xscale('log')
    plt.xlim(0.04, 30)

    plt.ylabel(r'$\tau_{\gamma\gamma}$')
    plt.ylim(-0.1, 5)

    plt.grid(linestyle='dashed', color='lightgray')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    test_optical_depth()
