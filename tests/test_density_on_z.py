import matplotlib.pyplot as plt
import numpy as np
from src.ebl_photon_density import EBLSimple, EBLSaldanaLopez


def test_density_on_z():
    cmb_on: bool = False
    ebl_s = EBLSimple(cmb_on=cmb_on)
    ebl_SL = EBLSaldanaLopez(cmb_on=cmb_on)

    wvl = np.logspace(-4, 3, 1000, base=10)  # [mkm], background photon energy
    n = 6

    for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
        plt.plot(wvl, ebl_s.intensity(wvl, z) * 1e9, color='red', alpha=1 - i / n, label=f"S: z = {z}")

    for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
        plt.plot(wvl, ebl_SL.intensity(wvl, z) * 1e9, color='blue', alpha=1 - i / n, label=f"SL: z = {z}")

    plt.legend()
    plt.grid(linestyle='--', color='lightgray')
    plt.xlim(0.1, 1000)
    plt.xlabel(r'wavelength, $\mu m$')
    plt.ylabel(r'intensity, $nW~m^{-2}~sr^{-1}$')
    plt.xscale('log')
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    test_density_on_z()
