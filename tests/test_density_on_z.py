import matplotlib.pyplot as plt
import numpy as np

from src.ebl_photon_density import EBLSimple, EBLSaldanaLopez, CMBOnly
from config.settings import PICS_DIR


def test_density_on_z():
    cmb_on: bool = True
    ebl_s = EBLSimple(cmb_on=cmb_on)
    ebl_SL = EBLSaldanaLopez(cmb_on=cmb_on)
    ebl_CMB = CMBOnly()

    wvl = np.logspace(-4, 6, 10000, base=10)  # [mkm], background photon energy
    n = 6

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.secondary_xaxis('top', functions=(ebl_s.wvl_to_e, ebl_s.e_to_wvl))

    for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
        ax1.plot(wvl, ebl_s.intensity(wvl, z) * 1e9, color='red', alpha=1 - i / n, label=f"S: z = {z}")

    for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
        ax1.plot(wvl, ebl_SL.intensity(wvl, z) * 1e9, color='blue', alpha=1 - i / n, label=f"SL: z = {z}")

    for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
        ax1.plot(wvl, ebl_CMB.intensity(wvl, z) * 1e9, color='green', alpha=1 - i / n, label=f"CMB: z = {z}")

    plt.legend()
    plt.xlim(1e-2, 1e5)
    ax1.set_xlabel(r'wavelength, $\mu m$')
    ax2.set_xlabel(r'energy, $eV$')
    plt.ylabel(r'intensity, $nW~m^{-2}~sr^{-1}$')
    plt.xscale('log')
    plt.ylim(0, 40)

    plt.savefig(PICS_DIR + "/density_on_z.pdf")
    plt.savefig(PICS_DIR + "/density_on_z.png")

    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    test_density_on_z()
