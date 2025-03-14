import matplotlib.pyplot as plt
import numpy as np

from src.ebl_photon_density import EBLSimple, EBLSaldanaLopez, CMBOnly
from config.settings import PICS_DIR


def test_density_on_z():
    cmb_on: bool = False
    ebl_s = EBLSimple(cmb_on=cmb_on)
    ebl_SL = EBLSaldanaLopez(cmb_on=cmb_on)
    ebl_CMB = CMBOnly()

    wvl = np.logspace(-4, 6, 10000, base=10)  # [mkm], background photon wavelength
    e = ebl_s.wvl_to_e(wvl)  # [ev], background photon energy
    n = 3

    colors = ['#438086', '#53548A', '#A04DA3']

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.secondary_xaxis('top', functions=(ebl_s.e_to_wvl, ebl_s.wvl_to_e))

    z_range = [0.03, 0.14, 0.6]

    # for (i, z) in enumerate([0.0, 0.01, 0.05, 0.1, 0.3, 0.5]):
    #     ax1.plot(wvl, ebl_s.intensity(wvl, z) * 1e9, color='red', alpha=1 - i / n, label=f"S: z = {z}")

    ax1.plot(0, 0, color='darkred', linewidth=0, label="EBL (SL)")
    for (i, z) in enumerate(z_range):
        ax1.plot(e, ebl_SL.intensity(wvl, z) * 1e9, color=colors[1], alpha=1 - i / n, label=f"z={z}", linewidth=2)

    ax1.plot(0, 0, color='darkgreen', linewidth=0, label="CMB")
    for (i, z) in enumerate(z_range):
        ax1.plot(e, ebl_CMB.intensity(wvl, z) * 1e9, color=colors[2], alpha=1 - i / n,
                 label=f"z={z}", linewidth=2, linestyle="--")

    plt.legend(ncol=2, framealpha=0)
    plt.xlim(2e-3, 2e1)
    # plt.ylim(5e-1, 5e1)
    ax2.set_xlabel(r'wavelength, $\mu m$')
    ax1.set_xlabel(r'energy, $eV$')
    plt.ylabel(r'intensity, $\mathrm{nW~m^{-2}~sr^{-1}}$')
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(0, 20)

    plt.grid(linestyle='dashed', color='lightgray')
    plt.savefig(PICS_DIR + "/density_on_z.pdf")
    plt.tight_layout()
    # plt.savefig(PICS_DIR + "/density_on_z2.png", transparent=True, dpi=600)

    plt.show()

    return


if __name__ == '__main__':
    test_density_on_z()
