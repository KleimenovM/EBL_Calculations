import os.path
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from gammapy.modeling.models import (
    EBL_DATA_BUILTIN,
    EBLAbsorptionNormSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

from config.settings import PICS_DIR, DATA_SL_DIR
from src.ebl_photon_density import EBLSaldanaLopez, EBLSimple
from src.optical_depth import OpticalDepth


def load_pck(folder: str, filename: str):
    path = os.path.join(folder, filename)
    with open(path, "rb") as f:
        my_dict = pickle.load(f)

    redshift = my_dict["redshift"]
    lg_energy = my_dict["lg_energy"]
    interpolate = my_dict["interp"]
    return redshift, lg_energy, interpolate


def test_optical_depth():
    n_e0: int = 100

    lg_e0_line = np.linspace(10.5, 13.5, n_e0)  # [DL], incident photon energy, lg(e/eV), 0.01 >>> 100 TeV
    e0_line = 10**lg_e0_line  # [eV]

    n_z = 100
    n_e = 200
    n_mu = 100

    ebl_SL = EBLSaldanaLopez(cmb_on=True)
    ebl_S = EBLSimple(cmb_on=True)

    od_SL = OpticalDepth(ebl_SL, simple=True)
    od_S = OpticalDepth(ebl_S, simple=True)

    _, _, tau_interpolator = load_pck(DATA_SL_DIR, "interpolated_optical_depth_SL.pck")

    colors = ['royalblue', 'red', 'green']

    plt.figure(figsize=(10, 6))

    for i, z0 in enumerate([0.03, 0.14, 0.60]):
        res = np.zeros((2, n_e0))
        t1 = time.time()
        for j, e0 in enumerate(e0_line):
            res[0, j] = od_SL.get(e0, z0, n_z, n_e, n_mu)
            res[1, j] = od_S.get(e0, z0, n_z, n_e, n_mu)
        print(time.time() - t1)
        e1_line = e0_line * 1e-12

        plt.plot(e1_line, res[0], label=f"SL: $z_0$ = {z0}", linestyle="solid", color=colors[i])
        plt.plot(e1_line, res[1], label=f"S: $z_0$ = {z0}", linestyle="dashed", color=colors[i])

        plt.plot(e1_line, tau_interpolator((z0, lg_e0_line - 12)),
                 color=colors[i], alpha=0.3, linewidth=5,
                 label=r"SL-interp, $z_0$" + f" = {z0}")

        saldana21 = EBLAbsorptionNormSpectralModel.read_builtin(
            "saldana-lopez21", redshift=z0
        )

        energy = e1_line * u.TeV  # TeV
        plt.plot(energy, -np.log(saldana21.evaluate(energy, z0, 1.0)), color=colors[i], linestyle="dotted",
                 label=r"$\gamma py$" + f" SL: $z_0$ = {z0}")

    plt.title("Gamma ray optical depth")
    plt.legend()
    plt.xlabel('E, TeV')
    plt.xscale('log')
    plt.xlim(0.04, 30)

    plt.ylabel(r'$\tau_{\gamma\gamma}$')
    plt.ylim(0, 5)

    plt.grid(linestyle='dashed', color='lightgray')
    plt.tight_layout()

    plt.savefig(os.path.join(PICS_DIR, "optical_depth2.png"))
    plt.savefig(os.path.join(PICS_DIR, "optical_depth2.pdf"))
    plt.show()
    return


if __name__ == '__main__':
    test_optical_depth()
