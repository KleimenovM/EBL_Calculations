import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from config.settings import BS_SAMPLES_DIR
from src.ebl_photon_density import EBLSaldanaLopez, EBLBasis
from src.functional_basis import FunctionalBasis, BSplineBasis
from src.optical_depth import BasisOpticalDepth, OpticalDepth, OpticalDepthInterpolator
from analysis.fit_saldana_lopez import fit_saldana_lopez_vector, fit_saldana_lopez_evolution


def save_basis_optical_depth(n: int = 17):
    fb = BSplineBasis(n, m=40000)

    ebl_model = fit_saldana_lopez_evolution(EBLBasis(basis=fb, v=fit_saldana_lopez_vector(fb)))

    bod: BasisOpticalDepth = BasisOpticalDepth(ebl_model=ebl_model,
                                               name=f"BSpline_{n}",
                                               series_expansion=True)
    bod.save()
    return


def load_basis_optical_depth(filename: str):
    file_pck = os.path.join(BS_SAMPLES_DIR, filename)
    with open(file_pck, "rb") as f:
        bod: BasisOpticalDepth = pickle.load(f)

    print(bod.name)
    fb: FunctionalBasis = bod.basis

    ebl_SL = EBLSaldanaLopez(cmb_on=True, if_err=0)
    od_SL = OpticalDepth(ebl=ebl_SL, series_expansion=True)
    odi_SL = OpticalDepthInterpolator(od_SL)

    plt.figure(figsize=(12, 8))
    z_0s = [0.01, 0.033, 0.06, 0.1, 0.5, 0.9]
    for i, z0 in enumerate(z_0s):
        plt.subplot(3, 2, i+1)
        plt.title(f"z={z0}")
        lg_energy, energy = odi_SL.lg_e0, odi_SL.e0
        z_line = z0 * np.ones_like(energy)
        tau_SL = odi_SL.interpolator((z_line, lg_energy))

        tau_BOD_matrix = bod.get_basis_components(z0=z0, e0=lg_energy)

        bod.vector, _ = fit_saldana_lopez_vector(fb, z_fit=z0)
        higher_vector = 1.1 * bod.vector
        lower_vector = 0.9 * bod.vector
        avg_vector = np.ones_like(bod.vector) * np.mean(bod.vector)

        plt.plot(lg_energy, np.exp(-tau_SL), label='SL')
        plt.plot(lg_energy, np.exp(-bod.vector @ tau_BOD_matrix), label='BOD')
        plt.fill_between(lg_energy,
                         np.exp(-higher_vector @ tau_BOD_matrix), np.exp(-lower_vector @ tau_BOD_matrix),
                         alpha=.5)

        """
        # Show the input of each basis component
        for i in range(bod.dim):
            tau_i = bod.get(z0=z0, e0=lg_energy, vector=bod.unit_matrix[i] * bod.vector)
            plt.plot(lg_energy, tau_i, label=f'BOD {i}', color='black', alpha=.3)
        plt.yscale('log')
        """

        plt.legend()

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    save_basis_optical_depth(n=17)
    load_basis_optical_depth("BSpline_17.pck")
