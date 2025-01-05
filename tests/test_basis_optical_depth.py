import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from config.settings import DATA_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.functional_basis import FunctionalBasis
from src.optical_depth import BasisOpticalDepth, OpticalDepth, OpticalDepthInterpolator
from tests.fit_saldana_lopez import fit_saldana_lopez_vector


def load_basis_optical_depth():
    file_pck = os.path.join(DATA_DIR, "BSpline_17.pck")
    with open(file_pck, "rb") as f:
        bod: BasisOpticalDepth = pickle.load(f)

    print(bod.name)

    z0 = 0.0

    ebl_SL = EBLSaldanaLopez(cmb_on=True, if_err=0)
    od_SL = OpticalDepth(ebl=ebl_SL, series_expansion=True)
    odi_SL = OpticalDepthInterpolator(od_SL)

    lg_energy, energy = odi_SL.lg_e0, odi_SL.e0
    z_line = z0 * np.ones_like(energy)
    tau_SL = odi_SL.interpolator((z_line, lg_energy))

    fb: FunctionalBasis = bod.basis
    tau_BOD_matrix = bod.get_basis_components(z0=z0, e0=lg_energy)

    bod.vector, _ = fit_saldana_lopez_vector(fb, z_fit=z0)
    higher_vector = 1.1 * bod.vector
    lower_vector = 0.9 * bod.vector
    avg_vector = np.ones_like(bod.vector) * np.mean(bod.vector)

    plt.plot(lg_energy, tau_SL, label='SL')
    plt.plot(lg_energy, bod.vector @ tau_BOD_matrix, label='BOD')
    # plt.plot(lg_energy, higher_vector @ tau_BOD_matrix, label='BOD higher')
    # plt.plot(lg_energy, lower_vector @ tau_BOD_matrix, label='BOD lower')
    # plt.plot(lg_energy, avg_vector @ tau_BOD_matrix, label='BOD avg')

    """
    # Show the input of each basis component
    for i in range(bod.dim):
        tau_i = bod.get(z0=z0, e0=lg_energy, vector=bod.unit_matrix[i] * bod.vector)
        plt.plot(lg_energy, tau_i, label=f'BOD {i}', color='black', alpha=.3)
    plt.yscale('log')
    """

    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    load_basis_optical_depth()
