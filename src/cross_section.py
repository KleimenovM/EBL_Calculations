import numpy as np
from numba import njit
from config.constants import SIGMA_TH, M_E2


@njit
def beta(e0, e, z, mu):
    """
    Get dimensionless [DL] energy parameter 'beta' from energies e0 (gamma0), e1 (gamma1), redshift z and angle mu
    :param e0: [eV], first incident photon energy
    :param e: [eV], second incident photon energy
    :param z: [DL], redshift
    :param mu: [DL], cos(theta)
    :return: [DL], reaction parameter 'beta'
    """
    b = 1.0 - 2.0 * M_E2 / (e0 * e * (1 + z) * (1 - mu))
    return np.sqrt((b + np.abs(b)) / 2.0)


@njit
def gamma_gamma_cross_section(e0, e, z, mu):
    """
    Get the gamma + gamma -> e+ e- cross-section (rest frame)
    :param e0: [eV], first incident photon energy
    :param e: [eV], second incident photon energy
    :param z: [DL], redshift
    :param mu: [DL], cos(theta)
    :return: [m2], cross-section
    """
    b2 = 1.0 - 2.0 * M_E2 / (e0 * e * (1 + z) * (1 - mu))
    b = ((b2 + np.abs(b2)) / 2.0) ** 0.5  # = b2 if b2 > 0; = 0 if b2 < 0
    multiplicator = -4 * b + 2 * b ** 3 + (3 - b ** 4) * np.log((1 + b) / (1 - b))
    return 3 / 16 * SIGMA_TH * (1 - b ** 2) * multiplicator


if __name__ == '__main__':
    print("Not for direct use")
