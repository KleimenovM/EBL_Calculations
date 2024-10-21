import numpy as np
from numba.experimental import jitclass

from config.constants import SIGMA_TH, M_E2


@jitclass
class CrossSection:
    """
    A class responsible for the first-order appoximation of 2gamma->e+e- cross-section
    """

    def __init__(self):
        return

    @staticmethod
    def beta(e0, e, z, mu):
        """
        Get dimensionless [DL] energy parameter 'beta' from energies e0 (gamma0), e1 (gamma1), redshift z and angle mu
        :param e0: [eV], first incident photon energy
        :param e: [eV], second incident photon energy
        :param z: [DL], redshift
        :param mu: [DL], cos(theta)
        :return: [DL], reaction parameter 'beta'
        """
        b = 1.0 - 2.0 * M_E2 / (e0 * e) / ((1 + z) * (1 - mu))
        return np.sqrt((b + np.abs(b)) / 2.0)

    @staticmethod
    def sigma_beta(beta: np.ndarray):
        """
        Get the gamma + gamma -> e+ e- cross-section (rest frame) from beta-parameter
        :param beta: [DL], energy parameter
        :return: [m2], cross-section
        """
        multiplicator = -4 * beta + 2 * beta ** 3 + (3 - beta ** 4) * np.log((1 + beta) / (1 - beta))
        return 3 / 16 * SIGMA_TH * (1 - beta ** 2) * multiplicator

    def cs(self, e0, e, z, mu):
        """
        Get the gamma + gamma -> e+ e- cross-section (rest frame)
        :param e0: [eV], first incident photon energy
        :param e: [eV], second incident photon energy
        :param z: [DL], redshift
        :param mu: [DL], cos(theta)
        :return: [m2], cross-section
        """
        return self.sigma_beta(self.beta(e0, e, z, mu))


if __name__ == '__main__':
    print("Not for direct use")
