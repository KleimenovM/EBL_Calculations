import numpy as np
import scipy.integrate as scpint
from numba import jit
from numba.experimental import jitclass

from config.constants import H0, C, OMEGA_M, OMEGA_DE, MPC_M
from src.cross_section import gamma_gamma_cross_section
from src.ebl_photon_density import EBL


class OpticalDepth:
    def __init__(self, ebl: EBL):
        self.lg_e_low: float = -6.0  # [DL], lg(e/eV), lower background photon energy limit
        self.lg_e_high: float = 4.0  # [DL], lg(e/eV), upper background photon energy limit

        self.ebl_model: EBL = ebl

        self.integrate = scpint.trapezoid

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z)**3)**(-1/2) * MPC_M

    def standard_lines(self, z0, n_z: int = 100, n_e: int = 100, n_mu: int = 100):
        """
        Construct a standard grid for integration
        :param z0: object redshift
        :param n_z: number of knots on z
        :param n_e: number of knots on e
        :param n_mu: number of knots on mu (cos theta)
        :return: z line, z_e matrix, lg(e) line, lg(e)_z matrix, mu_line
        """
        z_line = np.linspace(0, z0, n_z)  # [DL], redshift
        lg_e_line = np.linspace(self.lg_e_low, self.lg_e_high, n_e)  # [DL], lg(E/eV)
        e_line = 10 ** lg_e_line  # [eV], background photons energy
        mu_line = np.linspace(-1, 1, n_mu, endpoint=False)  # [DL], interaction cosine range
        return z_line, lg_e_line, e_line, mu_line

    @staticmethod
    def standard_matrices(z_line, lg_e_line, e_line, mu_line):
        """
        Construct a number of standard matrices for integration
        :param z_line: [DL], redshift 1D grid
        :param lg_e_line: [DL], lg(energy) 1D grid
        :param e_line: [eV], energy 1D grid
        :param mu_line: [DL], cos(theta) 1D grid
        :return: z_matrix, lg_e_z_matrix, 10**lg_e_z_matrix, e_mu_matrix, mu_matrix
        """
        z_matrix, lg_e_z_matrix = np.meshgrid(z_line, lg_e_line, indexing='ij')
        e_mu_matrix, mu_matrix = np.meshgrid(e_line, mu_line, indexing="ij")  # [DL], [DL]
        return z_matrix, lg_e_z_matrix, 10**lg_e_z_matrix, e_mu_matrix, mu_matrix

    def sigma_n_integration(self, e0, z, e_mu_matrix, mu_matrix):
        """
        Get an integral via interaction angle
        :param e_mu_matrix: [eV], background photon energy matrix
        :param mu_matrix: [DL], cos(interaction angle) matrix
        :param e0: [eV], incident photon energy
        :param z: [DL], redshift
        :return: [m-1], optical length of a unit partition of the LOS
        """
        mu_integration_matrix = (1 - mu_matrix) / 2 * gamma_gamma_cross_section(e0, e_mu_matrix, z, mu_matrix)  # [m2]

        return self.integrate(mu_integration_matrix, mu_matrix, axis=1)

    def get(self, e0, z0, n_z: int = 100, n_e: int = 100, n_mu: int = 100):
        """
        Get total optical depth of the interstellar medium by integrating along the line of sight (LOS)
        :param e0: [eV], incident photon energy
        :param z0: [DL], redshift of the object
        :param n_z: <int> number of splits along the LOS, linear scaling
        :param n_e: <int> number of splits in energy, log scaling
        :param n_mu: <int> number of splits in cos(theta), linear scaling
        :return: [DL], optical depth value
        """

        z_line, lg_e_line, e_line, mu_line = self.standard_lines(z0, n_z, n_e, n_mu)
        z_matrix, lg_e_z_matrix, e_z_matrix, e_mu_matrix, mu_matrix = self.standard_matrices(z_line, lg_e_line,
                                                                                             e_line, mu_line)
        density_matrix = self.ebl_model.density_e(e_z_matrix, z_matrix)

        result_matrix = np.zeros([n_z, n_e])
        for i, z in enumerate(z_line):
            result_matrix[i] = self.sigma_n_integration(e0, z, e_mu_matrix, mu_matrix)

        result_line = self.integrate(np.log(10) * result_matrix * e_z_matrix * density_matrix, lg_e_line, axis=1)

        return self.integrate(result_line * self.dist_element(z_line), z_line, axis=0)


if __name__ == '__main__':
    print("Not for direct use")
