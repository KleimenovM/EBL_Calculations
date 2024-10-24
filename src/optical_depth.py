import numpy as np
import scipy.integrate as scpint

from config.constants import H0, C, OMEGA_M, OMEGA_DE, MPC_M
from src.cross_section import gamma_gamma_cross_section, total_cross_section
from src.ebl_photon_density import EBL


class OpticalDepth:
    def __init__(self, ebl: EBL, simple: bool = False):
        self.lg_e_low: float = -5.0  # [DL], lg(e/eV), lower background photon energy limit
        self.lg_e_high: float = 2.0  # [DL], lg(e/eV), upper background photon energy limit

        self.ebl_model: EBL = ebl

        self.integrate = scpint.trapezoid

        self.simple = simple

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z)**3)**(-1/2) * MPC_M

    def angle_integration(self, e0, z, e_mu_matrix, mu_matrix):
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
        z_line = np.linspace(0, z0, n_z)  # [DL], redshift
        lg_e_line = np.linspace(self.lg_e_low, self.lg_e_high, n_e)  # [DL], lg(E/eV)
        e_line = 10 ** lg_e_line  # [eV], background photons energy

        z_matrix, lg_e_z_matrix = np.meshgrid(z_line, lg_e_line, indexing='ij')  # [DL], [DL]
        e_z_matrix = 10**lg_e_z_matrix  # [eV]

        density_matrix = self.ebl_model.density_e(e_z_matrix, z_matrix)

        if self.simple:
            result_matrix = total_cross_section(e0, z_matrix, e_z_matrix)
        else:
            mu_line = np.linspace(-1, 1, n_mu, endpoint=False)  # [DL], interaction cosine range
            e_mu_matrix, mu_matrix = np.meshgrid(e_line, mu_line, indexing="ij")  # [DL], [DL]
            result_matrix = np.zeros([n_z, n_e])
            for i, z in enumerate(z_line):
                result_matrix[i] = self.angle_integration(e0, z, e_mu_matrix, mu_matrix)

        result_line = self.integrate(np.log(10) * result_matrix * e_z_matrix * density_matrix, lg_e_line, axis=1)

        return self.integrate(result_line * self.dist_element(z_line), z_line, axis=0)


if __name__ == '__main__':
    print("Not for direct use")
