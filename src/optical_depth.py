import numpy as np
import scipy.integrate as scpint

from config.constants import H0, C, OMEGA_M, OMEGA_DE, MPC_M
from src.cross_section import CrossSection
from src.ebl_photon_density import EBL


class OpticalDepth:
    def __init__(self, ebl: EBL, cs: CrossSection):
        self.lg_e_low: float = -5.0  # [DL], lg(e/eV), lower background photon energy limit
        self.lg_e_high: float = 3.0  # [DL], lg(e/eV), upper background photon energy limit

        self.cs: CrossSection = cs
        self.ebl_model: EBL = ebl

        self.integrate = scpint.trapezoid

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 * 1 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z)**3)**(-1/2) * MPC_M

    def sigma_n_integration(self, e0, z, lg_e_line, mu_line, density_line):
        """
        Get an integral via background photon energy and angle
        :param lg_e_line: [DL], background photon energy grid
        :param mu_line: [DL], cosine of the angle between incident and bg photon
        :param density_line: [m-3 ev-1], photon density for given z
        :param e0: [eV], incident photon energy
        :param z: [DL], redshift
        :return: [m-1], optical length of a unit partition of the LOS
        """

        lg_e_matrix, mu_matrix = np.meshgrid(lg_e_line, mu_line, indexing="ij")  # [DL], [DL]
        e_matrix = 10**lg_e_matrix  # [eV]

        density_matrix = np.tile(density_line, (mu_line.size, 1)).T

        sigma_matrix = self.cs.cs(e0, e_matrix, z, mu_matrix)  # [m2]

        integration_matrix = (1 - mu_matrix) / 2 * e_matrix * density_matrix * sigma_matrix  # [m-1]

        # d_lg_e = lg_e_line[1] - lg_e_line[0]
        # d_mu = mu_line[1] - mu_line[0]
        # total = np.sum(integration_matrix) * d_lg_e * d_mu  << SIMPLIFICATION, 2 times faster

        total_line_e = self.integrate(integration_matrix, mu_line, axis=1)  # [m-1]
        total = self.integrate(total_line_e, lg_e_line, axis=0)  # [m-1]

        return total

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
        mu_line = np.linspace(-1, 1, n_mu, endpoint=False)  # [DL], cosine energy range
        # rightmost edge is deleted to avoid divergence at theta = 0

        z_matrix, lg_e_matrix = np.meshgrid(z_line, lg_e_line, indexing='ij')
        e_matrix = 10**lg_e_matrix
        density_matrix = self.ebl_model.density_e(e_matrix, z_matrix)

        result_line = np.zeros(n_z)

        for i, z in enumerate(z_line):
            dL_dz = self.dist_element(z)
            result_line[i] = self.sigma_n_integration(e0, z, lg_e_line, mu_line, density_matrix[i]) * dL_dz

        return self.integrate(result_line, z_line)


if __name__ == '__main__':
    print("Not for direct use")
