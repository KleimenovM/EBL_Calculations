import numpy as np
from scipy.integrate import trapezoid

from constants import H0, C, OMEGA_M, OMEGA_DE, MPC_M
from cross_section import CrossSection
from ebl_photon_density import EBL


class OpticalDepth:
    def __init__(self):
        self.e_low = -4  # [DL], lg(e/eV) - lower background photon energy limit
        self.e_high = 2  # [DL], lg(e/eV) - upper background photon energy limit

        self.cs = CrossSection()
        self.ebl = EBL()

        self.integrate = trapezoid

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 * 1 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z)**3)**(-1/2) * MPC_M

    def sigma_n_integration(self, e0, z, n_e: int, n_mu: int):
        """
        Get an integral via background photon energy and angle
        :param e0: [eV], incident photon energy
        :param z: [DL], redshift
        :param n_e: <int>, number of divisions for energy scale (log)
        :param n_mu: <int>, number of divisions for cosine scale (lin)
        :return: [m-1], optical length of a unit partition of the LOS
        """
        lg_e_line = np.linspace(self.e_low, self.e_high, n_e)  # [DL], lg(E/eV)

        mu_line = np.linspace(-1, 1, n_mu + 1)[:-1]  # [DL], cosine energy range,
        # rightmost edge is deleted to avoid divergence at theta = 0

        lg_e_matrix, mu_matrix = np.meshgrid(lg_e_line, mu_line, indexing="ij")  # [DL], [DL]
        e_matrix = 10**lg_e_matrix  # [eV]

        density_matrix = self.ebl.density_e(e_matrix, z)  # [m-3 eV-1]
        beta = self.cs.beta(e0, e_matrix, z, mu_matrix)  # [DL]

        sigma_matrix = self.cs.sigma_beta(beta)  # [m2]
        integration_matrix = (1 - mu_matrix) / 2 * e_matrix * density_matrix * sigma_matrix  # [m-1]

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
        z_list = np.linspace(0, z0, n_z)
        result_line = np.zeros(n_z)

        for i, z in enumerate(z_list):
            result_line[i] = self.dist_element(z) * self.sigma_n_integration(e0, z, n_e, n_mu)

        return self.integrate(result_line, z_list)


if __name__ == '__main__':
    print("Not for direct use")
