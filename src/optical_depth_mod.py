import numpy as np
from scipy.integrate import dblquad, tplquad, trapezoid

from config.constants import H0, C, OMEGA_DE, OMEGA_M, MPC_M
from src.cross_section import CrossSection
from src.ebl_photon_density import EBL


class OpticalDepthMod:
    def __init__(self, ebl: EBL, cs: CrossSection, e0: float, z0: float):
        self.lg_e_low: float = -4.0  # [DL], lg(e/eV), lower background photon energy limit
        self.lg_e_high: float = 2.0  # [DL], lg(e/eV), upper background photon energy limit

        self.cs: CrossSection = cs
        self.ebl_model: EBL = ebl
        self.e0 = e0
        self.z0 = z0
        self.z = z0

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 * 1 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z) ** 3) ** (-0.5) * MPC_M

    def integrand2D(self, lg_e, mu):
        e = 10**lg_e
        return e * self.ebl_model.density_e(e, self.z) * (1 - mu) / 2 * self.cs.cs(self.e0, e, self.z, mu)

    def integrand3D(self, z, lg_e, mu):
        e = 10**lg_e
        return self.dist_element(z) * e * self.ebl_model.density_e(e, z) * (1 - mu) / 2 * self.cs.cs(self.e0, e, z, mu)

    def get2D(self, z):
        self.z = z
        return dblquad(self.integrand2D,
                       a=-1, b=0.95,
                       gfun=self.lg_e_low, hfun=self.lg_e_high,
                       epsrel=0.1)[0]

    def get(self, e0: float = None, z0: float = None, n_z=10):
        if z0 is not None:
            self.z0 = z0
        if e0 is not None:
            self.e0 = e0

        result_line = np.zeros(n_z)
        z_line = np.linspace(0, z0, n_z)  # [DL], redshift

        for i, z in enumerate(z_line):
            dL_dz = self.dist_element(z)
            ans = dL_dz * self.get2D(z)
            result_line[i] = ans

        return trapezoid(result_line, z_line)

        """if z0 is not None:
            self.z0 = z0
        if e0 is not None:
            self.e0 = e0

        return tplquad(self.integrand3D, a=-1, b=0.95,
                       gfun=self.lg_e_low, hfun=self.lg_e_high,
                       qfun=0, rfun=self.z0,
                       epsrel=0.1)"""
