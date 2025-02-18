import os
import pickle

import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

from config.constants import H0, C, OMEGA_M, OMEGA_DE, MPC_M
from config.settings import DATA_SL_DIR, DATA_DIR, BS_SAMPLES_DIR
from src.cross_section import gamma_gamma_cross_section, total_cross_section
from src.ebl_photon_density import EBL, EBLBasis
from src.interpolation_2D import interpolate, save_interpolator, plot_interpolated_values


class OpticalDepth:
    def __init__(self, ebl: EBL, series_expansion: bool = False):
        self.lg_e_low: float = -5.0  # [DL], lg(e/eV), lower background photon energy limit
        self.lg_e_high: float = 2.0  # [DL], lg(e/eV), upper background photon energy limit

        self.ebl_model: EBL = ebl

        self.integrate_inner = trapezoid
        self.integrate_outer = trapezoid

        self.series_expansion = series_expansion  # for simple calculation of the optical depth

    @staticmethod
    def dist_element(z):
        """
        Get distance element [in Mpc] for the LambdaCDM cosmology
        :param z: [DL], redshift
        :return: [m], dL/dz[z]
        """
        return C / H0 / (1 + z) * (OMEGA_DE + OMEGA_M * (1 + z) ** 3) ** (-1 / 2) * MPC_M

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

        return self.integrate_inner(mu_integration_matrix, mu_matrix, axis=1)

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
        e_z_matrix = 10 ** lg_e_z_matrix  # [eV]

        density_matrix = self.ebl_model.density_e(e_z_matrix, z_matrix)

        if self.series_expansion:
            result_matrix = total_cross_section(e0, z_matrix, e_z_matrix)
        else:
            mu_line = np.linspace(-1, 1, n_mu, endpoint=False)  # [DL], interaction cosine range
            e_mu_matrix, mu_matrix = np.meshgrid(e_line, mu_line, indexing="ij")  # [DL], [DL]
            result_matrix = np.zeros([n_z, n_e])
            for i, z in enumerate(z_line):
                result_matrix[i] = self.angle_integration(e0, z, e_mu_matrix, mu_matrix)

        result_line = self.integrate_inner(np.log(10) * result_matrix * e_z_matrix * density_matrix, lg_e_line, axis=1)

        return self.integrate_outer(result_line * self.dist_element(z_line), z_line, axis=0)


class OpticalDepthInterpolator:
    def __init__(self, optical_depth: OpticalDepth,
                 z0_max: float = 1.0, n_z0: int = 100,
                 lg_e0_range=None, n_e0: int = 100,
                 n_e: int = 100, n_mu: int = 100):
        self.n_z0 = n_z0
        self.z0 = z0_max
        self.z0_line = np.linspace(0, self.z0, n_z0)

        if lg_e0_range is None:
            lg_e0_range = [10.5, 13.5]  # [eV], incident photon energies
        self.n_e0 = n_e0
        self.lg_e0 = np.linspace(lg_e0_range[0], lg_e0_range[1], self.n_e0)
        self.e0 = 10 ** self.lg_e0

        # OpticalDepth parameters
        self.optical_depth = optical_depth
        self.optical_depth.integrate_outer = cumulative_trapezoid
        # OpticalDepth.get() parameters
        self.n_e = n_e
        self.n_mu = n_mu

        self.od_table = self.fill_the_optical_depth_table()
        self.interpolator = self.set_the_interpolator()
        return

    def fill_the_optical_depth_table(self):
        """
        Set the optical depth table by calculating the optical depth in every point of the (e, z) grid
        :return: optical depth table
        """
        od_table = np.zeros([self.n_z0, self.n_e0])
        for i, e_0i in enumerate(self.e0):
            v = self.optical_depth.get(e0=e_0i, z0=self.z0, n_z=self.n_z0 + 1,
                                       n_e=self.n_e, n_mu=self.n_mu)
            od_table[:, i] = v
        return od_table

    def set_the_interpolator(self):
        """
        Set a regular grid interpolator of the calculated values
        :return: <RegularGridInterpolator>
        """
        return interpolate(x=self.z0_line, y=self.lg_e0, z=self.od_table, if_log_z=False, bounds_error=False)

    def get(self, z0, lg_e0, parameter: float = 1.0):
        """
        Get the optical depth at the given z0 for a given energies
        :param z0: redshifts (float / np.array)
        :param lg_e0: (float / np.array)
        :param parameter: (float)
        :return: Optical depth at z0 and lg_e0
        """
        return self.interpolator((z0, lg_e0)) * parameter

    def save(self, filename: str = 'interp.pck', folder=DATA_DIR):
        """
        Save the optical depth interpolator to a pickle file
        :param filename: filename of the pickle file
        :param folder: folder to save the pickle file
        :return:
        """
        save_interpolator(self.lg_e0, self.z0, self.interpolator,
                          folder=folder, filename=filename,
                          x_name="redshift", y_name="lg_energy", interp_name="interp")
        pass


class BasisOpticalDepth:
    def __init__(self, ebl_model, name: str = "basis_optical_depth_model",
                 series_expansion: bool = True, if_empty: bool = False,
                 z0_max: float = 1.0, n_z0: int = 100,
                 lg_e0_range=None, n_e0: int = 100,
                 n_e: int = 100, n_mu: int = 100):
        self.name = name
        self.ebl_model: EBLBasis = ebl_model
        self.basis = ebl_model.basis
        self.vector = self.ebl_model.vector

        self.dim = self.ebl_model.dim
        self.unit_matrix = np.eye(self.dim)
        self.interpolator: list[RegularGridInterpolator] = []
        self.empty: bool = True

        # OpticalDepth parameters
        self.series_expansion = series_expansion
        self.n_e = n_e
        self.n_mu = n_mu

        # OpticalDepthInterpolator parameters
        self.z0_max = z0_max
        self.n_z0 = n_z0
        self.lg_e0_range = lg_e0_range
        self.n_e0 = n_e0

        if not if_empty:
            self.set()
        return

    def set(self):
        for i in range(self.dim):
            print(i)
            self.ebl_model.vector = self.unit_matrix[i]
            od_i = OpticalDepth(self.ebl_model, series_expansion=self.series_expansion)
            odi_i = OpticalDepthInterpolator(optical_depth=od_i,
                                             z0_max=self.z0_max, n_z0=self.n_z0,
                                             lg_e0_range=self.lg_e0_range, n_e0=self.n_e0,
                                             n_e=self.n_e, n_mu=self.n_mu)
            self.interpolator.append(odi_i.interpolator)
        self.empty = False
        return

    def get_basis_components(self, z0: float, lg_e0: np.ndarray):
        final_matrix = np.zeros([self.dim, lg_e0.size])

        z_line = z0 * np.ones(lg_e0.size)

        for i in range(self.dim):
            final_matrix[i] = self.interpolator[i]((z_line, lg_e0))

        return final_matrix

    def get(self, z0: float, lg_e0: np.ndarray, parameter=None):
        return parameter @ self.get_basis_components(z0, lg_e0)

    def save(self, filename: str = None, folder=BS_SAMPLES_DIR):
        if filename is None:
            filename = self.name + ".pck"
        if self.empty:
            raise Exception("Saving empty interpolator, aborted")

        path = os.path.join(folder, filename)

        with open(path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)
        return


def load_basis_optical_depth(n: int):
    file_pck = os.path.join(BS_SAMPLES_DIR, f"BSpline_{n}.pck")
    with open(file_pck, "rb") as f:
        bod: BasisOpticalDepth = pickle.load(f)
    return bod


if __name__ == '__main__':
    print("Not for direct use")
