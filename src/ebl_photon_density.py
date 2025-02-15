import os.path
import pickle

import numpy as np

from config.settings import DATA_SL_DIR
from config.constants import C, H_J, H_EV, EV_J, K_B_EV
from src.functional_basis import FunctionalBasis


class CMB:
    def __init__(self):
        self.t_cmb = 2.72548  # [K], CMB temperature, Source: https://en.wikipedia.org/wiki/Cosmic_microwave_background
        return

    def intensity(self, wvl, z):
        """
        Blackbody radiation at T = 2.73 K
        :param wvl: [mkm], wavelength
        :param z: [DL], redshift
        :return: [W m-2 sr-1], specific intensity by wavelength per wavelength
        """
        # wavelength redshift
        wvl_z = wvl * (1 + z)
        # temperature redshift
        t_cmb = self.t_cmb * (1 + z)
        # unit adjustment
        wvl_z = wvl_z * 1e-6  # [mkm -> m]
        return 2 * H_J * C ** 2 / wvl_z**4 / (np.exp(H_EV * C / (wvl_z * K_B_EV * self.t_cmb)) - 1)


class EBL:
    def __init__(self, cmb_on: bool = False):
        self.cmb = CMB()
        self.cmb_on = cmb_on

    @staticmethod
    def wvl_to_e(wvl):
        """
        Energy of a photon with wavelenth wvl (e = hc / wvl)
        :param wvl: [mkm], wavelength
        :return: [eV], energy
        """
        return H_EV * C / (wvl * 1e-6)  # [mkm -> m]

    @staticmethod
    def e_to_wvl(e):
        """
        Wavelength of a photon with energy e (wvl = hc / e)
        :param e: [eV], energy
        :return: [mkm], wavelength
        """
        return H_EV * C / e * 1e6  # [m -> mkm]

    def no_cmb_intensity(self, wvl, z):
        """
        Get the peculiar EBL intensity by wavelength (per wavelength unit)
        :param z: [DL], redshift
        :param wvl: [mkm], photon wavelength
        :return: [W m-2 sr-1], EBL intensity
        """
        return .0

    def intensity(self, wvl, z):
        """
        Get the EBL intensity by wavelength (per wavelength unit)
        :param z: [DL], redshift
        :param wvl: [mkm], photon wavelength
        :return: [W m-2 sr-1], EBL intensity
        """
        return self.no_cmb_intensity(wvl, z) + self.cmb.intensity(wvl, z) * self.cmb_on

    def density_e(self, e, z):
        """
        Get the EBL spectral number density [m-3 eV-1]
        :param e: [eV], photon energy
        :param z: [DL], redshift
        :return: [m-3 eV-1], spectral number density
        """
        return 4 * np.pi / C * self.intensity(self.e_to_wvl(e), z) * e**(-2) * EV_J * (1 + z)**3


class CMBOnly(EBL):
    def __init__(self):
        super().__init__(True)


class EBLSimple(EBL):
    """
    A simple EBL intensity parametrization as a sum of two equally high gaussians in IR and optical light
    """

    def __init__(self, f_evol: float = None, cmb_on: bool = False):
        super().__init__(cmb_on)
        self.lg_wvl1: float = 0.  # [DL], 1 mkm, close IR
        self.lg_wvl2: float = 2.  # [DL], 100 mkm, IR

        self.d0: float = 1.6e-8  # [W m-2 sr-1], EBL spectrum normalizaton coeffiient
        if f_evol is None:
            self.f_evol: float = 1.7  # [DL], cosmological evolution parameter
        else:
            self.f_evol: float = f_evol

    def no_cmb_intensity(self, wvl, z):
        lg_wvl = np.log10(wvl * (1 + z))
        inten = (np.exp(-(lg_wvl - self.lg_wvl1) ** 2 / 0.4) + np.exp(-(lg_wvl - self.lg_wvl2) ** 2 / 0.4)) * self.d0
        return inten * (1 + z) ** (3 - self.f_evol)


class EBLSaldanaLopez(EBL):
    """
    An EBL parametrization from the work of A.Saldana-Lopez et al. (2021)
    https://doi.org/10.1093/mnras/stab2393
    """

    def __init__(self, cmb_on: bool = False, if_err: int = 0):
        super().__init__(cmb_on)
        if if_err == 1:
            filename = "interpolated_intensity_SL_upper.pck"
        elif if_err == -1:
            filename = "interpolated_intensity_SL_lower.pck"
        else:
            filename = "interpolated_intensity_SL.pck"

        self.file_pck = os.path.join(DATA_SL_DIR, filename)
        self.redshift, self.lg_wavelength, self.interpolator = self.extract_interpolator()

    def extract_interpolator(self):
        with open(self.file_pck, "rb") as f:
            my_dict = pickle.load(f)

        redshift = my_dict["redshift"]
        lg_wavelength = my_dict["wavelength"]
        interpolate = my_dict["interp"]
        return redshift, lg_wavelength, interpolate

    def no_cmb_intensity(self, wvl, z):
        return 10 ** (self.interpolator((z, np.log10(wvl))) - 9)  # [nW -> W]


class EBLBasis(EBL):
    """
    An EBL parametrization by as a linear combination of basis functions
    Based on https://doi.org/10.1088/0004-637X/812/1/60
    """

    def __init__(self, basis: FunctionalBasis,
                 v: np.ndarray = None, cmb_on: bool = False,
                 f_evol=0.0, f_wvl=0.0):
        super().__init__(cmb_on)
        self.basis = basis
        self.dim = basis.n

        if v is None:
            v = np.zeros(self.dim)
            v[0] = 1.0
        self.vector = v

        # z-evolution parameters
        self.f_evol = f_evol
        self.f_wvl = f_wvl
        self.wvl0 = 10  # [mkm]

    def evolution_function(self, z, wvl):
        return (1 + z) ** (self.f_evol + self.f_wvl * np.log10(wvl))

    def no_cmb_intensity(self, wvl, z):
        wvl_z = wvl * (1 + z)
        lg_wvl, total_intensities = self.basis.get_distribution_list(lg_wvl=np.log10(wvl_z))
        result = np.zeros_like(lg_wvl)
        for i in range(self.dim):
            result += self.vector[i] * total_intensities[i]
        result *= self.evolution_function(z, wvl_z)
        return result


if __name__ == '__main__':
    print("Not for direct use")
