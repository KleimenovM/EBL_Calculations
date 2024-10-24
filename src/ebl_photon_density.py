import os.path
import pickle

import numpy as np

from config.settings import DATA_SL_DIR
from config.constants import C, H_J, H_EV, EV_J, K_B_EV


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
        wvl_z = wvl * (1 + z)
        return 2 * H_J * C ** 2 / (1e-6 * wvl_z)**4 / (np.exp(H_EV * C / (K_B_EV * 1e-6 * wvl_z * self.t_cmb)) - 1)


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
        return 4 * np.pi / C * self.intensity(self.e_to_wvl(e), z) * e**(-2) * EV_J


class CMBOnly(EBL):
    def __init__(self):
        super().__init__(True)


class EBLSimple(EBL):
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
        inten = (np.exp(-(lg_wvl - self.lg_wvl1)**2 / 0.4) + np.exp(-(lg_wvl - self.lg_wvl2)**2 / 0.4)) * self.d0
        return inten * (1 + z)**(3 - self.f_evol)


class EBLSaldanaLopez(EBL):
    def __init__(self, cmb_on: bool = False):
        super().__init__(cmb_on)
        self.file_pck = os.path.join(DATA_SL_DIR, "interpolated_intensity_SL.pck")
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


if __name__ == '__main__':
    print("Not for direct use")
