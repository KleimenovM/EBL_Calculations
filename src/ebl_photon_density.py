import os.path
import pickle

import numpy as np

from config.settings import DATA_SL_DIR
from config.constants import C, H_EV, EV_J


class EBL:
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

    def intensity(self, wvl, z):
        """
        Get the EBL intensity (per wavelength unit)
        :param z: [DL], wavelength
        :param wvl: [mkm], photon wavelength
        :return: [W m-2 sr-1], EBL intensity
        """
        return .0

    def density_e(self, e, z):
        """
        Get the EBL spectral number density [m-3 eV-1]
        :param e: [eV], photon energy
        :param z: [DL], redshift
        :return: [m-3 eV-1], spectral number density
        """
        return 4 * np.pi / C * self.intensity(self.e_to_wvl(e), z) * e**(-2) * EV_J


class EBLSimple(EBL):
    def __init__(self):
        self.lg_wvl1: float = 0.  # [DL], 1 mkm, close IR
        self.lg_wvl2: float = 2.  # [DL], 100 mkm, IR

        self.d0: float = 1.6e-8  # [W m-2 sr-1], EBL spectrum normalizaton coeffiient
        self.f_evol: float = 1.7  # [DL], cosmological evolution parameter

    def intensity(self, wvl, z):
        lg_wvl = np.log10(wvl)
        inten = np.exp(-(lg_wvl - self.lg_wvl1)**2 / 0.4) + np.exp(-(lg_wvl - self.lg_wvl2)**2 / 0.4)
        return inten * self.d0 * (1 + z)**(3 - self.f_evol)


class EBLSaldanaLopez(EBL):
    def __init__(self):
        self.file_pck = os.path.join(DATA_SL_DIR, "interpolated_intensity_SL.pck")
        self.redshift, self.wavelength, self.interpolator = self.extract_interpolator()

    def extract_interpolator(self):
        with open(self.file_pck, "rb") as f:
            my_dict = pickle.load(f)

        redshift = my_dict["redshift"]
        wavelength = my_dict["wavelength"]
        interpolate = my_dict["interp"]
        return redshift, wavelength, interpolate

    def intensity(self, wvl, z):
        return 10 ** self.interpolator((z, wvl)) * 1e-9  # [W -> nW]


if __name__ == '__main__':
    print("Not for direct use")
