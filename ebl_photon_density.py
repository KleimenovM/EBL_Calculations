import numpy as np

from constants import C, H_EV, EV_J


class EBL:
    def __init__(self):
        self.lg_wvl1 = 0.0  # [DL], 1 mkm, close IR
        self.lg_wvl2 = 2.0  # [DL], 100 mkm, IR

        self.d0 = 16 * 1e-9  # [W m-2 sr-1], EBL spectrum normalizaton coeffiient
        self.f_evol = 1.7  # [DL], cosmological evolution parameter

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

    def intensity(self, wvl):
        """
        Get the EBL intensity (per wavelength unit)
        :param wvl: [mkm], photon wavelength
        :return: [W m-2 sr-1], EBL intensity
        """
        lg_wvl = np.log10(wvl)
        inten = np.exp(-(lg_wvl - self.lg_wvl1) ** 2 / 0.4) + np.exp(-(lg_wvl - self.lg_wvl2) ** 2 / 0.4)
        return inten * self.d0

    def density_e(self, e, z):
        """
        Get the EBL spectral number density [m-3 GeV-1]
        :param e: [eV], photon energy
        :param z: [DL], redshift
        :return: [m-3 eV-1], spectral number density
        """
        return 4 * np.pi / C * self.intensity(self.e_to_wvl(e)) * e**(-2) * (1 + z)**(3 - self.f_evol) * EV_J


if __name__ == '__main__':
    print("Not for direct use")
