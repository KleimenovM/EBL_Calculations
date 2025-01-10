import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.interpolate import CubicSpline, interp1d

from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.source_base import SourceBase, Source
from src.source_spectra import GreauxModel


class FastSL:
    def __init__(self, source: Source):
        self.ebl = EBLSaldanaLopez()
        self.od = OpticalDepth(ebl=self.ebl, series_expansion=True)
        self.odi = OpticalDepthInterpolator(self.od)

        self.z0 = source.z
        self.e0 = source.e_ref
        self.lg_e_wide = np.linspace(np.log10(self.e0[0]) - 1, np.log10(self.e0[-1]) + 1, 10 ** 5)
        z_wide = self.z0 * np.ones_like(self.lg_e_wide)
        self.values = self.odi.interpolator((z_wide, self.lg_e_wide))

        self.pf = CubicSpline(self.lg_e_wide, np.log(self.values))
        return

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return np.exp(self.pf(np.log10(e)))

    def test(self):
        plt.plot(self.lg_e_wide, self.values)
        plt.plot(self.lg_e_wide, self(10 ** self.lg_e_wide))
        plt.yscale('log')
        plt.show()
        return


def one_parameter_SL_based_model(e, fast_SL: FastSL, gm: GreauxModel,
                                 alpha, gamma, beta, eta, lam, eps):
    mod_e = e * np.exp(eps)
    tau = fast_SL(mod_e)
    return np.exp(-alpha * tau) * gm.get(mod_e, gamma, beta, eta, lam)


def simple_sl_modification():
    return


if __name__ == '__main__':
    simple_sl_modification()
