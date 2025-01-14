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

        self.pf = np.poly1d(np.polyfit(self.lg_e_wide, np.log(self.values), 7))
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


def simple_saldana_lopez_modification():
    # load the source database (select the sources with >= 4 events)
    source_db = SourceBase(if_min_evt=True, min_evt=4)

    # take the first source
    source: Source = source_db(1)
    source.plot_spectrum()

    # create a fast Saldana-Lopez function for the chosen source
    fast_sl = FastSL(source)
    # fast_sl.test()

    gm = GreauxModel(name=source.title,
                     phi0=source.phi0,
                     e0=source.e0,
                     gamma=2.0, beta=0.0, eta=0.0, lam=0.0)

    e_ref = source.e_ref

    with pm.Model() as model:
        # Define priors
        sigma = (source.dnde_errp + source.dnde_errn) / 2

        alpha = pm.Normal("alpha", mu=1.0, sigma=0.2)

        eta = pm.Uniform("eta", lower=-4, upper=4)
        gamma = pm.Uniform("gamma", lower=-1, upper=5)
        beta = pm.Uniform("beta", lower=-2, upper=2)
        lam = pm.Uniform("lam", lower=-2, upper=2)

        eps = pm.Normal("eps", mu=0, sigma=0.1)

        # Define likelihood
        likelihood = pm.MvNormal("y",
                                 one_parameter_SL_based_model(e_ref, fast_sl, gm,
                                                              alpha, gamma, beta, eta, lam, eps),
                                 cov=np.eye(source.n) * sigma**2,
                                 observed=source.dnde)
        # model.debug()
        idata = pm.sample(3000)

    return


if __name__ == '__main__':
    simple_saldana_lopez_modification()
