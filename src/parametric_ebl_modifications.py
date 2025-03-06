import os
import pickle as pck
import time

import emcee as emc
import numpy as np

from config.settings import MCMC_DIR
from src.likelihood_elements import log_likelihood_single_source, log_uniform, log_norm
from src.source_base import Source
from src.source_spectra import GreauxModel


class FluxModel:
    def __init__(self, optical_depth_model,
                 source: Source, gm: GreauxModel):
        self.optical_depth_model = optical_depth_model
        self.gm = gm
        self.source = source

    def __call__(self, *args, **kwargs):
        return 1.0


class NoEpsParametricModel(FluxModel):
    def __init__(self, optical_depth_model, source: Source, gm: GreauxModel):
        super().__init__(optical_depth_model, source, gm)
        self.tau = optical_depth_model.get(source.z, source.lg_e_ref, parameter=None)

    def __call__(self, e, parameters):
        return np.exp(-parameters[:-4] @ self.tau) * self.gm.get(e, *parameters[-4:])

    def get(self, e, parameters):
        lg_e = np.log10(e)
        tau = self.optical_depth_model.get(self.source.z, lg_e, parameter=parameters[:-4])
        return (np.exp(-tau) * self.gm.get(e, *parameters[-4:]))[0]


class ParametricModel(FluxModel):
    def __init__(self, optical_depth_model, source: Source, gm: GreauxModel):
        super().__init__(optical_depth_model, source, gm)

    def __call__(self, e, parameters):
        # unpack source parameters
        gamma, beta, eta, lam, eps = parameters[-5:]
        # unpack EBL parameters
        alpha = parameters[:-5]

        # energy modification
        lg_e = np.log10(e)
        lg_mod_e = lg_e - eps
        mod_e = 10 ** lg_mod_e

        # setting the redshifts
        z_line = np.ones_like(lg_mod_e) * self.source.z

        # calculating the optical depth
        tau = self.optical_depth_model.get(z0=z_line, lg_e0=lg_mod_e, parameter=alpha)

        return np.exp(-tau) * self.gm.get(mod_e, gamma, beta, eta, lam)


class ParametricModification:
    def __init__(self, flux_model, optical_depth_model,
                 theta0: list[float] = None, sigmas: list[float] = None, dist_type: list[str] = None,
                 start_value = None, mean: float = None, width: float = None,
                 fitting_vector=None, roughness=0.5,
                 nwalkers: int = 32, nsteps: int = 5000):

        if mean is None or width is None:
            self.vdim = fitting_vector.shape[0]
            mean, width = self.define_fitting_bounds(fitting_vector, roughness)
        else:
            self.vdim = 1

        self.theta0 = np.array([mean] * self.vdim + [2.0, 0.0, 0.0, 0.0]) if theta0 is None else np.array(theta0)
        self.ndim = self.theta0.shape[0]
        print(self.ndim)

        self.sigmas = np.array([width] * self.vdim + [3.0, 2.0, 4.0, 2.0]) if sigmas is None else np.array(sigmas)

        if self.sigmas.shape[0] != self.ndim:
            raise TypeError("Sigmas must have same shape as theta0")

        self.dist_type = ['u'] * self.vdim + ['u', 'u', 'u', 'u'] if dist_type is None else dist_type
        if len(self.dist_type) != self.ndim:
            raise TypeError("Dist_type must have same shape as theta0")

        self.flux_model = flux_model
        self.optical_depth_model = optical_depth_model

        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.start_vector = self.theta0.copy()
        if start_value is not None:
            self.start_vector[:self.vdim] = start_value
        self.start = self.start_vector + 0.5 * self.sigmas * np.random.randn(self.nwalkers, self.ndim)

        return

    @staticmethod
    def define_fitting_bounds(fitting_vector, roughness):
        """
        Define the fitting bounds for BSpline parameters
        :param fitting_vector: SL fitting vector
        :param roughness: in [0, 1]; the larger, the wider are the boundaries
        :return: mean, width
        """
        if abs(roughness - 0.5) > 0.5:  # check if roughness is in [0, 1]
            raise ValueError("Roughness should be between 0 and 1")

        lower, mean, upper = min(fitting_vector), np.mean(fitting_vector), max(fitting_vector)
        width = max(mean - lower, upper - mean)
        return mean, width * (1 + roughness)

    def log_prior_source(self, theta):
        """
        Calculate the 'source' log prior
        :param theta:
        :return:
        """
        result = 0
        for i in range(self.ndim):
            if self.dist_type[i] == 'u':
                result += log_uniform(theta[i], self.theta0[i], self.sigmas[i])
            elif self.dist_type[i] == 'n':
                result += log_norm(theta[i], self.theta0[i], self.sigmas[i])
            else:
                raise ValueError(f"Unknown dist_type {self.dist_type[i]}, allowed types: 'u' (uniform), 'n' (normal)")
        return result

    def log_probability(self, theta, source, model):
        lp = self.log_prior_source(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = model(source.e_ref, theta)
        return lp + log_likelihood_single_source(source=source, model=model)

    def run(self, source: Source):
        pos = self.start

        gm = GreauxModel(name=source.title,
                         phi0=source.phi0,
                         e0=source.e0,
                         gamma=self.theta0[-5], beta=self.theta0[-4],
                         eta=self.theta0[-3], lam=self.theta0[-2])

        op_SL_model = self.flux_model(self.optical_depth_model, source, gm)

        sampler = emc.EnsembleSampler(self.nwalkers, self.ndim,
                                      self.log_probability, args=(source, op_SL_model))

        # sampler.get_autocorr_time()

        sampler.run_mcmc(pos, self.nsteps, progress=True)

        return op_SL_model, sampler.get_chain(discard=int(0.2 * self.nsteps), thin=25, flat=True)


def save_as_pck(n, nwalkers, nsteps, data, mode, folder: str = None):
    t = time.strftime("%Y%m%d")
    if folder is None:
        folder = MCMC_DIR

    with open(os.path.join(folder, f"{t}_{mode}_{n}n_{nwalkers}w_{nsteps}st.pck"), "wb") as pickle_file:
        pck.dump(data, pickle_file)

    return
