import os
import time
import matplotlib.pyplot as plt
import numpy as np
import emcee as emc
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import pickle as pck

from config.settings import DATA_DIR, MCMC_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.source_base import SourceBase, Source
from src.source_spectra import GreauxModel


class OneParameterSLBasedModel:
    def __init__(self, odi: OpticalDepthInterpolator, source: Source, gm: GreauxModel):
        self.odi = odi
        self.gm = gm
        self.source = source

    def __call__(self, e, alpha, gamma, beta, eta, lam, eps):
        lg_e = np.log10(e)
        lg_mod_e = lg_e - eps
        mod_e = 10 ** lg_mod_e
        z_line = np.ones_like(lg_mod_e) * self.source.z
        tau = self.odi.interpolator((z_line, lg_mod_e))
        return np.exp(-alpha * tau) * self.gm.get(mod_e, gamma, beta, eta, lam)


# --------------------------------
# PROBABILISTIC SETTING
# --------------------------------


def log_likelihood_single_source(source: Source, model: np.ndarray):
    """
    Calculates the log-likelihood for a single source given a model and observed data.

    :param source: [Source] An instance of the `Source` class containing observed `dnde`
        data, upper errors (`dnde_errp`), and lower errors (`dnde_errn`).
    :param model: [numpy.array] the predicted `dnde` values of the model corresponding to the observation points
    :return: [float]  the negative total log-likelihood based on the observed data and model predictions
    """
    delta = model - source.dnde
    log_likelihood_plus = np.heaviside(delta, 0.5) * (delta ** 2 / (2 * source.dnde_errp ** 2))
    log_likelihood_minus = np.heaviside(-delta, 0.5) * (delta ** 2 / (2 * source.dnde_errn ** 2))
    return - np.sum(log_likelihood_plus + log_likelihood_minus)


def log_norm(x, mu, sigma):
    """
    Calculate the log pdf of a normal distribution
    :param x: value
    :param mu: center of a normal distribution
    :param sigma: variance of a normal distribution
    :return: parabolic_function - const(sigma)
    """
    # TODO: is there a need to calculate the logs? They always give a constant contribution!
    # return -(x - mu) ** 2 / (2 * sigma ** 2)
    return -(x - mu) ** 2 / (2 * sigma ** 2) - 0.5 * np.log(2 * np.pi * sigma ** 2)


def log_uniform(x, center, halfwidth):
    """
    Calculate the log pdf of a uniform distribution
    :param x: value
    :param center: center of a uniform distribution
    :param halfwidth: halfwidth of a uniform distribution
    :return: log(1/2hw) if x is in the uniform distribution, -np.inf otherwise
    """
    # TODO: is there a need to calculate the logs? They always give a constant contribution!
    # return if abs(x - center) < halfwidth else - np.inf
    return np.log(1 / (2 * halfwidth)) if abs(x - center) < halfwidth else -np.inf


def log_prior_source(gamma, beta, eta, lam, eps):
    """
    Calculate the 'source' log prior
    :param gamma: U(2, 3)
    :param beta: U(0, 2)
    :param eta: U(0, 4)
    :param lam: U(0, 2)
    :param eps: N(0, 0.1)
    :return:
    """
    return (log_uniform(gamma, 2, 3) + log_uniform(beta, 0, 2)
            + log_uniform(eta, 0, 4) + log_uniform(lam, 0, 2)
            + log_norm(eps, 0, 0.1))


def log_alpha(alpha):
    """
    Calculate the 'alpha' log prior
    :param alpha: N(1, 0.2)
    :return:
    """
    return log_norm(alpha, 1, 0.2)


def log_probability(theta, source, model_class):
    lp = log_alpha(theta[0]) + log_prior_source(*theta[1:])
    if not np.isfinite(lp):
        return -np.inf
    model = model_class(source.e_ref, *theta)
    return lp + log_likelihood_single_source(source=source, model=model)


def long_log_probability(theta, sources, models):
    lp = log_alpha(theta[0])
    for i, s in enumerate(sources):
        theta_s = theta[1 + 5 * i: 1 + 5 * (i+1)]

        lps = log_prior_source(*theta_s)
        if not np.isfinite(lps):
            return -np.inf
        model_s = models[i](s.e_ref, theta[0], *theta_s)
        lp += log_likelihood_single_source(source=s, model=model_s)
    if not np.isfinite(lp):
        return -np.inf
    return lp


# --------------------------------
# MCMC VARIOUS ORGANIZATIONS
# --------------------------------


class SimpleSLModification:
    def __init__(self, optical_depth_interpolator: OpticalDepthInterpolator,
                 nwalkers: int = 32, nsteps: int = 1000):
        self.odi = optical_depth_interpolator

        self.ndim = 6
        self.theta0 = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        self.sigmas = np.array([0.2, 3.0, 2.0, 4.0, 2.0, 0.1])

        self.nwalkers = nwalkers
        self.nsteps = nsteps
        return

    def run(self, source: Source):
        pos = self.theta0 + self.sigmas / 2 * np.random.randn(self.nwalkers, self.ndim)

        gm = GreauxModel(name=source.title,
                         phi0=source.phi0,
                         e0=source.e0,
                         gamma=2.0, beta=0.0, eta=0.0, lam=0.0)

        op_SL_model = OneParameterSLBasedModel(self.odi, source, gm)

        sampler = emc.EnsembleSampler(self.nwalkers, self.ndim, log_probability,
                                      args=(source, op_SL_model))

        sampler.run_mcmc(pos, self.nsteps, progress=True)

        return op_SL_model, sampler.get_chain(discard=200, thin=25, flat=True)


class GeneralModification:
    def __init__(self, optical_depth_interpolator: OpticalDepthInterpolator,
                 sources: list[Source],
                 nwalkers: int = 32, nsteps: int = 1000):
        self.odi = optical_depth_interpolator

        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.sources = sources
        self.nsources = len(sources)
        self.ndim = 1 + 5 * self.nsources  # general alpha and 5 source-parameters
        self.theta0 = np.array([1.0] + [2.0, 0.0, 0.0, 0.0, 0.0] * self.nsources)
        self.sigmas = np.array([0.2] + [3.0, 2.0, 4.0, 2.0, 0.1] * self.nsources)

        self.models = []
        self.source_data()
        return

    def source_data(self):
        for source in self.sources:
            gm = GreauxModel(name=source.title,
                             phi0=source.phi0,
                             e0=source.e0,
                             gamma=2.0, beta=0.0, eta=0.0, lam=0.0)

            self.models.append(OneParameterSLBasedModel(self.odi, source, gm))
        return

    def run(self):
        pos = self.theta0 + self.sigmas / 4 * np.random.randn(self.nwalkers, self.ndim)

        with mp.Pool(mp.cpu_count()) as pool:
            sampler = emc.EnsembleSampler(self.nwalkers, self.ndim, long_log_probability,
                                          args=(self.sources, self.models), pool=pool)
            sampler.run_mcmc(pos, self.nsteps, progress=True)

        return sampler.get_chain(discard=200, thin=25, flat=True)


# --------------------------------
# PLOTTING TOOLS
# --------------------------------


def plot_all_the_spectra(flat_samples, source, op_SL_model, if_show=False):
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0]), np.log10(source.e_ref[-1]), 100)

    for s in flat_samples:
        res = op_SL_model(e1, *s)
        # print(res)
        plt.plot(e1, res, alpha=0.1, color='orange')
        # print(res)
    plt.xscale('log')
    plt.yscale('log')
    source.plot_spectrum(if_show=if_show)
    return


def plot_hist(flat_samples):
    columns = ["alpha", "gamma", "beta", "eta", "lam", "eps"]
    fs = pd.DataFrame(flat_samples, columns=columns)

    sns.pairplot(fs, diag_kind="kde", kind="hist", corner=True)
    # plt.hist(fs["alpha"], bins=30)
    print(fs["alpha"].mean(), fs["alpha"].std())
    plt.show()
    return


# --------------------------------
# FILE MANAGEMENT
# --------------------------------


def save_as_pck(nwalkers, nsteps, data, mode: int = 1):
    t = time.strftime("%Y%m%d")

    if mode != 1 and mode != 2:
        raise ValueError("mode must be 1 (short) or 2 (long)")
    modeline = "short" if mode == 1 else "long"

    with open(os.path.join(MCMC_DIR, f"{t}_{modeline}_{nwalkers}w_{nsteps}st.pck"), "wb") as pickle_file:
        pck.dump(data, pickle_file)
    return


def plot_all_the_spectra_from_file1():
    with open(os.path.join(DATA_DIR, "results.pck"), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources, results = data[0], data[1]

    plt.figure(figsize=(10, 10))
    joint_result = pd.DataFrame(results[0][1], columns=["alpha", "gamma", "beta", "eta", "lam", "eps"])
    for i, s in enumerate(sources):
        plt.subplot(4, 3, i + 1)
        plot_all_the_spectra(results[i][1], s, results[i][0])
        pd.concat([joint_result, pd.DataFrame(results[i][1])])

    plt.tight_layout()
    plt.show()
    return


def plot_all_the_spectra_from_file2(filename, mode: int = 2):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    gm: GeneralModification = data[0]
    results = data[1]

    sources = gm.sources
    models = gm.models

    print(results.shape)

    if mode == 1:
        plt.figure(figsize=(10, 10))
        for i, s in enumerate(sources):
            plt.subplot(4, 3, i + 1)
            s.plot_spectrum(if_show=False)
            e_wide = np.linspace(s.e_ref[0] * 0.8, s.e_ref[-1]*1.2, 100)
            for j in range(results.shape[0]):
                if j % 9 != 0:
                    continue
                plt.plot(e_wide, models[i](e_wide, results[j][0], *results[j][1+i*5:1+(i+1)*5]),
                         alpha=0.02, color='orange')

        plt.tight_layout()
        plt.show()

    if mode == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.kdeplot(results[:, 0], color='orange', linewidth=2, linestyle='--', ax=ax)
        ax2 = ax.twinx()
        ax2.hist(results[:, 0], bins=100, alpha=0.5)
        plt.show()

    return


# --------------------------------
# EXECUTABLE METHODS
# --------------------------------


def single_source_main(i: int = 1):
    sb = SourceBase()
    source: Source = sb(i)

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = SimpleSLModification(odi, nwalkers=32, nsteps=1500)
    op_SL_model, flat_samples = ssl_mod.run(source)

    # plot_all_the_spectra(flat_samples, source, op_SL_model)
    plot_hist(flat_samples, source, op_SL_model)
    return


def several_sources_main(n: int = 12, nwalkers: int = 32, nsteps: int = 1500):
    sb = SourceBase(if_min_evt=True, min_evt=5)
    n = min(n, sb.n)
    source_ids = np.random.choice(np.arange(0, sb.n), size=n, replace=False)
    sources = [sb(i) for i in source_ids]

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = SimpleSLModification(odi, nwalkers=32, nsteps=1500)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(ssl_mod.run, sources)

    save_as_pck(nwalkers, nsteps, [sources, results], mode=1)

    plt.figure(figsize=(10, 10))
    joint_result = pd.DataFrame(results[0][1], columns=["alpha", "gamma", "beta", "eta", "lam", "eps"])
    for i, s in enumerate(sources):
        plt.subplot(4, 3, i + 1)
        plot_all_the_spectra(results[i][1], s, results[i][0])
        pd.concat([joint_result, pd.DataFrame(results[i][1])])

    plot_hist(joint_result)
    # plt.show()
    return


def several_sources_long_main(n: int = 12, nwalkers: int = 150, nsteps: int = 1000):
    sb = SourceBase(if_min_evt=True, min_evt=5)
    n = min(n, sb.n)
    source_ids = np.random.choice(np.arange(0, sb.n), size=n, replace=False)
    sources = [sb(i) for i in source_ids]

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    gen_mod = GeneralModification(odi, sources, nwalkers=nwalkers, nsteps=nsteps)

    flat_samples = gen_mod.run()

    save_as_pck(nwalkers, nsteps, [gen_mod, flat_samples], mode=2)
    return


if __name__ == '__main__':
    # single_source_main()
    # several_sources_main()
    # plot_all_the_spectra_from_file1()
    # several_sources_long_main()
    plot_all_the_spectra_from_file2("20250128_long_300w_5000st.pck", mode=1)
