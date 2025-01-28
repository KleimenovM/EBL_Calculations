import os
import matplotlib.pyplot as plt
import numpy as np
import emcee as emc
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import pickle as pck

from scipy.interpolate import CubicSpline

from config.settings import DATA_DIR
from src.ebl_photon_density import EBLSaldanaLopez, EBL
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


class FastOneParameterSLBasedModel:
    def __init__(self, fast_SL: FastSL, gm: GreauxModel):
        self.fast_SL = fast_SL
        self.gm = gm

    def __call__(self, e, alpha, gamma, beta, eta, lam, eps):
        mod_e = e * np.exp(-eps)
        tau = self.fast_SL(mod_e)
        return np.exp(-alpha * tau) * self.gm.get(mod_e, gamma, beta, eta, lam)


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


def log_likelihood_single_source(source: Source, model: np.ndarray):
    delta = model - source.dnde
    log_likelihood_plus = np.heaviside(delta, 0.5) * (delta ** 2 / (2 * source.dnde_errp ** 2))
    log_likelihood_minus = np.heaviside(-delta, 0.5) * (delta ** 2 / (2 * source.dnde_errn ** 2))
    return - np.sum(log_likelihood_plus + log_likelihood_minus)


def log_prior(alpha, gamma, beta, eta, lam, eps):
    """
    Calculate the log prior
    :param alpha: N(1, 0.2)
    :param gamma: U(2, 3)
    :param beta: U(0, 2)
    :param eta: U(0, 4)
    :param lam: U(0, 2)
    :param eps: N(0, 0.1)
    :return:
    """

    def log_norm(x, mu, sigma):
        return -(x - mu) ** 2 / (2 * sigma ** 2) - 0.5 * np.log(2 * np.pi * sigma ** 2)

    def log_uniform(x, center, halfwidth):
        return np.log(1 / 2 * halfwidth) if abs(x - center) < halfwidth else -np.inf

    return (log_norm(alpha, 1, 0.2) + log_uniform(gamma, 2, 3) +
            log_uniform(beta, 0, 2) + log_uniform(eta, 0, 4) +
            log_uniform(lam, 0, 2) + log_norm(eps, 0, 0.1))


def log_probability(theta, source, model_class):
    lp = log_prior(*theta)
    if not np.isfinite(lp):
        return -np.inf
    model = model_class(source.e_ref, *theta)
    return lp + log_likelihood_single_source(source=source, model=model)


def check_the_sampler(sampler, ndim):
    fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
    samples = sampler.get_chain()
    labels = ["alpha", "gamma", "beta", "eta", "lam", "eps"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()
    return


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


def simple_sl_modification(source, nsteps: int = 1000):
    theta0 = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    sigmas = np.array([0.2, 3.0, 2.0, 4.0, 2.0, 0.1])

    pos = theta0 + sigmas / 2 * np.random.randn(32, 6)
    nwalkers, ndim = pos.shape

    gm = GreauxModel(name=source.title,
                     phi0=source.phi0,
                     e0=source.e0,
                     gamma=2.0, beta=0.0, eta=0.0, lam=0.0)

    op_SL_model = FastOneParameterSLBasedModel(FastSL(source), gm)

    sampler = emc.EnsembleSampler(nwalkers, ndim, log_probability,
                                  args=(source, op_SL_model))

    sampler.run_mcmc(pos, nsteps, progress=True)

    # print(sampler.get_autocorr_time()
    # check_the_sampler(sampler, ndim)

    flat_samples = sampler.get_chain(discard=200, thin=25, flat=True)

    return [op_SL_model, flat_samples]


def plot_all_the_spectra(flat_samples, source, op_SL_model, if_show=False):
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0]), np.log10(source.e_ref[-1]), 100)

    for s in flat_samples:
        res = op_SL_model(e1, *s)
        # print(res)
        plt.plot(e1, res, alpha=0.1)
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


def single_source_main():
    sb = SourceBase()
    source: Source = sb(1)
    op_SL_model, flat_samples = simple_sl_modification(source)
    # plot_all_the_spectra(flat_samples, source, op_SL_model)
    plot_hist(flat_samples, source, op_SL_model)
    return


def several_sources_main():
    sb = SourceBase(if_min_evt=True, min_evt=5)
    # source_ids = np.arange(1, 100, 1)
    sources = [sb(i) for i in range(sb.n)]

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = SimpleSLModification(odi, nwalkers=32, nsteps=1500)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(ssl_mod.run, sources)

    pck.dump(results, os.path.join(DATA_DIR, "results.pck"))

    plt.figure(figsize=(10, 10))
    joint_result = pd.DataFrame(results[0][1], columns=["alpha", "gamma", "beta", "eta", "lam", "eps"])
    for i, s in enumerate(sources):
        # plt.subplot(4, 3, i + 1)
        # plot_all_the_spectra(results[i][1], s, results[i][0])
        pd.concat([joint_result, pd.DataFrame(results[i][1])])

    plot_hist(joint_result)
    # plt.show()
    return


if __name__ == '__main__':
    # single_source_main()
    several_sources_main()
