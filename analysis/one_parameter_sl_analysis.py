import os
import time
import pickle as pck

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from scipy.integrate import cumulative_simpson
from scipy.interpolate import make_splrep, interp1d
from scipy.stats import gaussian_kde

from config.plotting import STD_colors, STD_cmaps

from analysis.plot_spectra import plot_all_the_spectra
from config.settings import MCMC_DIR, PICS_DIR
from src.source_base import Source


def get_pdf(values, s_cdf: float = 1e-3, s_pdf: float = 0):
    """
    A non-binned approach for constructing the probability density function
    :param values: sampled values
    :param s_cdf: the smoothing condition for cdf spline (from make_splrep)
    :param s_pdf: the smoothing condition for pdf spline (-//-)
    :return: pdf (callable)
    """
    values_sorted = np.sort(values)  # the sampled values
    n = values_sorted.shape[0]  # number of sampled values

    # take unique only
    values_unique, indices_unique = np.unique(values_sorted, return_index=True)
    x = (np.arange(0, n, 1) / n)[indices_unique]  # numbers corresponding to the values

    cdf = make_splrep(values_unique, x, s=s_cdf)

    alphas = np.linspace(min(values), max(values), 10**4)
    mid_alphas = (alphas[1:] + alphas[:-1]) / 2.0

    pdf_num = np.diff(cdf(alphas)) / np.diff(alphas)
    pdf = make_splrep(mid_alphas, pdf_num, s=s_pdf)
    return pdf


def plot_pdf_and_kde(values, pdf, kde, axis: Axis = None,
                     if_show: bool = False, name: str = "", if_legend: bool = True,
                     if_pdf: bool = False):
    """
    Compare the manually reconstructed pdf (get_pdf) and the gaussian-based kde estimation (scipy.gaussian_kde)
    :param values: (np.ndarray) values
    :param pdf: (callable) pdf function
    :param kde: (callable) kde function
    :param axis: axis to plot on
    :param if_show: whether to show the plot
    :param name: title of the plot
    :param if_legend: (bool) whether to plot the legend
    :param if_pdf: (bool) whether to plot the pdf
    :return:
    """
    if axis is None:
        plt.figure(figsize=(8, 6))
        axis = plt.subplot(111)

    sns.histplot(values, bins=20, ax=axis, color='orange', alpha=.5, linewidth=0)
    axis.set_title(name)

    x_hom = np.linspace(min(values), max(values), 10**4)

    axis2 = axis.twinx()
    if if_pdf:
        axis2.plot(x_hom, pdf(x_hom), color='royalblue', label='pdf')
    axis2.plot(x_hom, kde(x_hom), color='red', label='kde')
    axis2.plot(x_hom, np.zeros_like(x_hom), color='black', linestyle='dashed', linewidth=.7)

    if if_legend:
        plt.legend()

    if if_show:
        plt.show()
    return


def plot_12_pdfs_and_kdes(filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources: list[Source] = data[0]
    results = data[1]

    plt.figure(figsize=(12, 8))

    source_counter = 0
    for i, s in enumerate(sources):
        if source_counter >= 48:
            break
        # if s.title[:3] == "Mkn":
        #     print(s.title)
        #     continue
        flux_model_i, flat_samples_i = results[i][0], results[i][1]
        values = flat_samples_i[:, 0]  # alpha parameter is the 1st one
        if np.var(values) < 0.2:
            continue
        source_counter += 1
        t0 = time.time_ns()
        # pdf_i = get_pdf(values, s_cdf=1e-2)
        # t1 = time.time_ns()
        # dt1 = t1 - t0
        kde_i = gaussian_kde(values)
        t2 = time.time_ns()
        dt2 = t2 - t0
        # print(f"Time for pdf_{i} = {dt1:.1f} ns, kde_{i} = {dt2:.1f} ns, ratio: {(dt1 / dt2):.1f}")
        ax_i = plt.subplot(8, 6, source_counter)
        plot_pdf_and_kde(values, 0.0, kde_i, axis=ax_i, name=s.title, if_show=False, if_legend=False)
        plt.xlim(0, 3)

    plt.tight_layout()
    # plt.savefig(os.path.join(PICS_DIR, "pdf-kde-plot-12sources.png"))
    plt.show()
    return


def one_parameter_sl_analysis(filename: str, min_var: list[float] = None):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources: list[Source] = data[0]
    results = data[1]

    if min_var is None:
        min_var = [0.1]

    n: int = len(sources)
    m: int = 10**3
    k: int = len(min_var)

    pdfs = np.ones([n, m])
    var = np.ones(n)
    alphas = np.linspace(0, 1.5, m)
    for i, s in enumerate(sources):
        if i % 20 == 0:
            print(f"{i+1} sources processed")
        alpha_i = results[i][1][:, 0]
        kde_i = gaussian_kde(alpha_i)
        var[i] = np.var(alpha_i)
        pdfs[i, :] = kde_i(alphas)

    true_vars = np.zeros(k)
    posteriors = np.zeros([k, m])

    for j in range(k):
        true_vars_j = var > min_var[j]
        true_vars[j] = np.sum(true_vars_j)
        true_pdf_j = pdfs[true_vars_j]
        posteriors[j] = np.prod(true_pdf_j, axis=0)

    print(f"Sources processed: {true_vars} sources")
    print(f"Posterior: {alphas[np.argmax(posteriors, axis=1)]}")

    plt.figure(figsize=(10, 4))
    for j in range(k):
        norm_j = np.trapezoid(posteriors[j], alphas)
        normed_posterior_j = posteriors[j] / norm_j
        plt.subplot(1, 2, 1)
        plt.plot(alphas, normed_posterior_j,
                 color=STD_colors[0], linewidth=2)
        plt.subplot(1, 2, 2)
        plt.plot(alphas[1:], cumulative_simpson(normed_posterior_j, x=alphas))
        plt.grid(linestyle='--', color="lightgrey")
        plt.xlabel(r"$\alpha$")

    plt.tight_layout()
    plt.show()

    return


def find_a_source(name: str, filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources: list[Source] = data[0]
    results = data[1]

    for i,s in enumerate(sources):
        # print(s.title)
        if s.title == name:
            print(i)
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plot_all_the_spectra(results[i][1], s, results[i][0])
            alpha_i = results[i][1][:, 0]
            print(np.mean(alpha_i))
            print(np.sqrt(np.var(alpha_i)))
            plt.subplot(1, 2, 2)
            plt.hist(alpha_i, bins=20)
            plt.tight_layout()
            plt.show()
    return


def study_means_and_stds(filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources: list[Source] = data[0]
    results = data[1]

    n = len(sources)

    means = np.zeros(n)
    var = np.zeros(n)

    for i, s in enumerate(sources):
        if i % 40 == 0:
            print(f"{i+1} sources processed")
        alpha_i = results[i][1][:, 0]
        means[i] = np.mean(alpha_i)
        var[i] = np.var(alpha_i)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(means, bins=20, color=STD_colors[0])
    plt.xlabel("mean")

    plt.subplot(1, 3, 2)
    plt.hist(var, bins=20, color=STD_colors[1])
    plt.xlabel("variance")

    plt.subplot(1, 3, 3)
    plt.hist2d(means, var, 20, cmap=STD_cmaps[2], vmax=10, density=True)
    plt.xlabel("mean")
    plt.ylabel("variance")

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    fn = "20250309_noeps_258n_32w_10000st.pck"
    study_means_and_stds(fn)
    # plot_12_pdfs_and_kdes(fn)
    # one_parameter_sl_analysis(fn, min_var=[0.02])
    # find_a_source(filename="20250302_short_258n_32w_6000st.pck", name="1H 1013+498")
