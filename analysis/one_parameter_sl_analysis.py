import os
import pickle as pck

import numpy as np
from scipy.interpolate import make_splrep

from config.settings import MCMC_DIR


def get_cdf(values):
    """
    A non-binned approach for constructing the cumulative  distribution function
    :param values: sampled values
    :return: pdf function (callable)
    """
    values_sorted = np.sort(values)  # the sampled values
    n = values_sorted.shape[0]  # number of sampled values

    # take unique only
    values_unique, indices_unique = np.unique(values_sorted, return_index=True)
    x = (np.arange(0, n, 1) / n)[indices_unique]  # numbers corresponding to the values

    cdf = make_splrep(values_unique, x, s=1e-2)
    return cdf


def get_pdf(x, cdf):
    """
    A non-binned approach for constructing the probability density function
    :param x: values
    :param cdf: cumulative distribution function (callable)
    :return: pdf (callable)
    """
    alphas = np.linspace(x[0], x[-1], 10 ** 6)
    mid_alphas = (alphas[1:] + alphas[:-1]) / 2.0

    pdf_num = np.diff(cdf(alphas)) / np.diff(alphas)
    pdf = make_splrep(mid_alphas, pdf_num, s=1e-2)
    return pdf


def one_parameter_sl_analysis(filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources, results = data[0], data[1]

    for i, s in enumerate(sources):
        if i != 0:
            continue
        flux_model_i, flat_samples_i = results[i][0], results[i][1]
        get_cdf(flat_samples_i[:, 0])

    return


if __name__ == '__main__':
    one_parameter_sl_analysis(filename="20250219_short_1n_32w_12000st.pck")
