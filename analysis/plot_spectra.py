import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pck
from marginalized_modification import ParametricModel

from config.settings import MCMC_DIR
from src.source_base import Source


def plot_all_the_spectra(flat_samples, source: Source,
                         op_SL_model: ParametricModel, if_show=False, if_scale=False):
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0]), np.log10(source.e_ref[-1]), 100)

    scale = 1.0
    if if_scale:
        scale = e1**2

    for i, s in enumerate(flat_samples):
        if i % 20 != 0:
            continue
        obs = op_SL_model(e1, s)[0]
        s_empty = [0] * (len(s) - 5) + s[-5:].tolist()
        intrinsic = op_SL_model(e1, s_empty)[0]
        plt.plot(e1, scale * obs, alpha=0.1, color='orange')
        plt.plot(e1, scale * intrinsic, alpha=0.1, color='royalblue')
    source.plot_spectrum(if_show=if_show)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{source.title}, z={source.z}")
    return


def plot_all_the_spectra_from_file1(filename, mode: int = 1, n_s: int = 1):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources, results = data[0], data[1]

    if mode == 1:
        plt.figure(figsize=(12, 8))
    joint_result = pd.DataFrame(results[0][1][:, :-5],
                                columns=[f"alpha{i+1}" for i in range(n_s)])
    for i, s in enumerate(sources):
        if mode == 1:
            plt.subplot(4, 3, i + 1)
            plot_all_the_spectra(results[i][1], s, results[i][0])
        joint_result = pd.concat([joint_result, pd.DataFrame(results[i][1][:, :-5],
                                                             columns=[f"alpha{i+1}" for i in range(n_s)])])

    if mode == 1:
        plt.tight_layout()
        plt.show()
        return

    if mode == 2:
        sns.pairplot(joint_result, corner=True, kind='hist')
        plt.tight_layout()
        plt.show()
        return

    return


if __name__ == '__main__':
    plot_all_the_spectra_from_file1("20250218_short_12n_32w_12000st.pck", mode=1, n_s=1)

