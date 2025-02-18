import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pck
from emcee_marginalized import ParametricModel

from config.settings import MCMC_DIR
from src.source_base import Source


def plot_all_the_spectra(flat_samples, source: Source,
                         op_SL_model: ParametricModel, if_show=False):
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0]), np.log10(source.e_ref[-1]), 100)

    for i, s in enumerate(flat_samples):
        if i % 4 != 0:
            continue
        obs = op_SL_model(e1, s)
        s_empty = [0] * (len(s) - 5) + s[-5:].tolist()
        intrinsic = op_SL_model(e1, s_empty)
        plt.plot(e1, obs, alpha=0.1, color='orange')
        plt.plot(e1, intrinsic, alpha=0.1, color='royalblue')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{source.title}, z={source.z}")
    source.plot_spectrum(if_show=if_show)
    return


def plot_all_the_spectra_from_file1(filename, mode: int = 1):
    with open(os.path.join(MCMC_DIR, filename), "rb") as pickle_file:
        data = pck.load(pickle_file)

    sources, results = data[0], data[1]

    if mode == 1:
        plt.figure(figsize=(12, 8))
    joint_result = pd.DataFrame(results[0][1][:, :-5],
                                columns=[f"alpha{i+1}" for i in range(8)])
    for i, s in enumerate(sources):
        if mode == 1:
            plt.subplot(4, 3, i + 1)
            plot_all_the_spectra(results[i][1], s, results[i][0])
        joint_result = pd.concat([joint_result, pd.DataFrame(results[i][1][:, :-5], columns=[f"alpha{i+1}" for i in range(8)])])

    # print(joint_result.head())

    if mode == 2:
        sns.pairplot(joint_result, corner=True, kind='hist')
        plt.tight_layout()
        plt.show()
    return


if __name__ == '__main__':
    plot_all_the_spectra_from_file1("20250218_mrg_12n_32w_6000st.pck", mode=2)

