import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import MCMC_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.parametric_ebl_modifications import ParametricModification, NoEpsParametricModel, save_as_pck
from src.source_base import SourceBase, Source


def plot_hist(flat_samples):
    columns = [r"$\alpha$", r"$\gamma$", r"$\beta$", r"$\eta$", r"$\lambda$"]
    fs = pd.DataFrame(flat_samples, columns=columns)

    sns.pairplot(fs, diag_kind="kde", kind="hist", corner=True)
    print(fs[r"$\alpha$"].mean(), fs[r"$\alpha$"].std())
    plt.show()
    return

def plot_all_the_spectra(flat_samples, source, op_SL_model, if_show=False, if_scale=False):
    mod = 1.1
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0] / mod), np.log10(source.e_ref[-1] * mod), 100)

    scale = 1.0
    if if_scale:
        scale = e1**2

    obs_f, int_f = np.zeros([flat_samples.shape[0], e1.size]), np.zeros([flat_samples.shape[0], e1.size])
    for i, s in enumerate(flat_samples):
        obs = op_SL_model.get(e1, s)
        s[0] = 0
        intrinsic = op_SL_model.get(e1, s)
        obs_f[i] = scale * obs
        int_f[i] = scale * intrinsic
        if i % 5 != 0:
            continue
        plt.plot(e1, obs_f[i], alpha=0.05, color='orange', linewidth=0.5)
        plt.plot(e1, int_f[i], alpha=0.05, color='royalblue', linewidth=0.5)

    #plt.plot(e1, np.mean(obs_f, axis=0), color='firebrick', linewidth=2)
    #plt.plot(e1, np.mean(int_f, axis=0), color='navy', linewidth=2)
    plt.plot((e1[0], e1[0]), (1e14, 1e14), color='orange', label='observed')
    plt.plot((e1[0], e1[0]), (1e14, 1e14), color='royalblue', label='intrinsic')

    plt.title(f"{source.title}, z={source.z}")
    plt.grid(color='lightgray', linestyle='dashed')
    source.plot_spectrum(if_show=False, if_scale=if_scale, ax=plt.gca())
    plt.legend()
    if if_show:
        plt.show()
    return


def show_for_a_source(filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as file_open:
        data = pickle.load(file_open)

    source, results = data[0], data[1]
    op_SL_model, flat_samples = data[1][0], data[1][1]

    plot_hist(flat_samples)
    plot_all_the_spectra(flat_samples, source=source, op_SL_model=op_SL_model, if_show=True, if_scale=True)
    return


def single_source_main(i: int = 1, nwalkers: int = 32, nsteps: int = 10000):
    sb = SourceBase(min_evt=5)

    source: Source = sb(i)
    print(source.title)

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez(), series_expansion=True))
    ssl_mod = ParametricModification(flux_model=NoEpsParametricModel, optical_depth_model=odi,
                                     mean=1.5, width=1.5, start_value=None,
                                     nwalkers=nwalkers, nsteps=nsteps)
    results = ssl_mod.run(source, get_time=False)

    save_as_pck(1, nwalkers, nsteps, [source, results], mode=f"single{i}")
    return


if __name__ == "__main__":
    # single_source_main(i=30, nsteps=10000)
    show_for_a_source(filename="20250309_single31_1n_32w_10000st.pck")
