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
    columns = ["alpha", "gamma", "beta", "eta", "lam"]
    fs = pd.DataFrame(flat_samples, columns=columns)

    sns.pairplot(fs, diag_kind="hist", kind="hist", corner=True)
    print(fs["alpha"].mean(), fs["alpha"].std())
    plt.show()
    return

def plot_all_the_spectra(flat_samples, source, op_SL_model, if_show=False, if_scale=False):
    mod = 1.1
    e1 = 10 ** np.linspace(np.log10(source.e_ref[0] / mod), np.log10(source.e_ref[-1] * mod), 100)

    scale = 1.0
    if if_scale:
        scale = e1**2

    for i, s in enumerate(flat_samples):
        if i % 8 != 0:
            continue
        obs = op_SL_model.get(e1, s)
        s[0] = 0
        intrinsic = op_SL_model.get(e1, s)
        plt.plot(e1, scale * obs, alpha=0.1, color='orange')
        plt.plot(e1, scale * intrinsic, alpha=0.1, color='royalblue')

    plt.title(f"{source.title}, z={source.z}")
    source.plot_spectrum(if_show=if_show, if_scale=if_scale)
    return


def show_for_a_source(filename: str):
    with open(os.path.join(MCMC_DIR, filename), "rb") as file_open:
        data = pickle.load(file_open)

    source, results = data[0], data[1]
    op_SL_model, flat_samples = data[1][0], data[1][1]

    plot_hist(flat_samples)
    plot_all_the_spectra(flat_samples, source=source, op_SL_model=op_SL_model, if_show=True, if_scale=True)
    return


def single_source_main(i: int = 1):
    sb = SourceBase(min_evt=5)

    source: Source = sb(i)
    print(source.title)

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = ParametricModification(flux_model=NoEpsParametricModel, optical_depth_model=odi,
                                     mean=1.5, width=1.5, start_value=None,
                                     nwalkers=32, nsteps=10000)
    results = ssl_mod.run(source)

    save_as_pck(1, 32, 10000, [source, results], mode=f"single{i}")
    return


if __name__ == "__main__":
    single_source_main(i=31)
    show_for_a_source(filename="20250306_single31_1n_32w_10000st.pck")
