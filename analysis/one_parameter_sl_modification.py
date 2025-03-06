import os
import pickle as pck
import time

import numpy as np
import multiprocessing as mp

from config.settings import MCMC_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepthInterpolator, OpticalDepth
from src.parametric_ebl_modifications import ParametricModification, ParametricModel, save_as_pck, NoEpsParametricModel
from src.source_base import SourceBase


def several_sources_main(n: int = 12, nwalkers: int = 32, nsteps: int = 1500,
                         sb: SourceBase = None, source_indices: list[int] = None,
                         folder: str = None, mode: str = "short", if_random: bool = False):
    """
    Independent Markov-Chain Monte-Carlo for several sources from a given SourceBase
    :param n: (int) number of sources, if -1 the whole base is analyzed
    :param nwalkers: (int) number of walkers for each source
    :param nsteps: (int) number of steps in the Markov chain for each source
    :param sb: (SourceBase)
    :param source_indices: (list[int]) source indices from the base to use
    :param folder: (str) folder to store the generated chain
    :param mode: (str) addition to the filename
    :param if_random: (bool) whether to use random sampling or not
    :return:
    """

    t1 = time.time()

    if sb is None:
        sb = SourceBase(if_min_evt=True, min_evt=5)

    if n == -1:
        n = sb.n
    elif n > sb.n:
        raise Exception("not enough sources in the SourceBase")

    if source_indices is None:
        if if_random:
            source_indices = np.random.choice(np.arange(0, sb.n), size=n, replace=False)
        else:
            source_indices = np.arange(0, n)

    sources = [sb(i) for i in source_indices]

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = ParametricModification(flux_model=NoEpsParametricModel, optical_depth_model=odi,
                                     mean=1.5, width=1.5,
                                     nwalkers=nwalkers, nsteps=nsteps)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(ssl_mod.run, sources)

    save_as_pck(n, nwalkers, nsteps, [sources, results], mode=mode, folder=folder)

    t2 = time.time()
    print(f"Total time: {t2 - t1} s")
    return


if __name__ == '__main__':
    several_sources_main(n=-1, nwalkers=32, nsteps=10000,
                         mode="noeps", source_indices=None, if_random=True)
