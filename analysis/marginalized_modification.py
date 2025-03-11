import multiprocessing as mp

import numpy as np

from src.optical_depth import BasisOpticalDepth, load_basis_optical_depth
from src.parametric_ebl_modifications import ParametricModification, save_as_pck, ParametricModel, NoEpsParametricModel
from src.source_base import SourceBase


def emcee_marginalized(n_fb: int = 8, n_s: int = 258, min_evt: int = 5,
                       nwalkers: int = 32, nsteps: int = 1500, thin: int = 100):
    sb = SourceBase(if_min_evt=True, min_evt=min_evt)
    n_s = min(n_s, sb.n)
    source_ids = np.random.choice(np.arange(0, sb.n), size=n_s, replace=False)
    sources = [sb(i) for i in source_ids]

    bod: BasisOpticalDepth = load_basis_optical_depth(n_fb)

    ssl_mod = ParametricModification(nwalkers=nwalkers, nsteps=nsteps,
                                     flux_model=NoEpsParametricModel, optical_depth_model=bod,
                                     fitting_vector=bod.vector[0], thin=thin)

    """
    ssl_mod = ParametricModification(nwalkers=nwalkers, nsteps=nsteps,
                                     flux_model=NoEpsParametricModel, optical_depth_model=bod,
                                     mean=0, width=0,
                                     theta0=bod.vector[0].tolist() + [2.0, 0.0, 0.0, 0.0],
                                     sigmas=bod.vector[0].tolist() + [3.0, 2.0, 4.0, 2.0],
                                     dist_type=['u'] * (8 + 4))
    """

    print("Setting finished")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(ssl_mod.run, sources)

    save_as_pck(n_s, nwalkers, nsteps, [sources, results], mode=f"mrg2{n_fb}")
    return


if __name__ == '__main__':
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    emcee_marginalized(n_s=258, nsteps=40000, thin=2000)
