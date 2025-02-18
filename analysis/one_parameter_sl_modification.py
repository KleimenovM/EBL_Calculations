import numpy as np
import multiprocessing as mp

from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepthInterpolator, OpticalDepth
from src.parametric_ebl_modifications import ParametricModification, ParametricModel, save_as_pck
from src.source_base import SourceBase


def several_sources_main(n: int = 12, nwalkers: int = 32, nsteps: int = 1500,
                         source_indices: list[int] = None):
    sb = SourceBase(if_min_evt=True, min_evt=5)
    if n == -1:
        n = sb.n
    elif n > sb.n:
        raise Exception("not enough sources in the SourceBase")

    if source_indices is None:
        source_indices = np.random.choice(np.arange(0, sb.n), size=n, replace=False)

    sources = [sb(i) for i in source_indices]

    odi = OpticalDepthInterpolator(OpticalDepth(ebl=EBLSaldanaLopez()))
    ssl_mod = ParametricModification(flux_model=ParametricModel, optical_depth_model=odi,
                                     mean=1.5, width=1.5,
                                     nwalkers=nwalkers, nsteps=nsteps)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(ssl_mod.run, sources)

    save_as_pck(n, nwalkers, nsteps, [sources, results], mode="short")
    return


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    several_sources_main(n=1, nwalkers=32, nsteps=12000, source_indices=[15])
