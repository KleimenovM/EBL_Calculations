from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.parametric_ebl_modifications import ParametricModification, NoEpsParametricModel, save_as_pck
from src.source_base import SourceBase, Source


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
    single_source_main(i=30, nsteps=10000)
