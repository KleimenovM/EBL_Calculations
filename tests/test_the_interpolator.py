import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.ebl_photon_density import EBLBasis, EBLSaldanaLopez, EBL
from src.functional_basis import FunctionalBasis, BSplineBasis
from src.optical_depth import OpticalDepthInterpolator, OpticalDepth
from analysis.fit_saldana_lopez import fit_saldana_lopez_vector


def test_the_interpolator(ebl_model: EBL, z_value: float = 0.01):
    optical_depth_fb = OpticalDepth(ebl=ebl_model, series_expansion=True)
    optical_depth_fb_interp = OpticalDepthInterpolator(optical_depth_fb, z0_max=max(0.6, z_value + 0.01))
    interpolator: RegularGridInterpolator = optical_depth_fb_interp.interpolator

    lg_energy, energy = optical_depth_fb_interp.lg_e0, optical_depth_fb_interp.e0
    z_line = z_value * np.ones_like(energy)
    values = interpolator((z_line, lg_energy))

    return energy, values


def compare_bs_and_sl():
    functional_basis: FunctionalBasis = BSplineBasis(n=16)
    ebl_model = EBLBasis(basis=functional_basis, v=fit_saldana_lopez_vector(functional_basis)[0])

    ebl_sl = EBLSaldanaLopez()
    ebl_sl1 = EBLSaldanaLopez(if_err=-1)
    ebl_sl2 = EBLSaldanaLopez(if_err=1)

    z_value = 0.7
    e, v0 = test_the_interpolator(ebl_model, z_value=z_value)
    e, v1 = test_the_interpolator(ebl_sl, z_value=z_value)
    _, vp = test_the_interpolator(ebl_sl1, z_value=z_value)
    _, vm = test_the_interpolator(ebl_sl2, z_value=z_value)

    plt.fill_between(e, vm, vp, alpha=.3)
    plt.plot(e, v1, label='sl')
    plt.plot(e, v0, label='fit')

    plt.legend()
    plt.ylim(0, 10)
    plt.xscale('log')
    plt.show()
    return


if __name__ == '__main__':
    compare_bs_and_sl()
