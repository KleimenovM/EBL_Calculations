import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth, OpticalDepthInterpolator
from src.source_base import SourceBase, Source


def get_sl_interpolators(z0_max: float = 1.0):
    od0 = OpticalDepthInterpolator(optical_depth=OpticalDepth(ebl=EBLSaldanaLopez(cmb_on=True, if_err=0)),
                                   z0_max=z0_max).interpolator
    odm = OpticalDepthInterpolator(optical_depth=OpticalDepth(ebl=EBLSaldanaLopez(cmb_on=True, if_err=-1)),
                                   z0_max=z0_max).interpolator
    odp = OpticalDepthInterpolator(optical_depth=OpticalDepth(ebl=EBLSaldanaLopez(cmb_on=True, if_err=1)),
                                   z0_max=z0_max).interpolator
    return [odm, od0, odp]


def check_one_source():
    sb = SourceBase(if_min_evt=True, min_evt=5)
    print(sb.n)
    zs = np.zeros(sb.n)
    for i in range(sb.n):
        zs[i] = sb(i).z
    print(max(zs))
    s: Source = sb(18)

    e, v, dv_m, dv_p = s.e_ref, s.dnde, s.dnde_errn, s.dnde_errp

    plt.errorbar(e, v, yerr=[dv_m, dv_p],
                 linestyle='', marker='s')

    od0, odm, odp = get_sl_interpolators()
    e, u, du_m, du_p = deconvolute_the_source(s, od0)

    plt.errorbar(e, u, yerr=[du_m, du_p], linestyle='', marker='o')

    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def deconvolute_the_source(s: Source, od0: RegularGridInterpolator):
    t0 = od0((s.z, s.lg_e_ref))
    e, v, vm, vp = s.e_ref, s.dnde, s.dnde_errn, s.dnde_errp

    u = v * np.exp(t0)
    du_m = vm * np.exp(t0)
    du_p = vp * np.exp(t0)

    return s.lg_e_ref, u, du_m, du_p


def fitting_function(t, a, b, c):
    return -a * t ** 2 + b * t + c  # parabolic approximation


def parabolic_fit(lg_e, u, du_m, du_p):
    lg_u = np.log10(u)
    a0 = .0
    b0 = (lg_u[-1] - lg_u[0]) / (lg_e[-1] - lg_e[0])
    c0 = b0 * lg_e[0] - lg_u[0]
    p = curve_fit(f=fitting_function, xdata=lg_e, ydata=lg_u, p0=(a0, b0, c0), sigma=np.max([du_m, du_p], axis=0) / u)
    return p[0], np.sqrt(np.diag(p[1]))


def calculate_for_sources(a0: int = 1):
    # import sources
    sb = SourceBase(if_min_evt=True, min_evt=5)
    print(f"Number of sources loaded from STeVCat: {sb.n}")

    # calculate gamma/gamma optical depth
    od: list[RegularGridInterpolator] = get_sl_interpolators()

    counter1: int = 0
    ks: list[int] = [1, 2, 3, 4, 5]
    counters: list[int] = [0, 0, 0, 0, 0]

    for i in range(sb.n):
        s = sb(i)
        lg_e, u, du_m, du_p = deconvolute_the_source(s, od[1 + a0])
        p, ers = parabolic_fit(lg_e, u, du_m, du_p)
        rel = abs(ers[0] / p[0])
        if p[0] < 0:
            counter1 += 1
            for j, k in enumerate(ks):
                if rel < 1/k:
                    counters[j] += 1
            if rel < 1/4:
                # print(f"a = {'%.3f' % p[0]} +- {'%.3f' % ers[0]}, rel = {'%.3f' % rel}", )
                plt.errorbar(10**lg_e, u, yerr=[du_m, du_p], linestyle='', marker='s')
                plt.plot(10**lg_e, 10 ** fitting_function(lg_e, *p), alpha=.3)

    # print(f"Oh, wow, {counter1} positive curvature sources, and for {counter2} of them"
    #       f" the deviation is more than {m} sigma")

    print(f"total = {counter1}, {counters}")

    plt.yscale("log")
    plt.xscale("log")
    plt.show()

    return


if __name__ == '__main__':
    check_one_source()
    # calculate_for_sources(1)
    # calculate_for_sources(-1)
