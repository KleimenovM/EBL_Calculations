import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from config.settings import PICS_DIR
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


def check_one_source(n: list[int], attenuation: bool = False):
    sb = SourceBase(if_min_evt=True, min_evt=8)
    print(sb.n)
    zs = np.zeros(sb.n)
    for i in range(sb.n):
        zs[i] = sb(i).z
    print(max(zs))

    colors = ['darkred', 'darkgreen']
    loc = [1, 3]

    if attenuation:
        od0, odm, odp = get_sl_interpolators()

    for i, n_i in enumerate(n):
        s: Source = sb(n_i)

        e, v, dv_m, dv_p = s.e_ref, s.dnde, s.dnde_errn, s.dnde_errp

        p0 = plt.scatter(1, 1e-11, linewidth=0, marker='')

        p1 = plt.errorbar(e * 1e-12, v, yerr=[dv_m, dv_p], linestyle='', marker='o', color=colors[i],
                          alpha=2 ** (-attenuation), label=f"observed")

        if attenuation:
            lg_e, u, du_m, du_p = deconvolute_the_source(s, od0)
            p2 = plt.errorbar(10 ** (lg_e - 12), u, yerr=[du_m, du_p], linestyle='', marker='s', color=colors[i],
                              label=f"{s.title}, intrinsic")

            legend1 = plt.legend([p0, p1, p2], [f"{s.title}\nz={np.round(s.z, 3)}", "observed", "intrinsic"],
                                 loc=loc[i], framealpha=0.0)
            plt.gca().add_artist(legend1)

        if not attenuation:
            legend1 = plt.legend([p0, p1], [f"{s.title}\nz={np.round(s.z, 3)}", "observed"], loc=loc[i], framealpha=0.0)
            plt.gca().add_artist(legend1)

    plt.xscale('log')
    plt.xlabel('$E,$ TeV')
    plt.ylabel(r'$dn/dE, \mathrm{TeV^{-1}~cm^{-2}~s^{-1}}$')
    plt.yscale('log')

    return


def deconvolute_the_source(s: Source, od0: RegularGridInterpolator, slice_last: bool = False):
    t0 = od0((s.z, s.lg_e_ref))
    e, v, vm, vp = s.e_ref, s.dnde, s.dnde_errn, s.dnde_errp

    u = v * np.exp(t0)
    du_m = vm * np.exp(t0)
    du_p = vp * np.exp(t0)

    if slice_last:
        n = 3
        return s.lg_e_ref[:-n], u[:-n], du_m[:-n], du_p[:-n]
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

    plt.figure(figsize=(4, 4))

    for i in range(sb.n):
        s = sb(i)
        wrong_list = [1, 11, 137, 171]

        def is_in_wrong_list(a):
            for w in wrong_list:
                if a == w:
                    return True
            return False

        colors = ['#A04DA3', '#53548A', '#438086']
        loc = [1, 3]

        lg_e, u, du_m, du_p = deconvolute_the_source(s, od[1 + a0], is_in_wrong_list(i))
        p, ers = parabolic_fit(lg_e, u, du_m, du_p)

        lg_e = np.copy(lg_e) - 12
        lg_e_fine = np.linspace(lg_e[0], lg_e[-1], 100)
        rel = abs(ers[0] / p[0])
        if p[0] < 0:
            counter1 += 1
            for j, k in enumerate(ks):
                if rel < 1 / k:
                    counters[j] += 1
            if rel < 1 / 4:
                print(i)
                k = counters[3] - 1
                c = colors[k]
                # print(f"a = {'%.3f' % p[0]} +- {'%.3f' % ers[0]}, rel = {'%.3f' % rel}", )
                print(s.lg_e_ref)
                p3 = plt.errorbar(10 ** (s.lg_e_ref - 12), s.dnde, yerr=[s.dnde_errn, s.dnde_errp], linestyle='',
                                  marker='o', color=c, alpha=0.5)
                p0 = plt.scatter(10 ** lg_e[0], u[0], linestyle='', marker='')
                p1 = plt.errorbar(10 ** lg_e, u, yerr=[du_m, du_p], linestyle='', marker='s', color=c)
                p2 = plt.errorbar(10 ** lg_e_fine, 10 ** fitting_function(lg_e_fine + 12, *p), alpha=.5, color=c,
                                  linestyle='--', linewidth=2, marker='')
                legend1 = plt.legend([p0, p3, p1, p2], [f"{s.title}\nz = {s.z}", "observed", "intrinsic", "fit"],
                                     loc=loc[k], framealpha=0.0)
                plt.gca().add_artist(legend1)

    # print(f"Oh, wow, {counter1} positive curvature sources, and for {counter2} of them"
    #       f" the deviation is more than {m} sigma")

    print(f"total = {counter1}, {counters}")

    plt.xlabel(r"$E$, TeV")
    plt.xscale("log")

    plt.ylabel(r'$dn/dE, \mathrm{TeV^{-1}~cm^{-2}~s^{-1}}$')
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(PICS_DIR, "bad-sources-stevecat.png"), dpi=600, transparent=True)
    plt.show()

    return


def make_a_nice_picture():
    plt.figure(figsize=(4, 4))
    check_one_source([23, 1], True)

    plt.xlim(0.1, 20)
    # plt.legend(fancybox=True, framealpha=0.0)

    plt.tight_layout()
    # plt.savefig(os.path.join(PICS_DIR, "double-stevecat-source-no-attenuation.png"), dpi=600, transparent=True)
    plt.show()
    return


if __name__ == '__main__':
    make_a_nice_picture()
    # calculate_for_sources(1)
    # calculate_for_sources(0)
    # calculate_for_sources(-1)
