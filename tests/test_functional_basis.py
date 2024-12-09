import numpy as np
import matplotlib.pyplot as plt

from src.functional_basis import FunctionalBasis, ExpParabolicBasis, BSplineBasis


def test_b_spline_box():
    b_fb: BSplineBasis = BSplineBasis(n=10, m=1000)
    lg_wvl = b_fb.get_lg_wvl_range()
    b1 = b_fb.box(lg_wvl, 6)

    plt.plot((lg_wvl - lg_wvl[0]) / b_fb.h, b1)
    plt.show()
    return


def test_b_spline_basis_function():
    b_fb: BSplineBasis = BSplineBasis(n=10, m=1000)

    lg_wvl = b_fb.get_lg_wvl_range()
    dist = b_fb.b_spline_basis_function(lg_wvl, 6)
    plt.plot((lg_wvl - lg_wvl[0]) / b_fb.h, dist)
    plt.grid(True)
    plt.show()

    return


def test_a_basis(basistype, a, ifplot: bool = True):
    n = a.size
    fb: FunctionalBasis = basistype(n=n, m=1000)

    lg_wvl, dist = fb.get_distribution_list(m=1000)

    for i in range(fb.n):
        plt.plot(lg_wvl, a[i] * dist[i])

    plt.plot(lg_wvl, a @ dist, color='black', linewidth=3)
    plt.grid(True)
    if ifplot:
        plt.show()
    return


if __name__ == '__main__':
    # test_b_spline_box()
    # test_b_spline_basis_function()
    plt.figure(figsize=(10, 6))

    a_vector = np.random.rand(20)

    plt.subplot(1, 2, 1)
    plt.title("ExpParabolic Basis")
    test_a_basis(ExpParabolicBasis, a_vector, ifplot=False)
    plt.subplot(1, 2, 2)
    plt.title("B-Spline Basis")
    test_a_basis(BSplineBasis, a_vector, ifplot=False)
    plt.tight_layout()
    plt.show()
