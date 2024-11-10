import numpy as np
import matplotlib.pyplot as plt

from src.functional_basis import FunctionalBasis, ParabolicBasis


def test_functional_basis():
    pb_fb: FunctionalBasis = ParabolicBasis(n=10, m=1000)

    lg_wvl, dist = pb_fb.get_distribution_list(m=1000)

    for i in range(pb_fb.n):
        plt.plot(lg_wvl, dist[i])

    plt.plot(lg_wvl, np.sum(dist, axis=0), color='black', linewidth=3)
    plt.show()
    return


if __name__ == '__main__':
    test_functional_basis()
