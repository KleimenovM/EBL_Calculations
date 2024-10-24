import numpy as np
import matplotlib.pyplot as plt

from src.cross_section import beta, gamma_gamma_cross_section


def test_cross_section():
    e = 1.0  # [eV], optical
    n = 10000

    e0_line = 10**np.linspace(11.45, 13, 5)  # [eV] (0.3 - 10 TeV, incident)
    mu_line = np.linspace(-1, 1, n)[:n-1]  # [DL], cosines
    theta_line = np.arccos(mu_line)
    factor = (1 - mu_line) / 2

    colors = ['#f99', '#f22', '#c00', '#800', '#000']

    plt.figure(figsize=(15, 10))
    for i, z0 in enumerate([0, 0.1, 0.5]):
        for j, e0 in enumerate(e0_line):
            sigmas = gamma_gamma_cross_section(e0, e, z0, mu_line)
            plt.subplot(2, 3, i + 1, projection='polar')
            plt.title(f"z0 = {z0}, pure")
            plt.polar(theta_line, sigmas, color=colors[j], label=f"{round(e0 * 1e-12, 1)} TeV")
            plt.polar(-theta_line, sigmas, color=colors[j])
            plt.legend(loc=4)
            plt.subplot(2, 3, i + 4, projection='polar')
            plt.polar(theta_line, factor * sigmas, color=colors[j], label=f"{round(e0 * 1e-12, 1)} TeV")
            plt.polar(-theta_line, factor * sigmas, color=colors[j])
            plt.legend(loc=4)
            plt.title(f"z0 = {z0}, transport")

    plt.tight_layout()
    plt.savefig("../pics/cross-sections.png")
    plt.savefig("../pics/cross-sections.pdf")
    plt.show()

    return


if __name__ == '__main__':
    test_cross_section()
