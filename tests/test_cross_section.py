import numpy as np
import matplotlib.pyplot as plt

from config.constants import SIGMA_TH
from src.cross_section import beta, gamma_gamma_cross_section


ColorsM = ['#CDBACE', '#BB96AC', '#A971AB', '#581A5A', '#300632', '#100012']
ColorsR = ['#f99', '#f22', '#c00', '#800', '#400']
ColorsB = ['#99f', "#22f", "#00c", "#008", "#004"]


def plot_nice():
    e = 1.0  # [eV], optical
    n = 10000

    e0_line = 10 ** np.linspace(11.45, 13, 6)  # [eV] (0.3 - 10 TeV, incident)
    mu_line = np.linspace(-1, 1, n)[:n - 1]  # [DL], cosines
    theta_line = np.arccos(mu_line)
    factor = (1 - mu_line) / 2

    fontsize = 14
    plt.figure(figsize=(7, 7))
    for i, z0 in enumerate([0]):
        colors = [ColorsM, ColorsR, ColorsB]
        for j, e0 in enumerate(e0_line):
            e0 *= (1 + z0)
            sigmas = gamma_gamma_cross_section(e0, e, mu_line) / SIGMA_TH
            # plt.polar(theta_line, sigmas, color=colors[i][j], label=f"{round(e0 * 1e-12, 1)} TeV")
            # plt.polar(-theta_line, sigmas, color=colors[i][j])
            plt.polar(theta_line, factor * sigmas, color=colors[i][j], label=f"E = {round(e0 * 1e-12, 1)} TeV")
            plt.polar(-theta_line, factor * sigmas, color=colors[i][j])
            plt.legend(loc=4, fontsize=fontsize)
            plt.tick_params(labelsize=fontsize)

    plt.tight_layout()
    # plt.savefig("../pics/cross-sections.png")
    # plt.savefig("../pics/cross-sections.pdf")
    plt.show()
    return


def test_cross_section():
    e = 1.0  # [eV], optical
    n = 10000

    e0_line = 10**np.linspace(11.45, 13, 5)  # [eV] (0.3 - 10 TeV, incident)
    mu_line = np.linspace(-1, 1, n)[:n-1]  # [DL], cosines
    theta_line = np.arccos(mu_line)
    factor = (1 - mu_line) / 2

    fontsize = 12
    plt.figure(figsize=(15, 10))
    for i, z0 in enumerate([0, 0.1, 0.5]):
        for j, e0 in enumerate(e0_line):
            sigmas = gamma_gamma_cross_section(e0, e, z0, mu_line)
            plt.subplot(2, 3, i + 1, projection='polar')
            plt.title(f"z0 = {z0}, pure")
            plt.polar(theta_line, sigmas, color=ColorsR[j], label=f"{round(e0 * 1e-12, 1)} TeV")
            plt.polar(-theta_line, sigmas, color=ColorsR[j])
            plt.legend(loc=4, fontsize=fontsize)
            plt.subplot(2, 3, i + 4, projection='polar')
            plt.polar(theta_line, factor * sigmas, color=ColorsR[j], label=f"{round(e0 * 1e-12, 1)} TeV")
            plt.polar(-theta_line, factor * sigmas, color=ColorsR[j])
            plt.legend(loc=4)
            plt.title(f"z0 = {z0}, transport")

    plt.tight_layout()
    plt.savefig("../pics/cross-sections.png")
    plt.savefig("../pics/cross-sections.pdf")
    plt.show()

    return


if __name__ == '__main__':
    plot_nice()
    # test_cross_section()
