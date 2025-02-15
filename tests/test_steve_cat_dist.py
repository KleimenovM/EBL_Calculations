import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.source_base import SourceBase


def test_steve_cat_dist():
    sb = SourceBase()
    z = np.zeros(sb.n)
    for i in range(sb.n):
        z[i] = sb(i).z

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax2 = ax.twinx()
    sns.kdeplot(np.log10(z), ax=ax2, color='orange', fill=False)
    sns.histplot(np.log10(z), ax=ax, bins=25)
    ax.set_yscale('log')
    ax2.set_yscale('log')
    plt.show()
    return


if __name__ == '__main__':
    test_steve_cat_dist()
