import os.path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from config.settings import ASTRO_SRC_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import OpticalDepth


class SourceBase:
    def __init__(self):
        self.source_base = []
        self.n = 0

        self.import_all_the_sources()

    def import_all_the_sources(self):
        ref_table = Table.read(os.path.join(ASTRO_SRC_DIR, 'table_spectra.csv'))

        for line in ref_table:
            filename = f"{ASTRO_SRC_DIR}/{line['reference']}/{line['file_id']}.ecsv"
            spectrum = Table.read(filename)
            self.source_base.append(spectrum)

        self.n = len(ref_table)
        return

    def get_the_source(self, i: int):
        try:
            return self.source_base[i]
        except IndexError:
            print("Alles kaput")


def calculate_optical_depth(ebl, od):
    return


if __name__ == '__main__':
    sb = SourceBase()
    # ebl = EBLSaldanaLopez()
    # TODO: implement the calculation of absorption on EBL photons
    # print(sb.source_meta)
    # print(sb.source_base)
    source = sb.get_the_source(3)
    print(source.meta['source_name'])
    plt.scatter(source['e_ref'], source['dnde'])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

