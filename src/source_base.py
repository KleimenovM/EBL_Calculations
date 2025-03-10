import os.path
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from config.settings import ASTRO_SRC_DIR
from src.ebl_photon_density import EBLSaldanaLopez
from src.optical_depth import load_basis_optical_depth, OpticalDepthInterpolator, OpticalDepth


class Source:
    def __init__(self, title: str, z: float,
                 e_ref: np.ndarray, dnde: np.ndarray,
                 dnde_errn: np.ndarray, dnde_errp: np.ndarray,
                 od_sl: OpticalDepthInterpolator = None):
        self.title: str = title
        self.z: float = z
        self.e_ref: np.ndarray = e_ref * 1e12  # TeV -> eV
        self.lg_e_ref: np.ndarray = np.log10(self.e_ref)  # lg(e/eV)
        self.dnde: np.ndarray = dnde
        self.n = self.dnde.shape[0]
        self.dnde_errn: np.ndarray = dnde_errn
        self.dnde_errp: np.ndarray = dnde_errp

        self.e0 = np.sqrt(self.e_ref[0] * self.e_ref[-1])
        self.phi0 = np.sqrt(self.dnde[0] * self.dnde[-1])
        if od_sl is not None:
            self.phi0 *= np.exp(0.5 * (od_sl.get(z, self.lg_e_ref[0]) + od_sl.get(z, self.lg_e_ref[-1])))

    def plot_spectrum(self, ax, if_show: bool = False, if_scale: bool = False, xscale: float = 1.0):
        scale = 1.0
        if if_scale:
            scale = self.e_ref**2
        ax.errorbar(self.e_ref * xscale, scale * self.dnde,
                     yerr=[scale * self.dnde_errn, scale * self.dnde_errp],
                     linestyle='', marker='o', color='red', label=f'measured')
        ax.plot(self.e_ref * xscale, self.phi0 * scale * (self.e_ref / self.e0)**(-2),
                 color='black', linestyle='--', label=r'$\gamma=-2$ spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        mod = 1.1
        ax.set_xlim(self.e_ref[0] * xscale / mod, self.e_ref[-1] * xscale * mod)
        if if_show:
            plt.show()
        return


class SourceBase:
    def __init__(self, if_min_evt: bool = True, min_evt: int = 3,
                 if_max_rsh: bool = False, max_rsh: float = 1.0):
        self.source_base: list[Source] = []
        self.n = 0

        # selection criteria
        self.if_min_evt = if_min_evt
        self.minimum_number_of_events = min_evt

        self.if_max_rsh = if_max_rsh
        self.max_rsh = max_rsh

        self.import_all_the_sources()

    def event_number_criterion(self, source) -> bool:
        if self.if_min_evt:
            return source.n > self.minimum_number_of_events
        return True

    def rsh_criterion(self, source: Source) -> bool:
        if self.if_min_evt:
            return source.z < self.max_rsh
        return True

    def import_all_the_sources(self):
        ref_table = Table.read(os.path.join(ASTRO_SRC_DIR, 'table_spectra.csv'))

        odi = OpticalDepthInterpolator(optical_depth=OpticalDepth(ebl=EBLSaldanaLopez(), series_expansion=True))

        for line in ref_table:
            filename = f"{ASTRO_SRC_DIR}/{line['reference']}/{line['file_id']}.ecsv"
            spectrum = Table.read(filename)
            # set a source
            z = line['redshift']
            e_ref: np.ndarray = spectrum['e_ref'].data
            dnde: np.ndarray = spectrum['dnde'].data
            real_values: np.ndarray[bool] = ~np.isnan(e_ref) * ~np.isnan(dnde)
            source = Source(title=spectrum.meta['source_name'],
                            z=z, e_ref=e_ref[real_values],
                            dnde=spectrum['dnde'][real_values].data,
                            dnde_errn=spectrum['dnde_errn'][real_values].data,
                            dnde_errp=spectrum['dnde_errp'][real_values].data,
                            od_sl=odi)
            if self.event_number_criterion(source) and self.rsh_criterion(source):
                self.source_base.append(source)

        self.n = len(self.source_base)
        return

    def __call__(self, i: int):
        try:
            return self.source_base[i]
        except IndexError:
            print("Wrong index! You either went too far, or entered something weird.")


if __name__ == '__main__':
    print("Not for direct use")
