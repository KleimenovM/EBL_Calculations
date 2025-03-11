from pathlib import Path

# Check package versions
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from gammapy.visualization import plot_spectrum_datasets_off_regions
from regions import CircleSkyRegion

# %matplotlib inline
import matplotlib.pyplot as plt

from IPython.display import display
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.estimators.utils import resample_energy_edges
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    create_crab_spectral_model,
)

from config.settings import GAMMAPY_DATA


# ---------
# LOAD DATA
# ---------

datastore = DataStore.from_dir(f"{GAMMAPY_DATA}/hess-dl3-dr1/")
obs_ids = [23523, 23526, 23559, 23592]
observations = datastore.get_observations(obs_ids)


# ---------
# DEFINE TARGET REGION
# ---------

target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# ---------
# CREATE EXCLUSION MASK
# ---------

exclusion_region = CircleSkyRegion(
    center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
    radius=0.5 * u.deg,
)

skydir = target_position.galactic
geom = WcsGeom.create(npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs")

exclusion_mask = ~geom.region_mask([exclusion_region])
exclusion_mask.plot()
plt.show()


# ---------
# RUN DATA REDUCTION CHAIN
# ---------

energy_axis = MapAxis.from_energy_bounds(
    0.1, 40, nbin=10, per_decade=True, unit="TeV", name="energy"
)
energy_axis_true = MapAxis.from_energy_bounds(
    0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

datasets = Datasets()

for obs_id, observation in zip(obs_ids, observations):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_maker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)

print(datasets)


# ---------
# PLOT OFF REGIONS
# ---------

plt.figure()
ax = exclusion_mask.plot()
on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
plt.show()


# --------
# SOURCE STATISTICS
# --------

info_table = datasets.info_table(cumulative=True)

display(info_table)

# And make the corresponding plots

fig, (ax_excess, ax_sqrt_ts) = plt.subplots(figsize=(10, 4), ncols=2, nrows=1)
ax_excess.plot(
    info_table["livetime"].to("h"),
    info_table["excess"],
    marker="o",
    ls="none",
)

ax_excess.set_title("Excess")
ax_excess.set_xlabel("Livetime [h]")
ax_excess.set_ylabel("Excess events")

ax_sqrt_ts.plot(
    info_table["livetime"].to("h"),
    info_table["sqrt_ts"],
    marker="o",
    ls="none",
)

ax_sqrt_ts.set_title("Sqrt(TS)")
ax_sqrt_ts.set_xlabel("Livetime [h]")
ax_sqrt_ts.set_ylabel("Sqrt(TS)")
plt.show()


# --------
# FIT SPECTRUM
# --------

spectral_model = ExpCutoffPowerLawSpectralModel(
    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
    index=2,
    lambda_=0.1 * u.Unit("TeV-1"),
    reference=1 * u.TeV,
)
model = SkyModel(spectral_model=spectral_model, name="crab")

datasets.models = [model]

fit_joint = Fit()
result_joint = fit_joint.run(datasets=datasets)

# we make a copy here to compare it later
model_best_joint = model.copy()


# --------
# FIT QUALITY AND MODEL RESIDUALS
# --------

# We can access the results dictionary to see if the fit converged:
print(result_joint)

# and check the best-fit parameters
display(result_joint.models.to_parameters_table())

# A simple way to inspect the model residuals is using the function plot_fit()
ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)
plt.show()

# For more ways of assessing fit quality, please refer to the dedicated Fitting tutorial.


# --------
# COMPUTE FLUX POINTS
# --------

fpe = FluxPointsEstimator(
    energy_edges=energy_axis.edges, source="crab", selection_optional="all"
)
flux_points = fpe.run(datasets=datasets)

display(flux_points.to_table(sed_type="dnde", formatted=True))

fig, ax = plt.subplots()
flux_points.plot(ax=ax, sed_type="e2dnde", color="darkorange")
flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde")
ax.set_xlim(0.6, 40)
plt.show()

flux_points_dataset = FluxPointsDataset(
    data=flux_points, models=model_best_joint.copy()
)
ax, _ = flux_points_dataset.plot_fit()
ax.set_xlim(0.6, 40)
plt.show()


