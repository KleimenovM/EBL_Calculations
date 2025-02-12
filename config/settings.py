# config/settings.py

from pathlib import Path
import os

GAMMAPY_DATA = '/home/maxkl/Documents/Applications/GammaPy/gammapy-datasets/1.2'

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')

MCMC_DIR = os.path.join(DATA_DIR, 'mcmc_samples')
DATA_SL_DIR = os.path.join(DATA_DIR, 'Saldana-Lopez')
ASTRO_SRC_DIR = os.path.join(DATA_DIR, 'STeVECat')
BS_SAMPLES_DIR = os.path.join(DATA_DIR, 'BSpline_samples')
