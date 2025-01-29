# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')
MCMC_DIR = os.path.join(ROOT_DIR, 'mcmc_samples')

DATA_SL_DIR = os.path.join(DATA_DIR, 'Saldana-Lopez')

ASTRO_SRC_DIR = os.path.join(DATA_DIR, 'STeVECat')
