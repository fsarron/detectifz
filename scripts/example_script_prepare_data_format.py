#!/usr/bin/env python

import os, sys
import re
import time
#import multiprocessing, logging



import sys
import numpy as np
from astropy.table import Table
import scipy.stats
import numba as nb
import h5py
import scipy.interpolate
from numpy.random import RandomState

from pathlib import Path

import detectifz.setup_data as data
import detectifz.detectifz as detectifz

import argparse
import configparser

from dataclasses import dataclass
from pydantic import validate_arguments





@validate_arguments
@dataclass
class ConfigDetectifz:
    detection_type: st

    obsmag_lim: float
    lgmass_comp: float
    lgmass_lim: float
    zmin: float
    zmax: float
    dzmap: float
    pixdeg: float
    Nmc: int
    SNmin: float
    dclean: float
    radnoclus : float
    conflim_1sig: str
    conflim_2sig: str
    lgM_dens: bool
    avg: str
    nprocs: int

    gal_id_colname: str
    ra_colname: str
    dec_colname : str
    obsmag_colname: str
    z_med_colname: str
    z_l68_colname: str
    z_u68_colname: str
    M_med_colname: str
    M_l68_colname: str
    M_u68_colname: str
    
    zmin_pdf: float
    zmax_pdf: float
    pdz_dz: float
    Mmin_pdf: float
    Mmax_pdf: float
    pdM_dM: float
    
    selection: str
    fit_Olga: bool
    fitdir_Olga: str
    
    field: str
    release: str
    rootdir: str

parser = argparse.ArgumentParser()

parser.add_argument("-c","--configuration", type=str,
                    help = "Configuration file")

args = parser.parse_args()
quiet = args.quiet

os.system('pwd')

config = configparser.ConfigParser()
config.read(args.configuration)
config_detectifz = ConfigDetectifz(*[config['DETECTIFz'][f] 
                                     for f in list(config['DETECTIFz'])], 
                                     *[config['GENERAL'][f] for f in list(config['GENERAL'])])


#%% You need 3 things to run witout tiling

#1. Put the detectifz config.ini file in a folder

#2. In that folder put you galaxy FITS Table 
# containing at least the columns indicated in 
# the config.ini file

#3. In that folder put a FITS Table 
# containing samples of p(M, z) or p(z)
# The file should contain at least one column 
# 'lgz_samples'

## If you run in galaxy-density mode, 'lgM_samples' columns
## will be ignored and replaced by ones.

## If you run in stellar mass-density mode, you should have two columns 
# 'lgz_samples' and 'lgM_samples', each containing 
# an array with Nmc values per galaxy : these must be 
# samples of the PDF(M, z).

#%%

@nb.njit()
def rvs(pdf, x, xmin, xmax, size=1):
    cdf = pdf  # in place
    cdf[1:] = np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(x)) #, out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(xmin, x, cdf)
    t_upper = np.interp(xmax, x, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    samples = np.interp(u, cdf, x)
    return samples

@nb.njit(parallel=True)
def sample_tpdf(tpdf_z, zz, size):
    np.random.seed(123456)
    N = len(tpdf_z)
    zmin, zmax = zz.min(), zz.max()
    samples = np.zeros((N, size))
    for i in nb.prange(N):
        samples[i] = rvs(tpdf_z[i], zz[i], zmin, zmax, size=size)
    return samples

ncpus = int(sys.argv[1])

nb.set_num_threads(config_detectifz.nprocs)

### redshift 
table_pdz = Table.read(datadir.joinpath('CEERS_pz_Mz_catalog.fits'))
      
print('convert to array...')
tpdf_z = np.array(table_pdz['p_z'], np.float32)
print('done.')

zz = np.array(table_pdz['Z'], np.float32)

Nmc = 100

samples_z  = sample_tpdf(tpdf_z, zz, Nmc)

t_samples = table_pdz
t_samples['Z_SAMPLES'] = samples_z

#for galaxy-density, put lgM to one for all relaisations of all galaxies
t_samples['lgM_SAMPLES'] = np.ones((len(t_samples), Nmc))

t_samples.keep_columns(['NUMBER',
                        'ALPHA_J2000',
                        'DELTA_J2000', 
                        'Z_SAMPLES',
                        'lgM_SAMPLES'])

t_samples.write(config_detectifz.rootdir+
                '/galaxies.'+config_detectifz.field+
                '.'+str(config_detectifz.Nmc)+'MC.Mz.fits', 
                overwrite=True)

















