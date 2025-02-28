#!/usr/bin/env python

import os, sys
import re
import time
#import multiprocessing, logging

import detectifz.setup_data as data
import detectifz.detectifz as detectifz

#import detectifz as dfz
#print(dfz.__path__)

import argparse
import configparser

from dataclasses import dataclass
from pydantic import validate_arguments

@validate_arguments
@dataclass
class ConfigDetectifz:
    detection_type: str
    use_mass_density: bool

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
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")

args = parser.parse_args()
quiet = args.quiet

os.system('pwd')


    
#config.t = args.t
#print(args.tile)


config = configparser.ConfigParser()
config.read(args.configuration)
config_detectifz = ConfigDetectifz(*[config['DETECTIFz'][f] 
                                     for f in list(config['DETECTIFz'])], 
                                     *[config['GENERAL'][f] for f in list(config['GENERAL'])])

print('')    
print('PREPARE DATA')
d = data.Data(config=config_detectifz, tile_id='none')
print('')


print('SET UP DETECTIFz')
detect = detectifz.DETECTIFz(config_detectifz.field, data=d, config=config_detectifz)
detect.run_main()


## curently broken functions 
## (worked on the legacy 2021 version, but you need grids of P(M, z) for each gal
## (please contact the author if you have such data and need to make it work)
#
# Hopefully a next version will make them work from MC samples of P(M, z).
# 
#detect.run_R200()  
#detect.run_Pmem()