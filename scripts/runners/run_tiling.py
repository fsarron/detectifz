#!/usr/bin/env python

import os, sys
import re
import time
#import multiprocessing, logging

import detectifz.setup_data as data
import detectifz.detectifz as detectifz
import detectifz.tiling as tiling

#import detectifz as dfz
#print(dfz.__path__)


import argparse
import configparser

from dataclasses import dataclass
from pydantic import validate_arguments

@validate_arguments
@dataclass
class ConfigTile:
    field: str
    release: str
    datadir: str
    tiles_rootdir: str
    masksfile: str
    maskstype: str

    nMC: int

    ra_colname: str
    dec_colname: str

    ramin: float
    ramax: float
    decmin: float
    decmax: float

    max_area: float
    border_width: float
    pixdeg: float
    
    lgmass_lim: float
    zmin: float
    zmax: float
    conflim_1sig: str
    avg: str
    nprocs: int
    
    z_l68_colname: str
    z_u68_colname: str
    
    zmin_pdf: float
    zmax_pdf: float
    pdz_dz: float

    Mmin_pdf: float
    Mmax_pdf: float
    pdM_dM: float

    selection: str
    fit_Olga: bool
    fitdir_Olga: str
    


parser = argparse.ArgumentParser()
parser.add_argument("-c","--configuration", type=str,
                    help = "Configuration file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()

print('')
print('read config')

config = configparser.ConfigParser()
config.read(args.configuration)
config_tile = ConfigTile(*[config['TILING'][f] for f in list(config['TILING'])])


print('')
print('init tile')
tiles = tiling.Tiles(config_tile=config_tile)

print('')
print('get_tiles')
tiles.get_tiles()

print('')
print('run_tiling')
tiles.run_tiling()

#print('')
#print('get sigz')
#tiles.run_get_sig()

#detect.run()