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
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", type=str,
                    help = "Config file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet

config_root = re.split(".py", args.config)[0]
if os.path.isfile(config_root+".pyc"):
    os.remove(config_root+".pyc")

import importlib
try:
    config = importlib.import_module(config_root)
    print('Successfully loaded "{0}" as config'.format(args.config))
    #reload(params)
except:
    print('Failed to load "{0}" as config'.format(args.config))
    raise
    

tiles = tiling.Tiles(config_tile=config)
tiles.get_tiles()
tiles.run_tiling()
#detect.run()