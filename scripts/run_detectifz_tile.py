import os, sys
import re
import time
#import multiprocessing, logging

import detectifz.setup_data as data
import detectifz.detectifz as detectifz

#import detectifz as dfz
#print(dfz.__path__)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config", type=str,
                    help = "Configuration file")
parser.add_argument("-t", "--tile", type=str,
                    help = "tile id")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")

args = parser.parse_args()
quiet = args.quiet

os.system('pwd')


config_root = re.split(".py", args.config)[0]
if os.path.isfile(config_root+".pyc"):
    os.remove(config_root+".pyc")

import importlib
try:
    config = importlib.import_module(config_root)
    print('Successfully loaded "{0}" as params'.format(args.config))
    #reload(params)
except:
    print('Failed to load "{0}" as params'.format(args.config))
    raise
    
#config.t = args.t
print(args.tile)
    
print('')    
print('PREPARE DATA')
d = data.Data(config=config, tile_id=args.tile)
print('')

print('SET UP DETECTIFz')
detect = detectifz.DETECTIFz(config.field, data=d, config=config)
detect.run()