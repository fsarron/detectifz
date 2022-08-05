import sys
import numpy as np
from astropy.table import Table

from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units

from astropy.io import fits
from astropy import wcs

import subprocess
from pathlib import Path


class Tile(object):
    
    def __init__(self, tileid=None, corners=None, field=None, tilesdir=None):
        if not (corners is None): 
            self.id = tileid
            self.bottom_left = corners[0][0]
            self.top_left = corners[0][1]
            self.bottom_right = corners[1][0]
            self.top_right = corners[1][1]
            self.field = field
            self.tilesdir = tilesdir
        else:
            raise TypeError('Could not init Tile object without corners')
            
            
    def run_venice(self):
        
        if not Path(self.tilesdir).is_dir():
            #process = subprocess.Popen(["mkdir", self.tilesdir])
            #process.wait()
            #result = process.communicate()
            raise OSError('working directory does not exists, '+
                          'you should create it and put'+
                          'a config_detectifz_master.py file inside')
            
        if not Path(self.thistile_dir).is_dir():
            process = subprocess.Popen(["mkdir", self.thistile_dir])
            process.wait()
            result = process.communicate()

            
        self.FITSmasks.writeto(self.FITSmasks_filename, overwrite=True)

        coords = SkyCoord(ra = self.galcat_raw['ra'],
                 dec = self.galcat_raw['dec'], unit = 'deg', frame='fk5')
        pix = wcs.utils.skycoord_to_pixel(coords, wcs = wcs.WCS(self.FITSmasks.header))
        self.galcat_raw['xpix_mask'] = pix[0]
        self.galcat_raw['ypix_mask'] = pix[1]

        self.galcat_raw.write(self.galcat_raw_filename, overwrite=True)
        
                
        process = subprocess.Popen(["venice", "-cat", 
                                    self.galcat_raw_filename, 
                                    "-catfmt", 
                                    "fits", 
                                    "-xcol", 
                                    "xpix_mask", 
                                    "-ycol", 
                                    "ypix_mask", 
                                    "-m", 
                                    self.FITSmasks_filename, 
                                    "-o", 
                                    self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        process.wait()
        result = process.communicate()
        
        
        galcat = Table.read(self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits")
        self.galcat = galcat[galcat['flag'] == 1]
        self.galcat_filename = self.thistile_dir+"/galaxies."+self.field+".galcat.fits"
        self.galcat.write(self.galcat_filename, overwrite=True) 
        
        process = subprocess.Popen(["rm", self.thistile_dir+"/galcat_mask."+self.field+".tmp.fits"])
        process.wait()
        result = process.communicate()
        
        
    def write_config_detectifz(self):
        
        f=open(self.tilesdir+'/config_master_detectifz.py','r')
        self.config = f.readlines()
        f.close()
        self.config.append("\n")
        self.config.append("field='"+self.field+"'\n")
        self.config.append("rootdir='"+self.thistile_dir+"'\n")
        
        with open(self.thistile_dir+'/config_detectifz.py', 'w') as f:
            for line in self.config:
                f.write(line)
            
            
class Tiles(object):

    def __init__(self, config_tile=None):
        self.config_tile = config_tile
        if self.config_tile is None:
            raise TypeError('Could not init Tiles object without config file')   
            
    def get_tiles(self):
        ramin, ramax, decmin, decmax = self.config_tile.sky_lims
        max_area = self.config_tile.max_area
        borderwidth = self.config_tile.border_width
        
        total_area = (ramax-ramin)*(decmax-decmin)
        Nmin_tiles =  total_area/max_area
        Nsplit = np.ceil(np.sqrt(Nmin_tiles)).astype(int) + 1 
        
        ras = np.linspace(ramin, ramax, Nsplit)
        rainfs, rasups = ras[:-1]-borderwidth, ras[1:]+borderwidth

        decs = np.linspace(decmin, decmax, Nsplit)
        decinfs, decsups = decs[:-1]-borderwidth, decs[1:]+borderwidth
        
        c00 = np.meshgrid(rainfs, decinfs)
        c01 = np.meshgrid(rainfs, decsups)
        c10 = np.meshgrid(rasups, decinfs)
        c11 = np.meshgrid(rasups, decsups)
        
        self.tiles = [] #np.empty((Nsplit-1,Nsplit-1), dtype='object')
        l = 0

        for i in range(Nsplit-1):
            for j in range(Nsplit-1):
                corners = [[(c00[0][i,j], c00[1][i,j]) , (c01[0][i,j], c01[1][i,j])],
                           [(c10[0][i,j], c10[1][i,j]), (c11[0][i,j], c11[1][i,j])]]
                
                self.tiles.append(Tile(tileid=l, corners=corners, 
                                       field=self.config_tile.field,
                                      tilesdir = self.config_tile.tilesdir))

                l += 1
            
    def run_tiling(self):
        
        galcat_main = Table.read(self.config_tile.galcatfile)

        for i, tile in enumerate(self.tiles):
            FITSmasks = fits.open(self.config_tile.FITSmasksfile)[0]
            (rainf, rasup, 
             decinf, decsup) = (
                tile.bottom_left[0], tile.top_right[0], 
                tile.bottom_left[1], tile.top_right[1])
            
            ##GALAXY CATALOGUE
            maskgal_tile  = ((galcat_main[self.config_tile.ra_colname] > rainf) &
                             (galcat_main[self.config_tile.ra_colname] < rasup) & 
                             (galcat_main[self.config_tile.dec_colname] > decinf) &
                             (galcat_main[self.config_tile.dec_colname] < decsup))
            
            tile.galcat_raw = galcat_main[maskgal_tile]
            #tile.galcat_raw_filename = 'tile'

    
            ##SKY MASK (BAD REGIONS -- FITS IMAGE)
            racentre = (rainf + rasup)/2
            deccentre = (decinf + decsup)/2

            size_sky_ra = (rasup - rainf) * units.deg
            size_sky_dec = (decsup - decinf) * units.deg

            centre_sky = SkyCoord(ra=racentre, 
                                  dec=deccentre, unit='deg', frame='icrs')
            
            FITScutout = Cutout2D(data = FITSmasks.data,
                              position = centre_sky,
                              size = (size_sky_dec, size_sky_ra),
                              wcs = wcs.WCS(FITSmasks.header))
            
            tile.FITSmasks = FITSmasks
            tile.FITSmasks.data = FITScutout.data
            tile.FITSmasks.header.update(FITScutout.wcs.to_header())
            
            tile.thistile_dir = tile.tilesdir+'/tile'+'{:04d}'.format(tile.id)
            
            
            tile.galcat_raw_filename = tile.thistile_dir+'/galaxies.'+tile.field+'.galcat_raw.fits'
            tile.FITSmasks_filename = tile.thistile_dir+'/masks.'+tile.field+'.fits'
            
            tile.run_venice()
            
            tile.write_config_detectifz()
            
            
#        
#        ###DETECTIFz CONFIG FILE
#
#    
#    #def save_tiling(self):
#            
#class Tiles(object):
#    
#    def __init__(self, field='UDS', params=None):
#        self.field = field
#        self.data = data
#        self.params = params
 
