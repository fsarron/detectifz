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
    
    def __init__(self, tileid=None, corners=None, border_width=None, field=None, tilesdir=None):
        if not (corners is None): 
            self.id = tileid
            self.corners = corners
            self.border_width = border_width
            self.bottom_left = corners[0][0]
            self.top_left = corners[0][1]
            self.bottom_right = corners[1][0]
            self.top_right = corners[1][1]
            self.field = field
            self.tilesdir = tilesdir
            
            
            self._core_bottom_left = tuple(np.add(corners[0][0] , (border_width, border_width)))
            self._core_top_left = tuple(np.add(corners[0][1] , (border_width, -border_width)))
            self._core_bottom_right =  tuple(np.add(corners[1][0], (-border_width, border_width)))
            self._core_top_right = tuple(np.add(corners[1][1], (-border_width, -border_width)))
            
        else:
            raise TypeError('Could not init Tile object without corners')
            
            
    def run_venice_inout(self):
 
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
        
        
        
    def run_venice_pixelize(self):
        
        (rainf, rasup, 
             decinf, decsup) = (
                self.bottom_left[0], self.top_right[0], 
                self.bottom_left[1], self.top_right[1] )
            
        ##lance venice to get pixelized mask at giuven resolution with given (ra,dec) limits
        process = subprocess.Popen(["venice", 
                                    "-m", 
                                    self.master_masksfile, 
                                    "-nx",
                                    str(int((rasup-rainf)/self.pixdeg)),
                                    "-ny",
                                    str(int((decsup-decinf)/self.pixdeg)),
                                    "-xmin", 
                                    str(rainf),
                                    "-xmax",
                                    str(rasup),
                                    "-ymin",
                                    str(decinf),
                                    "-ymax",
                                    str(decsup),
                                    "-o", 
                                    self.thistile_dir+"/masks."+self.field+".tmp.mat"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        #print("the commandline is {}".format(process.args))
        process.wait()
        result = process.communicate()
        #print(result)
        flag = np.loadtxt(self.thistile_dir+'/masks.'+self.field+'.tmp.mat')
        
        ### from the pixelized matrix, make a fits hdu with WCS information
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [1.0, 1.0]
        w.wcs.cdelt = np.array([self.pixdeg, self.pixdeg])
        w.wcs.crval = [rainf, decinf]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        self.FITSmasks = fits.PrimaryHDU(data=flag.astype(np.int8),header=w.to_header())
        self.FITSmasks_filename = self.thistile_dir+'/masks.'+self.field+'.radec.fits'
        self.FITSmasks.writeto(self.FITSmasks_filename, overwrite=True)

        
    def run_cutout(self):

        (rainf, rasup, 
             decinf, decsup) = (
                self.bottom_left[0], self.top_right[0], 
                self.bottom_left[1], self.top_right[1] )
        
        FITSmasks = fits.open(self.config_tile.masksfile)[0]
        
        ##SKY MASK (BAD REGIONS -- FITS IMAGE)
        racentre = (rainf + rasup)/2.
        deccentre = (decinf + decsup)/2.

        size_sky_ra = (rasup - rainf) * units.deg
        size_sky_dec = (decsup - decinf) * units.deg

        centre_sky = SkyCoord(ra=racentre, 
                                dec=deccentre, unit='deg', frame='icrs')
            
        FITScutout = Cutout2D(data = FITSmasks.data,
                              position = centre_sky,
                              size = (size_sky_dec, size_sky_ra),
                              wcs = wcs.WCS(FITSmasks.header))
            
        self.FITSmasks = FITSmasks
        self.FITSmasks.data = FITScutout.data
        self.FITSmasks.header.update(FITScutout.wcs.to_header())
                    
            
        self.galcat_raw_filename = self.thistile_dir+'/galaxies.'+self.field+'.galcat_raw.fits'
        self.FITSmasks_filename = self.thistile_dir+'/masks.'+self.field+'.radec.fits'
        
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
        border_width = self.config_tile.border_width
        
        total_area = (ramax-ramin)*(decmax-decmin)
        Nmin_tiles =  total_area/max_area
        Nsplit = np.ceil(np.sqrt(Nmin_tiles)).astype(int) + 1 
        
        ras = np.linspace(ramin, ramax, Nsplit)
        rainfs, rasups = ras[:-1]-border_width, ras[1:]+border_width

        decs = np.linspace(decmin, decmax, Nsplit)
        decinfs, decsups = decs[:-1]-border_width, decs[1:]+border_width
        
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
                
                tile = Tile(tileid=l, corners=corners, 
                                       border_width = border_width,
                                       field=self.config_tile.field,
                                      tilesdir = self.config_tile.tilesdir)
                self.tiles.append(tile)

                l += 1
                
                
            
    def run_tiling(self):
        
        galcat_main = Table.read(self.config_tile.galcatfile)

        for i, tile in enumerate(self.tiles):
            (rainf, rasup, 
             decinf, decsup) = (
                tile.bottom_left[0], tile.top_right[0], 
                tile.bottom_left[1], tile.top_right[1])
            
            ##GALAXY CATALOGUE
            maskgal_tile  = ((galcat_main[self.config_tile.ra_colname] > rainf) &
                             (galcat_main[self.config_tile.ra_colname] <= rasup) & 
                             (galcat_main[self.config_tile.dec_colname] > decinf) &
                             (galcat_main[self.config_tile.dec_colname] <= decsup))
            
            tile.galcat = galcat_main[maskgal_tile]
            #tile.galcat_raw_filename = 'tile'
            if self.config_tile.Mz_MC_exists:
                tile.Mz_MC = np.load(self.config_tile.Mz_MC_file)['Mz'][maskgal_tile]

            tile.thistile_dir = tile.tilesdir+'/tile'+'{:04d}'.format(tile.id)
            tile.master_masksfile = self.config_tile.masksfile
            tile.pixdeg = self.config_tile.pixdeg
            
            if not Path(tile.tilesdir).is_dir():
                #process = subprocess.Popen(["mkdir", self.tilesdir])
                #process.wait()
                #result = process.communicate()
                raise OSError('working directory does not exists, '+
                          'you should create it and put'+
                          'a config_detectifz_master.py file inside')
            
            if not Path(tile.thistile_dir).is_dir():
                process = subprocess.Popen(["mkdir", tile.thistile_dir])
                process.wait()
                result = process.communicate()
            
            
            tile.galcat_filename = tile.thistile_dir+"/galaxies."+tile.field+".galcat.fits"
            tile.galcat.write(tile.galcat_filename, overwrite=True) 
            
            if self.config_tile.Mz_MC_exists:
                tile.Mz_MC_filename = ( tile.thistile_dir+"/galaxies."+tile.field+"."+
                                       str(int(self.config_tile.nMC))+"MC.Mz.npz" )

                np.savez(tile.Mz_MC_filename, Mz=tile.Mz_MC)
            
            if self.config_tile.maskstype == 'ds9' :
                tile.run_venice_pixelize()
                
            elif self.config_tile.maskstype == 'fits' :
                tile.run_cutout()
                
            elif self.config_tile.maskstype == 'none' :
                f = open(self.config_tile.rootdir+'/'+self.config_tile.field+'_data/none.reg', "w")
                f.write("# FILTER HSC-G\n")
                f.write("wcs; fk5\n")
                f.write("circle("+str(np.median(tile.galcat[self.config_tile.ra_colname]))+
                        ","+str(np.median(tile.galcat[self.config_tile.dec_colname]))+
                        ",0.00000001d)")
                f.close()
                self.master_masksfile = "none.reg"
                
                tile.run_venice_pixelize()
                
                
    
            #tile.run_venice_inout()
            
            tile.write_config_detectifz()
            
            np.savez(tile.thistile_dir+'/tile_object_init.npz', 
                     tileid=tile.id, 
                     corners=tile.corners, 
                     border_width = tile.border_width,
                     field=tile.field,
                     tilesdir = tile.tilesdir)

            
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
 
